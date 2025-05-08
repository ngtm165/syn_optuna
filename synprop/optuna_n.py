# finetune_optuna_limited.py
# Chỉ sửa đổi file này, giữ nguyên gine.py và model.py gốc
import copy
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import optuna # Import Optuna
import shutil # Để dọn dẹp file log tạm

# --- Giả định cấu trúc thư mục gốc và import ---
root_dir=Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
os.chdir(str(root_dir))

from synprop.model import model, train, inference # Import bản gốc
from synprop.data_wrapper_7 import data_wrapper_7


# --- Hàm đọc best_val_loss từ file log ---
def get_best_val_loss_from_log(log_path):
    """Đọc file log JSON và trả về val_loss nhỏ nhất."""
    min_val_loss = float('inf')
    try:
        if not os.path.exists(log_path):
            print(f"Warning: Log file not found at {log_path}")
            return min_val_loss # Trả về vô cùng nếu file không tồn tại

        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    # Kiểm tra xem có key 'val_loss' và giá trị hợp lệ không
                    if 'val_loss' in log_entry and isinstance(log_entry['val_loss'], (int, float)):
                        min_val_loss = min(min_val_loss, log_entry['val_loss'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {log_path}: {line.strip()}")
                    continue
                except KeyError:
                     # Bỏ qua nếu dòng không có key 'val_loss'
                     continue

        if min_val_loss == float('inf'):
             print(f"Warning: No valid 'val_loss' found in {log_path}")

        return min_val_loss

    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        return float('inf') # Trả về vô cùng nếu có lỗi đọc file

# --- Hàm Objective cho Optuna ---
def objective(trial, args):
    """Hàm mục tiêu cho Optuna (phiên bản giới hạn)."""

    # --- 1. Đề xuất siêu tham số có thể điều chỉnh ---
    # Lưu ý: Chỉ có thể tune các tham số mà model() và data_wrapper_7() nhận vào.
    predict_hidden_feats = trial.suggest_int("predict_hidden_feats", 128, 1024, step=64)
    # Sử dụng đúng tên tham số 'drop_ratio' như trong model.__init__ gốc
    drop_ratio = trial.suggest_float("drop_ratio", 0.0, 0.5, step=0.05)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16) # Ví dụ tune batch_size
    # lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # depth = trial.suggest_int("depth", 2, 6) # Số lớp / bước lặp T
    node_hid_feats = trial.suggest_categorical("node_hid_feats", [128, 256, 300, 512])
    # edge_hid = trial.suggest_categorical("edge_hid_feats", [128, 256, 300, 512])
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    readout_option = trial.suggest_categorical("readout_option", [True, False])
    # readout_feats = trial.suggest_int("readout_feats", 512, 2048, step=512) if readout_option else node_hid_feats # Kích thước readout phụ thuộc optio

    
    # Tạo đường dẫn file log tạm thời cho trial này
    # Điều này QUAN TRỌNG để tránh xung đột khi đọc/ghi log giữa các trial
    temp_log_dir = os.path.join(args.monitor_folder, "optuna_temp_logs")
    os.makedirs(temp_log_dir, exist_ok=True)
    temp_log_path = os.path.join(temp_log_dir, f"trial_{trial.number}_monitor.json")

    # Tạo một bản sao của args để sửa đổi đường dẫn log cho trial này
    # trial_args = args.copy()
    # Tạm thời ghi đè đường dẫn log để hàm train gốc ghi vào file tạm
    # Giả sử hàm train sử dụng args.monitor_folder + args.monitor_name
    # Chúng ta cần đảm bảo đường dẫn cuối cùng là temp_log_path
    args.monitor_folder = temp_log_dir + "/" # Đảm bảo có dấu /
    args.monitor_name = f"trial_{trial.number}_monitor.json"

    # --- 2. Thiết lập môi trường ---
    device = (
        torch.device(f"cuda:{args.device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"--- Trial {trial.number}: Using device {device} ---")
    seed = args.seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed) # Đặt seed cho numpy nữa (quan trọng cho data split)


    # --- 3. Tải và chuẩn bị dữ liệu ---
    print(f"--- Trial {trial.number}: Loading data with batch_size={batch_size}... ---")
    try:
        # Sử dụng batch_size đã được sample
        data_loader = data_wrapper_7(args.data_path, args.graph_path, args.y_column, batch_size, 4, 0.1, 0.1)
        train_loader, val_loader, _ = data_loader.get_data_loaders()
        node_attr = train_loader.dataset[0].x.shape[1]
        edge_attr = train_loader.dataset[0].edge_attr.shape[1]
    except Exception as e:
        print(f"Error loading data in trial {trial.number}: {e}")
        return float('inf')

    # --- 4. Khởi tạo mô hình GỐC với siêu tham số đã sample ---
    try:
        # Chỉ truyền các tham số mà model() gốc chấp nhận
        net = model(
            node_feat=node_attr,
            edge_feat=edge_attr,
            out_dim=1,
            predict_hidden_feats=predict_hidden_feats,
            drop_ratio=drop_ratio,
            # lr=lr, 
            # depth=depth,
            node_hid_feats=node_hid_feats,
            # edge_hid = edge_hid_feats,
            # weight_decay=weight_decay,
            readout_option=readout_option,
            # readout_feats=readout_feats,

        ).to(device)
    except Exception as e:
         print(f"Error initializing model in trial {trial.number}: {e}")
         return float('inf')

    # --- 5. Huấn luyện mô hình sử dụng hàm train GỐC ---
    print(f"--- Trial {trial.number}: Starting training with params: {trial.params} ---")
    try:
        # Gọi hàm train gốc. Nó sẽ tự tạo optimizer với lr/wd cố định
        # và ghi log vào đường dẫn trong trial_args (tức là temp_log_path)
        # Hàm train gốc không trả về gì (hoặc trả về model đã train)
        train(
            trial, # Sử dụng args đã sửa đổi đường dẫn log trial_args, trial
            net,
            train_loader,
            val_loader,
            model_path="temp_model.pth", # Đường dẫn này không quan trọng lắm vì ta đọc log
            device=device,
            # lr=lr, ##mới thêm 
            # depth=depth, ##mới thêm
            epochs=args.epochs
            # Không thể truyền lr, weight_decay
            # Không thể bật save_model=False
            # Không thể dùng pruning của Optuna vì train gốc không hỗ trợ
        )
    except Exception as e:
        print(f"Error during training in trial {trial.number}: {e}")
        # Xóa file log tạm nếu có lỗi để tránh đọc nhầm
        if os.path.exists(temp_log_path):
            os.remove(temp_log_path)
        return float('inf')

    # --- 6. Đọc kết quả từ file log tạm ---
    print(f"--- Trial {trial.number}: Reading results from log file: {temp_log_path} ---")
    best_val_loss = get_best_val_loss_from_log(temp_log_path)

    # --- 7. Dọn dẹp file log tạm (tùy chọn) ---
    # try:
    #     if os.path.exists(temp_log_path):
    #         os.remove(temp_log_path)
    # except OSError as e:
    #     print(f"Warning: Could not remove temporary log file {temp_log_path}: {e}")

    # --- 8. Trả về chỉ số cần tối ưu ---
    print(f"--- Trial {trial.number}: Finished. Best Validation Loss from log: {best_val_loss:.4f} ---")
    # Kiểm tra nếu không đọc được loss hợp lệ
    if best_val_loss == float('inf'):
         print(f"Warning: Trial {trial.number} did not produce a valid validation loss.")
         # Optuna sẽ coi trial này là thất bại nếu trả về vô cùng

    return best_val_loss


# --- Hàm chính để chạy finetune với Optuna (phiên bản giới hạn) ---
def finetune_with_optuna_limited(args):
    """Hàm chính để thiết lập và chạy Optuna study (phiên bản giới hạn)."""

    # Tạo study object, chỉ định hướng tối ưu (minimize loss)
    study = optuna.create_study(direction="minimize")

    # Bắt đầu quá trình tối ưu
    try:
        # Sử dụng args.copy() để đảm bảo mỗi trial nhận bản sao độc lập
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("Optimization stopped by user.")
    finally:
        # Dọn dẹp thư mục log tạm sau khi study kết thúc hoặc bị ngắt
        temp_log_dir = os.path.join(args.monitor_folder, "optuna_temp_logs")
        if os.path.exists(temp_log_dir):
            print(f"Cleaning up temporary log directory: {temp_log_dir}")
            shutil.rmtree(temp_log_dir, ignore_errors=True)


    # --- In kết quả ---
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n" + "="*20 + " Optuna Study Statistics " + "="*20)
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    print(f"  Number of fail trials: {len(fail_trials)}")

    if complete_trials and study.best_trial: # Kiểm tra xem có best_trial không
        best_trial = study.best_trial
        print("\n" + "="*20 + " Best Trial Summary " + "="*20)
        print(f"  Value (Best Validation Loss from logs): {best_trial.value:.4f}")
        print("  Best Parameters Found (Limited Scope): ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # --- Huấn luyện lại mô hình cuối cùng với tham số tốt nhất và đánh giá trên test set ---
        print("\n" + "="*20 + " Training Final Model with Best Params " + "="*20)

        # Lấy các tham số tốt nhất (chỉ những cái đã tune)
        best_params = best_trial.params
        device = (
            torch.device(f"cuda:{args.device}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        seed = args.seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        else:
             torch.manual_seed(seed)
        np.random.seed(seed)

        # Tải lại dữ liệu (sử dụng batch_size tốt nhất)
        print(f"Reloading data for final run with batch_size={best_params['batch_size']}...")
        try:
            final_batch_size = best_params['batch_size']
            data_loader = data_wrapper_7(args.data_path, args.graph_path, args.y_column, final_batch_size, 4, 0.1, 0.1)
            train_loader, val_loader, test_loader = data_loader.get_data_loaders()
            node_attr = train_loader.dataset[0].x.shape[1]
            edge_attr = train_loader.dataset[0].edge_attr.shape[1]
        except Exception as e:
            print(f"Error reloading data for final run: {e}")
            return

        # Khởi tạo model gốc với tham số tốt nhất (chỉ những cái đã tune)
        try:
            final_net = model(
                node_feat=node_attr,
                edge_feat=edge_attr,
                out_dim=1,
                predict_hidden_feats=best_params["predict_hidden_feats"],
                drop_ratio=best_params["drop_ratio"],
                # lr=best_params['lr'], 
                # depth=best_params['depth'],
                node_hid_feats=best_params['node_hid_feats'],
                # edge_hid=best_params['edge_hid_feats'],
                # weight_decay=best_params['weight_decay'], ##decayo
                readout_option=best_params['readout_option'],
                # readout_feats=best_params['readout_feats'],

            ).to(device)
        except Exception as e:
            print(f"Error initializing final model: {e}")
            return

        # Huấn luyện lần cuối bằng hàm train gốc
        # Lần này sẽ ghi log và lưu model vào đường dẫn cuối cùng trong args
        print("Starting final training...")
        final_model_path = args.model_path + args.model_name # Đường dẫn lưu model cuối cùng
        final_monitor_path = args.monitor_folder + args.monitor_name # Đường dẫn log cuối cùng

        # Xóa file log cuối cùng cũ nếu tồn tại để bắt đầu log mới
        if os.path.exists(final_monitor_path):
            try:
                os.remove(final_monitor_path)
                print(f"Removed old final log file: {final_monitor_path}")
            except OSError as e:
                print(f"Warning: Could not remove old final log file {final_monitor_path}: {e}")


        try:
            # Gọi hàm train gốc, nó sẽ lưu model tốt nhất dựa trên val loss nội bộ của nó
            train(
                args, # Sử dụng args gốc với đường dẫn cuối cùng
                final_net,
                train_loader,
                val_loader,
                model_path=final_model_path, # Đường dẫn lưu model cuối cùng
                device=device,
                epochs=args.epochs # Số epochs cho final training
            )
        except Exception as e:
             print(f"Error during final training: {e}")
             return

        # Đánh giá trên tập test với model tốt nhất vừa được lưu bởi hàm train gốc
        print("\n" + "="*20 + " Evaluating Final Model on Test Set " + "="*20)
        # Tải lại model tốt nhất đã được lưu bởi lần train cuối cùng
        try:
            # Cần khởi tạo lại model trước khi load state dict
            eval_net = model(
                node_feat=node_attr,
                edge_feat=edge_attr,
                out_dim=1,
                predict_hidden_feats=best_params["predict_hidden_feats"],
                drop_ratio=best_params["drop_ratio"],
                # lr=best_params['lr'], 
                # depth=best_params['depth'],
                node_hid_feats=best_params['node_hid_feats'],
                # edge_hid=best_params['edge_hid_feats'],
                # weight_decay=best_params['weight_decay'], ##decayo
                readout_option=best_params['readout_option'],
                # readout_feats=best_params['readout_feats'],
            ).to(device)
            # Load state dict
            checkpoint = torch.load(final_model_path, map_location=device)
            eval_net.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError:
            print(f"Error: Could not load final model from {final_model_path}. Was it saved correctly by train()?")
            return
        except Exception as e:
            print(f"Error loading final model state_dict: {e}")
            return

        try:
            test_rmse, test_mae = inference(args, eval_net, test_loader, device, loss_fn=None, desc="Final Testing")
            print("-- FINAL TEST RESULT --")
            print(f"--- RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f} ---")

            # Ghi kết quả test vào file log cuối cùng
            test_dict = {
                "Name": "Test Evaluation with Best Limited Params",
                "test_rmse": test_rmse if not np.isnan(test_rmse) else 'NaN',
                "test_mae": test_mae if not np.isnan(test_mae) else 'NaN',
                "best_limited_params": best_params
            }
            if args.monitor_folder and args.monitor_name:
                with open(final_monitor_path, "a") as f:
                    # Ghi thêm thông tin về trial tốt nhất vào đầu file nếu muốn
                    f.write(json.dumps({"best_trial_info": study.best_trial.params, "best_value": study.best_trial.value}) + "\n")
                    f.write(json.dumps(test_dict) + "\n")
            else:
                 print(f"Final test results: {test_dict}")

        except Exception as e:
            print(f"Error during final testing: {e}")

    else:
        print("\nNo complete trials found or study did not yield a best trial. Could not run final evaluation.")


# --- Entry Point ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finetune GNN model with Optuna HPO (Limited Scope - Original model.py/gine.py)")
    # --- Giữ nguyên các tham số dòng lệnh như cũ ---
    
    parser.add_argument("--data_path", type=str, default='./Data/regression/e2sn2/e2sn2.csv')
    parser.add_argument("--graph_path", type=str, default='./Data/regression/e2sn2/its_new/e2sn2.pkl.gz')
    parser.add_argument("--y_column", type=str, default="ea")
    parser.add_argument("--model_path", type=str, default="./Data/model/") # Sửa mô tả
    parser.add_argument("--model_name", type=str, default="model.pt")
    parser.add_argument("--monitor_folder", type=str, default="./Data/monitor/")
    parser.add_argument("--monitor_name", type=str, default="monitor.txt")
    # Bỏ batch_size khỏi args nếu muốn tune nó bằng Optuna
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for each Optuna trial AND final training")
    parser.add_argument("--device", type=int, default=0, help="GPU device index to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # --- Thêm tham số cho Optuna ---
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials to run") # Giảm số trial mặc định vì scope hẹp hơn

    # Parse args
    args = parser.parse_args()

    # # # Lấy batch_size mặc định nếu không tune nó
    # # if 'batch_size' not in args:
    # #      args.batch_size = 128 # Đặt giá trị mặc định ở đây nếu không có trong parser

    # Chạy hàm tối ưu
    finetune_with_optuna_limited(args)
