import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.pool import global_add_pool
import time
import json
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, root_mean_squared_error, mean_absolute_error
import sys
import os
import argparse
import random
import optuna # Import Optuna

import pandas as pd
from pathlib import Path

root_dir=Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
os.chdir(str(root_dir))
from synprop.data_wrapper_7 import data_wrapper_7

#**THIẾT KẾ MẠNG GINE**
class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=5,
        node_hid_feats=300,
        readout_feats=1024,
        dr=0.1,
        readout_option=True,
    ):
        super(GNN, self).__init__()
        self.depth = depth
        self.readout_option = readout_option

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )
        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats) # GINE yêu cầu edge_feats có cùng dim với node_feats
        )

        self.gnn_layers = nn.ModuleList(
            [
                GINEConv(
                    nn=torch.nn.Sequential(
                        nn.Linear(node_hid_feats, node_hid_feats),
                        nn.ReLU(),
                        nn.Linear(node_hid_feats, node_hid_feats),
                    )
                )
                for _ in range(self.depth)
            ]
        )

        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )

        self.dropout = nn.Dropout(dr)
        self.readout_option = readout_option

    def forward(self, data):
        node_feats_orig = data.x
        edge_feats_orig = data.edge_attr
        batch = data.batch

        node_feats_init = self.project_node_feats(node_feats_orig)
        node_feats = node_feats_init
        edge_feats = self.project_edge_feats(edge_feats_orig)

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](node_feats, data.edge_index, edge_feats)

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = global_add_pool(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout

#** THIẾT KẾ MODEL**
class Model(nn.Module): # Đổi tên lớp từ model thành Model để tránh xung đột
    def __init__(
        self,
        node_in_feats, # Từ data
        edge_in_feats, # Từ data
        out_dim=1,
        # GNN params
        gnn_depth=3,
        gnn_node_hid_feats=300,
        gnn_readout_feats=1024, # Sẽ là emb_dim cho MLP nếu gnn_readout_option=True
        gnn_dr=0.1,
        gnn_readout_option=True,
        # MLP params
        mlp_predict_hidden_feats=512,
        mlp_drop_ratio=0.1,
    ):
        super(Model, self).__init__()

        self.gnn = GNN(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            depth=gnn_depth,
            node_hid_feats=gnn_node_hid_feats,
            readout_feats=gnn_readout_feats,
            dr=gnn_dr,
            readout_option=gnn_readout_option
        )

        # Xác định kích thước đầu vào cho MLP dựa trên output của GNN
        if gnn_readout_option:
            current_emb_dim = gnn_readout_feats
        else:
            current_emb_dim = gnn_node_hid_feats # Output của GNN là node_hid_feats nếu không có sparsify

        self.predict = nn.Sequential(
            torch.nn.Linear(current_emb_dim, mlp_predict_hidden_feats),
            torch.nn.PReLU(),
            torch.nn.Dropout(mlp_drop_ratio),
            torch.nn.Linear(mlp_predict_hidden_feats, mlp_predict_hidden_feats),
            torch.nn.PReLU(),
            torch.nn.Dropout(mlp_drop_ratio),
            torch.nn.Linear(mlp_predict_hidden_feats, out_dim),
        )

    def forward(self, mols):
        graph_feats = self.gnn(mols)
        out = self.predict(graph_feats)
        return out


def train_and_evaluate( # Đổi tên từ train để rõ ràng hơn cho Optuna
    trial, # Thêm trial cho Optuna pruning (nếu dùng)
    args, # Các args cố định
    net,
    optimizer, # Truyền optimizer vào
    loss_fn, # Truyền loss_fn vào
    train_loader_global, # Sử dụng global loader
    val_loader_global,   # Sử dụng global loader
    device,
    epochs_per_trial, # Số epochs cho mỗi trial Optuna
):
    # Không lưu model hoặc monitor file trong mỗi trial của Optuna trừ khi cần thiết
    # best_val_loss = 1e10 # Reset cho mỗi trial

    for epoch in range(epochs_per_trial):
        net.train()
        start_time = time.time()
        train_loss_list = []
        targets_train = []
        preds_train = []

        for batchdata in train_loader_global: # Không dùng tqdm ở đây để log đỡ rối
            batchdata = batchdata.to(device)
            pred = net(batchdata)
            labels = batchdata.y.to(device) # Đảm bảo labels trên cùng device
            
            # Kiểm tra và reshape nếu cần, output của model là (batch_size, out_dim)
            # labels cũng nên là (batch_size, out_dim) hoặc (batch_size,)
            # loss_fn mong đợi pred và labels có shape tương thích
            if pred.shape != labels.shape:
                 if labels.ndim == 1: # nếu labels là [N] và pred là [N,1]
                     labels = labels.view_as(pred)
                 elif pred.ndim == 1 and labels.ndim == 2 and labels.shape[1] == 1: # nếu pred là [N] và labels là [N,1]
                     labels = labels.squeeze(1)
                 # Thêm các trường hợp khác nếu cần


            targets_train.extend(labels.cpu().tolist()) # Chuyển về CPU trước khi tolist
            preds_train.extend(pred.cpu().detach().tolist()) # detach và chuyển về CPU
            
            loss = loss_fn(pred.view(-1), labels.view(-1)) # view(-1) để đảm bảo flat tensors

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())

        # Không in nhiều trong các epoch của Optuna trials để tránh quá nhiều output
        # rmse_train = root_mean_squared_error(targets_train, preds_train)
        # mae_train = mean_absolute_error(targets_train, preds_train)
        # print(f"Trial {trial.number} Epoch {epoch}, Train Loss: {np.mean(train_loss_list):.3f}, RMSE: {rmse_train:.3f}")


    # Validation sau khi hoàn thành tất cả các epoch cho trial này
    net.eval()
    val_rmse, val_mae, val_loss_mean = inference(args, net, val_loader_global, device, loss_fn)
    
    print(f"Trial {trial.number} - Val Loss: {val_loss_mean:.3f}, Val RMSE: {val_rmse:.3f}, Val MAE: {val_mae:.3f}")

    # Optuna Pruning (tùy chọn):
    # trial.report(val_rmse, epoch) # Báo cáo metric trung gian
    # if trial.should_prune():
    #     raise optuna.exceptions.TrialPruned()
        
    return val_rmse # Optuna sẽ tối thiểu hóa giá trị này


def inference(args, net, test_loader, device, loss_fn=None):
    net.eval()
    inference_loss_list = []
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batchdata in test_loader: # Không dùng tqdm ở đây
            batchdata = batchdata.to(device)
            pred = net(batchdata)
            labels = batchdata.y.to(device)

            if pred.shape != labels.shape:
                 if labels.ndim == 1:
                     labels = labels.view_as(pred)
                 elif pred.ndim == 1 and labels.ndim == 2 and labels.shape[1] == 1:
                     labels = labels.squeeze(1)

            targets_all.extend(labels.cpu().tolist())
            preds_all.extend(pred.cpu().tolist())

            if loss_fn is not None:
                loss = loss_fn(pred.view(-1), labels.view(-1))
                inference_loss_list.append(loss.item())
    
    # Đảm bảo preds_all và targets_all không rỗng trước khi tính metrics
    if not targets_all or not preds_all:
        print("Cảnh báo: Không có dữ liệu trong targets_all hoặc preds_all để tính metrics.")
        # Trả về giá trị mặc định hoặc báo lỗi tùy theo logic mong muốn
        nan_metric = float('nan')
        if loss_fn is None:
            return nan_metric, nan_metric
        else:
            return nan_metric, nan_metric, nan_metric


    rmse = root_mean_squared_error(targets_all, preds_all)
    mae = mean_absolute_error(targets_all, preds_all)

    if loss_fn is None:
        return rmse, mae
    else:
        mean_loss = np.mean(inference_loss_list) if inference_loss_list else float('nan')
        return rmse, mae, mean_loss

# --- Biến toàn cục cho data loaders và thuộc tính dữ liệu ---
# Sẽ được khởi tạo trong khối __main__
train_loader_global = None
val_loader_global = None
test_loader_global = None # Thêm test_loader nếu bạn muốn test sau HPO
node_attr_dim_global = 0
edge_attr_dim_global = 0
args_global = None # Để lưu các args cố định


def objective(trial: optuna.trial.Trial) -> float:
    global train_loader_global, val_loader_global, node_attr_dim_global, edge_attr_dim_global, args_global
    
    # Device setup
    device = (
        torch.device(f"cuda:{args_global.device}")
        if torch.cuda.is_available() and args_global.device is not None # Kiểm tra args_global.device
        else torch.device("cpu")
    )

    # --- Đề xuất các siêu tham số từ Optuna ---
    # GNN parameters
    gnn_depth = trial.suggest_int("gnn_depth", 2, 5) # Sửa đổi từ (2,5) trong code gốc là depth=5, opt=3
    gnn_node_hid_feats = trial.suggest_int("gnn_node_hid_feats", 64, 512, step=32) # Sửa (300, 1700) thành khoảng nhỏ hơn
    gnn_readout_option = trial.suggest_categorical("gnn_readout_option", [True, False])
    
    if gnn_readout_option:
        gnn_readout_feats = trial.suggest_int("gnn_readout_feats", 128, 1024, step=64) # (default=1024)
    else:
        gnn_readout_feats = gnn_node_hid_feats # Không dùng đến, nhưng cần gán giá trị

    gnn_dr = trial.suggest_float("gnn_dr", 0.0, 0.5, step=0.05) # (default=0.1, opt=0.2)

    # MLP (Predictor) parameters
    mlp_predict_hidden_feats = trial.suggest_int("mlp_predict_hidden_feats", 128, 512, step=64) # (default=512)
    mlp_drop_ratio = trial.suggest_float("mlp_drop_ratio", 0.0, 0.5, step=0.05) # (default=0.1)

    # Optimizer parameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) # (default=5e-4)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True) # (default=1e-5)

    # Khởi tạo mô hình
    net = Model(
        node_in_feats=node_attr_dim_global,
        edge_in_feats=edge_attr_dim_global,
        out_dim=1, # Giả sử out_dim là 1 cho regression
        gnn_depth=gnn_depth,
        gnn_node_hid_feats=gnn_node_hid_feats,
        gnn_readout_feats=gnn_readout_feats,
        gnn_dr=gnn_dr,
        gnn_readout_option=gnn_readout_option,
        mlp_predict_hidden_feats=mlp_predict_hidden_feats,
        mlp_drop_ratio=mlp_drop_ratio,
    ).to(device)

    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss() # Hoặc L1Loss (MAE) tùy bài toán

    # Huấn luyện và đánh giá
    # args_global.epochs ở đây là số epochs cho mỗi Optuna trial
    val_rmse = train_and_evaluate(
        trial,
        args_global,
        net,
        optimizer,
        loss_fn,
        train_loader_global,
        val_loader_global,
        device,
        epochs_per_trial=args_global.epochs_per_trial # Sử dụng epochs_per_trial
    )
    
    return val_rmse # Optuna sẽ cố gắng tối thiểu hóa giá trị này


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=48) # Giữ nguyên
    arg_parser.add_argument("--epochs_per_trial", type=int, default=20, help="Number of epochs for each Optuna trial") # Giảm để HPO nhanh hơn
    arg_parser.add_argument("--n_optuna_trials", type=int, default=50, help="Number of Optuna trials")
    arg_parser.add_argument("--device", type=int, default=0, help="GPU device index, None for CPU")
    
    # Các args không dùng trực tiếp trong objective nhưng cần cho data loading hoặc các cài đặt khác
    arg_parser.add_argument("--monitor_folder", type=str, default="./Data/monitor_optuna/") # Đổi folder
    arg_parser.add_argument("--monitor_name", type=str, default="monitor_optuna.txt")
    arg_parser.add_argument("--model_path_root", type=str, default="./Data/model_optuna/") # Để lưu model tốt nhất
    arg_parser.add_argument("--data_path", type=str, default='./Data/regression/cycloaddition/cycloaddition.csv')
    arg_parser.add_argument("--graph_path", type=str, default='./Data/regression/cycloaddition/its_new/cycloaddition.pkl.gz')
    arg_parser.add_argument("--y_column", type=str, default="G_act")
    arg_parser.add_argument("--seed", type=int, default=42)
    
    args = arg_parser.parse_args()
    args_global = args # Gán cho biến global

    # Tạo thư mục nếu chưa tồn tại
    Path(args.monitor_folder).mkdir(parents=True, exist_ok=True)
    Path(args.model_path_root).mkdir(parents=True, exist_ok=True)


    # --- Cài đặt seed ---
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device is not None:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False # Tắt để đảm bảo reproducibility
        torch.backends.cudnn.deterministic = True # Tắt để đảm bảo reproducibility


    # --- Tải dữ liệu một lần ---
    print("--- Loading Data ---")

    data_path=args.data_path
    graph_path=args.graph_path
    target=args.y_column
    batch_size=args.batch_size

    data_provider = data_wrapper_7(data_path, graph_path, target, batch_size,4, 0.1, 0.1)
    train_loader_global, val_loader_global, test_loader_global = data_provider.get_data_loaders()

    # Lấy node_attr_dim và edge_attr_dim từ data_provider hoặc dataset
    # Điều này phụ thuộc vào cách data_wrapper_7 của bạn được thiết kế
    # Ví dụ:
    if hasattr(data_provider, 'node_attr_dim') and hasattr(data_provider, 'edge_attr_dim'):
        node_attr_dim_global = data_provider.node_attr_dim
        edge_attr_dim_global = data_provider.edge_attr_dim
    elif hasattr(train_loader_global.dataset, 'num_node_features') and hasattr(train_loader_global.dataset, 'num_edge_features'):
         node_attr_dim_global = train_loader_global.dataset.num_node_features
         edge_attr_dim_global = train_loader_global.dataset.num_edge_features
    else: # Fallback nếu không tìm thấy, bạn cần cung cấp giá trị đúng
        print("Không thể tự động xác định node_attr_dim và edge_attr_dim. Sử dụng giá trị giả định.")
        node_attr_dim_global = 5 # Cần thay thế bằng giá trị đúng từ dữ liệu của bạn
        edge_attr_dim_global = 3 # Cần thay thế bằng giá trị đúng từ dữ liệu của bạn

    print(f"Node feature dimension: {node_attr_dim_global}, Edge feature dimension: {edge_attr_dim_global}")
    print(f"Device for Optuna: cuda:{args.device}" if torch.cuda.is_available() and args.device is not None else "cpu")


    # --- Chạy Optuna Study ---
    study_name = "GINE_Optimization" # Đặt tên cho study của bạn
    storage_name = f"sqlite:///{args.monitor_folder}/{study_name}.db" # Lưu trữ study vào SQLite database

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name, # Cho phép resume study nếu bị gián đoạn
        load_if_exists=True, # Tải study nếu đã tồn tại
        direction="minimize", # Tối thiểu hóa val_rmse
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1) # Ví dụ pruner
    )

    print(f"Sampler is {study.sampler.__class__.__name__}")
    
    # Callback để lưu model tốt nhất
    def save_best_model_callback(study, trial):
        if study.best_trial.number == trial.number:
            # Xóa model cũ nếu có
            for f_name in os.listdir(args.model_path_root):
                if f_name.startswith("best_model_trial_"):
                    os.remove(os.path.join(args.model_path_root, f_name))
            
            # Lưu model mới (chỉ state_dict)
            model_save_path = os.path.join(args.model_path_root, f"best_model_trial_{trial.number}_params.pt")
            # Không lưu toàn bộ model, chỉ lưu params cho Optuna
            # torch.save(trial.user_attrs["model_state_dict"], model_save_path) # Cần truyền model vào user_attrs
            print(f"New best trial: {trial.number} with value: {trial.value:.4f}. Params saved (conceptually).")
            # Thực tế, bạn sẽ xây dựng lại model với best_params và lưu nó sau khi study hoàn tất.

    study.optimize(objective, n_trials=args.n_optuna_trials, callbacks=[save_best_model_callback])

    # --- In kết quả ---
    print("\nOptimization Finished!")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (minimized val_rmse): {best_trial.value}")
    print("  Best hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- Huấn luyện lại model tốt nhất với toàn bộ dữ liệu huấn luyện (tùy chọn) và test ---
    print("\n--- Training and Testing with Best Hyperparameters ---")
    best_params = best_trial.params
    
    final_device = (
        torch.device(f"cuda:{args.device}")
        if torch.cuda.is_available() and args.device is not None
        else torch.device("cpu")
    )

    best_model = Model(
        node_in_feats=node_attr_dim_global,
        edge_in_feats=edge_attr_dim_global,
        out_dim=1,
        gnn_depth=best_params["gnn_depth"],
        gnn_node_hid_feats=best_params["gnn_node_hid_feats"],
        gnn_readout_feats=best_params.get("gnn_readout_feats", best_params["gnn_node_hid_feats"]), # Xử lý nếu readout_option=False
        gnn_dr=best_params["gnn_dr"],
        gnn_readout_option=best_params["gnn_readout_option"],
        mlp_predict_hidden_feats=best_params["mlp_predict_hidden_feats"],
        mlp_drop_ratio=best_params["mlp_drop_ratio"],
    ).to(final_device)

    # Huấn luyện lại với số epochs lớn hơn nếu muốn
    final_epochs = args_global.epochs_per_trial * 2 # Ví dụ: gấp đôi số epochs của trial
    print(f"Training best model for {final_epochs} epochs...")
    
    final_optimizer = Adam(best_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    final_loss_fn = torch.nn.MSELoss()

    # Vòng lặp huấn luyện cuối cùng (tương tự train_and_evaluate nhưng không có trial)
    best_val_rmse_overall = float('inf')
    final_model_path = os.path.join(args.model_path_root, "final_best_model.pt")

    for epoch in range(final_epochs):
        best_model.train()
        for batchdata in train_loader_global:
            batchdata = batchdata.to(final_device)
            pred = best_model(batchdata)
            labels = batchdata.y.to(final_device)
            if pred.shape != labels.shape:
                 if labels.ndim == 1: labels = labels.view_as(pred)
                 elif pred.ndim == 1 and labels.ndim == 2 and labels.shape[1] == 1: labels = labels.squeeze(1)
            loss = final_loss_fn(pred.view(-1), labels.view(-1))
            final_optimizer.zero_grad()
            loss.backward()
            final_optimizer.step()
        
        # Đánh giá trên val set
        current_val_rmse, _, current_val_loss = inference(args, best_model, val_loader_global, final_device, final_loss_fn)
        print(f"Final Training Epoch {epoch+1}/{final_epochs}, Val RMSE: {current_val_rmse:.4f}, Val Loss: {current_val_loss:.4f}")
        if current_val_rmse < best_val_rmse_overall:
            best_val_rmse_overall = current_val_rmse
            torch.save(best_model.state_dict(), final_model_path)
            print(f"Saved best model to {final_model_path} with Val RMSE: {best_val_rmse_overall:.4f}")


    # Tải model tốt nhất đã lưu và đánh giá trên test set
    print(f"\nLoading best model from {final_model_path} for final testing...")
    best_model.load_state_dict(torch.load(final_model_path))

    test_rmse, test_mae, test_loss = inference(args, best_model, test_loader_global, final_device, final_loss_fn)
    print(f"--- Test Results with Best Hyperparameters ---")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")

    # Lưu kết quả test
    test_results_log_path = os.path.join(args.monitor_folder, "final_test_results.txt")
    with open(test_results_log_path, "w") as f:
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTest Metrics:\n")
        f.write(f"  Test RMSE: {test_rmse:.4f}\n")
        f.write(f"  Test MAE: {test_mae:.4f}\n")
        f.write(f"  Test Loss: {test_loss:.4f}\n")
    print(f"Final test results saved to {test_results_log_path}")