import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, root_mean_squared_error, mean_absolute_error
from pathlib import Path
import sys
import os
root_dir=Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
os.chdir(str(root_dir))
from synprop.gine import GNN



class model(nn.Module):
    def __init__(
        self,
        node_feat,
        edge_feat,
        out_dim=1, #default = 1
        num_layer=3, #default = 3
        node_hid_feats=300, #default = 300
        readout_feats=1024, #default = 1024
        predict_hidden_feats=512, #default = 512
        readout_option=False,
        drop_ratio=0.1, #default = 0.1
        # lr, ##mới thêm
        # depth, ##mới thêm
    ):
        super(model, self).__init__()
        emb_dim=1024
        self.gnn = GNN(node_feat,edge_feat)  


        self.predict = nn.Sequential(
            torch.nn.Linear(emb_dim, predict_hidden_feats),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(predict_hidden_feats, predict_hidden_feats),
            torch.nn.PReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(predict_hidden_feats, out_dim),
        )

    def forward(self, mols):
        graph_feats = self.gnn(mols)
        out = self.predict(graph_feats)
        return out


def train(
    args,
    net,
    train_loader,
    val_loader,
    model_path,
    device,
    # lr=5e-4, ##mới thêm 5e-4
    # weight_decay=1e-5, ##mới thêm 1e-5
    # optimizer, 
    epochs=20,
    current_epoch=0,
    best_val_loss=1e10,
):
    monitor_path = args.monitor_folder + args.monitor_name
    n_epochs = epochs

    loss_fn = torch.nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-5) ##default: lr=5e-4, weight_decay=1e-5, chạy main_finetune (2. 1.3023669362312975e-05;  1.1848812109693355e-06)
    # optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay) ##default: lr=5e-4, weight_decay=1e-5

    for epoch in range(n_epochs):
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []
        targets = []
        preds = []

        for batchdata in tqdm(train_loader, desc="Training"):
            batchdata=batchdata.to(device)
            pred = net(batchdata)
            # print(pred.shape)
            labels = batchdata.y
            # print(labels.shape)
            # assert 1==2
            targets.extend(labels.tolist())
            labels = labels.to(device)

            preds.extend(pred.tolist())
            loss = loss_fn(pred.view(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        rmse = root_mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)
        print(
            "--- training epoch %d, loss %.3f, rmse %.3f, mae %.3f, time elapsed(min) %.2f---"
            % (
                epoch,
                np.mean(train_loss_list),
                rmse,
                mae,
                (time.time() - start_time) / 60,
            )
        )

        # validation
        net.eval()
        val_rmse, val_mae, val_loss = inference(args, net, val_loader, device, loss_fn)

        print(
            "--- validation at epoch %d, val_loss %.3f, val_rmse %.3f, val_mae %.3f ---"
            % (epoch, val_loss, val_rmse, val_mae)
        )
        print("\n" + "*" * 100)

        dict = {
            "epoch": epoch + current_epoch,
            "train_loss": np.mean(train_loss_list),
            "val_loss": val_loss,
            "train_rmse": rmse,
            "val_rmse": val_rmse,
        }
        with open(monitor_path, "a") as f:
            f.write(json.dumps(dict) + "\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + current_epoch,
                    "model_state_dict": net.state_dict(),
                    "val_loss": best_val_loss,
                },
                model_path,
            )


def inference(args, net, test_loader, device, loss_fn=None):
    # batch_size = test_loader.batch_size

    net.eval()
    inference_loss_list = []
    preds = []
    targets = []

    with torch.no_grad():
        for batchdata in tqdm(test_loader, desc="Testing"):
            batchdata=batchdata.to(device)
            pred = net(batchdata)
            labels = batchdata.y
            targets.extend(labels.tolist())
            labels = labels.to(device)

            preds.extend(pred.tolist())

            if loss_fn is not None:
                inference_loss = loss_fn(pred.view(-1), labels)
                inference_loss_list.append(inference_loss.item())

    rmse = root_mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)

    if loss_fn is None:
        return rmse, mae
    else:
        return rmse, mae, np.mean(inference_loss_list)
