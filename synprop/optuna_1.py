#**THIẾT KẾ MẠNG GINE**
import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.pool import global_add_pool


class GNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=5, #default = 5; opt = 3
        node_hid_feats=300, #default = 300; opt = 1700
        readout_feats=1024, #default = 1024
        dr=0.1, #default = 0.1; opt = 0.2
        readout_option=True, #default = True
        # lr=lr, ##mới thêm
    ):
        super(GNN, self).__init__()

        self.depth = depth

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats)
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
            # print('node: ',node_feats.shape)
            # print('edge_index: ',data.edge_index)
            # print('edge_feats: ',edge_feats.shape)
            node_feats = self.gnn_layers[i](node_feats, data.edge_index, edge_feats)

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = global_add_pool(node_feats, batch)

        if self.readout_option:
            readout = self.sparsify(readout)

        return readout


#** THIẾT KẾ MODEL**
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
        # lr=lr, ##mới thêm
        # depth=depth, ##mới thêm
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
    lr=5e-4, ##mới thêm 5e-4
    weight_decay=1e-5, ##mới thêm 1e-5
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


#** THIẾT KẾ FINE_TUNE

import pandas as pd
from pathlib import Path

root_dir=Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
os.chdir(str(root_dir))
from synprop.data_wrapper_7 import data_wrapper_7

def finetune(args):
    batch_size = args.batch_size
    model_path = args.model_path + args.model_name
    monitor_path = args.monitor_folder + args.monitor_name
    epochs = args.epochs
    data_path =args.data_path
    graph_path=args.graph_path
    # column_rxn=args.column_rxn #ver 3,4,5
    target=args.y_column
    # reaction_mode_str = args.reaction_mode_str #mới cho ver4

    data = pd.read_csv(data_path)
    out_dim = 1
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device is\t", device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # data_loader = data_wrapper_3(data_path, column_rxn, target, batch_size,4, 0.1, 0.1)
    # data_loader = data_wrapper_5(data_path, graph_path, column_rxn, target, reaction_mode_str, batch_size,4, 0.1, 0.1) #ver 4,5
    data_loader = data_wrapper_7(data_path, graph_path, target,  batch_size,4, 0.1, 0.1) #ver 6,7
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    node_attr=train_loader.dataset[0].x.shape[1]
    edge_attr=train_loader.dataset[0].edge_attr.shape[1]

    print("--- model_path:", model_path)

    # training
    if not os.path.exists(model_path):
        net = model(node_attr,edge_attr).to(device)
        print("-- TRAINING")
        net = train(
            args, net, train_loader, val_loader, model_path, device, epochs=epochs
        )
    else:
        net = model(node_attr,edge_attr).to(device)
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        current_epoch = checkpoint["epoch"]
        epochs = epochs - current_epoch
        net = train(
            args,
            net,
            train_loader,
            val_loader,
            model_path,
            device,
            epochs=epochs,
            current_epoch=current_epoch,
            best_val_loss=checkpoint["val_loss"],
        )

    # test
    net = model(node_attr,edge_attr).to(device)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    rmse, mae = inference(args, net, test_loader, device)
    print("-- RESULT")
    print("--- rmse: %.3f, MAE: %.3f," % (rmse, mae))
    dict = {
        "Name": "Test",
        "test_rmse": rmse,
        "test_mae": mae,
    }
    with open(monitor_path, "a") as f:
        f.write(json.dumps(dict) + "\n")


#** THIẾT KẾ MAIN_FINETUNE
import argparse
import random

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=48)
    arg_parser.add_argument("--epochs", type=int, default=100)
    arg_parser.add_argument("--device", type=int, default=0)
    arg_parser.add_argument("--monitor_folder", type=str, default="./Data/monitor/")
    arg_parser.add_argument("--monitor_name", type=str, default="monitor.txt")
    arg_parser.add_argument("--Data_folder", type=str, default="./Data/")
    arg_parser.add_argument("--model_path", type=str, default="./Data/model/")
    arg_parser.add_argument("--data_path", type=str, default='./Data/regression/e2sn2/e2sn2.csv')
    # arg_parser.add_argument("--column_rxn", type=str, default="AAM") #ver3,4,5
    # arg_parser.add_argument("--reaction_mode_str", type=str, default="reac_diff") #mới cho ver4   
    arg_parser.add_argument("--graph_path", type=str, default='./Data/regression/e2sn2/its_new/e2sn2.pkl.gz') #binh thuong bo cho nay
    arg_parser.add_argument("--model_name", type=str, default="model.pt")
    arg_parser.add_argument("--y_column", type=str, default="ea")
    arg_parser.add_argument("--seed", type=int, default=42) #default 27407
    args = arg_parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    finetune(args)

