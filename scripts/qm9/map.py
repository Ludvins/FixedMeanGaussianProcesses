# Global Imports
import sys

import numpy as np
import pandas as pd
import torch
import os.path as osp

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus

import copy
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
import os

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops

# Local Imports
sys.path.append(".")

import argparse
from time import time

from bayesipy.fmgp import FMGP
from bayesipy.utils.metrics import Regression, score
from bayesipy.utils import gaussian_logdensity

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--seed", type=int, help="Seed")

args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
SEED = 2147483647 - args.seed

torch.manual_seed(SEED)

target = 0
dim = 64


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = torch.nn.Linear(11, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
        
    def embedding_h(self, data):
        out = F.relu(self.lin0(data.x.to(self.device).to(self.dtype)))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index.to(self.device), data.edge_attr.to(self.device).to(self.dtype)))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        return out.squeeze(), data.y
    
    def forward(self, X):
        X = X.to(self.device).to(self.dtype)
        out = F.relu(self.lin1(X))
        out = self.lin2(out)

        return out
    
    

model = Net()

model.device = DEVICE
model.dtype = DTYPE
checkpoint = torch.load("QM9_models/CNN_100.pth", weights_only=True)
model.load_state_dict(checkpoint)
model = model.to(DEVICE).to(DTYPE)

class MyTransform:
    def __call__(self, data):
        data = copy.copy(data)
        data.y = data.y[:, target]  # Specify target.
        return data


class Complete:
    def __call__(self, data):
        data = copy.copy(data)
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data', 'QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False), model.embedding_h])
dataset = QM9(path, transform=transform).shuffle()


# Split datasets.
test_dataset = dataset[:10_000]
val_dataset = dataset[10_000:20_000]
train_dataset = dataset[20_000:]

# Normalize targets to mean = 0 and std = 1.
mean = train_dataset.data.y.mean(dim=0, keepdim=True)
std = train_dataset.data.y.std(dim=0, keepdim=True)

train_dataset.data.y = (train_dataset.data.y - mean) / std
val_dataset.data.y = (val_dataset.data.y - mean) / std
test_dataset.data.y = (test_dataset.data.y - mean) / std

mean, std = mean[:, target].item(), std[:, target].item()

batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

from properscoring import crps_gaussian

def test(loader, noise):
    model.eval()
    error = 0
    nll = 0
    crps = 0

    for data in loader:
        F_mean = model(data[0])
        F_var = torch.ones_like(F_mean) * noise
        F_mean = F_mean.detach().cpu() * std
        F_var = F_var.detach().cpu() * std ** 2
        label = data[1] * std

        error += (F_mean - label).abs().sum().item()  # MAE
        nll -= torch.sum(
            gaussian_logdensity(F_mean.squeeze(), F_var.squeeze(), label.squeeze())
        )
        crps += crps_gaussian(
            label.squeeze().detach().cpu(), 
            F_mean.squeeze().detach().cpu(), 
            np.sqrt(F_var.squeeze().detach().cpu())
        ).sum()
    return error / len(loader.dataset), nll / len(loader.dataset), crps / len(loader.dataset)

train_start = time()
sigma_noises = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]

best_val_NLL = float("inf")
from tqdm import tqdm
for sigma_noise in tqdm(sigma_noises):

    val_MAE, val_NLL, _ = test(val_loader, sigma_noise)
    
    print("noise: {}, val: {}".format(sigma_noise, val_NLL))
    print(val_MAE)
    if val_NLL < best_val_NLL:
        best_val_NLL = val_NLL
        best_sigma_noise = sigma_noise


train_end = time()

# Test the model
test_start = time()
error, nll, crps = test(test_loader, best_sigma_noise)
test_end = time()


test_metrics = {}
test_metrics["method"] = "map"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["mae"] = error
test_metrics["nll"] = nll.detach().cpu().numpy()
test_metrics["crps"] = crps
test_metrics["dataset"] = "QM9"
test_metrics["seed"] = args.seed


# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
df.to_csv(
    path_or_buf=f"results/QM9/map_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)
