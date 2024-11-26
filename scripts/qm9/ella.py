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
from bayesipy.laplace import ELLA

import argparse
from time import time

from bayesipy.fmgp import FMGP
from bayesipy.utils.metrics import Regression, score

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
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

class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
        
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
model.train()


model2 = Net2()

model2.device = DEVICE
model2.dtype = DTYPE
model2 = model2.to(DEVICE).to(DTYPE)

lin1_sd = copy.deepcopy(model.lin1.state_dict())
lin2_sd = copy.deepcopy(model.lin2.state_dict())

model2.lin1.load_state_dict(lin1_sd)
model2.lin2.load_state_dict(lin2_sd)

model2.train()


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

# Create the VFSUE model
ella = ELLA(
    model=model2,
    likelihood="regression",
    subsample_size=2000,
    n_eigenvalues=20,
    seed=args.seed,
)


prior_precisions = [0.5, 0.75, 1, 1.25, 1.5]
sigma_noises = [0.1, 0.075, 0.05, 0.025, 0.01]

combinations = []
for prior_precision in prior_precisions:
    for sigma_noise in sigma_noises:
        combinations.append((prior_precision, sigma_noise))

# Train the model
train_start = time()
ella.fit(train_loader=train_loader, verbose=True)
#ella.optimize_hyperparameters(val_loader, combinations, verbose=True)

best_sigma = -1
best_precision = -1
best_nll = 100000000
model.eval()

from bayesipy.utils import gaussian_logdensity
from properscoring import crps_gaussian
for prior, sigma in combinations:
    ella.prior_precision = prior
    ella.sigma_noise = sigma

    nll = 0
    for data in val_loader:
        F_mean, F_var = ella.predict(data[0])
        F_mean = F_mean.detach().cpu() * std
        F_var = F_var.detach().cpu() * std ** 2
        label = data[1] * std
        nll -= torch.sum(
            gaussian_logdensity(F_mean.squeeze(), F_var.squeeze(), label.squeeze())
        ).detach().cpu().numpy()
    nll /= len(val_loader.dataset)
    
    print("Sigma: {}, Prior Precision: {}, NLL: {}".format(sigma, prior, nll))
    
    if nll < best_nll:
        best_nll = nll
        best_sigma = sigma
        best_precision = prior

    print("Best Configuration: {} {}".format(best_sigma, best_precision))


ella.prior_precision = best_precision
ella.sigma_noise = best_sigma
train_end = time()


# Test the model
test_start = time()



nll = 0
crps = 0
error = 0
for data in test_loader:
    F_mean, F_var = ella.predict(data[0])
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
    
error /= len(test_loader.dataset)
crps /= len(test_loader.dataset)
nll = nll.detach().cpu().numpy() / len(test_loader.dataset)
test_end = time()


test_metrics = {}
test_metrics["method"] = "ella"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["mae"] = error
test_metrics["nll"] = nll
test_metrics["crps"] = crps
test_metrics["dataset"] = "QM9"
test_metrics["seed"] = args.seed


# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
df.to_csv(
    path_or_buf=f"results/QM9/ella_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)

