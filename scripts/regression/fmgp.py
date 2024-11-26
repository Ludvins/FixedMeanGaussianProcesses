# Global Imports
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local Imports
sys.path.append(".")

import argparse
from time import time

from bayesipy.fmgp import FMGP
from bayesipy.utils.datasets import Airline_Dataset, Taxi_Dataset, Year_Dataset
from bayesipy.utils.metrics import Regression, score
from bayesipy.utils.pretrained_models import Airline_MLP, Taxi_MLP, Year_MLP

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--dataset", type=str)
parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
SEED = 2147483647 - args.seed

torch.manual_seed(SEED)

if args.dataset == "airline":
    dataset = Airline_Dataset()
    model = Airline_MLP().to(DEVICE).to(DTYPE)
elif args.dataset == "year":
    dataset = Year_Dataset()
    model = Year_MLP().to(DEVICE).to(DTYPE)
elif args.dataset == "taxi":
    dataset = Taxi_Dataset()
    model = Taxi_MLP().to(DEVICE).to(DTYPE)
else:
    raise ValueError("Dataset not found")

train_dataset, test_dataset = dataset.train_test_splits()

y_mean = dataset.y_mean
y_std = dataset.y_std
y_mean_t = torch.tensor(y_mean, device=DEVICE, dtype=DTYPE)
y_std_t = torch.tensor(y_std, device=DEVICE, dtype=DTYPE)


# Create data loaders
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

ue = FMGP(
    model=model,
    likelihood="regression",
    kernel="RBF",
    inducing_locations="kmeans",
    num_inducing=100,
    noise_variance=np.exp(-5),
    subrogate_regularizer=True,
    y_mean=y_mean,
    y_std=y_std,
)
train_start = time()
loss = ue.fit(iterations=70000, lr=0.001, train_loader=train_loader, verbose=True)
train_end = time()


# Test the model
test_start = time()
test_metrics = score(ue, test_loader, Regression, verbose=True)
test_end = time()

test_metrics["method"] = "fmgp"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["dataset"] = args.dataset
test_metrics["seed"] = args.seed


# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
df.to_csv(
    path_or_buf=f"results/{args.dataset}/fmgp_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)
