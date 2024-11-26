# Global Imports
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Local Imports
sys.path.append(".")

import argparse
from time import time

from bayesipy.laplace import ELLA
from bayesipy.utils.datasets import Airline_Dataset, Taxi_Dataset, Year_Dataset
from bayesipy.utils.metrics import Regression
from bayesipy.utils.pretrained_models import Airline_MLP, Taxi_MLP, Year_MLP

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="ELLA")
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

val_dataset = dataset.validation_split()

# Create data loaders
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

ella = ELLA(
    model=model,
    likelihood="regression",
    subsample_size=2000,
    n_eigenvalues=20,
    prior_precision=0.01 * len(train_dataset),
    sigma_noise=1.0,
    seed=args.seed,
    y_mean=y_mean,
    y_std=y_std,
)

# Train the model
train_start = time()
metrics = ella.fit(
    train_loader=train_loader,
    val_loader=test_loader,
    val_steps=len(train_loader) // 100,
    metrics_cls=Regression,
    verbose=True,
)
train_end = time()
nll = np.array([m["NLL"] for m in metrics])
crps = np.array([m["CRPS"] for m in metrics])
CQM = np.array([m["CQM"] for m in metrics])
evaluations = np.arange(0, len(train_dataset), 100 * batch_size)

np.savetxt(f"results/{args.dataset}/ella_{args.seed}_nll.txt", nll, delimiter=",")
np.savetxt(f"results/{args.dataset}/ella_{args.seed}_crps.txt", crps, delimiter=",")
np.savetxt(f"results/{args.dataset}/ella_{args.seed}_cqm.txt", CQM, delimiter=",")
np.savetxt(
    f"results/{args.dataset}/ella_{args.seed}_evaluations.txt",
    evaluations,
    delimiter=",",
)
