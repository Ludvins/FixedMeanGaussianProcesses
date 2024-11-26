# Global Imports
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local Imports
sys.path.append(".")

import argparse
from time import time

import tqdm

from bayesipy.laplace import Laplace
from bayesipy.utils.datasets import Airline_Dataset, Taxi_Dataset, Year_Dataset
from bayesipy.utils.metrics import Regression, score
from bayesipy.utils.pretrained_models import Airline_MLP, Taxi_MLP, Year_MLP

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="LLA")
parser.add_argument("--dataset", type=str)
parser.add_argument("--hessian", type=str, help="Hessian approximation")
parser.add_argument("--subset", type=str, help="Subset of weights")
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

model = model.to(DEVICE).to(DTYPE)
train_dataset, test_dataset = dataset.train_test_splits()
# Split the dataset
val_dataset = dataset.validation_split()

# Create data loaders
batch_size = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create the VFSUE model
lla = Laplace(
    model=model,
    likelihood="regression",
    subset_of_weights=args.subset,
    hessian_structure=args.hessian,
    y_mean=dataset.y_mean,
    y_std=dataset.y_std,
)

# Train the model
train_start = time()
lla.fit(train_loader=train_loader)


log_sigma = torch.zeros(1, requires_grad=True)
log_prior = torch.zeros(1, requires_grad=True)

hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)

for i in tqdm.tqdm(range(100)):
    hyper_optimizer.zero_grad()
    neg_marglik = -lla.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

prior_precision = log_prior.exp().item()
sigma_noise = log_sigma.exp().item()
train_end = time()

# Test the model
test_start = time()
test_metrics = score(lla, test_loader, Regression, verbose=True)
test_end = time()

test_metrics["method"] = f"lla {args.subset} {args.hessian}"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["dataset"] = args.dataset
test_metrics["prior_precision"] = lla.prior_precision.item()
test_metrics["sigma_noise"] = sigma_noise
test_metrics["seed"] = args.seed

# Rotation angles

# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
# df.to_csv(
#     path_or_buf=f"results/{args.dataset}/lla_{args.subset}_{args.hessian}_{args.seed}.csv",
#     encoding="utf-8",
#     index=False,
# )
