# Global Imports
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local Imports
sys.path.append(".")

import argparse
from time import time

from bayesipy.laplace import ELLA
from bayesipy.utils.datasets import Airline_Dataset, Taxi_Dataset, Year_Dataset
from bayesipy.utils.metrics import Regression, score
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
    seed=args.seed,
    y_mean=y_mean,
    y_std=y_std,
)


prior_precisions = [1000, 100, 10, 1, 0.1, 0.01]
sigma_noises = [0.5, 1, 1.5]

combinations = []
for prior_precision in prior_precisions:
    for sigma_noise in sigma_noises:
        combinations.append((prior_precision, sigma_noise))

# Train the model
train_start = time()
ella.fit(train_loader=train_loader, verbose=True)
ella.optimize_hyperparameters(val_loader, combinations, verbose=True)
train_end = time()


# Test the model
test_start = time()
test_metrics = score(ella, test_loader, Regression, verbose=True)
test_end = time()

test_metrics["method"] = "ella"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["dataset"] = args.dataset
test_metrics["prior_precision"] = ella.prior_precision.item()
test_metrics["sigma_noise"] = ella.sigma_noise.item()
test_metrics["seed"] = args.seed


# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
# df.to_csv(
#     path_or_buf=f"results/{args.dataset}/ella_{args.seed}.csv",
#     encoding="utf-8",
#     index=False,
# )
