# Global Imports
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local Imports
sys.path.append(".")

import argparse
from time import time

from tqdm import tqdm

from bayesipy.utils.datasets import Airline_Dataset, Taxi_Dataset, Year_Dataset
from bayesipy.utils.metrics import Regression, score
from bayesipy.utils.pretrained_models import Airline_MLP, Taxi_MLP, Year_MLP

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="MAP")
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
    model = Airline_MLP()
elif args.dataset == "year":
    dataset = Year_Dataset()
    model = Year_MLP()
elif args.dataset == "taxi":
    dataset = Taxi_Dataset()
    model = Taxi_MLP()
else:
    raise ValueError("Dataset not found")

model.to(DEVICE).to(DTYPE)
model.device = DEVICE
model.dtype = DTYPE
model.eval()

train_dataset, test_dataset = dataset.train_test_splits()
val_dataset = dataset.validation_split()

y_mean = dataset.y_mean
y_std = dataset.y_std
y_mean_t = torch.tensor(y_mean, device=DEVICE, dtype=DTYPE)
y_std_t = torch.tensor(y_std, device=DEVICE, dtype=DTYPE)


# Create data loaders
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


train_start = time()
sigma_noises = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

best_val_NLL = float("inf")

for sigma_noise in tqdm(sigma_noises):

    def predict(self, X):
        F_mean = self(X)
        F_var = torch.ones_like(F_mean).to(DEVICE).to(DTYPE) * 1e-6 + sigma_noise
        return y_mean_t + y_std_t * F_mean, F_var * y_std_t**2

    model.predict = predict.__get__(model)

    val_NLL = score(model, val_loader, Regression, verbose=False)["NLL"]
    if val_NLL < best_val_NLL:
        best_val_NLL = val_NLL
        best_sigma_noise = sigma_noise


def predict(self, X):
    F_mean = self(X)
    F_var = torch.ones_like(F_mean).to(DEVICE).to(DTYPE) * 1e-6 + best_sigma_noise
    return y_mean_t + y_std_t * F_mean, F_var * y_std_t**2


model.predict = predict.__get__(model)

train_end = time()

# Test the model
test_start = time()
test_metrics = score(model, test_loader, Regression, verbose=True)
test_end = time()

test_metrics["method"] = "map"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["sigma_noise"] = best_sigma_noise
test_metrics["dataset"] = args.dataset
test_metrics["seed"] = args.seed


# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)

# df.to_csv(
#     path_or_buf=f"results/{args.dataset}/map_{args.seed}.csv",
#     encoding="utf-8",
#     index=False,
# )
