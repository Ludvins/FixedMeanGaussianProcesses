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

from tqdm import tqdm

from bayesipy.laplace import ELLA
from bayesipy.utils.datasets import CIFAR10_Dataset, CIFAR10_Rotated_Dataset, CIFAR10_OOD_Dataset
from bayesipy.utils.metrics import SoftmaxClassification, OOD, score
from bayesipy.utils.pretrained_models import CIFAR10_Resnet

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--resnet", type=str, default="resnet20", help="Resnet model")
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--lower_da", type=float, default=0.5)

parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
SEED = 2147483647 - args.seed

torch.manual_seed(SEED)

# Load Models
resnet = CIFAR10_Resnet(args.resnet).to(DEVICE).to(DTYPE)

# Load the data transform
data_transform = CIFAR10_Resnet(args.resnet, get_transform=True)

# Load the dataset
dataset = CIFAR10_Dataset(
    transform=data_transform,
)

# Split the dataset
train_dataset, test_dataset = dataset.train_test_splits()
val_dataset = dataset.validation_split(lower=args.lower_da)

# Create data loaders
batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create the VFSUE model
ella = ELLA(
    model=resnet,
    likelihood="classification",
    subsample_size=2000,
    n_eigenvalues=20,
    prior_precision=1 / 0.04,
    seed=args.seed,
)

# Train the model
train_start = time()
ella.fit(train_loader=train_loader, val_loader=val_loader, val_steps=1, verbose=True)
train_end = time()

# Test the model
test_start = time()
test_metrics = score(ella, test_loader, SoftmaxClassification, verbose=True)
test_end = time()

test_metrics["method"] = "ella"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["resnet"] = args.resnet
test_metrics["prior_precision"] = ella.prior_precision
test_metrics["seed"] = args.seed
test_metrics["lower_da"] = args.lower_da
test_metrics["batch_size"] = batch_size

# Rotation angles
angles = np.arange(10, 190, 10)

for angle in tqdm(angles):
    dataset = CIFAR10_Rotated_Dataset(angle=angle, transform=data_transform)
    _, test_dataset = dataset.train_test_splits()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    metrics = score(
        ella,
        test_loader,
        SoftmaxClassification,
        verbose=False,
    )
    for key, value in metrics.items():
        test_metrics[f"rotated_{angle}_{key}"] = value

# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
df.to_csv(
    path_or_buf=f"results/cifar10/ella_{batch_size}_{args.lower_da}_{args.resnet}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)

dataset = CIFAR10_OOD_Dataset(transform=data_transform)
_, test_dataset = dataset.train_test_splits()

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

predictions = []
metrics = score(
    ella,
    test_loader,
    OOD,
    verbose=False,
)

labels_ood = metrics["labels"]
predictions_ood = metrics["preds"]

np.savetxt("results/cifar10_ood/labels.txt", labels_ood)
np.savetxt(
    f"results/cifar10_ood/ella_{batch_size}_{args.lower_da}_{args.resnet}_{args.seed}.txt",
    predictions_ood,
)
