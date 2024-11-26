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

from bayesipy.laplace import Laplace
from bayesipy.utils.datasets import CIFAR10_Dataset, CIFAR10_Rotated_Dataset, CIFAR10_OOD_Dataset
from bayesipy.utils.metrics import SoftmaxClassification, OOD, score
from bayesipy.utils.pretrained_models import CIFAR10_Resnet

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--resnet", type=str, default="resnet20", help="Resnet model")
parser.add_argument("--hessian", type=str, help="Hessian approximation")
parser.add_argument("--subset", type=str, help="Subset of weights")
parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
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
val_dataset = dataset.validation_split()

# Create data loaders
batch_size = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create the VFSUE model
lla = Laplace(
    model=resnet,
    likelihood="classification",
    subset_of_weights=args.subset,
    hessian_structure=args.hessian,
)

# Train the model
train_start = time()
lla.fit(train_loader=train_loader)
lla.optimize_prior_precision()
train_end = time()

# Test the model
test_start = time()
test_metrics = score(lla, test_loader, SoftmaxClassification, verbose=True)
test_end = time()

test_metrics["method"] = f"lla {args.subset} {args.hessian}"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["hessian"] = args.hessian
test_metrics["prior_precision"] = lla.prior_precision.item()
test_metrics["seed"] = args.seed

# Rotation angles
angles = np.arange(10, 190, 10)

for angle in angles:
    dataset = CIFAR10_Rotated_Dataset(angle=angle, transform=data_transform)
    _, test_dataset = dataset.train_test_splits()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    metrics = score(
        lla,
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
    path_or_buf=f"results/cifar10/lla_{args.subset}_{args.hessian}_{args.resnet}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)


dataset = CIFAR10_OOD_Dataset(transform=data_transform)
_, test_dataset = dataset.train_test_splits()

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

predictions = []
metrics = score(
    lla,
    test_loader,
    OOD,
    verbose=False,
)

labels_ood = metrics["labels"]
predictions_ood = metrics["preds"]

np.savetxt("results/cifar10_ood/labels.txt", labels_ood)
np.savetxt(
    f"results/cifar10_ood/lla_{args.subset}_{args.hessian}_{args.resnet}_{args.seed}.txt",
    predictions_ood,
)
