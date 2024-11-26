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

import tqdm
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from bayesipy.laplace import Laplace
from bayesipy.utils.datasets import Imagenet_Dataset, Rotated_Imagenet_Dataset
from bayesipy.utils.metrics import SoftmaxClassification, score

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
DATA_ROOT = "/scratch/ludvins/ImageNet/"

torch.manual_seed(SEED)

if args.resnet == "resnet18":
    weights = ResNet18_Weights.IMAGENET1K_V1
    resnet = resnet18(weights=weights)

if args.resnet == "resnet34":
    weights = ResNet34_Weights.IMAGENET1K_V1
    resnet = resnet34(weights=weights)

if args.resnet == "resnet50":
    weights = ResNet50_Weights.IMAGENET1K_V1
    resnet = resnet50(weights=weights)

if args.resnet == "resnet101":
    weights = ResNet101_Weights.IMAGENET1K_V1
    resnet = resnet101(weights=weights)

if args.resnet == "resnet152":
    weights = ResNet152_Weights.IMAGENET1K_V1
    resnet = resnet152(weights=weights)


resnet.to(DEVICE).to(DTYPE)
resnet.eval()

dataset = Imagenet_Dataset(
    data_dir=DATA_ROOT,
    transform=weights.transforms(),
)

# Split the dataset
train_dataset, test_dataset = dataset.train_test_splits()


# Create data loaders
batch_size = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
print(test_metrics)
# Rotation angles
angles = np.arange(10, 190, 10)

for angle in tqdm(angles):
    dataset = Rotated_Imagenet_Dataset(
        angle=angle, data_dir=DATA_ROOT, transform=weights.transforms()
    )
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
    path_or_buf=f"results/imagenet/lla_{args.subset}_{args.hessian}_{args.resnet}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)
