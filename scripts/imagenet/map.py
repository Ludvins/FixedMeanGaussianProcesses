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

import torchvision
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
from tqdm import tqdm

from bayesipy.utils.datasets import Rotated_Imagenet_Dataset
from bayesipy.utils.metrics import SoftmaxClassification, score

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--resnet", type=str, default="resnet20", help="Resnet model")
parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
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

test_dataset = torchvision.datasets.ImageNet(
    root=DATA_ROOT, split="val", transform=weights.transforms()
)

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)


def predict(self, X):
    F_mean = self(X)
    F_var = torch.eye(1000).to(DEVICE).to(DTYPE) * 1e-6
    F_var = F_var.unsqueeze(0).tile(F_mean.shape[0], 1, 1)
    return F_mean, F_var


resnet.predict = predict.__get__(resnet)
resnet.device = DEVICE
resnet.dtype = DTYPE

# Train the model
train_start = time()
train_end = time()

# Test the model
test_start = time()
test_metrics = score(resnet, test_loader, SoftmaxClassification, verbose=True)
test_end = time()

test_metrics["method"] = "map"
test_metrics["train_time"] = 0
test_metrics["test_time"] = test_end - test_start
test_metrics["resnet"] = args.resnet
test_metrics["seed"] = args.seed

# Rotation angles
angles = np.arange(10, 190, 10)

for angle in tqdm(angles):
    dataset = Rotated_Imagenet_Dataset(
        angle=angle, data_dir=DATA_ROOT, transform=weights.transforms()
    )
    _, test_dataset = dataset.train_test_splits()
    test_loader = DataLoader(test_dataset, batch_size=32)

    metrics = score(
        resnet,
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
    path_or_buf=f"results/imagenet/map_{args.resnet}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)
