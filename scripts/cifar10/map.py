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

from bayesipy.utils.datasets import CIFAR10_Dataset, CIFAR10_Rotated_Dataset, CIFAR10_OOD_Dataset
from bayesipy.utils.metrics import SoftmaxClassification, OOD, score
from bayesipy.utils.pretrained_models import CIFAR10_Resnet

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

# Create data loaders
batch_size = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


def predict(self, X):
    F_mean = self(X)
    F_var = torch.eye(10).to(DEVICE).to(DTYPE) * 1e-6
    F_var = F_var.unsqueeze(0).tile(F_mean.shape[0], 1, 1)
    return F_mean, F_var


resnet.predict = predict.__get__(resnet)


# Train the model
train_start = time()
train_end = time()

# Test the model
test_start = time()
test_metrics = score(
    resnet,
    test_loader,
    SoftmaxClassification,
    verbose=True,
    device=DEVICE,
    dtype=DTYPE,
)
test_end = time()

test_metrics["method"] = "map"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["resnet"] = args.resnet
test_metrics["seed"] = args.seed

# Rotation angles
angles = np.arange(10, 190, 10)

for angle in tqdm(angles):
    dataset = CIFAR10_Rotated_Dataset(angle=angle, transform=data_transform)
    _, test_dataset = dataset.train_test_splits()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
    path_or_buf=f"results/cifar10/map_{args.resnet}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)

dataset = CIFAR10_OOD_Dataset(transform=data_transform)
_, test_dataset = dataset.train_test_splits()

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

predictions = []
metrics = score(
    resnet,
    test_loader,
    OOD,
    verbose=False,
)

labels_ood = metrics["labels"]
predictions_ood = metrics["preds"]

np.savetxt("results/cifar10_ood/labels.txt", labels_ood)
np.savetxt(
    f"results/cifar10_ood/map_{args.resnet}_{args.seed}.txt",
    predictions_ood,
)
