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
from bayesipy.utils.datasets import (
    CIFAR10_Dataset,
    CIFAR10_Rotated_Dataset,
    Precomputed_Output_Embedding_Dataset,
)
from bayesipy.utils.metrics import SoftmaxClassification, score
from bayesipy.utils.pretrained_models import CIFAR10_Resnet

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--resnet", type=str, default="resnet20", help="Resnet model")
parser.add_argument(
    "--initial_lr", type=float, default=1e-4, help="Initial learning rate"
)
parser.add_argument(
    "--scheduler_gamma", type=float, default=0.85, help="Scheduler gamma"
)
parser.add_argument(
    "--iterations", type=int, default=50000, help="Number of iterations"
)
parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
SEED = 2147483647 - args.seed

torch.manual_seed(SEED)

# Load Models
resnet = CIFAR10_Resnet(args.resnet).to(DEVICE).to(DTYPE)
embedding = CIFAR10_Resnet(args.resnet, embedding=True).to(DEVICE).to(DTYPE)
classifier = CIFAR10_Resnet(args.resnet, classifier=True).to(DEVICE).to(DTYPE)

# Load the data transform
data_transform = CIFAR10_Resnet(args.resnet, get_transform=True)

# Load the dataset
dataset = Precomputed_Output_Embedding_Dataset(
    model=resnet,
    embedding=embedding,
    model_name=args.resnet,
    dataset=CIFAR10_Dataset,
    transform=data_transform,
)

# Split the dataset
train_dataset, test_dataset = dataset.train_test_splits()

# Create data loaders
batch_size = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create the VFSUE model
ue = FMGP(
    embedding=embedding,
    classifier=classifier,
    likelihood="classification",
    kernel="RBFxNTK",
    subrogate_regularizer=True,
    inducing_locations="kmeans",
    num_inducing=20,
    seed=args.seed,
)

# Train the model
train_start = time()
loss, val_metrics = ue.fit(
    iterations=args.iterations,
    lr=args.initial_lr,
    scheduler_gamma=args.scheduler_gamma,
    train_loader=train_loader,
)
train_end = time()

# Test the model
test_start = time()
test_metrics = score(ue, test_loader, SoftmaxClassification, verbose=True)
test_end = time()

test_metrics["method"] = "fmgp"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["resnet"] = args.resnet
test_metrics["initial_lr"] = args.initial_lr
test_metrics["scheduler_gamma"] = args.scheduler_gamma
test_metrics["iterations"] = args.iterations
test_metrics["seed"] = args.seed

# Rotation angles
angles = np.arange(10, 190, 10)

for angle in angles:
    dataset = CIFAR10_Rotated_Dataset(angle=angle, transform=data_transform)
    _, test_dataset = dataset.train_test_splits()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    metrics = score(ue, test_loader, SoftmaxClassification, verbose=False)
    for key, value in metrics.items():
        test_metrics[f"rotated_{angle}_{key}"] = value

# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
df.to_csv(
    path_or_buf=f"results/cifar10/fmgp_{args.resnet}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)
