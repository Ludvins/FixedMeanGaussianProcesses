# Global Imports
import sys

sys.path.append(".")
sys.path.append("..")

import argparse
import copy
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
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

from bayesipy.fmgp import FMGP
from bayesipy.fmgp.kernels import LastLayerNTK_SquaredExponential
from bayesipy.utils.datasets import Imagenet_Dataset, Rotated_Imagenet_Dataset

# Local imports
from bayesipy.utils.metrics import SoftmaxClassification, score

torch.backends.cudnn.benchmark = True


# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--resnet", type=str, default="resnet20", help="Resnet model")
parser.add_argument(
    "--initial_lr", type=float, default=1e-3, help="Initial learning rate"
)
parser.add_argument(
    "--scheduler_gamma", type=float, default=0.8, help="Scheduler gamma"
)
parser.add_argument(
    "--iterations", type=int, default=10000, help="Number of iterations"
)
parser.add_argument(
    "--scheduler_steps", type=int, default=1000, help="Number of iterations"
)
parser.add_argument("--batch_size", type=int, default=32, help="Number of iterations")

parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()


DEVICE = "cuda:0"
DTYPE = torch.float64
SEED = 2147483647 - args.seed
DATA_ROOT = "/scratch/ludvins/ImageNet/"

torch.manual_seed(SEED)

if args.resnet == "resnet18":
    weights = ResNet18_Weights.IMAGENET1K_V1
    network = resnet18(weights=weights)

if args.resnet == "resnet34":
    weights = ResNet34_Weights.IMAGENET1K_V1
    network = resnet34(weights=weights)

if args.resnet == "resnet50":
    weights = ResNet50_Weights.IMAGENET1K_V1
    network = resnet50(weights=weights)

if args.resnet == "resnet101":
    weights = ResNet101_Weights.IMAGENET1K_V1
    network = resnet101(weights=weights)

if args.resnet == "resnet152":
    weights = ResNet152_Weights.IMAGENET1K_V1
    network = resnet152(weights=weights)

network.eval()
classifier = copy.deepcopy(network.fc).to(DEVICE).to(DTYPE)
embedding = network.to(DEVICE).to(DTYPE)
embedding.fc = torch.nn.Identity()

dataset = Imagenet_Dataset(
    data_dir=DATA_ROOT,
    transform=weights.transforms(),
)

# Split the dataset
train_dataset, test_dataset = dataset.train_test_splits()

# Create data loaders
batch_size = args.batch_size
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
)


IMAGENET_MEDIAN = np.exp(5.7466025)
IMAGENET_AMPLITUDE = 0.001
kernel = LastLayerNTK_SquaredExponential(
    initial_length_scale=IMAGENET_MEDIAN,
    initial_amplitude=IMAGENET_AMPLITUDE,
    n_features=(3, 224, 224),
    n_outputs=1000,
    device=DEVICE,
    dtype=DTYPE,
)


# Create the VFSUE model
fmgp = FMGP(
    embedding=embedding,
    classifier=classifier,
    likelihood="classification",
    kernel=kernel,
    subrogate_regularizer=True,
    inducing_locations="random",
    num_inducing=20,
    seed=args.seed,
)

# Train the model
train_start = time()
loss = fmgp.fit(
    iterations=args.iterations,
    lr=args.initial_lr,
    scheduler_gamma=args.scheduler_gamma,
    scheduler_steps=args.scheduler_steps,
    train_loader=train_loader
)
train_end = time()


# Test the model
test_start = time()
test_metrics = score(
    fmgp,
    test_loader,
    SoftmaxClassification,
    verbose=True,
)
test_end = time()

test_metrics["method"] = "fmgp"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["resnet"] = args.resnet
test_metrics["initial_lr"] = args.initial_lr
test_metrics["scheduler_gamma"] = args.scheduler_gamma
test_metrics["scheduler_steps"] = args.scheduler_steps
test_metrics["batch_size"] = args.batch_size
test_metrics["iterations"] = args.iterations
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
        fmgp,
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
    path_or_buf=f"results/imagenet/fmgp_{args.resnet}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)
