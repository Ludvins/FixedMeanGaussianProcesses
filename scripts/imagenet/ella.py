# Global Imports
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Local Imports
sys.path.append(".")

import argparse
import torchvision
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import ResNet34_Weights, resnet34
from time import time

from tqdm import tqdm

from bayesipy.laplace import ELLA
from bayesipy.utils.datasets import Imagenet_Dataset, Rotated_Imagenet_Dataset
from bayesipy.utils.metrics import SoftmaxClassification, score

torch.backends.cudnn.benchmark = True

# Add parser for resnet model and seed
parser = argparse.ArgumentParser(description="VFSUE")
parser.add_argument("--resnet", type=str, default="resnet20", help="Resnet model")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lower_da", type=float, default=0.5)

parser.add_argument("--seed", type=int, help="Seed")
args = parser.parse_args()

# Set the seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
SEED = 2147483647 - args.seed
DATA_ROOT = "/scratch/ludvins/ImageNet/"

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


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


dataset = Imagenet_Dataset(data_dir = DATA_ROOT,
    transform=weights.transforms(),
)

# Split the dataset
train_dataset, test_dataset = dataset.train_test_splits()
val_dataset = dataset.validation_split_ella(lower = args.lower_da)

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
    prior_precision=1e-4 * 300,
    seed=args.seed,
)
wd = 1e-4

# Train the model
train_start = time()
data_seen = ella.fit(train_loader=train_loader,
                     val_loader = val_loader,
                     val_steps = 1,
                     balanced = True,
                     update_prior_precision=True,
                     weight_decay=wd,
                     verbose=True)
train_end = time()

# Test the model
test_start = time()
test_metrics = score(ella, test_loader, SoftmaxClassification, verbose=True)
test_end = time()

test_metrics["method"] = "ella"
test_metrics["train_time"] = train_end - train_start
test_metrics["test_time"] = test_end - test_start
test_metrics["resnet"] = args.resnet
test_metrics["prior_precision"] = ella._prior_precision.detach().cpu().item()
test_metrics["seed"] = args.seed
test_metrics["batch_size"] = batch_size

print(test_metrics)

# Rotation angles
# angles = np.arange(10, 190, 10)

# for angle in tqdm(angles):
#     dataset = Rotated_Imagenet_Dataset(angle=angle,data_dir = DATA_ROOT, transform=weights.transforms())
#     _, test_dataset = dataset.train_test_splits()
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     metrics = score(
#         ella,
#         test_loader,
#         SoftmaxClassification,
#         verbose=False,
#     )
#     for key, value in metrics.items():
#         test_metrics[f"rotated_{angle}_{key}"] = value

# Save csv
df = pd.DataFrame.from_dict(test_metrics, orient="index").transpose()
print(df)
df.to_csv(
    path_or_buf=f"results/imagenet/ella_{batch_size}_{args.resnet}_{args.lower_da}_{args.seed}.csv",
    encoding="utf-8",
    index=False,
)
