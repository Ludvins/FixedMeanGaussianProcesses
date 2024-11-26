import copy
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
import os


import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops

target = 0
dim = 64


class MyTransform:
    def __call__(self, data):
        data = copy.copy(data)
        data.y = data.y[:, target]  # Specify target.
        return data


class Complete:
    def __call__(self, data):
        data = copy.copy(data)
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data', 'QM9')
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
print(path)
dataset = QM9(path, transform=transform).shuffle()


# Split datasets.
test_dataset = dataset[:10_000]
val_dataset = dataset[10_000:20_000]
train_dataset = dataset[20_000:]

# Normalize targets to mean = 0 and std = 1.
mean = train_dataset.data.y.mean(dim=0, keepdim=True)
std = train_dataset.data.y.std(dim=0, keepdim=True)

train_dataset.data.y = (train_dataset.data.y - mean) / std
val_dataset.data.y = (val_dataset.data.y - mean) / std
test_dataset.data.y = (test_dataset.data.y - mean) / std

mean, std = mean[:, target].item(), std[:, target].item()

batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader_no_norm = DataLoader(dataset[20_000:], batch_size=batch_size, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)
        
    def embedding_h(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        return out
    
    
    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        return out.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)

from tqdm import tqdm

best_val_error = None
for epoch in tqdm(range(1, 101)):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    test_error = test(test_loader)
    train_error = test(train_loader_no_norm)
    scheduler.step(val_error)


    print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
          f'Train MAE: {train_error:.7f}', f'Val MAE: {val_error:.7f}', f'Test MAE: {test_error:.7f}')

    # Create dir for saved models
    path = "QM9_models/"
    if not osp.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), path + 'CNN_{}.pth'.format(epoch))


print("Train MAE: {}".format(test(train_loader_no_norm)))
print("Val MAE: {}".format(test(val_loader)))
print("Test MAE: {}".format(test(test_loader)))