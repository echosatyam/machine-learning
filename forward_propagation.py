# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

torch.set_printoptions(linewidth=120)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        # t = F.softmax(t, dim=1)
        return t


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

network = Network()

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=100
)
optimizer = optim.Adam(network.parameters(), lr=.01)

for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for batch in train_loader:
        batch = next(iter(train_loader))
        images, labels = batch
        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = network(images)
        loss = F.cross_entropy(preds, labels)
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
    print(
        f"epoch: {epoch} total_correct: {total_correct} \
                    total_loss: {total_loss}"
    )
    print(f"accuracy of epoch {epoch}: {total_correct/len(train_set)}")
# %%
len(train_set)
len(train_set.targets)


def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds


prediction_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=10000
)
train_preds = get_all_preds(network, prediction_loader)
print(train_preds.shape)
