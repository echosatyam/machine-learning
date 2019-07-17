import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)

        self.fc1 = nn.Linear(in_features = 12*4*4,out_features=120)
        self.fc2 = nn.Linear(in_features = 120,out_features=60)
        self.out = nn.Linear(in_features = 60,out_features=10)

        self.layer = None

    def forward(self, t):
        t = self.layer(t)
        return t
network = Network()
print(network)
print(network.conv1.weight.shape)
print(network.conv2.weight.shape)
print(network.fc1.weight.shape)
print(network.fc2.weight.shape)
print(network.out.weight.shape)
# print(network.conv2.weight)
print(network.fc1)
in_features = torch.tensor([1,2,3,4,5],dtype = torch.float32)
weight_matrix = torch.tensor([
    [1,2,3,4,1],
    [2,3,4,5,7],
    [21,3,4,5,7]
],dtype=torch.float32)
print(weight_matrix.matmul(in_features))


for name, param in network.named_parameters():
    print(name,param.shape)
