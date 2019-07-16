# %%
import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
print(len(train_set))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
print(train_loader)
print(len(train_set.train_labels))
print(train_set.train_labels.bincount())

sample = next(iter(train_set))

print(len(sample))
print(type(sample))

image, label = sample

plt.imshow(image.squeeze(), cmap='gray')
print('label: ', label)

batch = next(iter(train_loader))
print(len(batch))
type(batch)
images, labels = batch
grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
print('labels: ', labels)
plt.show()
print(images.shape)
# %%
