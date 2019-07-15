import torch
t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]
], dtype=torch.float32)
print(t.shape)
x = torch.tensor(t.shape).prod()
print(x)
print(t.reshape(1, 48))


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


print(flatten(t))
