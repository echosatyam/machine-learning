import torch

t1 = torch.ones(4, 4, dtype=torch.int64)
t2 = t1*2
t3 = t1*3
print(t1)
print(t2)
print(t3)
t = torch.stack((t1, t2, t3))
print(t.shape)
print(t)
print(t.reshape(3, 1, 4, 4))
print(t.flatten(start_dim=1))
print(t.reshape(3, 1, 16).squeeze())
