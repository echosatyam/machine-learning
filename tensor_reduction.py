import torch
import numpy as np
t = torch.tensor([
    [0,1,0],
    [2,0,2],
    [0,3,0]
    ],dtype=torch.float32)
print(t.sum())
print(t.sum().numel())
print(t.numel())
print(t.numel()>(t.sum().numel()))
print(t.prod())
print(t.mean())
print(t.std())
t1 = torch.ones(4,3,dtype=torch.float32)*torch.tensor([1,2,3],dtype=torch.float32)
print(t1)
print(t1.sum(dim=0))
print(t1.sum(dim=1))
print(t1[0].sum())
print(t1[1].sum())
print(t1[0].sum())