# %%
import torch
import numpy as np
t = torch.Tensor()
print(type(t))

print(t.dtype)
print(t.device)
print(t.layout)

device = torch.device('cuda:0')
print(device)

t1 = torch.tensor([1, 2, 3])
t2 = t1.cuda()
# print(t1+t2) this will show error
t1 = t1.cuda()
print(t1+t2)

data = np.array([1, 2, 3])
print(type(data))

print(torch.Tensor(data))

print(torch.tensor(data))

print(torch.from_numpy(data))

print(torch.eye(4))

print(torch.zeros(3, 5))
print(torch.ones(2, 5))
print(torch.rand(2, 3))
