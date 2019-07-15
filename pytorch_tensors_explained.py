import torch
import numpy as np
data = np.array([1, 2, 3])
t1 = torch.Tensor(data)
t2 = torch.tensor(data)
t3 = torch.as_tensor(data)
t4 = torch.from_numpy(data)

print(t1)
print(t2)
print(t3)
print(t4)
print(type(t1), t1.dtype)
print(type(t2), t2.dtype)
print(type(t3), t3.dtype)
print(type(t4), t4.dtype)
print(torch.get_default_dtype())
#t1 = torch.tensor([1,2,3],dtype=torch.float64)
print(t1, t1.dtype)
data[0] = 0
data[1] = 0
data[2] = 0
print(t1)
print(t2)
print(t3)
print(t4)
print(type(t1), t1.dtype)
print(type(t2), t2.dtype)
print(type(t3), t3.dtype)
print(type(t4), t4.dtype)
