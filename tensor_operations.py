import numpy as numpy
import torch

t1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
t2 = torch.tensor([[9, -2], [-7, 6]], dtype=torch.float32)
print(t1[0][0])
print(t1+t2)
print(t1+torch.tensor([1, 2], dtype=torch.float32))
print((torch.tensor(t2.ge(2), dtype=torch.float32)*t2))
print(t1.abs())
t3 = t2.sqrt()
t3[t3 != t3] = 0

print("sqrt", t3)
print(t2.neg())
print(t2.neg().abs())
