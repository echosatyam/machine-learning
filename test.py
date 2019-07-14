import torch
t = torch.tensor([[1,2,3],[1,235,1],[12,123,1235]])
print(t)
print(t.shape)

print(t.reshape(9,1))
t = t.cuda()
print(t)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
