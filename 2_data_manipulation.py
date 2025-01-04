import torch


x = torch.arange(12, dtype=torch.float32) \
    .reshape(3,-1)
x

torch.randn((2, 3, 4))

torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

x[-1], x[0:2]

x[1, 2] = 17
x