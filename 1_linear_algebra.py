%matplotlib inline
import torch
import torchvision
from IPython import display
from torchvision import transforms
from d2l import torch as d2l


def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))



M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M



torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))



# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape



# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v



ex1 = torch.tensor([[1,0],[2,1]], dtype=torch.float32)
ex2 = torch.tensor([[1,0],[-2,1]], dtype=torch.float32)
ex1 @ ex2