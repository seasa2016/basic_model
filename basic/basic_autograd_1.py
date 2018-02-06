import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

x = Variable(torch.Tensor([1]),requires_grad = True)
w = Variable(torch.Tensor([2]),requires_grad = True)
b = Variable(torch.Tensor([3]),requires_grad = True)

y = w * x + b
y.backward()

print(w.grad)
print(x.grad)
print(b.grad)