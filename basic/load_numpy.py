import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

a = np.array([[1,2],[3,4]])
b = torch.from_numpy(a)
c = b.numpy()

print(a)
print(b)
print(c)