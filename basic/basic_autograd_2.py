import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

x = Variable(torch.randn(5,3))
y = Variable(torch.randn(5,2))

linear = nn.Linear(3,2)
print('w: ',linear.weight)
print('b: ',linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred,y)

print('loss:',loss.data[0])

loss.backward()

print('dL/dw: ',linear.weight.grad)
print('dL/db: ',linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred,y)
print('loss after 1 step: ',loss.data[0])
