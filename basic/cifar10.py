import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

train_dataset = dsets.CIFAR10(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

image,label = train_dataset[0]
print(image.size())
print(label)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=2)

data_iter = iter(train_loader)

images,labels = data_iter.next()

for images,labels in train_loader:
    print(labels)
