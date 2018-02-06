import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 784
hidden_size = 500
num_class = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False
                                            )


class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_class):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.relu2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size,num_class)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

net = Net(input_size,hidden_size,num_class)
net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
    for i , (images,labels) in enumerate(train_loader):
        images = Variable(images.view(-1,28*28)).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs,labels)
        loss.backward()

        optimizer.step()


        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'%(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')