import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 784
num_class = 10
num_epoch = 10
batch_size = 100
learning_rate = 0.001

train_dataset =  dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download= True
                            )

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor()
                            )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False
                                            )
                                
class LogisticRegregression(nn.Module):
    def __init__(self,input_size,num_class):
        super(LogisticRegregression,self).__init__()
        self.linear = nn.Linear(input_size,num_class)
        
    def forward(self,x):
        out = self.linear(x)
        return out

model = LogisticRegregression(input_size,num_class)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for i,(images,labels) in enumerate(train_loader):
        images = Variable(images.view(-1,28*28)).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        if(i+1)%100 == 0:
            print('Epoch: [%d/%d],step:[%d/%d],loss:%.4f'%(epoch+1,num_epoch,i+1,len(train_dataset)//batch_size,loss.data[0]))

correct = 0
total = 0
for images,labels in test_loader:
    images = Variable(images.view(-1,28*28)).cuda()
    outputs = model(images)
    
    _,predicted = torch.max(outputs.data,1)

    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()


print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(model.state_dict(), 'model.pkl')