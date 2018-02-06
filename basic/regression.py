import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot  as plt
from torch.autograd import Variable

input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)


y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        out = self.linear(x)
        return out

model = LinearRegression(1,1)
model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
    inputs = Variable(torch.Tensor(x_train).cuda())
    target = Variable(torch.Tensor(y_train).cuda())

    outputs = model(inputs)

    optimizer.zero_grad()

    loss = criterion(outputs,target)
    loss.backward()
    optimizer.step()

    if((epoch % 5)==4):
        print('Epoch [%d/%d] loss :%f'%(epoch,num_epochs,loss.data[0]))

predicted = model(Variable(torch.from_numpy(x_train).cuda())).data.cpu().numpy()

plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')

plt.legend()
plt.show()

torch.save(model.state_dict(), 'model.pkl')

    