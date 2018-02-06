import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets 
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(   mean = (0.5,0.5,0.5),
                                            std = (0.5,0.5,0.5))])

mnist = datasets.MNIST( root = './data/',
                        train = True,
                        transform = transform,
                        download = True)

data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=100, 
                                          shuffle=True)


batch_size = 100

D = nn.Sequential(
    nn.Linear(784,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,1),
    nn.Sigmoid()
    )

G = nn.Sequential(
    nn.Linear(64,256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,256),
    nn.LeakyReLU(0.2,),
    nn.Linear(256,784),
    nn.Tanh()
    )

D = D.cuda()
G = G.cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

real_labels = to_var(torch.ones(batch_size))
fake_labels = to_var(torch.zeros(batch_size))


for epoch in range(200):
    for i,(images,_) in enumerate(data_loader):
        
        batch_size = images.size(0)
        images = to_var(images.view(batch_size,-1))

        
        outputs = D(images)
        d_loss_real = criterion(outputs,real_labels)
        real_score = outputs

        z = to_var(torch.randn(batch_size,64))
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs,fake_labels)
        fake_score = outputs

        d_loss = d_loss_fake + d_loss_real
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        z = to_var(torch.randn(batch_size,64))
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)

        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                  'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f' 
                  %(epoch, 200, i+1, 600, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.view(images.size(0), 1, 28, 28)
        save_image(denorm(images.data), './data/real_images.png')
    
    # Save sampled images
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './data/fake_images-%d.png' %(epoch+1))

# Save the trained parameters 
torch.save(G.state_dict(), './generator.pkl')
torch.save(D.state_dict(), './discriminator.pkl')