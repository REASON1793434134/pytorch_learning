import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import Flatten
from torch.nn import Sequential
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset=CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True)
writer=SummaryWriter("logs")

class Reason_module(nn.Module):
    def __init__(self):
        super(Reason_module,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2,dilation=1)
        self.maxpool1=MaxPool2d(kernel_size=2)
        self.conv2=Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2,dilation=1)
        self.maxpool2=MaxPool2d(kernel_size=2)
        self.conv3=Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2,dilation=1)
        self.maxpool3=MaxPool2d(kernel_size=2)
        self.flatten=Flatten()
        self.linear1=Linear(1024,64)
        self.linear2=Linear(64,10)
        self.model1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)
        x=self.model1(x)
        return x
resaon=Reason_module()
loss_cross=CrossEntropyLoss()
optim=torch.optim.SGD(params=resaon.parameters(),lr=0.01)

for epoch in range(10):
    running_loss=0
    for data in dataloader:
        imgs,targets=data
        outputs=resaon(imgs)
        loss=loss_cross(outputs,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss=running_loss+loss
    print(running_loss)






# print(imgs.shape)
# print(outputs)
# print(targets)
# print(lost_result)
# writer.close()