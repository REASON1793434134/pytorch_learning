import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64,shuffle=True)


class Reason_Module(nn.Module):
    def __init__(self):
        super(Reason_Module,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x



reason=Reason_Module()
writer=SummaryWriter("nn_conv")
step=0
for data in dataloader:
    imgs,targets=data
    output=reason(imgs)
    print(output.shape)
    writer.add_images("nn_conv1",imgs,step)
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("nn_conv2",output,step)
    step=step+1
writer.close()