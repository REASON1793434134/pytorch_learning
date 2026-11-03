import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64,shuffle=True)

class Reason_Module(nn.Module):
    def __init__(self):
        super(Reason_Module,self).__init__()
        self.conv=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    
    def forward(self,x):
        x=self.conv(x)
        return x

reason=Reason_Module()
writer=SummaryWriter("conv")

conv1=Conv2d(1,1,2,1,0)
data=torch.tensor([[1,2,1],[1,2,1],[1,2,1]],dtype=torch.float32)
data=torch.reshape(data,(1,1,3,3))
print(data)
output=conv1(data)
print(output)
# step=0
# for data in dataloader:
#     imgs,targets=data
#     output=reason(imgs)
#     print(output.shape)
#     writer.add_images("con1",imgs,step)
#     output=torch.reshape(output,(-1,3,30,30))
#     writer.add_images("con2",output,step)
#     step=step+1
# writer.close()



# imgs,targets=next(iter(dataloader))
# print(imgs.shape)
# print(targets)
# a=torch.tensor([[1,23,4,5,6,7],[1,23,4,5,6,7],[1,23,4,5,6,7],[1,23,4,5,6,7],[1,23,4,5,6,7],[1,23,4,5,6,7]], dtype=torch.float32)
# a = a.unsqueeze(0).unsqueeze(0)
# print(reason(a))

        