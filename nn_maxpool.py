import torch
from torch import nn
from torch.nn import MaxPool2d
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input =torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]],dtype=torch.float32)
# input =torch.reshape(input,(-1,1,5,5))
# print(input.shape)

dataset=CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True)
writer=SummaryWriter('pool_max')

class Reason_Modul(nn.Module):
    def __init__(self):
        super(Reason_Modul,self).__init__()
        self.maxpool=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool(input)
        return output
    
reason =Reason_Modul()
step=0



for data in dataloader:
    imgs,targets=data
    writer.add_images('imgs',imgs,global_step=step)
    output=reason(imgs)
    output=torch.reshape(output,(-1,3,11,11))
    print(output.shape)
    writer.add_images('max_o=pool',output,global_step=step)
    step=step+1
writer.close()
 

# imgs,target=next(iter(dataloader))
# output=reason(imgs)
# print(imgs.shape)
# print(output.shape)

