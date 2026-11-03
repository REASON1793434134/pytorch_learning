import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.nn import Sigmoid
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True)


input=torch.tensor([[1,2],[3,4]],dtype=torch.float32)

class Reason_Module(nn.Module):
    def __init__(self):
        super(Reason_Module,self).__init__()
        self.linear=Linear(4,2)
        

    def forward(self,input):
        output=self.linear(input)
        return output

reason=Reason_Module()


input=torch.flatten(input)
print(input.shape)
output=reason(input)
print(output.shape)
print(output)






# data=next(iter(dataloader))
# imgs,targets=data
# print(imgs.shape)
# # output=torch.reshape(imgs,(1,1,1,-1))
# output=torch.flatten(imgs)
# print(output.shape)
# output=reason(output)
# print(output.shape)

