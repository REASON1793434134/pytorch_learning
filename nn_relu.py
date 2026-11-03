import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
input=torch.tensor([[1,-1],[-2,4]],dtype=torch.float32)
input=torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset=CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True)
writer=SummaryWriter('nn_sigmoid')


class Reason_Module(nn.Module):
    def __init__(self):
        super(Reason_Module,self).__init__()
        self.relu=ReLU(inplace=False)
        self.sigmoid=Sigmoid()

    def forward(self,input):
        output=self.sigmoid(input)
        return output

reason=Reason_Module()
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images('imgs',imgs,global_step=step)
    output=reason(imgs)
    writer.add_images("sigmoid_imgs",output,global_step=step)
    step=step+1
writer.close()







# output=reason(input)
# print(output)
