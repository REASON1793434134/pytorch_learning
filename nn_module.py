from torch import nn
import torch


class Reason_Module(nn.Module):
    def  __init__(self):
        super(Reason_Module,self).__init__()


    def forward(self,input):
        output =input +1
        return output

reason=Reason_Module()
x=torch.tensor(1.0)
output=reason(x)
print(output)
 