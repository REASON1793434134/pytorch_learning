import torch
import torchvision
from model_save import *


# 方式一：模型+参数一起取
model1=torch.load("vgg16_method1.pth",weights_only=False)
print(model1)

# 方式二：取存的参数
model2=torch.load("vgg16_method2.pth",weights_only=False)
print(model2)

# 方式三：取存的参数放到模型中
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)