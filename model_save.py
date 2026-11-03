import torch
import torchvision

vgg16=torchvision.models.vgg16(pretrained=False)


# 方式一：模型结构+模型参数
torch.save(vgg16,'./vgg16_method1.pth')

# 方式二：模型参数（官方推荐）
torch.save(vgg16.state_dict(),"./vgg16_method2.pth")


