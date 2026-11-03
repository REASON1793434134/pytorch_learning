import torchvision
from torchvision import models
from torch.nn import Linear
# train_data=torchvision.datasets.ImageNet('./data_image_net',split='train',download=True,transform=torchvision.transforms.ToTensor())


vgg_true=models.vgg16(pretrained=True)
vgg_false=models.vgg16(pretrained=False)

vgg_true.classifier.add_module('add_linear',Linear(1000,10))
# print(vgg_true)

vgg_false.classifier[6]=Linear(4096,10)
print(vgg_false.classifier[0])



