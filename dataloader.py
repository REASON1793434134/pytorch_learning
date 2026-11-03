import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

# 测试数据集第一张样本
img,target=test_data[0]
print(target)
print(img.shape)
writer=SummaryWriter("test_loader")


for epoch in range(2):
    step=0
    for data in test_loader:
        imgs,targets=data
        writer.add_images(f"Epoch:{epoch}",imgs,step)
        step=step+1
        # print(imgs.shape)
        # print(targets)
writer.close()