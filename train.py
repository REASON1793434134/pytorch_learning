import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

# 选择设备
device=torch.device("cuda" if torch.cuda.is_available else "cpu")

# 准备数据集
train_data=torchvision.datasets.CIFAR10("./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

# 训练集和测试集长度
train_data_size=len(train_data)
test_data_size=len(test_data)
print(f"训练测试集的长度为{train_data_size}")
print(f"测试测试集的长度为{test_data_size}")

# 加载数据
train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)

# 创建网络
class Reason_module(nn.Module):
    def __init__(self):
        super(Reason_module,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model(x)
        return x

# 测试网络正确性
reason=Reason_module()
# if torch.cuda.is_available():
#     reason=reason.cuda()
reason.to(device)
input=torch.ones((64,3,32,32))
input=input.cuda()
output=reason(input)
print(output.shape)

# 损失函数
loss_fn=nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fn=loss_fn.cuda()
loss_fn.to(device)

# 优化器
learning_rate=0.01
optim=torch.optim.SGD(params=reason.parameters(),lr=learning_rate)

# tensorboard
writer=SummaryWriter("logs")

# 设置训练和测试的次数，训练轮数
total_train_step=0
total_test_step=0
epoch=50

for i in range(epoch):
    print(f"---------这是第{i+1}轮训练-----------")
    time_start=time.time()
    # 训练开始
    reason.train()
    for data in train_dataloader:
        imgs,targets=data
        # 选用设备写法一
        # if torch.cuda.is_available():
        #     imgs=imgs.cuda()
        #     targets=targets.cuda()
        # 选用设备写法二
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs=reason(imgs)
        # 损失函数模型
        loss=loss_fn(outputs,targets)
        # 优化器模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step+=1
        if total_train_step % 100==0:
            print(f'训练次数:{total_train_step},LOSS:{loss.item()}')
            writer.add_scalar("train_loss",scalar_value=loss,global_step=total_train_step)

    # 测试开始
    reason.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            # 选用设备写法一
            # if torch.cuda.is_available():
            #     imgs=imgs.cuda()
            #     targets=targets.cuda()
            # 选用设备写法二
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs=reason(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=loss.item()+total_test_loss
            accruacy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accruacy
    total_test_step+=1
    print(f"整体测试集的LOSS:{total_test_loss}")
    print(f"整体正确率为{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss",scalar_value=total_test_loss,global_step=total_test_step)
    writer.add_scalar("test_accuracy",scalar_value=total_accuracy,global_step=total_test_step)
    if (i+1) % 10==0 :
        torch.save(reason,f"./reason_model{i+1}.pth")
    time_end=time.time()
    print(f"第{i}轮的时间为{time_end-time_start}")


