import torchvision
from torch.utils.tensorboard import SummaryWriter
dataset_trans=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set =torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_trans,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_trans,download=True)

writer=SummaryWriter("P10")
for i in range(10):
    img,target=test_set[i]
    writer.add_image("test_dataset",img,i)
writer.close()





# print(test_set[0])
# print(test_set.classes)
# img,target=test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])