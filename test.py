import torch
from  PIL import Image
from torch import nn
from torchvision import transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dog_img_path="./test_imgs/dog.png"
plane_img_path="./test_imgs/plane.png"

dog_img=Image.open(dog_img_path)
plane_img=Image.open(plane_img_path)
dog_img=dog_img.convert('RGB')
plane_img=plane_img.convert('RGB')
print(dog_img.size)

transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

dog_img=transform(dog_img)
plane_img=transform(plane_img)


dog_img=torch.reshape(dog_img,(1,3,32,32))
plane_img=torch.reshape(plane_img,(1,3,32,32))


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

model=torch.load("./reason_model10.pth",weights_only=False)
model.to(device)
model.eval()
with torch.no_grad(): 
    dog_img=dog_img.to(device)  
    plane_img=plane_img.to(device)
    dog_output=model(dog_img)
    plane_output=model(plane_img)
dog_output=dog_output.argmax(1)
plane_output=plane_output.argmax(1)
print(dog_output)
print(plane_output)

