from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img=Image.open("test.png").convert("RGB")
# Totensor
trans_Totensor=transforms.ToTensor()
img_tensor=trans_Totensor(img)
writer.add_image("To_tensor",img_tensor)

# Normalize
trans_Normalize =transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_Normalize(img_tensor)
print(img_tensor[0][0][0])
print(img_norm[0][0][0])

# resize
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)
img_resize=trans_Totensor(img_resize)
writer.add_image("resize",img_resize,1)


# compose
trans_resize2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize2,trans_Totensor])
img_resize2=trans_compose(img)
print(img_resize2)
writer.add_image("resize",img_resize2,2)
writer.close()

# random
trans_random=transforms.RandomCrop(512)
trans_compose2=transforms.Compose([trans_random,trans_Totensor])
for i in range(10):
    img_random = trans_compose2(img)
    writer.add_image("random_crop",img_random,i)
