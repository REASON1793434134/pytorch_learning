from torch.utils.data import Dataset
import os
from PIL import Image

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path =os.path.join(self.root_dir,self.label_dir)
        self.img_path =os.listdir(self.path)

    def __getitem__(self,idx):
        img_name =self.img_path[idx]
        img_item_path= os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir ="hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir="bees"

ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)
train_dataset = ants_dataset + bees_dataset

img,label=bees_dataset[1]
print(label)
img.show()
print(ants_dataset.img_path[0])
print(len(ants_dataset))





#img_path="C:\\Users\\Acer\\Desktop\\pytorch-learning\\hymenoptera_data\\train\\ants\\0013035.jpg"
#img=Image.open(img_path)
#print(img.size)
#img.show()

