from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter

img_path=('hymenoptera_data\\train\\ants\\0013035.jpg')
img=Image.open(img_path)
img_array=np.array(img)
cv_img = cv2.imread(img_path)

tensor_trans=transforms.ToTensor()
Tensor_img=tensor_trans(img)    
# print(Tensor_img) 

writer=SummaryWriter('logs')
writer.add_image("Tensor_img",Tensor_img)
writer.close()
