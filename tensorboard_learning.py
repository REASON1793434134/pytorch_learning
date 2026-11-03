from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer= SummaryWriter('logs')
# y=2x
# for i in range(100):
      # writer.add_scalar("y=2x",2*i,i)


# 关于图片
img_path=('hymenoptera_data\\train\\ants\\5650366_e22b7e1065.jpg')
img=Image.open(img_path)
img_array=np.array(img)
print(img_array.shape)

writer.add_image("test",img_array,2,dataformats='HWC')
writer.close()