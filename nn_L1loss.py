import torch
from torch import nn
target=torch.tensor([1,2,5],dtype=torch.float32)
output=torch.tensor([1,2,3],dtype=torch.float32)

target=torch.reshape(target,(1,1,1,3))
output=torch.reshape(output,(1,1,1,3))


loss=torch.nn.L1Loss(reduction='sum')
loss_mse=nn.MSELoss()
loss_cross=nn.CrossEntropyLoss()

target1=torch.tensor([1])
outputs=torch.tensor([0.1,0.2,0.3])
outputs=torch.reshape(outputs,(1,3))

result_cross=loss_cross(outputs,target1)
result=loss(output,target)
result_mse=loss_mse(output,target)
print(result)
print(result_mse)

print(result_cross)