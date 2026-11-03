import torch

outputs=torch.tensor([[0.3,0.9],
                      [0.5,0.7]])
print(outputs.argmax(1))
targets=torch.tensor([1,0])
preds=outputs.argmax(1)
print(preds==targets)
print((preds==targets).sum())
