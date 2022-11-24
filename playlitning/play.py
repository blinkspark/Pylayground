import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# data = torch.randn(64, 3, 32, 32)
# data1 = torch.randn(64, 3, 32, 32)

data = torch.zeros(32)
data1 = torch.zeros(32)

# y = nn.Conv2d(3,16,kernel_size=5,padding=2)(data)
# print(y.shape)

dd = torch.cat([data, data1])
print(dd.shape)
