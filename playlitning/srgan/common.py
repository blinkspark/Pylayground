import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image

class SRDataset(Dataset):

  def __init__(self, srcDS: Dataset, target_size=16):
    super().__init__()
    self.srcDS = srcDS
    self.target_size = target_size
    self.resize = transforms.Resize(16)

  def __len__(self):
    return len(self.srcDS)

  def __getitem__(self, index):
    x, _ = self.srcDS[index]
    x0 = self.resize(x)
    return x0, x


class ConvBlk(nn.Module):

  def __init__(self,
               inputs: int,
               outputs: int,
               mid=None,
               kernel_size=5,
               padding=2):
    super().__init__()
    if mid is None:
      mid = outputs
    self.blk = nn.Sequential(
        nn.Conv2d(inputs, mid, kernel_size=kernel_size, padding=padding),
        nn.SiLU(inplace=True),
        nn.GroupNorm(1, mid),
        nn.Conv2d(mid, outputs, kernel_size=kernel_size, padding=padding),
        nn.SiLU(inplace=True),
        nn.GroupNorm(1, outputs),
    )

  def forward(self, x):
    return self.blk(x)