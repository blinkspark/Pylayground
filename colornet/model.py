# %%
import torch, torchvision
from torch import nn
from torch.nn import functional as F
# from torchvision.datasets import CIFAR10


class ColorNet(nn.Module):

  def __init__(self) -> None:
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 64, 7, 2, 3),
        nn.SiLU(),
        nn.GroupNorm(1, 64),
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(64, 128, 7, 2, 3),
        nn.SiLU(),
        nn.GroupNorm(1, 128),
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(128, 256, 7, 2, 3),
        nn.SiLU(),
        nn.GroupNorm(1, 256),
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(256, 512, 7, 2, 3),
        nn.SiLU(),
        nn.GroupNorm(1, 512),
    )

    self.dconv1 = nn.Sequential(
        nn.ConvTranspose2d(512, 256, 7, 2, padding=3, output_padding=1),
        nn.SiLU(),
        nn.GroupNorm(1, 256),
    )

    self.dconv2 = nn.Sequential(
        nn.ConvTranspose2d(512, 128, 7, 2, padding=3, output_padding=1),
        nn.SiLU(),
        nn.GroupNorm(1, 128),
    )

    self.dconv3 = nn.Sequential(
        nn.ConvTranspose2d(256, 64, 7, 2, padding=3, output_padding=1),
        nn.SiLU(),
        nn.GroupNorm(1, 64),
    )

    self.dconv4 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 7, 2, padding=3, output_padding=1),
        nn.SiLU(),
        nn.GroupNorm(1, 64),
    )

    self.outconv = nn.Sequential(
        nn.Conv2d(64, 2, 7, padding=3),
        nn.Tanh(),
    )

  def padcat(self, x1: torch.Tensor, x2: torch.Tensor):
    diff_h = x2.shape[2] - x1.shape[2]
    diff_w = x2.shape[3] - x1.shape[3]

    if diff_h > 0 or diff_w > 0:
      x1 = F.pad(x1, [
          diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2
      ])
    return torch.cat([x1, x2], dim=1)

  def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x = self.conv4(x3)

    x = self.dconv1(x)
    x = self.dconv2(self.padcat(x3, x))
    x = self.dconv3(self.padcat(x2, x))
    x = self.dconv4(self.padcat(x1, x))

    return self.outconv(x)
