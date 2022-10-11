# %%
from re import X
import torch, torchvision
from torch import nn
from torch.nn import functional as F
# from torchvision.datasets import CIFAR10


class DoubleConv2d(nn.Module):

  def __init__(self,
               in_channel: int,
               out_channel: int,
               mid_channel=None,
               kernel_size=7,
               padding=3) -> None:
    super().__init__()
    mid_channel = out_channel if mid_channel == None else mid_channel
    self.net = nn.Sequential(
        nn.Conv2d(in_channel,
                  mid_channel,
                  kernel_size=kernel_size,
                  padding=padding),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, mid_channel),
        nn.Conv2d(mid_channel,
                  out_channel,
                  kernel_size=kernel_size,
                  padding=padding),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, out_channel),
    )

  def forward(self, x):
    return self.net(x)


class Down(nn.Module):

  def __init__(self, in_channel, out_channel, mid_channel=None) -> None:
    super().__init__()
    mid_channel = out_channel if mid_channel == None else mid_channel
    self.net = nn.Sequential(
        nn.MaxPool2d(2, 2),
        DoubleConv2d(in_channel, out_channel, mid_channel),
    )

  def forward(self, x):
    return self.net(x)


class Up(nn.Module):

  def __init__(self, in_channel, out_channel, mid_channel=None) -> None:
    super().__init__()
    mid_channel = out_channel if mid_channel == None else mid_channel
    self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    self.db_conv = DoubleConv2d()

  def forward(self, x1: torch.Tensor, x2: torch.Tensor):
    x1 = self.up(x1)

    x = x1
    if x2:
      diff_h = x2.shape[2] - x1.shape[2]
      diff_w = x2.shape[3] - x1.shape[3]
      if diff_h > 0:
        x1 = F.pad(x1, [0, 0, diff_w // 2, diff_w - diff_w // 2])
      # elif diff_h < 0:
      #   diff_h = abs(diff_h)
      #   x2 = F.pad(x2, [0, 0, diff_w // 2, diff_w - diff_w // 2])

      if diff_w > 0:
        x1 = F.pad(x1, [diff_h // 2, diff_h - diff_h // 2, 0, 0])
      # elif diff_w < 0:
      #   diff_w = abs(diff_w)
      #   x1 = F.pad(x1, [diff_h // 2, diff_h - diff_h // 2, 0, 0])

      x = torch.cat([x1, x2], dim=1)
    return self.db_conv(x)


class ColorNet(nn.Module):

  def __init__(self, initial_channel=64, channel_factors=[1, 2, 4, 8]) -> None:
    super().__init__()
    channels = list(map(lambda f: initial_channel * f, channel_factors))
    self.d1 = Down(1, channels[0]),
    self.d2 = Down(channels[0], channels[1]),
    self.d3 = Down(channels[1], channels[2]),
    self.d4 = Down(channels[2], channels[3]),

    self.u1 = Up(channels[3], channels[2])
    self.u2 = Up(channels[2] * 2, channels[1])
    self.u3 = Up(channels[1] * 2, channels[1])
    self.u4 = Up(channels[0] * 2, channels[0])

    self.outconv = nn.Sequential(
        nn.Conv2d(channels[0], 2, 7, padding=3),
        nn.Tanh(),
    )

  def forward(self, x):
    x1 = self.d1(x)
    x2 = self.d2(x1)
    x3 = self.d3(x2)
    x4 = self.d4(x3)

    x = self.u1(x4, None)
    x = self.u2(x,x3)
    x = self.u3(x,x2)
    x = self.u4(x,x1)

    return self.outconv(x)
