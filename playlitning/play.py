import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


class MyModule(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, 64),
        nn.Conv2d(64, 128, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(128, 10),
    )

  def forward(self, x):
    return self.net(x)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), 1e-3)

  def training_step(self, batch, i):
    x, y = batch
    pred = self.net(x)
    loss = F.cross_entropy(pred, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, i):
    x, y = batch
    pred = self.net(x)
    loss = F.cross_entropy(pred, y)
    self.log('val_loss', loss)
    return loss


if __name__ == '__main__':
  train_ds = CIFAR10('data',
                     train=True,
                     download=True,
                     transform=transforms.ToTensor())
  test_ds = CIFAR10('data',
                    train=True,
                    download=True,
                    transform=transforms.ToTensor())
  train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
  test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
  data = torch.randn(1, 3, 32, 32)
  m = MyModule()
  print(m)

  tr = pl.Trainer(accelerator='cuda')
  tr.fit(m, train_dataloaders=train_dl, val_dataloaders=test_dl)