import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from common import ConvBlk, SRDataset


class Generator(pl.LightningModule):

  def __init__(self,
               up_factor=2,
               lr=2e-5,
               batch_size=640,
               num_workers=4) -> None:
    super().__init__()
    self.save_hyperparameters()
    self.generator = nn.Sequential(
        ConvBlk(3, 64),
        ConvBlk(64, 64),
        ConvBlk(64, 128),
        nn.Conv2d(128, 128 * up_factor**2, kernel_size=5, padding=2),
        nn.SiLU(inplace=True),
        nn.PixelShuffle(up_factor),
        nn.Conv2d(128, 3, kernel_size=5, padding=2),
        nn.Tanh(),
    )

  def train_dataloader(self):
    trs = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize(0.5, 0.5, inplace=True)
    ])
    ds = CIFAR10('data', train=True, download=True, transform=trs)
    ds = SRDataset(ds)
    dl = DataLoader(ds,
                    batch_size=self.hparams.batch_size,
                    shuffle=True,
                    num_workers=self.hparams.num_workers)
    return dl

  def val_dataloader(self):
    trs = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize(0.5, 0.5, inplace=True)
    ])
    ds = CIFAR10('data', train=False, download=True, transform=trs)
    ds = SRDataset(ds)
    ds = random_split(ds, [0.2, 0.8],
                      generator=torch.Generator().manual_seed(42))[0]
    dl = DataLoader(ds,
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                    num_workers=self.hparams.num_workers)
    return dl

  def configure_optimizers(self):
    return torch.optim.Adam(self.generator.parameters(), self.hparams.lr)

  def training_step(self, batch, step):
    x, y = batch
    pred = self.generator(x)
    loss = F.mse_loss(pred, y)
    self.log('train_loss', loss.item())
    return loss

  def validation_step(self, batch, idx):
    x, y = batch
    pred = self.generator(x)
    loss = F.mse_loss(pred, y)
    self.log('val_loss', loss.item())
    return loss

  def validation_epoch_end(self, outputs):
    for (x, _) in self.trainer.val_dataloaders[0]:
      x = torch.split(x, 2)[0]
      x: torch.Tensor = x.to(self.device)
      # y: torch.Tensor = y.to(self.device)
      self.eval()
      with torch.no_grad():
        toimg = transforms.ToPILImage()
        pred = self.generator(x)
        for i, img in enumerate(pred):
          img = img / 2 + 0.5
          img = toimg(img)
          img.save(f'outputs/{self.current_epoch}-{i}.jpg')
      # self.train()
      break

  def forward(self, x):
    return self.generator(x)


class GeneratorV2(pl.LightningModule):

  def __init__(self,
               up_factor=2,
               lr=2e-5,
               batch_size=640,
               num_workers=4) -> None:
    super().__init__()
    self.save_hyperparameters()
    self.generator = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=9, padding=4),
        nn.SiLU(inplace=True),
        nn.GroupNorm(64 // 4, 64),
        ConvBlk(64, 64),
        ConvBlk(64, 64),
        ConvBlk(64, 64),
        ConvBlk(64, 64),
        nn.Conv2d(64, 64 * up_factor**2, kernel_size=5, padding=2),
        nn.PixelShuffle(up_factor),
        nn.SiLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=9, padding=4),
        nn.Tanh(),
    )

  def train_dataloader(self):
    trs = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize(0.5, 0.5, inplace=True)
    ])
    ds = CIFAR10('data', train=True, download=True, transform=trs)
    ds = SRDataset(ds)
    dl = DataLoader(ds,
                    batch_size=self.hparams.batch_size,
                    shuffle=True,
                    num_workers=self.hparams.num_workers)
    return dl

  def val_dataloader(self):
    trs = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize(0.5, 0.5, inplace=True)
    ])
    ds = CIFAR10('data', train=False, download=True, transform=trs)
    ds = SRDataset(ds)
    ds = random_split(ds, [0.2, 0.8],
                      generator=torch.Generator().manual_seed(42))[0]
    dl = DataLoader(ds,
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                    num_workers=self.hparams.num_workers)
    return dl

  def configure_optimizers(self):
    return torch.optim.Adam(self.generator.parameters(), self.hparams.lr)

  def training_step(self, batch, step):
    x, y = batch
    pred = self.generator(x)
    loss = F.mse_loss(pred, y)
    self.log('train_loss', loss.item())
    return loss

  def validation_step(self, batch, idx):
    x, y = batch
    pred = self.generator(x)
    loss = F.mse_loss(pred, y)
    self.log('val_loss', loss.item())
    return loss

  def validation_epoch_end(self, outputs):
    for (x, _) in self.trainer.val_dataloaders[0]:
      x = torch.split(x, 2)[0]
      x: torch.Tensor = x.to(self.device)
      # y: torch.Tensor = y.to(self.device)
      self.eval()
      with torch.no_grad():
        toimg = transforms.ToPILImage()
        pred = self.generator(x)
        for i, img in enumerate(pred):
          img = img / 2 + 0.5
          img = toimg(img)
          img.save(f'outputs/{self.current_epoch}-{i}.jpg')
      # self.train()
      break

  def forward(self, x):
    return self.generator(x)