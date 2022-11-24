import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image


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
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, mid),
        nn.Conv2d(mid, outputs, kernel_size=kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, outputs),
    )

  def forward(self, x):
    return self.blk(x)


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


class SRGAN(pl.LightningModule):

  def __init__(self, lr=1e-5, batch_size=128, num_workers=2):
    super().__init__()
    self.save_hyperparameters()
    self.automatic_optimization = False
    self.generator = nn.Sequential(
        ConvBlk(1, 128, mid=64),
        nn.UpsamplingBilinear2d(scale_factor=2),
        ConvBlk(128, 64, mid=64),
        nn.Conv2d(64, 1, kernel_size=5, padding=2),
    )
    self.discriminator = nn.Sequential(
        ConvBlk(1, 64),
        nn.MaxPool2d(2, 2),
        ConvBlk(64, 128),
        nn.MaxPool2d(2, 2),
        ConvBlk(128, 256),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )

  def forward(self, x):
    return self.generator(x)

  def train_dataloader(self):
    trs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
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
        transforms.Grayscale(),
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
    return torch.optim.Adam(self.generator.parameters(),
                            self.hparams.lr), torch.optim.Adam(
                                self.discriminator.parameters(),
                                self.hparams.lr)

  def training_step(self, batch, _):
    x, y = batch
    g_opt, d_opt = self.optimizers()
    g_opt: torch.optim.Adam
    d_opt: torch.optim.Adam
    batch_size = x.shape[0]
    zeros = torch.zeros((batch_size, 1), device=self.device)
    ones = torch.ones((batch_size, 1), device=self.device)

    # train discriminator
    with torch.no_grad():
      generated = self.generator(x)
    d_opt.zero_grad()
    labels = torch.cat([zeros, ones])
    inputs = torch.cat([generated, y])
    pred = self.discriminator(inputs)
    d_loss = F.binary_cross_entropy_with_logits(pred, labels)
    self.log('train_d_loss', d_loss.item())
    self.manual_backward(d_loss)
    d_opt.step()

    # train generator
    g_opt.zero_grad()
    pred = self.discriminator(self.generator(x))
    g_loss = F.binary_cross_entropy_with_logits(pred, ones)
    self.log('train_g_loss', g_loss)
    self.manual_backward(g_loss)
    g_opt.step()

  def validation_step(self, batch, _):
    x, y = batch
    batch_size = x.shape[0]
    zeros = torch.zeros((batch_size, 1), device=self.device)
    ones = torch.ones((batch_size, 1), device=self.device)

    # train discriminator
    with torch.no_grad():
      generated = self.generator(x)
    labels = torch.cat([zeros, ones])
    inputs = torch.cat([generated, y])
    pred = self.discriminator(inputs)
    d_loss = F.binary_cross_entropy_with_logits(pred, labels)
    self.log('val_d_loss', d_loss.item())

    # train generator
    pred = self.discriminator(self.generator(x))
    g_loss = F.binary_cross_entropy_with_logits(pred, ones)
    self.log('val_g_loss', g_loss)


if __name__ == '__main__':
  m = SRGAN(batch_size=512)
  tr = pl.Trainer(
      accelerator='cuda',
      precision=16,
  )
  tr.fit(m)
