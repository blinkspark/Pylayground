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
        nn.SiLU(inplace=True),
        nn.GroupNorm(mid // 4, mid),
        nn.Conv2d(mid, outputs, kernel_size=kernel_size, padding=padding),
        nn.SiLU(inplace=True),
        nn.GroupNorm(outputs // 4, outputs),
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


class Generator(pl.LightningModule):

  def __init__(self,
               up_factor=2,
               lr=2e-5,
               batch_size=640,
               num_workers=4) -> None:
    super().__init__()
    self.save_hyperparameters()
    self.val_dl_ = None
    self.generator = nn.Sequential(
        ConvBlk(3, 64),
        ConvBlk(64, 64),
        ConvBlk(64, 128),
        nn.Conv2d(128, 128 * up_factor**2, kernel_size=5, padding=2),
        nn.SiLU(inplace=True),
        nn.PixelShuffle(up_factor),
        nn.Conv2d(128, 3, kernel_size=5, padding=2),
        # nn.Tanh(),
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


class SRGAN(pl.LightningModule):

  def __init__(self,
               generator:pl.LightningModule,
               lr=2e-5,
               up_factor=2,
               batch_size=128,
               num_workers=2):
    super().__init__()
    self.save_hyperparameters(ignore=['generator'])
    self.val_dl_ = None
    self.automatic_optimization = False
    self.generator = generator
    self.discriminator = nn.Sequential(
        ConvBlk(3, 64),
        nn.MaxPool2d(2, 2),
        ConvBlk(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )

  def forward(self, x):
    return self.generator(x)

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
    return torch.optim.Adam(self.generator.parameters(),
                            self.hparams.lr), torch.optim.Adam(
                                self.discriminator.parameters(),
                                self.hparams.lr)

  def training_step(self, batch, step):
    x, y = batch
    g_opt, d_opt = self.optimizers()
    g_opt: torch.optim.Adam
    d_opt: torch.optim.Adam
    batch_size = x.shape[0]
    zeros = torch.zeros((batch_size, 1), device=self.device)
    ones = torch.ones((batch_size, 1), device=self.device)

    d_loss = None
    # train discriminator
    # if step % 2 < 1:
    with torch.no_grad():
      generated = self.generator(x)
    d_opt.zero_grad()
    labels = torch.cat([zeros, ones])
    inputs = torch.cat([generated, y])
    pred = self.discriminator(inputs)
    d_loss = F.binary_cross_entropy_with_logits(pred, labels)
    self.log('train_d_loss', d_loss.item())
    self.manual_backward(d_loss)
    # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
    d_opt.step()

    # train generator
    g_opt.zero_grad()
    self.discriminator.requires_grad_(False)
    pred = self.discriminator(self.generator(x))
    g_loss = F.binary_cross_entropy_with_logits(pred, ones)
    self.log('train_g_loss', g_loss)
    self.manual_backward(g_loss)
    # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
    g_opt.step()
    self.discriminator.requires_grad_(True)
    ret = {
        'train_d_loss': d_loss.item(),
        'train_g_loss': g_loss.item()
    } if d_loss is not None else {
        'train_g_loss': g_loss.item()
    }
    return ret

  def training_epoch_end(self, outputs) -> None:
    if self.val_dl_ is None:
      self.val_dl_ = self.val_dataloader()
    for (x, _) in self.val_dl_:
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

  # def validation_step(self, batch, _):
  #   x, y = batch
  #   batch_size = x.shape[0]
  #   zeros = torch.zeros((batch_size, 1), device=self.device)
  #   ones = torch.ones((batch_size, 1), device=self.device)

  #   # train discriminator
  #   with torch.no_grad():
  #     generated = self.generator(x)
  #   labels = torch.cat([zeros, ones])
  #   inputs = torch.cat([generated, y])
  #   pred = self.discriminator(inputs)
  #   d_loss = F.binary_cross_entropy_with_logits(pred, labels)
  #   self.log('val_d_loss', d_loss.item())

  #   # train generator
  #   pred = self.discriminator(self.generator(x))
  #   g_loss = F.binary_cross_entropy_with_logits(pred, ones)
  #   self.log('val_g_loss', g_loss)


def train_generator():
  gen = Generator()
  # m = SRGAN(batch_size=512)
  tr = pl.Trainer(max_epochs=100,
                  accelerator='cuda',
                  precision=16,
                  fast_dev_run=False,
                  callbacks=[callbacks.EarlyStopping('val_loss')])
  tr.fit(gen)


def train_srgan():
  gen = Generator.load_from_checkpoint(
      './lightning_logs/version_0/checkpoints/epoch=40-step=3239.ckpt')
  m = SRGAN(gen, batch_size=512)
  tr = pl.Trainer(
      max_epochs=100,
      accelerator='cuda',
      precision=16,
      fast_dev_run=False,
  )
  tr.fit(m)


if __name__ == '__main__':
  train_srgan()
