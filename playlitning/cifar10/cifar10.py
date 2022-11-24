import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


class MyModule(pl.LightningModule):

  def __init__(self, lr=0.0002511886431509582, batch_size=2048):
    super().__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, 64),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, 128),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.GroupNorm(1, 256),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(256, 10),
    )

  def forward(self, x):
    return self.net(x)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.hparams.lr)

  def training_step(self, batch, i):
    x, y = batch
    pred = self.net(x)
    loss = F.cross_entropy(pred, y)
    item = torch.argmax(pred, 1)
    res = (item == y)
    res = torch.count_nonzero(res).item()
    self.log('train_acc', res / y.shape[0])
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, i):
    x, y = batch
    pred = self.net(x)
    loss = F.cross_entropy(pred, y)
    item = torch.argmax(pred, 1)
    res = (item == y)
    res = torch.count_nonzero(res).item()
    self.log('val_acc', res / y.shape[0])
    self.log('val_loss', loss)
    return loss

  def train_dataloader(self):
    train_ds = CIFAR10('data',
                       train=True,
                       download=True,
                       transform=transforms.ToTensor())
    train_dl = DataLoader(train_ds,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=2)
    return train_dl

  def val_dataloader(self):
    test_ds = CIFAR10('data',
                      train=True,
                      download=True,
                      transform=transforms.ToTensor())
    test_dl = DataLoader(test_ds,
                         batch_size=self.hparams.batch_size,
                         shuffle=False,
                         num_workers=2)
    return test_dl


if __name__ == '__main__':
  m = MyModule()

  tr = pl.Trainer(accelerator='cuda',
                  precision=16,
                  limit_val_batches=0.2,
                  log_every_n_steps=15,
                  auto_lr_find=True,
                  auto_scale_batch_size=True,
                  callbacks=[
                      callbacks.EarlyStopping('val_acc'),
                      callbacks.ModelCheckpoint(
                          monitor='val_acc',
                          mode='max',
                          filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}')
                  ])
  tr.fit(m)