import pytorch_lightning as pl
from pytorch_lightning import callbacks
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from common import ConvBlk, SRDataset
from generator import GeneratorV2
from srgan_model import SRGAN

def train_generator():
  gen = GeneratorV2()
  # m = SRGAN(batch_size=512)
  tr = pl.Trainer(max_epochs=100,
                  accelerator='cuda',
                  precision=16,
                  fast_dev_run=False,
                  callbacks=[callbacks.EarlyStopping('val_loss')])
  tr.fit(gen)


def train_srgan():
  gen = GeneratorV2.load_from_checkpoint(
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
  train_generator()
