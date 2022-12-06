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

# data = torch.zeros(32)
# data1 = torch.zeros(32)

# y = nn.Conv2d(3,16,kernel_size=5,padding=2)(data)
# print(y.shape)

# dd = torch.cat([data, data1])
# print(dd.shape)


def play_srgan():
  # from srgan.srgan_model import SRGAN
  from generator import GeneratorV2
  from PIL import Image
  m = GeneratorV2.load_from_checkpoint('./lightning_logs/version_2/checkpoints/epoch=40-step=3239.ckpt')
  m.eval()
  with torch.no_grad():
    img = Image.open('F:\\tmp\\test.jpg')
    # img1 = img.convert('YCbCr')
    # img1 = img1.resize((img.size[0] * 2, img.size[1] * 2))
    # img1: torch.Tensor = transforms.ToTensor()(img1)
    # img1 = torch.split(img1, [1, 2])[1]
    trs = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Grayscale(),
        transforms.Normalize(0.5, 0.5, inplace=True)
    ])
    toimg = transforms.ToPILImage()
    img: torch.Tensor = trs(img)
    img = img.unsqueeze(0).to('cuda')
    m.to('cuda')
    y = m(img)
    img = y / 2 + 0.5
    img = img[0]
    # print(img.shape,img1.shape)
    # img = torch.vstack([img, img1])
    img = toimg(img)
    img.save('outputs/test.jpg')


if __name__ == '__main__':
  play_srgan()