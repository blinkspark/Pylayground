from io import BytesIO
from PIL import Image
import torch, torchvision, os
from torch.utils.data import Dataset, Sampler
from zipfile import ZipFile, ZipInfo
from torch import nn
from torchvision import transforms as trs
from threading import Lock


class ZipImageDS(Dataset):

  def __init__(self,
               zip_path: str,
               exts=['.jpg', '.png', '.jpeg'],
               target_size=128,
               transforms=None) -> None:
    super().__init__()
    self.exts = exts
    self.zfile = ZipFile(zip_path, 'r')
    self.imglist = list(filter(self.file_filter, self.zfile.infolist()))
    self.transforms: nn.Module = transforms if transforms else trs.Compose([
        trs.Resize(target_size),
        trs.RandomCrop(target_size),
        trs.RandomHorizontalFlip(),
        trs.ToTensor(),
        trs.Normalize(0.5, 0.5),
    ])
    self.lock = Lock()

  def file_filter(self, finfo: ZipInfo):
    fname = finfo.filename
    _, ext = os.path.splitext(fname)
    return ext in self.exts

  def __len__(self):
    return len(self.imglist)

  def __getitem__(self, index):
    finfo = self.imglist[index]
    img = self.zfile.read(finfo.filename)
    img = BytesIO(img)
    # print(img,len(img))
    img = Image.open(img)
    return self.transforms(img)