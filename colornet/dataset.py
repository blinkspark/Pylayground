import torch, torchvision, os
from torch.utils.data import Dataset,Sampler
from zipfile import ZipFile
from torch import nn
from torchvision import transforms as trs


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

  def file_filter(self, fname: str):
    _, ext = os.path.splitext(fname)
    return ext in self.exts

  def __len__(self):
    return len(self.imglist)

  def __getitem__(self, index):
    pass