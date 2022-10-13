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


class ImgFolderDS(Dataset):

  def __init__(self,
               img_folder: str = None,
               index_path: str = None,
               trans:nn.Module=None,
               exts=['.jpg', '.png', '.webp', '.jpeg']) -> None:
    super().__init__()
    self.img_folder = img_folder
    self.exts = exts
    self.indexes:list[str]
    if index_path:
      self.indexes: list[str] = torch.load(index_path)
    elif img_folder:
      self.indexes = self.create_index()
  
  def __len__(self):
    return len(self.indexes)
  
  def __getitem__(self, index):
    pass

  def create_index(self) -> list[str]:
    indexes = []

    def is_img(fname):
      _, ext = os.path.splitext(fname)
      return ext in self.exts

    def full_path(base: str):

      def inner(fname):
        return os.path.join(base, fname)

      return inner

    for base, _, files in os.walk(self.img_folder):
      files = list(filter(is_img, files))
      files = list(map(full_path(base), files))
      indexes.extend(files)

    return indexes

  def filter_grayscale(self):
    tt = trs.ToTensor()

    def not_grayscale(fpath: str):
      img = Image.open(fpath).convert('YCbCr')
      img = tt(img)
      return not (img[1].std().item() == 0 and img[2].std().item() == 0)

    self.indexes = list(filter(not_grayscale, self.indexes))
    return self

  def save_indexes(self, fpath: str):
    torch.save(self.indexes, fpath)
