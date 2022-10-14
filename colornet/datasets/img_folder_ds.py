import os
from torch.utils.data import Dataset
import torchvision.transforms as trs


class InitialError(Exception):
  pass


class ImageFolderDataset(Dataset):
  img_folder: str
  index_path: str
  indexes: str
  exts: list[str]

  def __init__(
      self,
      img_folder: str = None,
      index_path: str = None,
      exts=['.jpg', '.png', '.jpeg', '.webp'],
  ) -> None:
    super().__init__()
    self.exts = exts
    if index_path:
      self.index_path = index_path
      self.indexes = self.create_indexes()
    elif img_folder:
      self.img_folder = img_folder
    else:
      raise InitialError('img_folder and index_path cannot both be None')

  def create_indexes(self):
    indexes = []

    def is_img(fname):
      _, ext = os.path.splitext(fname)
      return ext in self.exts

    def join(base: str):

      def inner(fname: str):
        return os.path.join(base, fname)

      return inner

    for base, _, fnames in os.walk(self.img_folder):
      tmp_indexes = list(filter(is_img, fnames))
      tmp_indexes = list(map(join(base), fnames))
      indexes.extend(tmp_indexes)
    return indexes