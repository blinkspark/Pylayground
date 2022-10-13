import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL.Image import Resampling

from model import ColorNet


class Trainer():

  def __init__(self,
               model: nn.Module,
               train_dl: nn.Module,
               val_dl: nn.Module,
               device=torch.device('cuda'),
               loss_fn=nn.L1Loss,
               optm=torch.optim.Adam,
               lr=1e-3) -> None:
    self.device = device
    self.model = model.to(device)
    self.loss_fn = loss_fn().to(device)
    self.optm = optm(model.parameters(), lr=lr)
    self.train_dl = train_dl
    self.val_dl = val_dl

  def load(self, fpath: str):
    self.model.load_state_dict(torch.load(fpath))

  def save_val(self, epoch, num=1):
    device = self.device
    self.model.eval()
    target_size = (200, 200)

    with torch.no_grad():
      for x, y in self.val_dl:
        x = x.to(device)
        y = y.to(device)
        toimg = transforms.ToPILImage(mode='YCbCr')
        pred = self.model(x)
        num = min(x.shape[0], num)
        for i, gen_chroma in enumerate(pred):
          if i >= num:
            break
          luma = x[i]
          luma = luma / 2 + 0.5
          real_chroma = y[i]
          gen_chroma = gen_chroma / 2 + 0.5
          real_chroma = real_chroma / 2 + 0.5

          gen_img = torch.cat([luma, gen_chroma])
          gen_img: Image.Image = toimg(gen_img)
          gen_img = gen_img.resize(target_size, Resampling.LANCZOS)

          real_img = torch.cat([luma, real_chroma])
          real_img: Image.Image = toimg(real_img)
          real_img = real_img.resize(target_size, Resampling.LANCZOS)

          gen_img.save(f'out/img-{epoch}-{i}.jpg', quality=100)
          real_img.save(f'out/img-{epoch}-{i}-r.jpg', quality=100)
        break
    self.model.train()

  def fit(self,
          epochs=100,
          start_epoch=1,
          steps_per_epoch=None,
          val_steps_per_epoch=None,
          patient=10):
    device = self.device
    min_v_loss = float('inf')
    rounds = 0
    for epoch in range(start_epoch, start_epoch + epochs):
      rounds += 1
      if rounds > patient:
        break
      losses = []
      for step, (x, y) in enumerate(self.train_dl):
        if steps_per_epoch and step >= steps_per_epoch:
          break
        self.model.zero_grad()
        x: torch.Tensor = x.to(device)
        y: torch.Tensor = y.to(device)
        pred = self.model(x)
        loss: torch.Tensor = self.loss_fn(pred, y)
        loss.backward()
        self.optm.step()
        losses.append(loss.item())

      train_loss = sum(losses) / len(losses)
      losses = []
      self.model.eval()
      for step, (x, y) in enumerate(self.val_dl):
        if val_steps_per_epoch and step >= val_steps_per_epoch:
          break
        with torch.no_grad():
          x: torch.Tensor = x.to(device)
          y: torch.Tensor = y.to(device)
          pred = self.model(x)
          loss: torch.Tensor = self.loss_fn(pred, y)
          losses.append(loss.item())
      self.save_val(epoch)
      self.model.train()
      val_loss = sum(losses) / len(losses)
      if val_loss < min_v_loss:
        rounds = 0
        min_v_loss = val_loss
        torch.save(self.model.state_dict(), f'colornet-{epoch}-{val_loss}.pth')
      print(
          f'epoch: {epoch}, loss: {train_loss}, v_loss: {val_loss}, r/p: {rounds}/{patient}'
      )


class MyDS(Dataset):

  def __init__(self, ds) -> None:
    super().__init__()
    self.ds = ds
    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        # transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
    ])

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, index):
    img, _ = self.ds[index]
    img = img.convert('YCbCr')
    img: torch.Tensor = self.transform(img)

    return img[0].unsqueeze(0), img[1:]


if __name__ == '__main__':
  from dataset import ZipImageDS
  train_ds = ZipImageDS(R'F:\Download\Compressed\train2014.zip')
  test_ds = ZipImageDS(R'F:\Download\Compressed\val2014.zip')
  
  train_dl = DataLoader(train_ds,batch_size=32,shuffle=True)
  test_dl = DataLoader(test_ds,batch_size=32,shuffle=True)

  # print(next(iter(train_dl))[0].shape)
  # print(next(iter(train_dl))[1].shape)

  trainer = Trainer(ColorNet(), train_dl, test_dl, lr=1e-3)
  # trainer.load('colornet-9-0.07162804681807756.pth')
  trainer.fit(patient=20,steps_per_epoch=10,val_steps_per_epoch=2)
  # from IPython.display import display
  # display(train_ds[0][0])
