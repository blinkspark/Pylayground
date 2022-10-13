# # %%
# import model
# from PIL import Image
# from torchvision import transforms
# import torch
# from torch.nn import functional as F

# from train import Trainer

# # %%
# net = model.ColorNet()
# net = net.cuda()
# trainer = Trainer(net, None, None)
# trainer.load('colornet-29-0.07059022851288319.pth')
# # net.load_state_dict(torch.load('./colornet-29-0.07059022851288319.pth'))
# img = Image.open('test.jpg').convert('YCbCr')

# img = transforms.ToTensor()(img)
# img: torch.Tensor = img[0].unsqueeze(0)
# img = img.unsqueeze(0)
# img = img.cuda()

# pred = net(img)
# print(img.shape)
# print(pred.shape)
# diff_h = pred.shape[2] - img.shape[2]
# diff_w = pred.shape[3] - img.shape[3]
# img = F.pad(
#     img,
#     [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

# img = torch.cat([img, pred], 1)
# print(img.shape)
# img = img / 2 + 0.5
# img = img.squeeze()
# img = transforms.ToPILImage('YCbCr')(img)
# img.save('colored.jpg', quality=100)

# # %%
# from zipfile import ZipFile
# import io
# from PIL import Image
# from IPython.display import display
# with ZipFile(R'E:\Download\train2014.zip') as f:
#   for finfo in f.infolist():
#     if not finfo.filename.endswith('.jpg'):
#       continue
#     print(finfo)
#     img_data = f.read(finfo.filename)
#     # print(img_data)
#     img_data = io.BytesIO(img_data)
#     img_data.filename = finfo.filename
#     img = Image.open(img_data)
#     display(img)
#     break

# # %%
# from os import path

# path.splitext('test.jpg')
# # %%
# import cv2
# from PIL import Image
# v = cv2.VideoCapture('test.webm')
# ok,frame = v.read()
# print(ok)
# if ok:
#   frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#   img = Image.fromarray(frame)
#   img.show()

# %% play
import dataset
from torch.utils.data import DataLoader
from time import perf_counter

if __name__ == '__main__':
  ds = dataset.ZipImageDS(R'E:\Download\train2014.zip')
  dl = DataLoader(ds,batch_size=64,shuffle=True,num_workers=2)
  before = perf_counter()
  next(iter(dl))
  print(perf_counter()-before)

# %%
import dataset

ds = dataset.ImgFolderDS(R'E:\tmp\training').filter_grayscale()
ds.save_indexes('pixiv_index.pth')

# %%
len(ds.indexes)
# %%
