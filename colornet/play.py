from email.mime import image
import model
from PIL import Image
from torchvision import transforms
import torch
from torch.nn import functional as F

from train import Trainer

net = model.ColorNet()
net = net.cuda()
trainer = Trainer(net, None, None)
trainer.load('colornet-29-0.07059022851288319.pth')
# net.load_state_dict(torch.load('./colornet-29-0.07059022851288319.pth'))
img = Image.open('test.jpg').convert('YCbCr')

img = transforms.ToTensor()(img)
img: torch.Tensor = img[0].unsqueeze(0)
img = img.unsqueeze(0)
img = img.cuda()

pred = net(img)
print(img.shape)
print(pred.shape)
diff_h = pred.shape[2] - img.shape[2]
diff_w = pred.shape[3] - img.shape[3]
img = F.pad(
    img,
    [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

img = torch.cat([img, pred], 1)
print(img.shape)
img = img/2+0.5
img = img.squeeze()
img = transforms.ToPILImage('YCbCr')(img)
img.save('colored.jpg',quality=100)