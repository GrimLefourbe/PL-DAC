import torch
from torch import nn as nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convmask = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5,stride=1, padding=2)

    def forward(self, obj, bg, coord):
        mask = self.convmask(obj)
#        mask = mask.byte()
        obj_w, obj_h = obj.shape[2:4]
        x, y = coord.data
#        print(obj_w, obj_h)
        print((mask.byte().data.numpy() > 0).any())
        im = bg.clone()
        im[:, :, x: x + obj_w, y:y + obj_h] = (mask < 0.5).float() * im[:, :, x: x + obj_w, y:y + obj_h].clone() + (mask >= 0.5).float() * obj
        return im


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 7 * 7, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=1)

    def forward(self, x):
#        print(x.shape)
        x = self.conv1(x)
#        print(x.shape)
        x = F.relu(x)
#        print(x.shape)
        x = self.pool(x)
#        print(x.shape)
        x = self.conv2(x)
#        print(x.shape)
        x = F.relu(x)
#        print(x.shape)
        x = self.pool(x)
#        print(x.shape)
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)