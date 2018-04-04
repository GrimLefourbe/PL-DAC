import torch
from torch import nn as nn
from torch.nn import functional as F


# class Generator(nn.Module):
#     def __init__(self, nb_embeddings, obj_format=(32, 32)):
#         super().__init__()
# #         self.convmask = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7,stride=1, padding=3)
#         self.embeddings = nn.Embedding(num_embeddings=nb_embeddings, embedding_dim=obj_format[0] * obj_format[1])
#
#     def forward(self, obj, bg, coord, obj_id):
#         mask = self.embeddings(obj_id)
#         mask = mask.byte()
#         obj_w, obj_h = obj.shape[2:4]
#         x, y = coord.data
#         im = bg.clone()
#         im[:, :, x: x + obj_w, y:y + obj_h] = (mask < 0.5).float() * im[:, :, x: x + obj_w, y:y + obj_h].clone() + (mask >= 0.5).float() * obj
#         return im

class Generator(nn.Module):
    def __init__(self, nb_embeddings, obj_format=(32, 32)):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=nb_embeddings, embedding_dim=obj_format[0]*obj_format[1])
#        self.embeddings.weight.data = torch.ones(nb_embeddings, obj_format[0]*obj_format[1]) * 0
        self.embeddings.weight.data = torch.rand(nb_embeddings, obj_format[0]*obj_format[1])*0.1 + 0.475

    def forward(self, obj, bg, coord, obj_id):
#        print(obj.shape, bg.shape)
        mask = self.embeddings(obj_id)
        batch_size, obj_w, obj_h = obj.shape[0], obj.shape[2], obj.shape[3]
#        mask.fill_(0)
        mask = mask.view((batch_size, obj_w, obj_h))
        x, y = coord.data
        im = bg.clone()
        for i in range(batch_size):
            im[i, :, x: x + obj_w, y:y+obj_h] = (mask[i] < 0.5).float() * im[i, :, x:x+obj_w, y:y+obj_h].clone() \
                                                + (mask[i] >= 0.5).float()*obj[i]
        return im

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=4):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)