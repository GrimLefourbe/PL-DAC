import datasets
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
import numpy as np

import itertools as it

from models import Generator, Discriminator

ToPIL = transforms.ToPILImage()

def test(data_loader, G, D):
    fake_res = []
    real_res = []
    for i, data in enumerate(data_loader):
        obj, bg = data['obj'], data['bg']
        real_res.append(D(Variable(bg)))
        v_obj, v_bg, v_coord = Variable(obj), Variable(bg), Variable(torch.IntTensor([10, 10]))
        fake_res.append(D(G(v_obj, v_bg, v_coord).detach()))
#    fake_res = np.vstack(fake_res)
#    real_res = np.vstack(real_res)
    return fake_res, real_res

def show_im(t):
    ToPIL(t).show()

if __name__ == '__main__':
    batch_size = 4

    dataloader = DataLoader(datasets.ObjectsDataset({'id':2, 'name':'bicycle'},
                                                    obj_transform=transforms.Compose([transforms.Resize((32,32)),
                                                                                      transforms.ToTensor()]),
                                                    bg_transform=transforms.Compose([transforms.Resize((64,64)),
                                                                                     transforms.ToTensor()]), nbperobj=1),
                            batch_size=batch_size, shuffle=False, num_workers=4)
    print(len(dataloader))
    G = Generator(nb_embeddings=len(dataloader)*batch_size)
    D = Discriminator()

    criterion = nn.BCELoss()


    outf = "bicycles"
    real_label = 0.9
    fake_label = 0.1
    coord = torch.IntTensor([10, 10])

    lr = 2*1e-4
    betas = (0.5, 0.999)

    niter = 25

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(G.parameters(), lr=lr*100, betas=betas)

    for epoch in range(niter):
        for i, data in it.islice(enumerate(dataloader), 5):

            #update D network
            D.zero_grad()

            obj, bg = data['obj'], data['bg']
            obj_id = torch.arange(i*batch_size, (i+1)*batch_size).long()

            # train with real
            real_inputv = Variable(bg)
            real_labelv = Variable(torch.ones((batch_size, 1))*real_label)

            real_output = D(real_inputv)
            errD_real = criterion(real_output, real_labelv)
            errD_real.backward()
            D_x = real_output.data.mean()

            #train with fake
            objv, bgv, coordv, obj_idv = Variable(obj), Variable(bg), Variable(coord), Variable(obj_id)
            fake_input = G(objv, bgv, coordv, obj_idv)
            fake_labelv = Variable(torch.ones((batch_size, 1))*fake_label)
            fake_output = D(fake_input.detach())

            if i == 0:
                show_im(real_inputv.data[0])
                show_im(fake_input.data[0])

            errD_fake = criterion(fake_output, fake_labelv)
            errD_fake.backward()

            D_G_z1 = fake_output.data.mean()

            errD = errD_real + errD_fake

            if errD.data[0] > 0:
                optimizerD.step()
            else:
                print('skipping Dstep')

            #update G network
            G.zero_grad()
            labelv = Variable(torch.ones((batch_size, 1))*real_label)

#            objv, bgv, coordv, obj_idv = Variable(obj), Variable(bg), Variable(coord), Variable(obj_id)
            output = D(fake_input)

            errG = criterion(output, labelv)
            errG.backward()

            print(G.embeddings.weight.grad)
            D_G_z2 = output.data.mean()

            optimizerG.step()

            # while errG.data[0] > 0.8:
            #     G.zero_grad()
            #     labelv = Variable(torch.ones((batch_size, 1)) * real_label)
            #
            #     objv, bgv, coordv, obj_idv = Variable(obj), Variable(bg), Variable(coord), Variable(obj_id)
            #     output = D(G(objv, bgv, coordv, obj_idv))
            #
            #     errG = criterion(output, labelv)
            #     errG.backward()
            #     D_G_z2 = output.data.mean()
            #
            #     optimizerG.step()
            #     print(f'Loss_G: {errG.data[0]}')

            print(f'[{epoch}/{niter}][{i}/{len(dataloader)}] Loss_D: {errD.data[0]:f} Loss_G: {errG.data[0]:f} '
                  f'D(x): {D_x:f} D(G(z)): {D_G_z1:f} / {D_G_z2:f}')

    torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
