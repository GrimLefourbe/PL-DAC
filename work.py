import datasets
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import numpy as np

import itertools as it

from models import Generator, Discriminator

ToPIL = transforms.ToPILImage()

def test_D(test_loader, G, D, batch_size):
    correct = 0
    for i, data in enumerate(test_loader):
        obj, bg, coord = data

        real_inputv = Variable(bg)
        real_pred = torch.round(D(real_inputv)).data.numpy()

        obj_id = torch.arange(i * batch_size, (i + 1) * batch_size).long()

        objv, bgv, coordv, obj_idv = Variable(obj), Variable(bg), Variable(coord), Variable(obj_id)

        fake_input = G(objv, bgv, coordv, obj_idv).detach()
        fake_pred = torch.round(D(fake_input)).data.numpy()

        correct += np.count_nonzero(real_pred == 1)
        correct += np.count_nonzero(fake_pred == 0)
    prec = correct / (len(test_loader) * batch_size * 2)
    print(prec)
    return prec


def train_test_indices(n, p_train):
    train_len = int(n * p_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices[:train_len], indices[train_len:]

def show_im(t):
    ToPIL(t).show()

if __name__ == '__main__':
    batch_size = 4
    D_ndf = 32
    nbperobj = 100000
    maxsize = 16000

    bg_format = (64, 64)
    obj_format = (8, 8)
    category = {'id':2, 'name': 'bicycle'}

    np.random.seed(47)

    dataset = datasets.ObjectsDataset(category, nbperobj=nbperobj, shuffled=True, maxsize=maxsize,
                                      obj_format=obj_format, bg_format=bg_format)

    train_indices, test_indices = train_test_indices(len(dataset), p_train=0.95)

    trainset = datasets.SubsetDataset(dataset, train_indices)
    testset = datasets.SubsetDataset(dataset, test_indices)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    print(len(trainloader))
    G = Generator(nb_embeddings=len(trainloader) * batch_size, obj_format=obj_format)
    D = Discriminator(ndf=D_ndf)

    criterion = nn.BCELoss()


    outf = "bicycles"

    real_label = 0.9
    fake_label = 0.1

    train_gen = False

    lr = 4*2*1e-4
    betas = (0.5, 0.999)

    niter = 25

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=betas)

    prec = [test_D(testloader, G, D, batch_size)]

    for epoch in range(niter):
        for i, data in enumerate(trainloader):# for each batch

            #update D network
            D.zero_grad()

            obj, bg, coord = data

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

            if train_gen:
                #update G network
                G.zero_grad()
                labelv = Variable(torch.ones((batch_size, 1))*real_label)

                output = D(fake_input)

                errG = criterion(output, labelv)
                errG.backward()

                print(G.embeddings.weight.grad.max().data[0])
                D_G_z2 = output.data.mean()

                optimizerG.step()

            print(f'[{epoch+1}/{niter}][{i+1}/{len(trainloader)}] Loss_D: {errD.data[0]:f} ' +
                  (f'Loss_G: {errG.data[0]:f} ' if train_gen else '') +
                  f'D(x): {D_x:f} ' +
                  f'D(G(z)): {D_G_z1:f} ' + ('/ {D_G_z2:f} ' if train_gen else ''))

        prec.append(test_D(test_loader=testloader, G=G, D=D, batch_size=batch_size))
        torch.save(G.state_dict(), f'{outf}/netG_epoch_{epoch}.pth')
        torch.save(D.state_dict(), f'{outf}/netD_ndf_{D_ndf}_epoch_{epoch}.pth')

    import pickle
    pickle.dump(prec, open(f'{outf}/Dprec_nbperobj_{nbperobj}_ndf_{D_ndf}_obj_{obj_format[0]}_{obj_format[1]}_maxsize_{maxsize}.pkl', 'wb'))

