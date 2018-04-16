import datasets
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import numpy as np

import itertools as it
import time

from models import Generator, Discriminator
import os
from pathlib import Path

ToPIL = transforms.ToPILImage()

def train_test_indices(n, p_train):
    train_len = int(n * p_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices[:train_len], indices[train_len:]

def show_im(t):
    ToPIL(t).show()

if __name__ == '__main__':
    start = time.perf_counter()
    np.random.seed(47)

    outf = Path("tests/G3")
    outf.mkdir(parents=True, exist_ok=True)
    nb_images = 30

    train_gen = True

    batch_size = 128
    D_ndf = 32
    nbperobj = 32000
    maxsize = 32000

    image_indices = np.random.choice(maxsize, replace=False, size=nb_images)

    bg_format = (64, 64)
    obj_format = (8, 8)
    category = {'id': 16, 'name': 'bird'}


    dataset = datasets.ObjectsDataset(category, nbperobj=nbperobj, shuffled=True, maxsize=maxsize,
                                      obj_format=obj_format, bg_format=bg_format,
                                      objects_folder=datasets.TrainStorageFolder, background_folder=datasets.Train_img_folder, annFile=datasets.instTrainFile)

    if train_gen:
        _, test_indices = train_test_indices(len(dataset), p_train=0.95)
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
        testloader = DataLoader(datasets.SubsetDataset(dataset, test_indices),
                                batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
        print('Loaders prepped')
    else:
        train_indices, test_indices = train_test_indices(len(dataset), p_train=0.95)

        trainloader = DataLoader(datasets.SubsetDataset(dataset, train_indices),
                                 batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
        testloader = DataLoader(datasets.SubsetDataset(dataset, test_indices),
                                batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    print(len(trainloader))

    use_gpu = torch.cuda.is_available()

    criterion = nn.BCELoss()

    real_label = 0.9
    fake_label = 0.1

    lr = 4*2*1e-4
    betas = (0.5, 0.999)

    niter = 25

    D = Discriminator(ndf=D_ndf)
    G = Generator(nb_embeddings=len(trainloader) * batch_size, obj_format=obj_format)

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(G.parameters(), lr=100*lr, betas=betas)

    bg = torch.FloatTensor(batch_size, 3, bg_format[0], bg_format[1])
    obj = torch.FloatTensor(batch_size, 3, obj_format[0], obj_format[1])
    coord = torch.IntTensor(batch_size, 2)
    obj_id = torch.LongTensor(batch_size)
    label = torch.FloatTensor(batch_size)

    if use_gpu:
        G.cuda()
        D.cuda()
        criterion.cuda()
        bg = bg.cuda()
        obj = obj.cuda()
        coord = coord.cuda()
        obj_id = obj_id.cuda()
        label = label.cuda()

    def test_D(test_loader, G, D):
        print('Testing')
        correct = 0
        mean = []
        std = []

        for i, (cpu_obj, cpu_bg, cpu_coord) in enumerate(test_loader):
            bg.copy_(cpu_bg)

            real_inputv = Variable(bg)
            real_pred = torch.round(D(real_inputv))

            if use_gpu:
                real_pred = real_pred.cpu()
            real_pred = real_pred.data.numpy()

            obj.copy_(cpu_obj)
            coord.copy_(cpu_coord)
            obj_id.copy_(torch.arange(i * batch_size, (i + 1) * batch_size).long())

            objv, bgv, coordv, obj_idv = Variable(obj), Variable(bg), Variable(coord), Variable(obj_id)

            fake_input, masks = G(objv, bgv, coordv, obj_idv)

            mean.append(torch.mean(masks).data[0])
            std.append(torch.std(masks).data[0])

            fake_pred = torch.round(D(fake_input.detach()))
            if use_gpu:
                fake_pred = fake_pred.cpu()
            fake_pred = fake_pred.data.numpy()

            correct += np.count_nonzero(real_pred == 1)
            correct += np.count_nonzero(fake_pred == 0)
        prec = correct / (len(test_loader) * batch_size * 2)
        mean = np.mean(mean)
        std = np.mean(std)
        return prec, mean, std

    def make_test_grid():
        bg = []
        obj = []
        coord = []
        for i in image_indices:
            cpu_obj, cpu_bg, cpu_coord = dataset[i]
            bg.append(cpu_bg)
            obj.append(cpu_obj)
            coord.append(cpu_coord)

        bg = Variable(torch.stack(bg).cuda())
        obj = Variable(torch.stack(obj).cuda())
        coord = Variable(torch.stack(coord).cuda())
        obj_id = Variable(torch.LongTensor(image_indices).cuda())

        fake, masks = G(obj, bg, coord, obj_id)

        grid = utils.make_grid(torch.cat((bg.data, fake.data)), nrow=10)
        return grid

    grid = make_test_grid()
    prec, mean, std = test_D(testloader, G, D)
    precisions = [prec]
    means = [mean]
    stds = [std]
    grids = [grid]
    grad_norms = []

    print(prec, mean, std, sep=' ')

    for epoch in range(niter):
        for i, (cpu_obj, cpu_bg, cpu_coord) in enumerate(trainloader):# for each batch

            #update D network
            D.zero_grad()

            bg.copy_(cpu_bg)
            obj.copy_(cpu_obj)
            coord.copy_(cpu_coord)
            obj_id.copy_(torch.arange(i*batch_size, (i+1)*batch_size).long())

            # train with real
            label.fill_(real_label)
            real_inputv = Variable(bg)
            real_labelv = Variable(label)

            real_output = D(real_inputv)
            errD_real = criterion(real_output, real_labelv)
            errD_real.backward()
            D_x = real_output.data.mean()

            #train with fake
            objv, bgv, coordv, obj_idv = Variable(obj), Variable(bg), Variable(coord), Variable(obj_id)

            fake_input, masks = G(objv, bgv, coordv, obj_idv)
            label.fill_(fake_label)
            fake_labelv = Variable(label)
            fake_output = D(fake_input.detach())

            if False:
                show_im(real_inputv.cpu().data[0])
                show_im(fake_input.cpu().data[0])

            errD_fake = criterion(fake_output, fake_labelv)
            errD_fake.backward()

            D_G_z1 = fake_output.data.mean()

            errD = errD_real + errD_fake

            if errD.data[0] > 0:
                optimizerD.step()
            else:
                print('skipping Dstep')

        if train_gen:

            for i, (cpu_obj, cpu_bg, cpu_coord) in enumerate(trainloader):  # for each batch
                bg.copy_(cpu_bg)
                obj.copy_(cpu_obj)
                coord.copy_(cpu_coord)
                obj_id.copy_(torch.arange(i * batch_size, (i + 1) * batch_size).long())
                #update G network
                G.zero_grad()

                label.fill_(real_label)
                labelv = Variable(label)

                output = D(fake_input)

                errG = criterion(output, labelv)
                errG.backward()

                grad_norm = torch.sum(G.embeddings.weight.grad.data**2)**0.5
                D_G_z2 = output.data.mean()

                optimizerG.step()

            print(f'[{epoch+1}/{niter}][{i+1}/{len(trainloader)}] Loss_D: {errD.data[0]:f} ' +
                  (f'Loss_G: {errG.data[0]:f} ' if train_gen else '') +
                  f'D(x): {D_x:f} ' +
                  f'D(G(z)): {D_G_z1:f} ' + (f'/ {D_G_z2:f} ' if train_gen else ''))

        prec, mean, std = test_D(testloader, G, D)
        grid = make_test_grid()

        precisions.append(prec)
        means.append(mean)
        stds.append(std)
        grids.append(grid)
        grad_norms.append(grad_norm)

        print(prec, mean, std, sep=' ')
        torch.save(G.state_dict(), f'{outf}/netG_epoch_{epoch}.pth')
        torch.save(D.state_dict(), f'{outf}/netD_ndf_{D_ndf}_epoch_{epoch}.pth')

    results = {'nbperobj': nbperobj, 'D_ndf': D_ndf, 'obj_format': obj_format, 'maxsize': maxsize, 'batch_size': batch_size,
               'train_gen': train_gen,
               'precision': precisions, 'means': means, 'stds': stds}

    import pickle
    name = f'glr100_Dprec_Gmean_Gstd_' + ('nog_' if not train_gen else '') +\
               f'nbperobj_{nbperobj}_ndf_{D_ndf}_obj_{obj_format[0]}_{obj_format[1]}_maxsize_{maxsize}_bsize_{batch_size}'
    outf = outf / name
    outf.mkdir(exist_ok=True)
    pickle.dump(results, open(outf / 'perf.pkl', 'wb'))
    for i, grid in enumerate(grids):
        utils.save_image(grid, str((outf / f'{i}.png').absolute()))

    end = time.perf_counter()

    print(f'{end - start} seconds taken')

