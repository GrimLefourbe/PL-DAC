import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import datasets
from models import Generator, Discriminator

ToPIL = transforms.ToPILImage()

def train_test_indices(n, p_train):
    train_len = int(n * p_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices[:train_len], indices[train_len:]


def show_im(t):
    ToPIL(t).show()

def get_higher_better_objective(obj_format):
    return torch.ones(obj_format)

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def get_gaussian_objective(n):
    def get_kernel(obj_format):
        kern = gkern(obj_format[0])**(1/n)
        max, min = np.max(kern), np.min(kern)
        kern = (kern - min) / (max - min)
        return torch.FloatTensor(kern)
    return get_kernel

if __name__ == '__main__':
    start = time.perf_counter()
    np.random.seed(47)

    outf = Path("tests/F_gen_gauss5_scaling")
    outf.mkdir(parents=True, exist_ok=True)

    extra_name = 'ratio0.1'

    nb_images = 30

    train_gen = True

    batch_size = 128
    D_ndf = 32
    nb_backgrounds = 16000
    maxsize = 16000

    objective = get_gaussian_objective(5)

    image_indices = np.random.choice(maxsize, replace=False, size=nb_images)

    bg_format = (64, 64)
    obj_format = (16, 16)
    category = {'id': 16, 'name': 'bird'}


    dataset = datasets.ObjectsDataset(category, nbperobj=nb_backgrounds, shuffled=True, maxsize=maxsize,
                                      obj_format=obj_format, bg_format=bg_format,
                                      objects_folder=datasets.TrainStorageFolder, background_folder=datasets.Train_img_folder, annFile=datasets.instTrainFile)

    if train_gen:
        _, test_indices = train_test_indices(len(dataset), p_train=0.95)
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=True)
        testloader = DataLoader(datasets.SubsetDataset(dataset, test_indices),
                                batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        print('Loaders prepped')
    else:
        train_indices, test_indices = train_test_indices(len(dataset), p_train=0.95)

        trainloader = DataLoader(datasets.SubsetDataset(dataset, train_indices),
                                 batch_size=batch_size, shuffle=False, num_workers=12, drop_last=True)
        testloader = DataLoader(datasets.SubsetDataset(dataset, test_indices),
                                batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    print(len(trainloader))

    use_gpu = torch.cuda.is_available()

    criterion = nn.BCELoss()
    mask_basis = torch.ones((batch_size, obj_format[0], obj_format[1])) * objective(obj_format)
    mask_criterion = nn.L1Loss()

    real_label = 0.9
    fake_label = 0.1

    lr = 2*1e-4
    g_to_d_lr = 125
    betas = (0.5, 0.999)

    niter = 50

    D = Discriminator(ndf=D_ndf)
    G = Generator(nb_embeddings=len(trainloader) * batch_size, obj_format=obj_format)

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(G.parameters(), lr=g_to_d_lr*lr, betas=betas)

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
        mask_basis = mask_basis.cuda()

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
        images = [fake[i//2].data if i%2 else bg[i//2].data for i in range(2*len(image_indices))]
        grid = utils.make_grid(torch.stack(images), nrow=10)
        print(grid.shape)
        return grid

    trainloader = list(trainloader)
    testloader= list(testloader)

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

            errD_fake = criterion(fake_output, fake_labelv)
            errD_fake.backward()

            D_G_z1 = fake_output.data.mean()

            errD = errD_real + errD_fake

            if errD.data[0] > 0:
                optimizerD.step()
            else:
                print('skipping Dstep')
            print(f'[{epoch+1}/{niter}][{i+1}/{len(trainloader)}] Loss_D: {errD.data[0]:f} ' +
                  f'D(x): {D_x:f} ' +
                  f'D(G(z)): {D_G_z1:f} ')

        prec, mean, std = test_D(testloader, G, D)
        precisions.append(prec)
        means.append(mean)
        stds.append(std)

        print(prec, mean, std, sep=' ')

        if train_gen:
            for i, (cpu_obj, cpu_bg, cpu_coord) in enumerate(trainloader):  # for each batch

                bg.copy_(cpu_bg)
                obj.copy_(cpu_obj)
                coord.copy_(cpu_coord)
                obj_id.copy_(torch.arange(i * batch_size, (i + 1) * batch_size).long())

                objv, bgv, coordv, obj_idv = Variable(obj), Variable(bg), Variable(coord), Variable(obj_id)

                fake_input, masks = G(objv, bgv, coordv, obj_idv)

                #update G network
                G.zero_grad()

                label.fill_(real_label)
                labelv = Variable(label)

                output = D(fake_input)

                mask_basisv = Variable(mask_basis)

                errG = criterion(output, labelv) + 0.1 * mask_criterion(masks, mask_basisv)
                errG.backward()

                grad_norm = torch.sum(G.embeddings.weight.grad.data**2)**0.5
                D_G_z2 = output.data.mean()

                optimizerG.step()


                print(f'[{epoch+1}/{niter}][{i+1}/{len(trainloader)}] ' +
                      f'Loss_G: {errG.data[0]:f} ' +
                      f'D(G(z)): {D_G_z2:f}')

        prec, mean, std = test_D(testloader, G, D)
        precisions.append(prec)
        means.append(mean)
        stds.append(std)
        print(prec, mean, std, sep=' ')

        grid = make_test_grid()

        grids.append(grid)
        if train_gen:
            grad_norms.append(grad_norm)

        torch.save(G.state_dict(), f'{outf}/netG_epoch_{epoch}.pth')
        torch.save(D.state_dict(), f'{outf}/netD_ndf_{D_ndf}_epoch_{epoch}.pth')

    results = {'nbperobj': nb_backgrounds, 'D_ndf': D_ndf, 'obj_format': obj_format, 'maxsize': maxsize, 'batch_size': batch_size, 'glr':g_to_d_lr, 'lr':lr, 'niter':niter,
               'train_gen': train_gen, 'category': category,
               'precision': np.array(precisions), 'means': np.array(means), 'stds': np.array(stds), 'grad_norms': np.array(grad_norms)}

    import pickle, random
    name = f'Dprec_Gmean_Gstd_grad_' + ('nog_' if not train_gen else '') + '_' + extra_name + '_' +\
           f'nbperobj_{nb_backgrounds}_ndf_{D_ndf}_obj_{obj_format[0]}_{obj_format[1]}_maxsize_{maxsize}_bsize_{batch_size}_{random.randint(0, 2**31)}'
    outf = outf / name
    outf.mkdir(exist_ok=True)
    pickle.dump(results, open(outf / 'perf.pkl', 'wb'))
    for i, grid in enumerate(grids):
        utils.save_image(grid, str((outf / f'{i}.png').absolute()))

    end = time.perf_counter()

    print(f'{end - start} seconds taken')

