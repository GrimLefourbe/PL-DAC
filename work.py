import datasets
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
import numpy as np

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
    batch_size = 8
    dataloader = DataLoader(datasets.ObjectsDataset({'id':2, 'name':'bicycle'},
                                                    obj_transform=transforms.Compose([transforms.Resize((20,20)),
                                                                                      transforms.ToTensor()]),
                                                    bg_transform=transforms.Compose([transforms.Resize((40,40)),
                                                                                     transforms.ToTensor()])),
                            batch_size=batch_size, shuffle=False, num_workers=4)
    G = Generator()
    D = Discriminator()
    i = dataloader.__iter__().next()
    obj, bg = i['obj'], i['bg']
    coord = torch.IntTensor([10, 10])

    o, b = Variable(obj), Variable(bg)
    c = Variable(coord)
    real_im = bg.clone()
    fake_im = G(o, b, c)

    show_im(real_im[0])
    show_im(o[0].data)
    show_im(fake_im[0].data)


    criterion = nn.BCELoss()
    d_optimizer = optim.SGD(D.parameters(), lr=0.001, momentum=0.9)

    d_learning_rate = 2e-4
    g_learning_rate = 2e-3
    optim_betas = (0.9, 0.999)
    num_epochs = 2
    print_interval = 200
    d_steps = 1
    g_steps = 2

    criterion = nn.BCELoss()

    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            for i, data in enumerate(dataloader):
                D.zero_grad()
                d_real_data = Variable(data['bg'])
                d_real_decision = D(d_real_data)
                d_real_error = criterion(d_real_decision, Variable(torch.ones(batch_size, 1)))
                d_real_error.backward()

                obj, bg, coord = Variable(data['obj']), Variable(data['bg']), Variable(torch.IntTensor([10, 10]))
                d_fake_data = G(obj, bg, coord).detach()
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(batch_size, 1)))
                d_fake_error.backward()
                d_optimizer.step()
                print(f'{epoch} {d_index} d_fake_error : {d_fake_error.data[0]}')
                print(f'{epoch} {d_index} d_real_error : {d_real_error.data[0]}')

        for g_index in range(g_steps):
            for i, data in enumerate(dataloader):
                G.zero_grad()

                obj, bg, coord = Variable(data['obj']), Variable(data['bg']), Variable(torch.IntTensor([10, 10]))
                g_fake_data = G(obj, bg, coord)
                dg_fake_decision = D(g_fake_data)
                g_error = criterion(dg_fake_decision, Variable(torch.ones(batch_size, 1)))
                g_error.backward()
                g_optimizer.step()

                print(f'{epoch} {g_index} g_error : {g_error.data[0]}')

    fake_im = G(o, b, c)

    show_im(real_im[0])
    show_im(o[0].data)
    show_im(fake_im[0].data)

