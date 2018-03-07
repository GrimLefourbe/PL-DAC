import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

def get_generator_input_sampler():
    return lambda m, n : torch.rand(m, n)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

if __name__ == '__main__':
    print('gan_tuto')
    data_mu = 75
    data_sigma = 10

    g_input_size = 1
    g_hidden_size = 100
    g_output_size = 1
    d_input_size = 100
    d_hidden_size = 50
    d_output_size = 1
    minibatch_size = d_input_size

    d_learning_rate = 2e-4
    g_learning_rate = 2e-4
    optim_betas = (0.9, 0.999)
    num_epochs = 30000
    print_interval = 200
    d_steps = 1
    g_steps = 1


    name, preprocess, d_input_func = "Raw data", lambda data: data, lambda x:x

    d_sampler = get_distribution_sampler(data_mu, data_sigma)
    gi_sampler = get_generator_input_sampler()

    G = Generator(g_input_size, g_hidden_size, g_output_size)
    D = Discriminator(d_input_func(d_input_size), d_hidden_size, d_output_size)

    criterion = nn.BCELoss()

    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

    for epoch in range(num_epochs):
        for d_index in range(d_steps):
            D.zero_grad()

            d_real_data = Variable(d_sampler(d_input_size))
            d_real_decision = D(preprocess(d_real_data))
            d_real_error = criterion(d_real_decision, Variable(torch.ones(1)))
            d_real_error.backward()

            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            d_fake_data = G(d_gen_input).detach()
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1)))
            d_fake_error.backward()
            d_optimizer.step()

        for g_index in range(g_steps):
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))
            g_error = criterion(dg_fake_decision, Variable(torch.ones(1)))
            g_error.backward()
            g_optimizer.step()

        if epoch % print_interval == 0:
            print(f"{epoch} : D: {d_real_error.data[0]}/{d_fake_error.data[0]} G: {g_error.data[0]} \n"
                  f"(Real: ({d_real_data.mean().data[0]}, {d_real_data.std().data[0]}\n"
                  f" Fake: ({d_fake_data.mean().data[0]}, {d_fake_data.std().data[0]}))")


