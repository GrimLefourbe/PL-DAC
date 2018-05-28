from matplotlib import pyplot as plt
import pickle
import numpy as np
from pathlib import Path

def plot_prec_single_param(param, title, data, param_name=None):
    if param_name is None:
        param_name = param
    plt.title(title)
    plt.ylim([0.5, 1])
    plt.xlabel('epoch')
    plt.ylabel('test prec')
    niter = data[0]['niter']
    lines = []
    data.sort(key=lambda x:x[param]['name'])
    for i, d in enumerate(data):
        prec_D = np.array([p for i, p in enumerate(d['precision'])])
        l_D, = plt.plot(prec_D, label=f'{param_name} = {d[param]}')
        lines.append(l_D)
    plt.legend([f'{param_name} = {d[param]["name"]}' for d in data])
    plt.show()

def plot_multiple_single_param(param, title_format, data, param_name=None, simult=False, labels=None):
    if param_name is None:
        param_name = param

    niter = data[0]['niter']
    plots = []
    if not labels:
        data.sort(key=lambda x:x[param])
    for i, d in enumerate(data):
        plt.figure()
        if labels:
            plt.title(labels[i])
        else:
            plt.title(title_format.format(**d))
        plt.ylim([0, 1])

        if simult:
            prec_D = np.array([p for i, p in enumerate(d['precision'])])
            mean_G = np.array([m for i, m in enumerate(d['means'])])
            std_G = np.array([s for i, s in enumerate(d['stds'])])
        else:
            prec_D = np.array([p for i, p in enumerate(d['precision']) if i%2])
            prec_G = np.array([p for i, p in enumerate(d['precision']) if not i%2])
            mean_G = np.array([m for i, m in enumerate(d['means']) if not i%2])
            std_G = np.array([s for i, s in enumerate(d['stds']) if not i%2])

        if simult:
            l_D, = plt.plot(prec_D, label='precision')
        else:
            l_G, = plt.plot(prec_G, label='post G_training prec')
            l_D, = plt.plot(np.arange(niter) + 0.5, prec_D, label='post D_training prec')
        l_mean, = plt.plot(mean_G, label='average mask')
        l_stdup, = plt.plot(mean_G + std_G, 'r--', label='mean + 1 std')
        l_stddown, = plt.plot(mean_G - std_G, 'r--', label='mean - 1 std')
        if simult:
            plt.legend(handles=[l_D, l_mean, l_stdup, l_stddown])
        else:
            plt.legend(handles=[l_G, l_D, l_mean, l_stdup, l_stddown])
        plt.show()

def plot_kernel(im, title):
    fig= plt.figure()
    plt.title(title)
    plt.imshow(im, interpolation='none')
    plt.colorbar()
    return fig

if __name__ == '__main__':
    folder = Path('tests/F_gen_kernels')
    # prec4 = pickle.load(open('bicycles/Dprec_ndf_4.pkl', 'rb'))
    # prec16 = pickle.load(open('bicycles/Dprec_ndf_16.pkl', 'rb'))
    # prec32 = pickle.load(open('bicycles/Dprec_ndf_32.pkl', 'rb'))
    # prec10 = pickle.load(open('bicycles/Dprec_nbperobj_10_ndf_16.pkl', 'rb'))
    # prec20 = pickle.load(open('bicycles/Dprec_nbperobj_20_ndf_16.pkl', 'rb'))
    # prec50 = pickle.load(open('bicycles/Dprec_nbperobj_50_ndf_16.pkl', 'rb'))
    # prec100 = pickle.load(open('bicycles/Dprec_nbperobj_100_ndf_16.pkl', 'rb'))
    # prec200 = pickle.load(open('bicycles/Dprec_nbperobj_200_ndf_16.pkl', 'rb'))
    folder_list = list(folder.glob('Dprec*'))

    data = [pickle.load(open(x/'perf.pkl', 'rb')) for x in folder_list]
    labels = ['Flat kernel', 'Gauss**1/3', 'Gauss**1/5', 'Gauss**1/7']

    plot_multiple_single_param('obj_format', 'Evolution over 75 epoch', data, labels=labels)
#    plot_prec_single_param('category', 'Impact of the object type over 25 epochs', data, 'obj_type')
#    plot_multiple_single_param('lr', 'Learning rate = {lr}', data)

    # from work import get_gaussian_objective, get_higher_better_objective
    # kernel1 = get_gaussian_objective(1)((16, 16))
    # kernel3 = get_gaussian_objective(3)((16, 16))
    # kernel5 = get_gaussian_objective(5)((16, 16))
    # kernel7 = get_gaussian_objective(7)((16, 16))
    # plot_kernel(kernel1, 'gauss1')
    # plot_kernel(kernel3, 'gauss3')
    # plot_kernel(kernel5, 'gauss5')
    # plot_kernel(kernel7, 'gauss7')

    #
    # for i, d in enumerate(data):
    #     plt.figure()
    #     plt.title(f"Impact of the number of backgrounds over 16000 (obj, bg) pairs")
    #     plt.ylim([0, 1])
    #     niter = d['niter']
    #     prec_G = np.array([p for i, p in enumerate(d['precision']) if not i%2])
    #     prec_D = np.array([p for i, p in enumerate(d['precision']) if i%2])
    #     mean_G = np.array([m for i, m in enumerate(d['means']) if not i%2])
    #     std_G = np.array([s for i, s in enumerate(d['stds']) if not i%2])
    #
    #     l_G, = plt.plot(prec_G, label='post G_training prec')
    #     l_D, = plt.plot(np.arange(niter)+0.5, prec_D, label='post D_training prec')
    #     l_mean, = plt.plot(mean_G, label='average mask')
    #     l_stdup, = plt.plot(mean_G + std_G, 'r--', label='mean + 1 std')
    #     l_stddown, = plt.plot(mean_G - std_G, 'r--', label='mean - 1 std')
    #     plt.legend(handles=[l_G, l_D, l_mean, l_stdup, l_stddown])
    #
    # plt.show()
    # precisions = [np.array(pickle.load(open(x, 'rb'))) for x in folder.glob('*Dprec*nbperobj_100000_*_maxsize_16000*.pkl')]
    # print(len(precisions))
    #
    # plt.figure()
    # for prec in precisions:
    #     plt.plot(prec[:, 0])
    #
    #
    # plt.legend(labels)
    # for prec in precisions:
    #     plt.plot(prec[:, 1])
    #
    # plt.title('64x64 backgrounds with 8x8 bicycles, nbperobj = all')
    # plt.xlabel('epochs')
    # plt.ylabel('precision')
    # plt.savefig('results/bicycles_G_impact_Dprec_nbperobj_100000_ndf_32_maxsize_16000.png')
    # plt.show()