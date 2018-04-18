from matplotlib import pyplot as plt
import pickle
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    folder = Path('tests/G8')
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
    labels = ['no G', 'glr = 1000', 'glr = 10', 'glr = 125', 'glr = 1']

    for i, d in enumerate(data):
        plt.figure()
        plt.ylim([0, 1])
        niter = len(d['grad_norms'])
        prec_G = np.array([d['precision'][i] for i in range(len(d['precision'])) if not i%2])
        prec_D = np.array([d['precision'][i] for i in range(len(d['precision'])) if i%2])
        mean_G = np.array([d['means'][i] for i in range(len(d['means'])) if not i%2])
        std_G = np.array([d['stds'][i] for i in range(len(d['stds'])) if not i%2])

        l_G, = plt.plot(prec_G, label='post G_training prec')
        l_D, = plt.plot(np.arange(niter)+0.5, prec_D, label='post D_training prec')
        l_mean, = plt.plot(mean_G, label='average mask')
        l_stdup, = plt.plot(mean_G + std_G, 'r--', label='mean + 1 std')
        l_stddown, = plt.plot(mean_G - std_G, 'r--', label='mean - 1 std')
        plt.legend(handles=[l_G, l_D, l_mean, l_stdup, l_stddown])

    plt.show()
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