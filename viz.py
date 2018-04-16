from matplotlib import pyplot as plt
import pickle
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    folder = Path('tests/G1')
    # prec4 = pickle.load(open('bicycles/Dprec_ndf_4.pkl', 'rb'))
    # prec16 = pickle.load(open('bicycles/Dprec_ndf_16.pkl', 'rb'))
    # prec32 = pickle.load(open('bicycles/Dprec_ndf_32.pkl', 'rb'))
    # prec10 = pickle.load(open('bicycles/Dprec_nbperobj_10_ndf_16.pkl', 'rb'))
    # prec20 = pickle.load(open('bicycles/Dprec_nbperobj_20_ndf_16.pkl', 'rb'))
    # prec50 = pickle.load(open('bicycles/Dprec_nbperobj_50_ndf_16.pkl', 'rb'))
    # prec100 = pickle.load(open('bicycles/Dprec_nbperobj_100_ndf_16.pkl', 'rb'))
    # prec200 = pickle.load(open('bicycles/Dprec_nbperobj_200_ndf_16.pkl', 'rb'))
    labels = ['with G learning', 'without G learning', 'with 100x G learning', 'with 10x G learning']
    precisions = [np.array(pickle.load(open(x, 'rb'))) for x in folder.glob('*Dprec*nbperobj_100000_*_maxsize_16000*.pkl')]
    print(len(precisions))

    plt.figure()
    for prec in precisions:
        plt.plot(prec[:, 0])


    plt.legend(labels)
    for prec in precisions:
        plt.plot(prec[:, 1])

    plt.title('64x64 backgrounds with 8x8 bicycles, nbperobj = all')
    plt.xlabel('epochs')
    plt.ylabel('precision')
    plt.savefig('results/bicycles_G_impact_Dprec_nbperobj_100000_ndf_32_maxsize_16000.png')
    plt.show()