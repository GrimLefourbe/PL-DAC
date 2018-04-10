from matplotlib import pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    prec4 = pickle.load(open('bicycles/Dprec_ndf_4.pkl', 'rb'))
    prec16 = pickle.load(open('bicycles/Dprec_ndf_16.pkl', 'rb'))
    prec32 = pickle.load(open('bicycles/Dprec_ndf_32.pkl', 'rb'))

    plt.figure()
    line4 = plt.plot(prec4)
    line16 = plt.plot(prec16)
    line32 = plt.plot(prec32)
    plt.legend(['ndf = 4', 'ndf = 16', 'ndf = 32'])

    plt.title('64x64 backgrounds with 8x8 objects and random position')
    plt.xlabel('epochs')
    plt.ylabel('precision')
    plt.savefig('bicycles/Dprec.png')
    plt.show()