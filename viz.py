from matplotlib import pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    # prec4 = pickle.load(open('bicycles/Dprec_ndf_4.pkl', 'rb'))
    # prec16 = pickle.load(open('bicycles/Dprec_ndf_16.pkl', 'rb'))
    # prec32 = pickle.load(open('bicycles/Dprec_ndf_32.pkl', 'rb'))
    # prec10 = pickle.load(open('bicycles/Dprec_nbperobj_10_ndf_16.pkl', 'rb'))
    # prec20 = pickle.load(open('bicycles/Dprec_nbperobj_20_ndf_16.pkl', 'rb'))
    # prec50 = pickle.load(open('bicycles/Dprec_nbperobj_50_ndf_16.pkl', 'rb'))
    # prec100 = pickle.load(open('bicycles/Dprec_nbperobj_100_ndf_16.pkl', 'rb'))
    # prec200 = pickle.load(open('bicycles/Dprec_nbperobj_200_ndf_16.pkl', 'rb'))
    prec4 = pickle.load(open('bicycles/Dprec_nbperobj_50_ndf_4_obj_8_8.pkl', 'rb'))
    prec8 = pickle.load(open('bicycles/Dprec_nbperobj_50_ndf_8_obj_8_8.pkl', 'rb'))
    prec16 = pickle.load(open('bicycles/Dprec_nbperobj_50_ndf_16_obj_8_8.pkl', 'rb'))
    prec32 = pickle.load(open('bicycles/Dprec_nbperobj_50_ndf_32_obj_8_8.pkl', 'rb'))

    plt.figure()
    line4 = plt.plot(prec4)
    line8 = plt.plot(prec8)
    line16 = plt.plot(prec16)
    line32 = plt.plot(prec32)

    plt.legend(['ndf = 4', 'ndf = 8', 'ndf = 16', 'ndf = 32'])

    plt.title('64x64 backgrounds with 8x8 bicycles, nbperobj = 50')
    plt.xlabel('epochs')
    plt.ylabel('precision')
    plt.savefig('results/bicycles_ndf_Dprec_nbperobj_50.png')
    plt.show()