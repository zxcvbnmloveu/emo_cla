#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Functions for Preprocessing
Extract the time-domain and frequency-domian features of original AMIGOS dataset
'''
import pickle
import pywt
from scipy import signal
nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064
from argparse import ArgumentParser
import os
import warnings
import numpy as np
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def convertData(adr = "C:\\Users\\zxcvb\\PycharmProjects\\new1\\data/features_1.dat" ):
    print("Program started"+"\n")
    fout_data = open(adr,'w')
    fout_labels0 = open("C:\\Users\\zxcvb\\PycharmProjects\\new1\\data\labels_0.dat",'w')
    fout_labels1 = open("C:\\Users\\zxcvb\\PycharmProjects\\new1\\data\labels_1.dat",'w')
    fout_labels2 = open("C:\\Users\\zxcvb\\PycharmProjects\\new1\\data\labels_2.dat",'w')
    fout_labels3 = open("C:\\Users\\zxcvb\\PycharmProjects\\new1\\data\labels_3.dat",'w')

    print("\n"+"Print Successful")
    for i in range(32):
    # for i in [i for i in range(32) if i != 1]:  #nUser #4, 40, 32, 40, 8064
    # for i in [1]:  # nUser #4, 40, 32, 40, 8064

        if(i%1 == 0):
            if i < 10:
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
        fname = "C:\\Users\\zxcvb\\PycharmProjects\\new1\\data_preprocessed_python\\s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
        f = open(fname, 'rb')
        x = pickle.load(f, encoding='latin1')
        print(fname)
        for tr in range(nTrial):
            if(tr%1 == 0):
                start = 128 * 3
                # features = gsr_features

                # for fea in features:
                #     fout_data.write(str(fea) + " ")

                fout_labels0.write(str(x['labels'][tr][0]) + "\n")
                fout_labels1.write(str(x['labels'][tr][1]) + "\n")
                fout_labels2.write(str(x['labels'][tr][2]) + "\n")
                fout_labels3.write(str(x['labels'][tr][3]) + "\n")
                fout_data.write("\n")#40个特征换行
    fout_labels0.close()
    fout_labels1.close()
    fout_labels2.close()
    fout_labels3.close()
    fout_data.close()
    print("\n"+"Print Successful")

if __name__ == '__main__':
    convertData()
