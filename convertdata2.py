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
    # 保存到features_1文件里
    # 标签值保存在label0等文件里， 用labelcalss0等文件二值化
    print("Program started"+"\n")
    fout_data = open(adr,'w')
    print("\n"+"Print Successful")
    each_seg = 15
    seg_num = 60 // each_seg
    fs = 128

    data = np.zeros((10,seg_num,32*nTrial,each_seg*fs))
    print(data.shape)
    for ch in range(10):
        start = 128 * 3
        for seg in range(seg_num):

            print("channel %d seg %d "%(ch,seg))

            next = start + each_seg * fs
            for i in range(32):
                if(i%1 == 0):
                    if i < 10:
                        name = '%0*d' % (2,i+1)
                    else:
                        name = i+1
                fname = "C:\\Users\\zxcvb\\PycharmProjects\\new1\\data_preprocessed_python\\s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
                f = open(fname, 'rb')
                x = pickle.load(f, encoding='latin1')
                for tr in range(nTrial):
                    if(tr%1 == 0):
                        data[ch][seg][i*40 + tr][:] = x['data'][tr][ch][start:next]

            start = next
    np.save('./data/seg_data.npy', data)

    fout_data.close()
    print("\n"+"Print Successful")

if __name__ == '__main__':
    convertData()
