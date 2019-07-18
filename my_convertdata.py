import pickle
import pywt

from scipy import signal

import numpy as np
from scipy import stats
from scipy.signal import chirp

from libs import detect_peaks
from cross_validation0 import run

nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064

def convertData(adr = "./data/features_clear_less.dat" ):
    print("Program started"+"\n")
    fout_data = open(adr,'w')
    fout_labels0 = open("./data\labels_0.dat",'w')
    fout_labels1 = open("./data\labels_1.dat",'w')
    fout_labels2 = open("./data\labels_2.dat",'w')
    fout_labels3 = open("./data\labels_3.dat",'w')

    print("\n"+"Print Successful")

    for i in range(32):  #nUser #4, 40, 32, 40, 8064
        if(i%1 == 0):
            if i < 10:
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
        fname = "./data_preprocessed_python\s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
        f = open(fname, 'rb')
        x = pickle.load(f, encoding='latin1')
        print(fname)
        for tr in range(nTrial):
            # noise = np.random.normal(0,3000, size=(8064,))
            start = 128*3 - 1
            if(tr%1 == 0):
                for dat in range(128*3,nTime):
                     if dat != 0:
                        if((dat-383)%768 == 0):
                            if(tr == 0 and i == 0):
                                print(dat)
                            for ch in [32,33,36,38]:
                                if(1 == 1):
                                    if ch == 36 or ch == 32 or ch == 33:
                                        data_fil = x['data'][tr][ch]
                                    else:
                                        data_oringin = x['data'][tr][ch]
                                        sos = signal.butter(4, 0.3, 'high', output='sos')
                                        data_fil = signal.sosfilt(sos, data_oringin)

                                    features = extract_fre_fea(data_fil[start:dat])
                                    for fea in features:
                                            fout_data.write(str(fea)+ " ")

                            start = dat

                            fout_labels0.write(str(x['labels'][tr][0]) + "\n")
                            fout_labels1.write(str(x['labels'][tr][1]) + "\n")
                            fout_labels2.write(str(x['labels'][tr][2]) + "\n")
                            fout_labels3.write(str(x['labels'][tr][3]) + "\n")
                            fout_data.write("\n")

                #个性化特征
                # fout_data.write(str(tr)+ " ")
                # fout_data.write(str(i)+ " ")
                # 总
                # print(x['data'][tr][39][:].shape)
                # for data in datas:
                #     fout_data.write(str(data)+ " ")
                # fout_labels0.write(str(x['labels'][tr][0]) + "\n")
                # fout_labels1.write(str(x['labels'][tr][1]) + "\n")
                # fout_labels2.write(str(x['labels'][tr][2]) + "\n")
                # fout_labels3.write(str(x['labels'][tr][3]) + "\n")
                # fout_data.write("\n")#40个特征换行
    fout_labels0.close()
    fout_labels1.close()
    fout_labels2.close()
    fout_labels3.close()
    fout_data.close()
    print("\n"+"Print Successful")

def extract_fre_fea(data,samplerate = 128):
    db4 = pywt.Wavelet('db4')
    c = pywt.wavedec(data, db4, mode='symmetric', level=2, axis=-1)
    fea = []
    for i in c:
        fea.append(i.sum())
        fea.append(i.mean())
        fea.append(np.std(i))
    # f, Pxx_den = signal.welch(data, samplerate, nperseg=None)
    # fea.append(Pxx_den.max())
    # fea.append(int(Pxx_den.argmax()))
    # fea.append(Pxx_den.mean())
    # fea.append(stats.kurtosis(Pxx_den))
    # fea.append(stats.skew(Pxx_den))
    return fea

def extract_data(target_data, a = 0):
    target_mean = target_data.mean(axis=a)
    target_median = np.median(target_data, axis=a)
    target_maximum = np.max(target_data, axis=a)
    target_minimum = np.min(target_data, axis=a)
    target_std = np.std(target_data, axis=a)
    target_var = np.var(target_data, axis=a)
    target_range = np.ptp(target_data, axis=a)
    target_skew = stats.skew(target_data, axis=a)
    target_kurtosis = stats.kurtosis(target_data, axis=a)

    features = [target_mean, target_median, target_maximum, target_minimum, target_std, target_var, target_range, target_skew, target_kurtosis]
    # features_rescale = rescale(features)
    return features

def extractFeatures(PPGdata):
    dataWindow = signal.medfilt(PPGdata, 3)

    # Power spectral density (self.sampleWindow/4 reading window, self.sampleWindow/(4*2) sequential window overlap)
    f, pxx = signal.welch(dataWindow, fs = 128, nperseg = 384/4)
    pxx = [10*np.log10(x) for x in pxx]

    # Mean amplitude of HF components (20 to 200Hz range)
    m = np.mean(pxx[20:200])

    # LF/HF component magnitude ratio
    lfhfratio = (pxx[1] + pxx[2]) / (pxx[4] + pxx[5])

    # Apply a 2nd order bandpass butterworth filter to attenuate heart rate components
    # other than heart signal's RR peaks.
    b, a = signal.butter(2, [1/100, 1/10], 'bandpass')
    dataWindow = signal.filtfilt(b, a, dataWindow)
    # Find the segment peaks' indices for peak occurrence variance feature.
    indices = detect_peaks.detect_peaks(dataWindow,mph = 0, mpd = 100,edge='rising',show=False)
    peakVariance = np.finfo(np.float64).max
    if len(indices) > 1:
        peakVariance = np.var(np.diff(indices))

    # Calculate the heart rate number from number of peaks and start:end timeframe
    timeDifferenceInMinutes =  384 / 128 / 60

    # print(timeDifferenceInMinutes)
    heartRate = float(len(indices)) / timeDifferenceInMinutes

    # features = [m, lfhfratio, peakVariance, heartRate]
    features = [m, lfhfratio, peakVariance, heartRate]


    return features

def rescale(features, EX, std):
    re_feas = np.zeros(288)
    for i,fea in enumerate(features):
        re_feas[i] = (fea - EX[i])/std[i]

    return re_feas

def noise_gen(f0, len = 21):
    t = np.arange(0, len, 1.0/128)
    w = chirp(t, f0, f1=0, t1=21, method = "quadratic")
    # pl.plot(t,w)
    # pl.show()
    return w

if __name__ == '__main__':
    file = "./data/features_clear1.dat"
    print(file)
    convertData(adr = file);
    # run(file)


