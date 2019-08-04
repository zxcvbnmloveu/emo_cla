import pickle

import biosppy
import pysnooper
import pywt
from scipy import signal
import numpy as np
from scipy import stats
from scipy.signal import chirp
from libs import detect_peaks
from util import getfreqs_power, getBand_Power
from scipy.stats import skew, kurtosis
from util import multiScaleEntropy
import matplotlib.pyplot as plt
from util import moving_average

def extract_DWT_fea(data, level = 2, fs = 128):

    features = []
    db4 = pywt.Wavelet('db4')
    c = pywt.wavedec(data, db4, mode='symmetric', level=level, axis=-1)
    for i in c:
        features.append(i.sum())
        features.append(i.mean())
        features.append(np.std(i))

    return features

def extract_PSD_fea(data, fs = 128):
    b, a = signal.butter(2, 20 * 2 / fs, 'lowpass')
    data = signal.filtfilt(b, a, data)

    # PSD features max,argmax
    features = []
    f, Pxx_den = signal.welch(data, fs, nperseg=data.size)
    features.append(Pxx_den.max())
    features.append(f[int(Pxx_den.argmax())])
    features.append(Pxx_den.mean())
    features.append(stats.kurtosis(Pxx_den))
    features.append(stats.skew(Pxx_den))
    return features

def extract_Temporal_fea(data, fs, a = 0):
    b, a = signal.butter(2, 20 * 2 / fs, 'lowpass')
    data = signal.filtfilt(b, a, data)

    target_mean = data.mean(axis=a)
    target_median = np.median(data, axis=a)
    target_maximum = np.max(data, axis=a)
    target_minimum = np.min(data, axis=a)
    target_std = np.std(data, axis=a)
    target_var = np.var(data, axis=a)
    target_range = np.ptp(data, axis=a)
    target_skew = stats.skew(data, axis=a)
    target_kurtosis = stats.kurtosis(data, axis=a)

    features = [target_mean, target_median, target_maximum, target_minimum, target_std, target_var, target_range, target_skew, target_kurtosis]
    return features

def extract_GSR_fea(GSR, fs, ampThresh = 1):
    #These are the timing threshold defined
    tThreshLow = 0.1
    tThreshUp = 20

    plt.subplot(211)
    plt.plot(np.arange(0,GSR.shape[0])/fs, GSR)
    # b, a = signal.butter(2, 1 * 2 / fs, 'lowpass')
    # GSR = signal.filtfilt(b, a, GSR)
    # GSR = signal.medfilt(GSR,61)
    # GSR = moving_average(GSR,61)


    #Search low and high peaks
    #low peaks are the GSR appex reactions (highest sudation)
    #High peaks are used as starting points for the reaction
    # dN = np.array(np.diff(GSR) <= 0 ,dtype=int)
    diff = moving_average(np.diff(GSR),50)
    dN = np.array(diff <= 0 ,dtype=int)

    plt.subplot(212)
    plt.plot(np.arange(0,GSR.shape[0])/fs, GSR)
    # plt.show()
    dN = np.diff(dN)
    idxL = np.where(dN < 0)[0] + 1; #+1 to account for the double derivative
    idxH = np.where(dN > 0)[0] + 1;


    #For each low peaks find it's nearest high peak and check that there is no
    #low peak between them, if there is then reject the peak (OR SEARCH FOR CURVATURE)
    riseTime = np.array([]) #vector of rise time for each detected peak
    ampPeaks = np.array([]) #vector of amplitude for each detected peak
    posPeaks = np.array([]) #final indexes of low peaks (not used but could be usefull for plot puposes)
    for iP in range(0,len(idxL)):
        #get list of high peak before the current low peak
        nearestHP = idxH[idxH < idxL[iP]]

        #if no high peak before (this is certainly the first peak detected in
        #the signal) don't do anything else process peak
        if len(nearestHP) > 0:
            #Get nearest high peak
            nearestHP = nearestHP[-1]

            #check if there is not other low peak between the nearest high and
            #the current low peaks. If not the case than compute peak features
            if not any( (idxL > nearestHP) & (idxL < idxL[iP]) ):
                rt = (idxL[iP] - nearestHP)/fs
                amp = GSR[nearestHP]- GSR[idxL[iP]]


                #if rise time and amplitude fits threshold then the peak is
                #considered and stored
                if (rt >= tThreshLow) and (rt <= tThreshUp) and (amp >= ampThresh):
                    riseTime = np.append(riseTime, rt)
                    ampPeaks = np.append(ampPeaks, amp)
                    posPeaks = np.append(posPeaks, idxL[iP])

    #Compute the number of positive peaks
    nbPeaks = len(posPeaks);

    #retype the arrays
    posPeaks = np.array(posPeaks,dtype=int)

    # print(nbPeaks, ampPeaks, riseTime, posPeaks)
    #return the values
    # rise time ratio
    rtr = (np.sum(riseTime)/GSR.shape[0]/fs)

    # top3 risetime average
    rt_sorted = sorted(riseTime) # sort the fisher from small to large
    rt_sorted = rt_sorted[::-1]  # arrange from large to small
    if len(rt_sorted)<3:
        top_rt = np.mean(rt_sorted)
    else:
        top_rt = np.mean((rt_sorted[:3]))

    # top3 amp average
    amp_sorted = sorted(riseTime)  # sort the fisher from small to large
    amp_sorted = amp_sorted[::-1]  # arrange from large to small
    if len(amp_sorted) < 3:
        top_amp = np.mean(amp_sorted)
    else:
        top_amp = np.mean((amp_sorted[:3]))

    features= [nbPeaks,rtr,top_rt,top_amp]
    return features

def extract_PPG_fea(PPGdata, fs = 128):

    PPGdata = signal.medfilt(PPGdata, 3)
    b, a = signal.butter(2, [0.5  * 2/fs, 50 * 2/fs], 'bandpass')
    dataWindow = signal.filtfilt(b, a, PPGdata)

    # Find the segment peaks' indices for peak occurrence variance feature.
    indices = detect_peaks.detect_peaks(dataWindow,mph = None, mpd = 50,edge='rising',show=False)
    IBI = np.array([])  # 两个峰值之之差的时间间隔
    for i in range(len(indices) - 1):
        IBI = np.append(IBI, (indices[i + 1] - indices[i]) / fs)

    # IBI feas 7
    mean_IBI = np.mean(IBI)
    rms_IBI = np.sqrt(np.mean(np.square(IBI)))
    std_IBI = np.std(IBI)
    skew_IBI = skew(IBI)
    kurt_IBI = kurtosis(IBI)
    per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size) / float(IBI.size)
    per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size) / float(IBI.size)
    IBI_feas = [mean_IBI,rms_IBI,std_IBI,skew_IBI,kurt_IBI,per_above_IBI,per_below_IBI]

    # HR feas 7
    heart_rate = np.array([])
    for i in range(len(IBI)):
        append_value = (PPGdata.size / fs) /IBI[i] if IBI[i] != 0 else 0
        heart_rate = np.append(heart_rate, append_value)

    mean_heart_rate = np.mean(heart_rate)
    std_heart_rate = np.std(heart_rate)
    skew_heart_rate = skew(heart_rate)
    kurt_heart_rate = kurtosis(heart_rate)
    per_above_heart_rate = float(heart_rate[heart_rate >
                                            mean_heart_rate + std_heart_rate].size) / float(heart_rate.size)
    per_below_heart_rate = float(heart_rate[heart_rate <
                                            mean_heart_rate - std_heart_rate].size) / float(heart_rate.size)
    # HRV
    peakVariance = np.finfo(np.float64).max
    if len(indices) > 1:
        peakVariance = np.var(np.diff(indices))
    HR_feas = [mean_heart_rate,std_heart_rate,skew_heart_rate,kurt_heart_rate,per_above_heart_rate,per_below_heart_rate,peakVariance]


    # power_0-0.6
    freqs, power = getfreqs_power(dataWindow, fs, nperseg=dataWindow.size, scaling='spectrum')
    power_0_6 = []
    for i in range(1,40):
        power_0_6.append(getBand_Power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))
    Power_0_6_fea = power_0_6

    # MSE fea
    MSE = multiScaleEntropy(indices, np.arange(1, 5), r=0.2 * np.std(indices), m=2)

    features = IBI_feas + HR_feas + MSE + power_0_6

    return features

def extract_ECG_fea(ECGdata, fs = 128):

    dataWindow = signal.medfilt(ECGdata, 3)
    b, a = signal.butter(2, [0.5  * 2/fs, 50 * 2/fs], 'bandpass')
    dataWindow = signal.filtfilt(b, a, ECGdata)

    # Find the segment peaks' indices for peak occurrence variance feature.
    ecg_all = biosppy.signals.ecg.ecg(signal=dataWindow, sampling_rate=fs, show=False)
    indices = ecg_all['rpeaks']
    IBI = np.array([])  # 两个峰值之之差的时间间隔
    for i in range(len(indices) - 1):
        IBI = np.append(IBI, (indices[i + 1] - indices[i]) / fs)

    # IBI feas 7
    mean_IBI = np.mean(IBI)
    rms_IBI = np.sqrt(np.mean(np.square(IBI)))
    std_IBI = np.std(IBI)
    skew_IBI = skew(IBI)
    kurt_IBI = kurtosis(IBI)
    per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size) / float(IBI.size)
    per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size) / float(IBI.size)
    IBI_feas = [mean_IBI,rms_IBI,std_IBI,skew_IBI,kurt_IBI,per_above_IBI,per_below_IBI]

    # HR feas 7
    heart_rate = np.array([])
    for i in range(len(IBI)):
        append_value = (ECGdata.size / fs) /IBI[i] if IBI[i] != 0 else 0
        heart_rate = np.append(heart_rate, append_value)

    mean_heart_rate = np.mean(heart_rate)
    std_heart_rate = np.std(heart_rate)
    skew_heart_rate = skew(heart_rate)
    kurt_heart_rate = kurtosis(heart_rate)
    per_above_heart_rate = float(heart_rate[heart_rate >
                                            mean_heart_rate + std_heart_rate].size) / float(heart_rate.size)
    per_below_heart_rate = float(heart_rate[heart_rate <
                                            mean_heart_rate - std_heart_rate].size) / float(heart_rate.size)
    # HRV
    peakVariance = np.finfo(np.float64).max
    if len(indices) > 1:
        peakVariance = np.var(np.diff(indices))
    HR_feas = [mean_heart_rate,std_heart_rate,skew_heart_rate,kurt_heart_rate,per_above_heart_rate,per_below_heart_rate,peakVariance]


    # power_0-0.6
    freqs, power = getfreqs_power(dataWindow, fs, nperseg=dataWindow.size, scaling='spectrum')
    power_0_6 = []
    for i in range(1,40):
        power_0_6.append(getBand_Power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))
    Power_0_6_fea = power_0_6

    # MSE fea
    MSE = multiScaleEntropy(indices,np.arange(1,5),r = 0.2 * np.std(indices),m = 2)

    features = IBI_feas+HR_feas+MSE+power_0_6


    return features

def GSR_preprocess(GSR, fs):
    b, a = signal.butter(2, 1 * 2 / fs, 'lowpass')
    GSR = signal.filtfilt(b, a, GSR)
    GSR = signal.medfilt(GSR, 61)
    GSR = moving_average(GSR, 61)
    return GSR

def ST_preprocess(data,fs):
    data = data - np.mean(data)
    data[np.where(data > 0.15)] = 0
    data[np.where(data < -0.15)] = 0
    plt.plot(np.arange(0, data.shape[0]) / fs, data)
    plt.subplot(212)
    data = signal.medfilt(data, 11)
    return data

def test():

    # ppgdata = np.load('./data/ppgdata.npy')#只有23个人数据正常
    data = np.load('data.npy')
    fea = extract_GSR_fea(data,fs =134)
    print(fea)

def test1():
    sort = (sorted([5, 2, 3, 1, 4]))
    print(sort[::-1])

if __name__ == '__main__':
    test()
