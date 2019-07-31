import pickle

import biosppy
import pywt
from scipy import signal
import numpy as np
from scipy import stats
from scipy.signal import chirp
from libs import detect_peaks
from util import getfreqs_power, getBand_Power
from scipy.stats import skew, kurtosis

def extract_DWT_fea(data, level = 2, samplerate = 128):
    features = []
    db4 = pywt.Wavelet('db4')
    c = pywt.wavedec(data, db4, mode='symmetric', level=level, axis=-1)
    print(len(c))
    for i in c:
        features.append(i.sum())
        features.append(i.mean())
        features.append(np.std(i))

    return

def extract_PSD_fea(data, fs = 128):
    # PSD features max,argmax
    features = []
    f, Pxx_den = signal.welch(data, fs, nperseg=data.size)
    features.append(Pxx_den.max())
    features.append(f[int(Pxx_den.argmax())])
    features.append(Pxx_den.mean())
    features.append(stats.kurtosis(Pxx_den))
    features.append(stats.skew(Pxx_den))
    return features

def extract_Temporal_fea(signals, a = 0):
    target_mean = signals.mean(axis=a)
    target_median = np.median(signals, axis=a)
    target_maximum = np.max(signals, axis=a)
    target_minimum = np.min(signals, axis=a)
    target_std = np.std(signals, axis=a)
    target_var = np.var(signals, axis=a)
    target_range = np.ptp(signals, axis=a)
    target_skew = stats.skew(signals, axis=a)
    target_kurtosis = stats.kurtosis(signals, axis=a)
    der_signals = np.gradient(signals)
    con_signals = 1.0 / signals
    nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)

    der_mean = np.mean(der_signals)
    neg_der_mean = np.mean(der_signals[der_signals < 0])
    neg_der_pro = float(der_signals[der_signals < 0].size) / float(der_signals.size)

    features = [target_mean, target_median, target_maximum, target_minimum, target_std, target_var, target_range, target_skew, target_kurtosis,der_mean,neg_der_mean,neg_der_pro,nor_con_signals]
    return features

def extract_PPG_fea(PPGdata, fs = 128):

    dataWindow = signal.medfilt(PPGdata, 3)
    b, a = signal.butter(2, [0.5 / 100, 1 / 10], 'bandpass')
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
    for i in range(60):
        power_0_6.append(getBand_Power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))
    Power_0_6_fea = power_0_6

    features = IBI_feas+HR_feas+Power_0_6_fea


    return features

def gen():
    fs = 128
    N = 10000
    amp = 2 * np.sqrt(2)
    freq = 40
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    x = amp * np.sin(2 * np.pi * freq * time)
    x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    return x

def test():
    x = gen()
    # extract_DWT_fea(x,level = 3)
    ppgdata = np.load('./data/ppgdata.npy')#只有23个人数据正常
    ppg1 = ppgdata[123,:]
    extract_PPG_fea(ppg1)
    # biosppy.signals.bvp.bvp(signal=ppg1, sampling_rate=128.0, show=True)
    extract_Temporal_fea(x)

if __name__ == '__main__':
    test()
