import os

import numpy
import numpy as np
from PyEMD import EMD
from scipy.signal import butter, lfilter, filtfilt, welch
from sklearn.feature_selection import f_classif

def getfreqs_power(signals, fs, nperseg, scaling):
    ''' Calculate power density or power spectrum density '''
    if scaling == "density":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='density')
        return freqs, power
    elif scaling == "spectrum":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='spectrum')
        return freqs, power
    else:
        return 0, 0


def getBand_Power(freqs, power, lower, upper):
    ''' Sum band power within desired frequency range '''
    low_idx = np.array(np.where(freqs <= lower)).flatten()
    up_idx = np.array(np.where(freqs > upper)).flatten()
    band_power = np.sum(power[low_idx[-1]:up_idx[0]])

    return band_power