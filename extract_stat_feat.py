import numpy as np
from scipy import stats

def extract_data(target_data, a = 0):
    target_res = []

    target_mean = target_data.mean(axis=a)
    target_median = np.median(target_data, axis=a)
    target_maximum = np.max(target_data, axis=a)
    target_minimum = np.min(target_data, axis=a)
    target_std = np.std(target_data, axis=a)
    target_var = np.var(target_data, axis=a)
    target_range = np.ptp(target_data, axis=a)
    target_skew = stats.skew(target_data, axis=a)
    target_kurtosis = stats.kurtosis(target_data, axis=a)

    return [target_mean, target_median, target_maximum, target_minimum, target_std, target_var, target_range, target_skew, target_kurtosis]
