import biosppy
import scipy.io as scio
import numpy as np
from extr_fea_readata import extract_PPG_fea
from extr_fea_readata import extract_ECG_fea
from extr_fea_readata import extract_Temporal_fea
from extr_fea_readata import extract_PSD_fea
from extr_fea_readata import extract_DWT_fea
from extr_fea_readata import extract_GSR_fea
from scipy import signal
from util import moving_average
import matplotlib.pyplot as plt

def myplot(data, fs = 134):

    plt.subplot(211)
    # data = data - np.mean(data)
    # data[np.where(data > 0.15)] = 0
    # data[np.where(data < -0.15)] = 0
    plt.plot(np.arange(0,data.shape[0])/fs, data)
    plt.subplot(212)
    # data = signal.medfilt(data, 11)
    # data = moving_average(data, 3)
    # b, a = signal.butter(2, 1 * 2 / fs, 'lowpass')
    # data = signal.filtfilt(b, a, data)


    f, Pxx_den = signal.welch(data, fs, nperseg=data.size)
    plt.plot(f,10*np.log10(Pxx_den))
    # plt.plot(np.arange(0,data.shape[0])/fs, data)

    plt.show()


if __name__ == '__main__':
    data = scio.loadmat('./data/selfdata.mat')
    emotion = data['emotion'][0]
    # print(emotion[10][4][0].shape)

    testee_num = 11
    experi_num = 40

    PPG_data = np.array([])
    ECG_data = np.array([])
    GSR_data = np.array([])
    ST_data = np.array([])
    label = np.array([])
    len = 440
    for i in range(testee_num):
        PPG_data = np.append(PPG_data, emotion[i][0][0].reshape(-1))
        ECG_data = np.append(ECG_data, emotion[i][1][0].reshape(-1))
        GSR_data = np.append(GSR_data, emotion[i][2][0].reshape(-1))
        ST_data = np.append(ST_data, emotion[i][3][0].reshape(-1))
        label = np.append(label, emotion[i][4][0].reshape(-1))

    fs = 134


    for i in range(len):
        if i == 0:
            PPG_fea = [extract_PPG_fea(PPG_data[i][0,:],fs=134)]
        else:
            PPG_fea.append(extract_PPG_fea(PPG_data[i][0,:],fs=134))
    PPG_fea = np.array(PPG_fea)
    print('PPG_fea.shape'+str(PPG_fea.shape))
    # 调用去噪
    for i in range(len):
        if i == 0:
            GSR_fea = [extract_DWT_fea(GSR_data[i][0,:]) + extract_Temporal_fea(GSR_data[i][0,:])]
        else:
            GSR_fea.append(extract_DWT_fea(GSR_data[i][0,:]) + extract_Temporal_fea(GSR_data[i][0,:]))
    GSR_fea = np.array(GSR_fea)
    # print('GSR_fea.shape'+str(GSR_fea.shape))
    #
    # for i in range(len):
    #     if i == 0:
    #         ECG_fea = [extract_ECG_fea(ECG_data[i][0,:],fs=134)]
    #     else:
    #         ECG_fea.append(extract_ECG_fea(ECG_data[i][0,:],fs=134))
    # ECG_fea = np.array(ECG_fea)
    # print('ECG_fea.shape'+str(ECG_fea.shape))
    #
    # for i in range(len):
    #     if i == 0:
    #         label_v = [label[i][0,0]]
    #     else:
    #         label_v.append(label[i][0,0])
    # label_v = np.array(label_v)
    #
    # for i in range(len):
    #     if i == 0:
    #         label_a = [label[i][0,1]]
    #     else:
    #         label_a.append(label[i][0,1])
    # label_a = np.array(label_a)

