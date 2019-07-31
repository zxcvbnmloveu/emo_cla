import scipy.io as scio
import numpy as np
from extr_fea_readata import extract_PPG_fea

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

    for i in range(testee_num):
        PPG_data = np.append(PPG_data, emotion[i][0][0].reshape(-1))
        ECG_data = np.append(ECG_data, emotion[i][1][0].reshape(-1))
        GSR_data = np.append(GSR_data, emotion[i][2][0].reshape(-1))
        ST_data = np.append(ST_data, emotion[i][3][0].reshape(-1))
        label = np.append(label, emotion[i][4][0].reshape(-1))



    # for i in range(len(PPG_data)):
    #     if i == 0:
    #         PPG_fea = [extract_PPG_fea(PPG_data[i][0,:],fs=134)]
    #     else:
    #         PPG_fea.append(extract_PPG_fea(PPG_data[i][0,:]))
    # print(np.array(PPG_fea).shape)