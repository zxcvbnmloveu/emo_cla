import numpy as np
import scipy.io as scio


def run():
    dataFile = 'mahnob_features.mat'
    data = scio.loadmat(dataFile)

    features = data['features']
    row_num, column_num = features.shape

    label_arousal = []
    label_valence = []
    for i in range(row_num):
        for j in range(column_num):
            try:
                label_arousal += list(features[i][j][0][0][0][2][0])
                label_valence += list(features[i][j][0][0][0][3][0])
            except IndexError:
                # 空数据置为-1
                label_arousal.append(-1)
                label_valence.append(-1)
    # 处理标签数据
    arousal_median = np.median([i for i in label_arousal if i != -1])
    valence_median = np.median([i for i in label_valence if i != -1])
    for i in range(len(label_arousal)):
        if label_arousal[i] != -1:
            label_arousal[i] = 0 if label_arousal[i] < arousal_median else 1
        if label_valence[i] != -1:
            label_valence[i] = 0 if label_valence[i] < valence_median else 1
    label_arousal = np.array(label_arousal)
    label_valence = np.array(label_valence)
    # print(len(label_arousal), label_arousal)
    print('label_arousal.shape:'+str(label_arousal.shape)+'\nlabel_valence.shape:'+str(label_valence.shape))

    # print(features.shape)
    ECG_fea = features[0][0][1].reshape(-1)
    tmp_shape = ECG_fea.shape
    for i in range(row_num):
        for j in range(column_num):
            if i is 0 and j is 0:
                continue
            try:
                ECG_fea = np.vstack((ECG_fea, features[i][j][1].reshape(-1)))
            except ValueError:
                # 空数据置为全1数组，下同
                ECG_fea = np.vstack((ECG_fea, np.ones(tmp_shape)))
    print("ECG_fea.shape:" + str(ECG_fea.shape))

    EEG_fea = features[0][0][3].reshape(-1)
    tmp_shape = EEG_fea.shape
    for i in range(row_num):
        for j in range(column_num):
            if i is 0 and j is 0:
                continue
            try:
                EEG_fea = np.vstack((EEG_fea, features[i][j][3].reshape(-1)))
            except ValueError:
                EEG_fea = np.vstack((EEG_fea, np.ones(tmp_shape)))
    print("EEG_fea.shape:" + str(EEG_fea.shape))

    GSR_fea = features[0][0][5].reshape(-1)
    tmp_shape = GSR_fea.shape
    for i in range(row_num):
        for j in range(column_num):
            if i is 0 and j is 0:
                continue
            try:
                GSR_fea = np.vstack((GSR_fea, features[i][j][5].reshape(-1)))
            except ValueError:
                GSR_fea = np.vstack((GSR_fea, np.ones(tmp_shape)))
    print("GSR_fea.shape:" + str(GSR_fea.shape))

    HST_fea = features[0][0][7].reshape(-1)
    tmp_shape = HST_fea.shape
    for i in range(row_num):
        for j in range(column_num):
            if i is 0 and j is 0:
                continue
            try:
                HST_fea = np.vstack((HST_fea, features[i][j][7].reshape(-1)))
            except ValueError:
                HST_fea = np.vstack((HST_fea, np.ones(tmp_shape)))
    print("HST_fea.shape:" + str(HST_fea.shape))

    RES_fea = features[0][0][9].reshape(-1)
    tmp_shape = RES_fea.shape
    for i in range(row_num):
        for j in range(column_num):
            if i is 0 and j is 0:
                continue
            try:
                RES_fea = np.vstack((RES_fea, features[i][j][9].reshape(-1)))
            except ValueError:
                RES_fea = np.vstack((RES_fea, np.ones(tmp_shape)))
    print("RES_fea.shape:" + str(RES_fea.shape))

    # peripheral_fea = np.hstack((ECG_fea, GSR_fea, RES_fea, HST_fea[:, :-4]))
    # print("peripheral_fea.shape:" + str(peripheral_fea.shape))
    # all_fea = np.hstack((EEG_fea, peripheral_fea))


if __name__ == '__main__':
    run()
    # 根据标签是否为-1排除空值
