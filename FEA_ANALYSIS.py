import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler



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
    return features


if __name__ == '__main__':
    file_x1 = "./data/features_clear.dat"
    file_x2 = "./data/features_noise.dat"
    fea_clear = np.genfromtxt(file_x1, delimiter=' ')
    fea_noise = np.genfromtxt(file_x2, delimiter=' ')
    fea_clear = StandardScaler().fit_transform(fea_clear)
    fea_noise = StandardScaler().fit_transform(fea_noise)

    file_y = './data/label_class_0_3class.dat'
    y = np.genfromtxt(file_y, delimiter=' ')
    indice_2 = np.where(y == 2)
    indice_1 = np.where(y == 1)
    indice_0 = np.where(y == 0)
    fea_clear_indice_0 = fea_clear[indice_0]
    fea_noise_indice_0 = fea_noise[indice_0]
    fea_clear_indice_1 = fea_clear[indice_1]
    fea_noise_indice_1 = fea_noise[indice_1]
    fea_clear_indice_2 = fea_clear[indice_2]
    fea_noise_indice_2 = fea_noise[indice_2]
    print(fea_clear_indice_0.shape, fea_clear_indice_1.shape, fea_noise_indice_0.shape, fea_noise_indice_1.shape)

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))
print("begin")
plt.subplots()
temp0 = pd.DataFrame(fea_clear_indice_0[:,(1)], columns=['clear_label0'])
# basic plot
temp0['X'] = pd.Series(['-1.5', '-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', 'B', 'B', 'B'])
boxplot = temp0.boxplot(sym='',vert=False,patch_artist=True,meanline=False,showmeans=True, by='X', return_type='axes')
print(type(boxplot))
plt.show()

    # for i in [0, 1, 2, 9, 10, 12, 13, 15, 20, 24, 27, 30, 31, 33]:
    #     pass
    #     fig = plt.figure();
    #     fig, axs = plt.subplots(1, 1)
    #     # ticks = axs.set_xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
    #     temp0 = pd.DataFrame(np.transpose(np.vstack((fea_clear_indice_0[:,(i)],
    #                                                 fea_noise_indice_0[:,(i)]))),
    #                    columns=['clear_label0', 'noise_label0'])
    #     axs[0, 0].boxplot(fea_clear_indice_0[:,(i)], sym='')
        # # temp0.boxplot(sym='',vert=False,patch_artist=True,meanline=False,showmeans=True)
        # ax = fig.add_subplot(3, 1, 2)
        # temp1 = pd.DataFrame(np.transpose(np.vstack((fea_clear_indice_1[:,(i)],
        #                                             fea_noise_indice_1[:,(i)]))),
        #                columns=['clear_label1', 'noise_label1'])
        # temp1.boxplot(sym='',vert=False,patch_artist=True,meanline=False,showmeans=True)
        # ax = fig.add_subplot(3, 1, 3)
        # temp2 = pd.DataFrame(np.transpose(np.vstack((fea_clear_indice_2[:,(i)],
        #                                             fea_noise_indice_2[:,(i)]))),
        #                columns=['clear_label2', 'noise_label2'])
        # temp2.boxplot(sym='',vert=False,patch_artist=True,meanline=False,showmeans=True)


        # plt.savefig('C:/Users/cuzzw/Desktop/pic3/feature ' + str(i) + '.jpg')


