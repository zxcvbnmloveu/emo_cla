import warnings

import numpy as np
import pysnooper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import time
import scipy.io as scio
import numpy
from sklearn import model_selection
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings(action="ignore")

def fisher_idx(features, labels):
    ''' Get idx sorted by fisher linear discriminant
        :param train data with shape: sample_num,fea_num
        :param train label
        :return select index 3
    '''
    labels = np.array(labels)
    labels0 = np.where(labels < 1)
    labels1 = np.where(labels > 0)
    labels0 = np.array(labels0).flatten()
    labels1 = np.array(labels1).flatten()
    features0 = np.delete(features, labels1, axis=0)
    features1 = np.delete(features, labels0, axis=0)
    mean_features0 = np.mean(features0, axis=0)
    mean_features1 = np.mean(features1, axis=0)
    std_features0 = np.std(features0, axis=0)
    std_features1 = np.std(features1, axis=0)
    std_sum = std_features1 ** 2 + std_features0 ** 2
    fisher = (abs(mean_features0 - mean_features1)) / std_sum
    # print(fisher)
    fisher[np.isnan(fisher)] = 0
    # print(fisher)
    fisher_sorted = np.argsort(np.array(fisher))  # sort the fisher from small to large
    sorted_feature_idx = fisher_sorted[::-1]  # arrange from large to small
    # select fisher larger than 0.3
    fisher_threshold = 0.3
    feature_idx = np.where(fisher > fisher_threshold)
    if (features[:, feature_idx].shape[2] < 3):
        return sorted_feature_idx[:3]
    return feature_idx

def delete_NAN(file_x):
    '''
    delete NAN features
    :param data
    :return processed data
    '''
    min = np.min(file_x, axis=0)
    delete_idx = np.isnan(min)
    file_x_new = file_x[:, ~delete_idx]
    return file_x_new

def normalization_binary(file_x):
    '''
    将file_x特征按median进行二进制normalization（大于为1，小于为1）
    :param filex for one subject
    :return normalization_data
    '''
    file_x_new = file_x.copy()
    median = np.median(file_x,axis=0)
    for i in range(file_x.shape[1]):
        file_x_new[:, i] = np.where(file_x[:, i] < median[i], 0, 1)
    return file_x_new

def nor_subject(file_x):
    '''
        将每个subject的特征按median进行二进制normalization（大于为1，小于为1）
        :param file_x
        :return normalization_data
    '''
    file_x_new = file_x.copy()
    for i in range(32):
        file_x_new[40 * i: 40 * (i + 1),:] = normalization_binary(file_x[40 * i: 40 * (i + 1),:])
    return file_x_new

def inter_model_eval(file_x, file_y, nClasses = 2):
    '''
    :param file_x: input data
    :param file_y: label
    :param nClasses: label class
    :return: average F1,ACC
    '''
    # model = LogisticRegression(solver='liblinear')
    model = KNeighborsClassifier()
    # model = LinearSVC(random_state=0, tol=1e-5)
    F1s =[]
    ACCs = []

    for i in range(32):
        test_idx = [i for i in range(40 * i, 40 * (i + 1))]
        train_idx = [i for i in range(1280) if i not in test_idx]
        test_data = file_x[test_idx, :]
        train_data = file_x[train_idx, :]
        test_label = file_y[test_idx]
        train_label = file_y[train_idx]

        numpy.random.seed(9)
        shuffle_indices = numpy.random.permutation(numpy.arange(len(train_data)))
        train_data = train_data[shuffle_indices]  # 将样本和标签打乱
        train_label = train_label[shuffle_indices]

        model.fit(train_data,train_label)
        y_pred = model.predict(test_data)
        # F1Score = []
        # for j in range(nClasses):
        #     # true positive
        #     TP = np.sum(np.logical_and(np.equal(test_label, j), np.equal(y_pred, j)))
        #     # false positive
        #     FP = np.sum(np.logical_and(np.not_equal(test_label, j), np.equal(y_pred, j)))
        #     # true negative
        #     TN = np.sum(np.logical_and(np.not_equal(test_label, j), np.not_equal(y_pred, j)))
        #     # false negative
        #     FN = np.sum(np.logical_and(np.equal(test_label, j), np.not_equal(y_pred, j)))
        #     precision = TP / (TP + FP)
        #     recall = TP / (TP + FN)
        #     F1Score.append(2 * precision * recall / (precision + recall))
        # F1Score = np.nan_to_num(F1Score).mean()
        F1Score2 = f1_score(test_label,y_pred,labels=[0,1],average='macro') #计算各类的TP再算平均值
        # F1Score3 = f1_score(test_label,y_pred, labels=[0,1],average='micro') #计算总体的TP
        # accuracy = np.sum(np.equal(y_pred, test_label))/len(test_data)
        accuracy2 = accuracy_score(y_pred, test_label)
        # print("test on subject%d, F1: %f, ACC: %f"% (i, F1Score, accuracy))
        F1s.append(F1Score2)
        ACCs.append(accuracy2)

    print("average score, F1: %f, ACC: %f" % (np.mean(F1s), np.mean(ACCs)))
    return np.mean(F1s), np.mean(ACCs)


if __name__ == '__main__':
    # read data
    EEG_fea = np.load('./data/EEG_fea.npy')
    Peri_fea = np.load('./data/Peri_fea.npy')
    label_v = np.load('./data/label_v.npy')
    label_a = np.load(('./data/label_a.npy'))

    # delete NAN
    Peri_fea = delete_NAN(Peri_fea)

    # normalize each subject
    EEG_data = nor_subject(EEG_fea)
    Peri_data = nor_subject(Peri_fea)

    #test
    inter_model_eval(EEG_fea, label_a)
    inter_model_eval(EEG_data, label_a)
    inter_model_eval(EEG_fea, label_v)
    inter_model_eval(EEG_data, label_v)
    inter_model_eval(Peri_fea, label_a)
    inter_model_eval(Peri_data, label_a)
    inter_model_eval(Peri_fea, label_v)
    inter_model_eval(Peri_data, label_v)



