import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import time
import scipy.io as scio
import numpy

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings(action="ignore")


def cross_validate(X, y, feaselect=False):
    # X = numpy.genfromtxt(file_x, delimiter=' ')
    X = StandardScaler().fit_transform(X)

    if feaselect == True:
        X = fisher_idx(X, y)
        # print(X.shape, y.shape)
    # #remove nan
    # d = []
    # l = []
    # for i, v in enumerate(X):
    #     if True in numpy.isnan(v):
    #         pass
    #         # print(i,v)
    #     else:
    #         l.append(y[i])
    #         d.append(v)
    # X = numpy.array(d)
    # y = numpy.array(l)

    model = ('LR', LogisticRegression(solver='liblinear'))
    # models.append(('SVC', SVC()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('DT', DecisionTreeClassifier()))
    # models.append(('RF', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456, criterion='entropy')))
    scoring = 'accuracy'
    # scoring = 'f1'

    # Split the data into training/testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

    numpy.random.seed(9)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(y)))
    x_shuffled = X[shuffle_indices]  # 将样本和标签打乱
    y_shuffled = y[shuffle_indices]
    # print(x_shuffled.shape, y_shuffled.shape)
    # Cross Validate
    results = []
    names = []
    timer = []
    # print('Model | Mean of CV | Std. Dev. of CV | Time')
    # for name, model in models:
    #     start_time = time.time()
    #     kfold = model_selection.KFold(n_splits=33, random_state=numpy.random.randint(1,100))
    #     cv_results = model_selection.cross_val_score(model, x_shuffled, y_shuffled, cv=kfold, scoring=scoring)
    #     t = (time.time() - start_time)
    #     timer.append(t)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f) %f s" % (name, cv_results.mean(), cv_results.std(), t)
    #     print(msg)

    # 留一法验证
    loo = LeaveOneOut()
    start_time = time.time()
    # [:,(25, 90, 91)]
    if(x_shuffled.shape[0] == 0):
        print(x_shuffled.shape[0])
        return -1
    cv_results = model_selection.cross_val_score(model[1], x_shuffled, y_shuffled, cv=loo, scoring=scoring)
    names.append(model[0])
    t = (time.time() - start_time)
    msg = "%s: %f (%f) %f s" % (model[0], cv_results.mean(), cv_results.std(), t)
    print(msg)
    return cv_results.mean()


def run():
    dataFile = './data/matlab.mat'
    data = scio.loadmat(dataFile)

    features = data['features']

    # print(features.shape)
    EEG_fea = features[0][0][2].reshape(-1)
    for i in range(32):
        for j in range(40):
            if i is 0 and j is 0:
                continue
            # print(EEG_fea.shape)
            # print(features[i][j][2].reshape(-1).shape)
            EEG_fea = numpy.vstack((EEG_fea, features[i][j][2].reshape(-1)))
    print("EEG_fea.shape:" + str(EEG_fea.shape))

    EMG_fea = features[0][0][0].reshape(-1)
    for i in range(32):
        for j in range(40):
            if i is 0 and j is 0:
                continue
            # print(EEG_fea.shape)
            # print(features[i][j][0].reshape(-1).shape)
            EMG_fea = numpy.vstack((EMG_fea, features[i][j][0].reshape(-1)))
    print("EMG_fea.shape:" + str(EMG_fea.shape))

    GSR_fea = features[0][0][4].reshape(-1)
    for i in range(32):
        for j in range(40):
            if i is 0 and j is 0:
                continue
            # print(EEG_fea.shape)
            # print(features[i][j][0].reshape(-1).shape)
            GSR_fea = numpy.vstack((GSR_fea, features[i][j][4].reshape(-1)))
    print("GSR_fea.shape:" + str(GSR_fea.shape))

    BVP_fea = features[0][0][6].reshape(-1)
    for i in range(32):
        for j in range(40):
            if i is 0 and j is 0:
                continue
            # print(EEG_fea.shape)
            # print(features[i][j][0].reshape(-1).shape)
            BVP_fea = numpy.vstack((BVP_fea, features[i][j][6].reshape(-1)))
    print("BVP_fea.shape:" + str(BVP_fea.shape))

    RES_fea = features[0][0][8].reshape(-1)
    for i in range(32):
        for j in range(40):
            if i is 0 and j is 0:
                continue
            RES_fea = numpy.vstack((RES_fea, features[i][j][8].reshape(-1)))
    print("RES_fea.shape:" + str(RES_fea.shape))

    peripheral_fea = numpy.hstack((EMG_fea, RES_fea, BVP_fea[:, :-4], GSR_fea))
    print("peripheral_fea.shape:" + str(peripheral_fea.shape))
    all_fea = numpy.hstack((EEG_fea, peripheral_fea))
    print("all_fea.shape:" + str(all_fea.shape))

    np.save("./data/filename.npy", all_fea)
    # b = np.load("filename.npy")

    # print("BVP_fea")
    # run_fea(BVP_fea[:,:-4])
    # print("GSR_fea")
    # run_fea(GSR_fea)
    # print("RES_fea")
    # run_fea(RES_fea)
    # print("EMG_fea")
    # run_fea(EMG_fea)
    # print("EEG_fea")
    # run_fea(EEG_fea, feaselect=True)
    # print("peripheral_fea")
    # run_fea(peripheral_fea, feaselect=True)
    # print("all_fea")
    # run_fea(all_fea, feaselect=True)


def fisher_idx(features, labels):
    ''' Get idx sorted by fisher linear discriminant '''
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
        return features[:, sorted_feature_idx[:3]]
    return features[:, feature_idx].reshape((features.shape[0], -1))


def run_fea(file_x, feaselect=False):
    print("2class:")
    file_y1 = './data/label_class_0.dat'
    file_y2 = './data/label_class_1.dat'
    file_y1 = numpy.genfromtxt(file_y1, delimiter=' ')
    file_y2 = numpy.genfromtxt(file_y2, delimiter=' ')
    result_a = []
    result_v = []
    print("Valence")
    for i in range(32):
        X = file_x[40*i:40*(i+1), :]
        y = file_y1[40*i:40*(i+1)]
        print(' subject%d'%(i+1), end=' ')
        result = cross_validate(X, y, feaselect=feaselect)
        if result is not -1:
            result_a.append(result)
    print("Average acc:%f（%f）"%(np.mean(result_a),np.std(result_a)))
    print("Arousal")
    for i in range(32):
        X = file_x[40 * i:40 * (i + 1), :]
        y = file_y2[40 * i:40 * (i + 1)]
        print(' subject%d' % (i + 1), end=' ')
        result =  cross_validate(X, y, feaselect=feaselect)
        if result is not -1:
            result_v.append(result)
    print("Average acc:%f（%f）" % (np.mean(result_v), np.std(result_v)))



if __name__ == '__main__':
    run()


