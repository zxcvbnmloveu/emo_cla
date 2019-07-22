import warnings

import numpy
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

def cross_validate0(file_x, file_y):

    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    X = StandardScaler().fit_transform(X)
    d = []
    l = []
    for i,v in enumerate(X):
        if True in numpy.isnan(v):
            print(i,v)
        else:
            l.append(y[i])
            d.append(v)
    X = numpy.array(d)
    y = numpy.array(l)
    print(X.shape)
    print(y.shape)

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear')))
    models.append(('SVC', SVC()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DT', DecisionTreeClassifier()))
    # models.append(('RF', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456, criterion='entropy')))
    scoring = 'accuracy'

    # Split the data into training/testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

    numpy.random.seed(9)
    shuffle_indices = numpy.random.permutation(numpy.arange(len(y)))
    x_shuffled = X[shuffle_indices]  # 将样本和标签打乱
    y_shuffled = y[shuffle_indices]

    # Cross Validate
    results = []
    names = []
    timer = []
    print('Model | Mean of CV | Std. Dev. of CV | Time')
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

    #留一法验证
    loo=LeaveOneOut()
    for name, model in models:
        start_time = time.time()
        # [:,(25, 90, 91)]
        cv_results = model_selection.cross_val_score(model, x_shuffled, y_shuffled, cv=loo, scoring=scoring)
        names.append(name)
        t = (time.time() - start_time)
        msg = "%s: %f (%f) %f s" % (name, cv_results.mean(), cv_results.std(), t)
        print(msg)

def run():
    print("2class:")
    file_x = './data/features_gsr.dat'
    file_y1 = './data/label_class_0.dat'
    file_y2 = './data/label_class_1.dat'

    print("valence")
    cross_validate0(file_x, file_y1)
    print("Arousal")
    cross_validate0(file_x, file_y2)


if __name__ == '__main__':
    run()


