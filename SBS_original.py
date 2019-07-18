from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


class SequentialBackwardSelection():
    """
    Feature selection using Sequential Backward Selection
    Parameter
    -----------
    estimator: a classifier or regressor that will be used to do
                feature selection.
    num_of_features: number of features for the feature selection
                    result
    scoring: the scoring method that will be used for selecting
                the best set of features
    test_size: test size ratio
    random_state: random seed value used for train_test_split
    """
    def __init__(self, estimator=KNeighborsClassifier(n_neighbors=3),
                 num_of_features=5, scoring=f1_score, test_size=0.25,
                 random_state=1):

        self.scoring = scoring
        self.estimator = clone(estimator)
        self.num_of_features = num_of_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fitting the estimator to the data, and finding the best set of
        features from the features of the data. Amount of features in the
        set will be equal to the num_of_features defined
        Parameter
        -----------
        X: Training Vectors, shape=[number_of_samples, number_of_features]
        Y: Target Values, shape=[number_of_samples]
        Attribute
        -----------
        best: integer, index of the feature set that scores the highest
                with the scoring method selected
        indices: best feature set from the combinations of feature set tested
        best_score = score from the test run on the estimator using the
                        best feature set
        scores: list, containing the score from the test run on the estimator
                using the feature sets tested in the method
        """

        self.scores = []
        self.subsets = []

        for feature_combination in combinations(range(X.shape[1]),
                                                r=self.num_of_features - 1):
            score = self.calc_score(X, y, feature_combination)
            self.scores.append(score)
            self.subsets.append(feature_combination)

        best = np.argmax(self.scores)
        self.indices = self.subsets[best]
        self.best_score = self.scores[best]
        return self

    def transform(self, X):
        """
        Transform a data vector into a new data vector containing only the
        features from the best set of features obtained from the fit function
        Parameter
        -----------
        X: Data Vectors, shape=[number_of_samples, number_of_features]
        Data with the same features as the ones used for fitting
        Attribute
        -----------
        indices: best feature set from the combinations of feature set tested
        """
        return X[:, self.indices]

    def calc_score(self, X, y, indices):
        """
        Scoring an estimator using data with only a specific set of features
        Parameter
        -----------
        X: Training Vectors, shape=[number_of_samples, number_of_features]
        Y: Target Values, shape=[number_of_samples]
        indices: column index, features that will be used from X
        """
        X_train, X_test, \
            y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                               random_state=self.random_state)
        self.estimator.fit(X_train[:, indices], y_train)
        y_prediction = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_prediction)
        return score


if __name__ == "__main__":
    print("this is the SBS library file, used to do feature selection,")
print("do not use this as the main file")
