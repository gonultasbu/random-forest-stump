import numpy as np 
import pandas as pd 
import os 
import sys 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn 
from matplotlib import pyplot as plt 
from scipy import stats 
from tqdm import tqdm

def mode(x):
    if (len(x)==0):
        return -1
    return stats.mode(x.flatten())[0][0]

class UnweightedStump():
    def __init__(self):
        pass

    def fit(self, X, y, max_features=3):
        if np.unique(y).size <= 1:
            return
        self.max_features = max_features
        N, D = X.shape
        class_vals, count = np.unique(y, return_counts=True)
        self.stump_positive_label = class_vals[np.argmax(count)]
        self.stump_negative_label = None
        self.stump_feature = None
        self.stump_threshold = None
        X = np.round(X)
        
        max_ig = float("-inf")
        for d in np.sort(np.random.choice(D, size=self.max_features, replace=False)):
            for value in np.unique(X):
                positive_label_candidate = mode(y[X[:,d] > value])
                negative_label_candidate = mode(y[X[:,d] <= value])
                y_pred = positive_label_candidate * np.ones(N)
                y_pred[X[:, d] <= value] = negative_label_candidate
                try:
                    positive_rate = np.unique(y[X[:,d] > value], return_counts=True)[1][1] / N
                except IndexError:
                    continue

                negative_rate = 1 - positive_rate
                y2 = count / y.size
                sub_y_yes = y[X[:,d] > value]
                y_yes = np.unique(sub_y_yes, return_counts=True)[1] / y.size
                y_no = 1 - y_yes

                ig = (stats.entropy(y2)) - ((positive_rate * stats.entropy(y_yes))) - ((negative_rate * stats.entropy(y_no)))
                if ig > max_ig:
                    max_ig = ig
                    self.stump_feature = d
                    self.stump_threshold = value
                    self.stump_positive_label = positive_label_candidate
                    self.stump_negative_label = negative_label_candidate
        return 

    def predict(self, X):
        M, _ = X.shape
        X = np.round(X)
        if self.stump_feature is None:
            exit("Error. None stump_feature encountered!")

        y_pred = np.zeros(M)
        y_pred[X[:, self.stump_feature] <= self.stump_threshold] = self.stump_negative_label
        y_pred[X[:, self.stump_feature] > self.stump_threshold] = self.stump_positive_label
        return y_pred

class RFClassifier(object):
    def __init__(self, n_estimators=100, max_features=3, max_samples=0.9):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.estimators = []

    def fit(self, X, y):
        # Generate the emtpy weak learner list and slice the bootstrap count.
        self.estimators = []
        n_bootstrap_samples = round(len(y)*self.max_samples)
        
        for _ in np.arange(self.n_estimators):
            # Randomly sample bootstrap samples.
            X,y = sklearn.utils.shuffle(X,y)
            X_bootstrapped = X[:n_bootstrap_samples]
            y_bootstrapped = y[:n_bootstrap_samples]
            # Fit a weak learner and append to the weak learner list.
            stump = UnweightedStump()
            stump.fit(X_bootstrapped, y_bootstrapped, max_features=3)
            self.estimators.append(stump)

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros([len(self.estimators), n_samples])
        for i in range(len(self.estimators)):
            predictions[i] = self.estimators[i].predict(X)
        
        y_pred = np.sign(np.mean(predictions, axis=0))

        return y_pred

    def score(self, X, y):
        preds = self.predict(X)
        labels = np.squeeze(y)
        assert preds.shape == labels.shape
        return np.mean(preds == labels)


def rf(dataset:str) -> None:
    # Read the data and replace question mark values with nans and then impute nans with ones.
    df = pd.read_csv(dataset, sep=',' , header=None).replace('?', np.nan).fillna(value=1).astype('int32')
    y = df[10].to_numpy()
    # Encode labels. 
    y[y==2] = 1 
    y[y==4] = -1 
    X = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].to_numpy().astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


    tr_errors = []
    test_errors = []

    # Sklearn sanity check.
    """
    for n_features in tqdm(np.arange(2, 10)):
        model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=1, max_features=n_features)
        model.fit(X_train, y_train)
        tr_errors.append(model.score(X_train, y_train))
        tr_errors.append(model.score(X_test, y_test))
    """
    for n_estimators in tqdm(np.arange(1, 101)):
        model = RFClassifier(n_estimators=n_estimators, max_features=3)
        model.fit(X_train, y_train)
        tr_error = 1.0-model.score(X_train, y_train)
        test_error = 1.0-model.score(X_test, y_test)
        tr_errors.append(tr_error)
        test_errors.append(test_error)
        print("Training error with", n_estimators, "estimators:",tr_error)
        print("Test error with", n_estimators, "estimators:",test_error)

    plt.plot(np.arange(1, 101), tr_errors, 'b-',
                np.arange(1, 101), test_errors, 'r-')
    plt.title("Random Forest training and test errors vs tree count")
    plt.xlabel("number of trees")
    plt.ylabel("error")
    plt.legend(['training error', 'test error'])
    plt.show()
    plt.clf()

    del tr_errors, test_errors

    tr_errors = []
    test_errors = []

    # Sklearn sanity check.
    """
    for n_features in tqdm(np.arange(2, 10)):
        model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=1, max_features=n_features)
        model.fit(X_train, y_train)
        tr_errors.append(model.score(X_train, y_train))
        tr_errors.append(model.score(X_test, y_test))
    """
    for n_features in tqdm(np.arange(2, 11)):
        model = RFClassifier(n_estimators=100, max_features=n_features)
        model.fit(X_train, y_train)
        tr_error = 1.0-model.score(X_train, y_train)
        test_error = 1.0-model.score(X_test, y_test)
        tr_errors.append(tr_error)
        test_errors.append(test_error)
        print("Training error with", n_features, "features:",tr_error)
        print("Test error with", n_features, "features:",test_error)

    plt.plot(np.arange(2, 11), tr_errors, 'b-',
                np.arange(2, 11), test_errors, 'r-')
    plt.title("Random Forest training and test errors vs feature count")
    plt.xlabel("number of features")
    plt.ylabel("error")
    plt.legend(['training error', 'test error'])
    plt.show()
    plt.clf()

    return 


if __name__ == "__main__":
    rf("data/breast-cancer-wisconsin.data")