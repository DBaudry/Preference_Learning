import numpy as np
from sklearn.svm import SVC
from random import shuffle
from sklearn.model_selection import GridSearchCV
from itertools import chain


##################  Tools  ##################
def Vec(x, i , k):
    "Vector of dim d embedded in a vector of dim kd"
    Vec = np.zeros((k+1)*len(x))
    Vec[i*len(x):(i+1)*len(x)] = x
    return Vec


def expansionTuple(x, i, j, k):
    return Vec(x, i, k) - Vec(x, j, k)


def sampleExt(x, prefs, k):
    new_sample = [(expansionTuple(x, pref[0], pref[1], k), 1) for pref in prefs]
    new_sample.extend([(expansionTuple(x, pref[1], pref[0], k), -1) for pref in prefs])
    return new_sample


##################  CC-SVM  ##################
class CCSVM_LL:
    def __init__(self, inputs, K, C):
        if not isinstance(K, list) and not isinstance(K, np.ndarray):
            self.classifier = SVC(gamma=K, C=C)
        else:
            parameters = {'C': C, 'gamma': K}
            self.classifier = GridSearchCV(SVC(), parameters, cv=5)
        self.X = inputs[0]
        self.D = inputs[1]
        self.classes = inputs[2]
        self.k = np.max(self.classes)
        Data = []
        for index in range(len(self.D)):
            Data.extend(sampleExt(self.X[index], self.D[index], self.k))
        Data = list(zip(*Data))
        self.Data, self.Label = Data[0], Data[1]

    def fit(self):
        self.classifier.fit(self.Data, self.Label)
        try:
            print(self.classifier.best_params_)
        except AttributeError:
            pass

    def score(self, Data=None, Label=None, Classes=None, train=True):
        Label_error = []
        Pref_error = []
        if train:
            Data = self.X
            Label = self.D
            Classes = self.classes
            k = self.k
        else:
            k = np.max(Classes)
        for index in range(len(Label)):
            Pref_error.append(
                np.mean(
                    [
                        self.classifier.predict(
                            [expansionTuple(Data[index], pref[0], pref[1], k)]
                        )[0] != -1 for pref in Label[index]
                    ]
                )
            )
            Label_error.append(
                any(self.classifier.predict(
                            [expansionTuple(Data[index], Classes[index], label, k)]
                        )[0] != -1 for label in range(k) if label != Classes[index])
            )
        return np.mean(Pref_error), np.mean(Label_error)
