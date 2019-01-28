import numpy as np
from sklearn.svm import SVC
from random import shuffle
from sklearn.model_selection import GridSearchCV
from itertools import chain


##################  Tools  ##################
def Vec(x, i , k):
    "Vector of dim d embedded in a vector of dim kd"
    Vec = np.zeros(k*len(x))
    Vec[i*len(x):(i+1)*len(x)] = x
    return Vec


def expansionTuple(x, i, j, k):
    return Vec(x, i, k) - Vec(x, j, k)


def sampleExt(x, prefs, k):
    new_sample = [(expansionTuple(x, pref[0], pref[1], k), 1) for pref in prefs]
    new_sample.extend([(expansionTuple(-x, pref[1], pref[0], k), -1) for pref in prefs])
    return new_sample


##################  CC-SVM  ##################
class CCSVM:
    def __init__(self, inputs, K, C):
        if not isinstance(K, list):
            self.classifier = SVC(gamma=K, C=C)
        else:
            parameters = {'C': C, 'gamma': K}
            self.classifier = GridSearchCV(SVC(), parameters, cv=5)
        self.X = inputs[0]
        self.D = inputs[1]
        k = np.max([np.max(list(chain(*prefs))) for prefs in self.D])
        Data = []
        for index in range(len(self.D)):
            Data.extend(sampleExt(self.X[index], self.D[index], k))
        Data = list(zip(*Data))
        self.Data, self.Label = Data[0], Data[1]

    def fit(self):
        self.classifier.fit(self.Data, self.Label)
        try:
            print(self.classifier.best_params_)
        except AttributeError:
            pass

    def score(self, Data=None, Label=None, train=True):
        if train:
            predic = []
            for pref in self.Data[::2]:
                predic.append(self.classifier.predict([pref]))
            return np.mean(predic)
        else:
            Data_, _ = ExpansionData(Data, Label)
            preds = self.classifier.predict(Data_[::2])
            return np.mean(preds)

    def predict(self, x):
        return self.classifier.predict(x)

    def predictions(self, Data, Label):
        Data_, _ = ExpansionData(Data, Label)
        a = self.predict(Data_)
        return np.count_nonzero(np.array(a[::2]) + np.array(a[1::2])), \
               [i for i, e in enumerate(np.array(a[::2]) + np.array(a[1::2])) if e != 0]



