import numpy as np
from sklearn.svm import SVC
from random import shuffle
from sklearn.model_selection import GridSearchCV
from itertools import chain

##################  Tools  ##################
def Vec(x, i , k):
    """Vector of dim d embedded in a vector of dim kd"""
    Vec = np.zeros((k+1)*len(x))
    Vec[i*len(x):(i+1)*len(x)] = x
    return Vec


def expansionTuple(x, i, j, k):
    """Translation of preference into a vector according to the methodology from Har-Peled et al."""
    return Vec(x, i, k) - Vec(x, j, k)


def sampleExt(x, prefs, k):
    """Generation of a dataset suited to the classification problem form the preferences in prefs, according to the
    methodology of Har-Peled et al."""
    new_sample = [(expansionTuple(x, pref[0], pref[1], k), 1) for pref in prefs]
    new_sample.extend([(expansionTuple(x, pref[1], pref[0], k), -1) for pref in prefs])
    return new_sample


def random_index(n):
    """Randomly select half of the integers between 0 and n-1."""
    if n % 2 != 0:
        index = np.arange(n - 1)
    else:
        index = np.arange(n)
    shuffle(index)
    splits = np.split(index, 2)
    return splits[0]


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
            k = self.k
        for index in range(len(Label)):
            Lrandom = random_index(len(Label[index]))
            Pref_error.append(
                np.mean(
                    [
                        self.classifier.predict(
                            [expansionTuple(Data[index], pref[i in Lrandom], pref[i not in Lrandom], k)]
                        )[0] != (-1)**(i in Lrandom) for i, pref in enumerate(Label[index])
                    ]
                )
            )
            Label_error.append(
                any(self.classifier.predict(
                            [expansionTuple(Data[index],
                                            [Classes[index], label][label not in Lrandom],
                                            [Classes[index], label][label in Lrandom],
                                            k)]
                        )[0] != (-1)**(label not in Lrandom) for label in range(k) if label != Classes[index])
            )
        return np.mean(Pref_error), np.mean(Label_error)
