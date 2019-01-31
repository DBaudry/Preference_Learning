import numpy as np
from sklearn.svm import SVC
from random import shuffle
from sklearn.model_selection import GridSearchCV


##################  Tools  ##################
def ProcessData(Data, Label):
    """
    :param Data: Set of covariates
    :param Label: Label for the classification
    :return: A data and a label set transformed according to the methodology presented by Herbrich et al., with a random
    inversion of the order of tuple preference (i.e. some u_k > v_k are seen as v_k > u_k and should be classified as
    -1 instead of 1, the others should be classified as 1).
    """
    Data_ = []
    Label_ = []
    if len(Label) % 2 != 0:
        index = np.arange(len(Label) - 1)
    else:
        index = np.arange(len(Label))
    shuffle(index)
    splits = np.split(index, 2)
    for index, pref in enumerate(Label):
        if index in splits[0]:
            Data_.append(
                Data[pref[0]] - Data[pref[1]]
            )
            Label_.append(
                1
            )
        else:
            Data_.append(
                Data[pref[1]] - Data[pref[0]]
            )
            Label_.append(
                -1
            )
    return Data_, Label_


def ExpansionData(Data, Label):
    """
    :param Data: Set of covariates
    :param Label: Label for the classification
    :return: A data and a label set fully transformed according to the methodology presented by Har-Peled et al.
    """
    Data_ = []
    Label_ = []
    for pref in Label:
            Data_.extend([
                np.append(Data[pref[0]], (-1)*Data[pref[1]]),
                np.append(Data[pref[1]], (-1) * Data[pref[0]])
            ])
            Label_.extend([
                1, -1
            ])
    return Data_, Label_


def randomTest(Data, Label):
    """
    :param Data: Set of covariates
    :param Label: Label for the classification
    :return: A data and a label set fully transformed according to the methodology presented by Har-Peled et al. Each
    sample results in only one sample (randomly seen as (u_k>v_k,1) or (v_k>u_k,-1)).
    """
    Data_ = []
    Label_ = []
    if len(Label) % 2 != 0:
        index = np.arange(len(Label) - 1)
    else:
        index = np.arange(len(Label))
    shuffle(index)
    splits = np.split(index, 2)
    for index, pref in enumerate(Label):
        if index in splits[0]:
            Data_.append(
                np.append(Data[pref[0]], (-1) * Data[pref[1]])
            )
            Label_.append(
                1
            )
        else:
            Data_.append(
                np.append(Data[pref[1]], (-1) * Data[pref[0]])
            )
            Label_.append(
                -1
            )
    return Data_, Label_



##################  Herbrich  ##################
class SVM_InstancePref:
    def __init__(self, inputs, K, C):
        if not isinstance(K, list) and not isinstance(K, np.ndarray):

            self.classifier = SVC(gamma=K, C=C)
        else:
            parameters = {'C': C, 'gamma': K}
            self.classifier = GridSearchCV(SVC(), parameters, cv=5)
        self.X = inputs[0]
        self.D = inputs[1]
        self.Data, self.Label = ProcessData(self.X, self.D)

    def fit(self):
        self.classifier.fit(self.Data, self.Label)
        try:
            print("(Herbrich) SVM CV-Hyperparameters: K={gamma} and C={C}".format(**self.classifier.best_params_))
        except AttributeError:
            pass

    def score(self, Data=None, Label=None, train=True):
        if train:
            predic = []
            for pref, lab in zip(self.D, self.Label):
                predic.append(self.classifier.predict([lab*(self.X[pref[0]]-self.X[pref[1]])])[0])
            return np.mean([predic[i] == self.Label[i] for i in range(len(predic))])
        else:
            Data_, Label_ = ProcessData(Data, Label)
            preds = self.classifier.score(Data_, Label_)
            return preds


##################  Har-Peled  ##################
class CCSVM_IL:
    def __init__(self, inputs, K, C):
        if not isinstance(K, list) and not isinstance(K, np.ndarray):
            self.classifier = SVC(gamma=K, C=C)
        else:
            parameters = {'C': C, 'gamma': K}
            self.classifier = GridSearchCV(SVC(), parameters, cv=5)
        self.X = inputs[0]
        self.D = inputs[1]
        self.Data, self.Label = ExpansionData(self.X, self.D)

    def fit(self):
        self.classifier.fit(self.Data, self.Label)
        try:
            print("(Har_Peled) SVM CV-Hyperparameters: K={gamma} and C={C}".format(**self.classifier.best_params_))
        except AttributeError:
            pass

    def score(self, Data=None, Label=None, train=True):
        if train:
            Data, Label = self.X, self.D
        Data_, Label_ = randomTest(Data, Label)
        preds = self.classifier.score(Data_, Label_)
        return preds
