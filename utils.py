import pandas as pd
import os
import numpy as np
import itertools
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

targets = {'abalone': 'rings', 'diabetes': 'c_peptide', 'housing': 'class',
           'machine': 'class', 'pyrim': 'activity', 'r_wpbc': 'Time', 'triazines': 'activity'}

min_max_scaler = preprocessing.MinMaxScaler()

def combinations(n2, n1=0):
    a = [[(i, j) for i in range(n1, j)] for j in range(n1, n2)]
    return np.array(list(itertools.chain.from_iterable(a)))


def read_data(data, n, d):
    """
    Create a dataFrame containing both data (using file .data) and columns' labels (using file .domain)
    :param data: string, choice between abalone, diabetes, housing, machine, pyrim, r_wpbc, triazines
    :return: pandas dataFrame
    """
    X = pd.read_csv(os.path.join('./Data/', data+'.data'), header=None, sep=',')
    col = list(pd.read_csv(os.path.join('./Data/', data+'.domain'), header=None, sep=':').iloc[:, 0].apply(lambda x: x.replace('\t', '').replace(' ','')))
    X.columns = col
    target = targets[data]
    idx = [col.index(i) for i in col if i != target] + [col.index(target)]
    X = pd.get_dummies(X).iloc[:, idx]
    X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns, index=X.index)
    n0 = X.shape[0] if n == -1 else n
    d0 = X.shape[1] if d == -1 else d
    print('\nDataset ' + data + ' of size ({}, {}) truncated to size ({}, {})'.format(X.shape[0], X.shape[1], n0, d0))
    return X


def distance(x, y):
    if x is None or y is None:
        return np.inf
    return np.sum((x-y)**2)


def n_pdf(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)


def gaussian_kernel(x, y, K):
    return np.exp(-K/2.*np.sum((x-y)**2))


def reshape_pref(pref):
    indices = np.unique(pref)
    mapping = {p: i for i, p in zip(range(len(indices)), indices)}
    new_pref = []
    for p in pref:
        new_pref.append((mapping[p[0]], mapping[p[1]]))
    return np.array(new_pref), indices

def ratio_n_obs(m_pref):
    return int(np.sqrt(2*m_pref))
