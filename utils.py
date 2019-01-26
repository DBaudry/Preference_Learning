import pandas as pd
import os
import numpy as np
import itertools

targets = {'abalone': 'rings', 'diabetes': 'c_peptide', 'housing': 'class',
           'machine': 'class', 'pyrim': 'activity', 'r_wpbc': 'Time', 'triazines': 'activity'}


def combinations(n):
    a = [[(i, j) for i in range(j)] for j in range(n)]
    return np.array(list(itertools.chain.from_iterable(a)))


def read_data(data):
    """
    Create a dataFrame containing both data (using file .data) and columns' labels (using file .domain)
    :param data: string, choice between abalone, diabetes, housing, machine, pyrim, r_wpbc, triazines
    :return: pandas dataFrame
    """
    print('Dataset: '+data+'...\n')
    X = pd.read_csv(os.path.join('./Data/', data+'.data'), header=None, sep=',')
    col = list(pd.read_csv(os.path.join('./Data/', data+'.domain'), header=None, sep=':').iloc[:, 0].apply(lambda x: x.replace('\t', '').replace(' ','')))
    X.columns = col
    target = targets[data]
    idx = [col.index(i) for i in col if i != target] + [col.index(target)]
    X = pd.get_dummies(X)
    X = X.iloc[:, idx]
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
    m = len(pref)
    indices = np.unique(pref)
    mapping = {p: i for i, p in zip(range(m), indices)}
    new_pref = []
    for p in pref:
        new_pref.append((mapping[p[0]], mapping[p[1]]))
    return np.array(new_pref), indices
