import pandas as pd
import os
import numpy as np


def read_data(data):
    """
    Create a dataFrame containing both data (using file .data) and columns' labels (using file .domain)
    :param data: string, choice between abalone, diabetes, housing, machine, pyrim, r_wpbc, triazines
    :return: pandas dataFrame
    """
    X = pd.read_csv(os.path.join('./Data/', data+'.data'), header=None, sep=',')
    col = pd.read_csv(os.path.join('./Data/', data+'.domain'), header=None, sep=':')
    X.columns = list(col.iloc[:, 0])
    for i, m in enumerate(X.dtypes):
        if m in ['object', 'str']:
            pd.to_numeric(X.iloc[i, :])
    return X


def distance(x, y):
    if x is None or y is None:
        return np.inf
    return np.sum((x-y)**2)


def n_pdf(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)


def gaussian_kernel(x, y, K):
    return np.exp(-K/2.*np.sum((x-y)**2))
