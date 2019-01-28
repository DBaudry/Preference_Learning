import pandas as pd
import os
import numpy as np
import itertools
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt

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


def to_preference(data, user):
    x = data.iloc[user, 0]
    return np.array(str.split(x)[2:]).astype('int').tolist()


def compute_linear_edges(a):
    nodes = []
    for i in range(len(a)-1):
        nodes.append((a[i], a[i+1]))
    return nodes


def compute_all_edges(a):
    nodes = []
    for i in range(len(a)):
        for j in range(i+1,len(a)):
           nodes.append((a[i], a[j]))
    return nodes


def read_sushi(type):
    users = pd.read_csv(os.path.join('./Data', 'sushi3.udata'), header=None, sep='\t')
    pref = pd.read_csv(os.path.join('./Data', 'sushi3'+type+'.5000.10.order'), header=None, sep='\t')
    graphs = []
    for user in range(users.shape[0]):
        graphs.append(compute_all_edges(to_preference(pref, user)))
    return users, pref, graphs


def train_test_split_sushi(users, graphs):
    idx = np.random.choice(range(users.shape[0]), users.shape[0], replace=False)
    train_idx, test_idx = idx[0:int(0.75*len(idx))], idx[(int(0.75*len(idx))+1):]
    users_train = users.iloc[train_idx, :]
    users_test = users.iloc[test_idx, :]
    graphs_train, graphs_test = [graphs[i] for i in train_idx], [graphs[i] for i in test_idx]
    train, test = [np.array(users_train), graphs_train], [np.array(users_test), graphs_test]
    return train, test


def get_positions(a, mode):
    if mode == 'compute_all_edges':
        return None
    else:
        positions = {}
        for x in a:
            positions[x] = (a.index(x), 0)
        return positions


def draw_graph(G, user, mode, a):
    w = 10 if mode == 'compute_all_edges' else 3
    plt.figure('User '+str(user), figsize=(15, w))
    nx.draw(G, pos=get_positions(a, mode), with_labels=True, font_weight='bold', node_size=1e3)
    plt.title('Preferences of user ' + str(user), fontsize=20)
    plt.show()


def pipeline_graph(data, user, mode):
    G = nx.DiGraph()
    a = to_preference(data, user)
    G.add_nodes_from(a)
    eval('G.add_edges_from(' + mode + '(a))')
    draw_graph(G, user, mode, a)


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


def gridsearchBool(param):
    if param == 'best':
        gridsearch=False
    else:
        if isinstance(param[0], list):
            gridsearch=True
        else:
            gridsearch=False
    return gridsearch


def get_alpha(dim):
    rho = np.random.uniform()
    coeff = np.array([rho**(i+1) for i in range(dim)])
    return np.random.permutation(coeff/coeff.sum())

