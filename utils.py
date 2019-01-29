import pandas as pd
import os
import numpy as np
import itertools
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

import warnings
warnings.filterwarnings("ignore")


targets = {'abalone': 'rings', 'diabetes': 'c_peptide', 'housing': 'class',
           'machine': 'class', 'pyrim': 'activity', 'r_wpbc': 'Time', 'triazines': 'activity'}

n_attributes = {'waveform': 40, 'dna': 180, 'mnist': 772, 'letter': 16, 'satimage': 36, 'usps': 256, 'segment': 19}

min_max_scaler = preprocessing.MinMaxScaler()


''' Short functions '''


def combinations(n2, n1=0):
    a = [[(i, j) for i in range(n1, j)] for j in range(n1, n2)]
    return np.array(list(itertools.chain.from_iterable(a)))


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


# Function to read data for instance learning


def read_data_IL(data, n, d):
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


# Function to read data for label learning
# Datasets are chosen among : 'sushia', 'sushib', 'movies', 'german2005', 'german2009', 'algae', 'dna', 'letter'
# 'mnist', 'satimage', 'segment', 'usps', 'waveform'


def read_data_LL(dataset, n):
    graphs = []
    if dataset == 'sushia':
        users = pd.read_csv(os.path.join('./Data', 'sushi3.udata'), header=None, sep='\t').iloc[1:(n+1), 1:]
        pref = pd.read_csv(os.path.join('./Data', 'sushi3a.5000.10.order'), header=None, sep='\t').iloc[1:(n+1), :]
        classes = np.ones(n).astype(int)
        for user in range(users.shape[0]):
            t = compute_all_edges(to_preference(pref, user))
            graphs.append(t)
            classes[user] = t[0][0]

    elif dataset == 'sushib':
        users = pd.read_csv(os.path.join('./Data', 'sushi3.udata'), header=None, sep='\t').iloc[1:(n+1), 1:]
        pref = pd.read_csv(os.path.join('./Data', 'sushi3b.5000.10.order'), header=None, sep='\t').iloc[1:(n+1), :]
        classes = np.ones(n).astype(int)
        for user in range(users.shape[0]):
            t = compute_all_edges(to_preference(pref, user))
            graphs.append(t)
            classes[user] = t[0][0]

    elif dataset == 'movies':
        X = pd.read_csv(os.path.join('./Data/', 'top7movies.txt'), sep=',').iloc[:n, :]
        users = pd.get_dummies(X.loc[:, ['gender', 'age', 'latitude', 'longitude', 'occupations']])
        classes = np.ones(n).astype(int)
        for user in range(users.shape[0]):
            g = get_pref(X.iloc[user, -1])
            graphs.append(g)
            classes[user] = g[0, 0]

    elif dataset in ['german2005', 'german2009']:
        X = pd.read_csv(os.path.join('./Data/', dataset+'.txt'), sep=',').iloc[:n, :]
        users = pd.get_dummies(X.loc[:, [col for col in X.columns if col not in ['State', 'Region', 'ranking']]])
        classes = np.ones(n).astype(int)
        for user in range(users.shape[0]):
            g = get_pref(X.iloc[user, -1])
            graphs.append(g)
            classes[user] = g[0, 0]

    elif dataset == 'algae':
        X = pd.read_csv(os.path.join('./Data/', 'algae.txt'), sep=',').iloc[:n, :]
        users = pd.get_dummies(X.iloc[:, :-2])
        classes = np.ones(n).astype(int)
        for user in range(users.shape[0]):
            g = get_pref(X.iloc[user, -1])
            graphs.append(g)
            classes[user] = g[0, 0]

    else:
        d = load_svmlight_file(os.path.join('./Data/', dataset+'.scale-0'), n_features=n_attributes[dataset])
        users = pd.DataFrame(d[0].todense()).iloc[:n, :]
        classes = np.array(d[1]).astype(int)-1
        labels = np.unique(classes)
        graphs = []
        for c in classes[:n]:
            graphs.append([(c, l) for l in labels if c != l])

    return users, graphs, classes



def train_test_split(users, graphs, classes):
    idx = np.random.choice(range(users.shape[0]), users.shape[0], replace=False)
    X = pd.DataFrame(min_max_scaler.fit_transform(users), columns=users.columns, index=users.index)
    train_idx, test_idx = idx[0:int(0.75*len(idx))], idx[(int(0.75*len(idx))):]
    users_train, users_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    graphs_train, graphs_test = [graphs[i] for i in train_idx], [graphs[i] for i in test_idx]
    classes_train, classes_test = classes[train_idx], classes[test_idx]
    train, test = [np.array(users_train), graphs_train, classes_train], [np.array(users_test), graphs_test, classes_test]
    return train, test


''' Functions to build preferences graphs'''


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


def letters_to_numbers(s):
    s = s.replace('a', '0').replace('b', '1').replace('c', '2')
    s = s.replace('d', '3').replace('e', '4').replace('f', '5')
    s = s.replace('g', '6')
    return s


def get_pref(s):
    s = letters_to_numbers(s)
    s = letters_to_numbers(s).split('>')
    b = [list(i) for i in s]
    b = [[int(float(j)) for j in i] for i in b]
    n = len(b)
    pref = []
    for j in range(0, n):
        for k in range(j + 1, n):
            pref.append([i for i in itertools.product(b[j], b[k])])
    return np.array(list(itertools.chain.from_iterable(pref)))


def get_positions(a, mode):
    if mode == 'compute_all_edges':
        return None
    else:
        positions = {}
        for x in a:
            positions[x] = (a.index(x), 0)
        return positions


def pipeline_graph(data, user, mode):
    G = nx.DiGraph()
    a = np.unique(data)
    G.add_nodes_from(a)
    G.add_edges_from(data)
    nx.draw(G, pos=get_positions(a, mode), with_labels=True, font_weight='bold', node_size=1e3)
    plt.title('Preferences of user ' + str(user), fontsize=20)


