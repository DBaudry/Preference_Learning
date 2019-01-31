import pandas as pd
import os
import pickle as pkl
import numpy as np
import itertools
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import warnings
warnings.filterwarnings("ignore")

targets = {'abalone': 'rings', 'housing': 'class', 'machine': 'class', 'pyrim': 'activity', 'triazines': 'activity'
           }

dataset_shapes = {'abalone': (4177, 9), 'housing': (506, 14), 'machine': (209, 7), 'pyrim': (74, 28),
                  'triazines': (186, 61)}

mapping_n_labels = {'sushia': 10, 'sushib': 100, 'movies': 7, 'german2005': 5, 'german2009': 5, 'algae': 7, 'dna': 3,
                    'letter': 26, 'mnist': 10, 'satimage': 6, 'segment': 7, 'usps': 10, 'waveform': 3}

authors_n_pref = {'pyrim': 100, 'triazines': 300, 'machine': 500, 'housing': 700, 'abalone': 1000
                  }

# some have not been determined (None, None)
best_parameters = {'pyrim': (0.005, 0.007), 'triazines': (0.007, 0.006), 'machine': (0.03, 0.0006),
                   'housing': (0.005, 0.001), 'abalone': (80, 0.025), 'sushia': (None, None), 'sushib': (None, None),
                   'movies': (None, None), 'german2005': (2, 0.2), 'german2009': (2, 0.2), 'algae': (None, None),
                   'dna': (0.1, 0.001), 'letter': (None, None), 'mnist': (None, None), 'satimage': (10, 0.1),
                   'segment': (10, 0.001), 'usps': (0.001, 0.005), 'waveform': (5, 0.001)}

n_attributes = {'waveform': 40, 'dna': 180, 'mnist': 772, 'letter': 16, 'satimage': 36, 'usps': 256, 'segment': 19
                }


def combinations(n2, n1=0):
    """ Instance Learning:
    Construct all possible pairs of preference in list [n1, n1+1, ..., n2] : [(n1, n1+1), (n1, n1+2), ..., (n2-1, n2)]
    :param n2: int, higher int
    :param n1: int, lower int
    :return: np.array, preferences tuples
    """
    a = [[(i, j) for i in range(n1, j)] for j in range(n1, n2)]
    return np.array(list(itertools.chain.from_iterable(a)))


def distance(x, y):
    """
    Compute L2 norm
    :param x: np.array
    :param y: np.array
    :return: float
    """
    if x is None or y is None:
        return np.inf
    return np.sum((x-y)**2)


def n_pdf(x):
    """
    Compute pdf of N(0,1)
    :param x: np.array, list of points
    :return: np.array, pdf evaluated at each point
    """
    return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)


def gaussian_kernel(x, y, K):
    """
    Compute Gaussian kernel
    :param x: np.array
    :param y: np.array
    :param K: float
    :return: np.array
    """
    return np.exp(-K/2.*np.sum((x-y)**2))


def reshape_pref(pref):
    """ Instance Learning:
    Reindex preferences to avoid out of range error
    :param pref: list of tuples
    :return: - new_pref: np.array, new re-indexed preferences
             - indices: np.array, unique item used in preferences ([(1,2), (3,2), (1,3)] -> [1, 2, 3])
    """
    indices = np.unique(pref)
    mapping = {p: i for i, p in zip(range(len(indices)), indices)}
    new_pref = []
    for p in pref:
        new_pref.append((mapping[p[0]], mapping[p[1]]))
    return np.array(new_pref), indices


def ratio_n_obs(m_pref, p=0.5):
    """ Instance Learning:
    Reduce number of observations for training. It relies on the authors' suggestion that n << m_pref
    :param m_pref: int, number of preferences
    :param p: float
    :return: int, number n of observations such that m_pref/p = n * (n-1)/2
    """
    return int(np.sqrt(2*m_pref/p))


def gridsearchBool(param):
    """
    Output if grid search is asked by user according to param
    :param param: list of values or pair of values
    :return: Bool
    """
    if param == 'best':
        gridsearch=False
    else:
        if isinstance(param[0], list):
            gridsearch=True
        else:
            gridsearch=False
    return gridsearch


def get_alpha(dim):
    """ Instance Learning:
    Compute coefficients for Cobb-Douglas utility function
    :param dim: int, number of coefficients
    :return: np.array
    """
    rho = np.random.uniform()
    coeff = np.array([rho**(i+1) for i in range(dim)])
    return np.random.permutation(coeff/coeff.sum())


def read_data_IL(data, n, d):
    """ Instance Learning:
    Create a dataFrame containing both data (using file .data) and columns' labels (using file .domain)
    :param data: string, choice between abalone, housing, machine, pyrim, triazines
    :param n: int, maximal number of observations to construct preferences pairs
    :param d: int, maximal number of variables
    :return: pandas dataframe
    """
    min_max_scaler = preprocessing.MinMaxScaler()
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


def read_data_LL(dataset, n):
    """ Label Learning:
    Load data from Data folder. Dataset can be taken in 'sushia', 'sushib', 'movies', 'german2005', 'german2009',
    'algae', 'dna', 'letter', 'mnist', 'satimage', 'segment', 'usps', 'waveform'.
    :param dataset: string, name of the dataset (see just above for possibilities)
    :param n: int, maximal number of observations
    :return: - users: np.array, observations used for constructing preferences
             - graphs: list of list of tuples, preferences graphs for each user (a graph is a list of 2d-tuples/edges)
             - classes: np.array, true classes for multi-classifcation or best preferred item (root of the graph)
    """
    n = dataset_shapes[dataset][0] if n == -1 else n
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
            classes[user] = g[0][0]

    elif dataset in ['german2005', 'german2009']:
        X = pd.read_csv(os.path.join('./Data/', dataset+'.txt'), sep=',').iloc[:n, :]
        users = pd.get_dummies(X.loc[:, [col for col in X.columns if col not in ['State', 'Region', 'ranking']]])
        classes = np.ones(n).astype(int)
        for user in range(users.shape[0]):
            g = get_pref(X.iloc[user, -1])
            graphs.append(g)
            classes[user] = g[0][0]

    elif dataset == 'algae':
        X = pd.read_csv(os.path.join('./Data/', 'algae.txt'), sep=',').iloc[:n, :]
        users = pd.get_dummies(X.iloc[:, :-2])
        classes = np.ones(n).astype(int)
        for user in range(users.shape[0]):
            g = get_pref(X.iloc[user, -1])
            graphs.append(g)
            classes[user] = g[0][0]

    elif dataset == 'dna':
        users, classes = pkl.load(open(os.path.join('./Data/', dataset + '.pkl'), 'rb'))
        classes -= 1
        labels = np.unique(classes)
        for c in classes:
            graphs.append([(c, l) for l in labels if c != l])

    else:
        d = load_svmlight_file(os.path.join('./Data/', dataset + '.scale-0'), n_features=n_attributes[dataset])
        users = pd.DataFrame(d[0].todense()).iloc[:n, :]
        classes = np.array(d[1]).astype(int) - 1
        if dataset == 'waveform':
            classes += 1
        labels = np.unique(classes)
        for c in classes[:n]:
            graphs.append([(c, l) for l in labels if c != l])
    return users, graphs, classes


def train_test_split(users, graphs, classes):
    """ Label Learning:
    Split users, graphs and classes inputs in 2 subsets : train and test sets. Here we apply 40% for training
    and 60% for testing as suggested in the article (4.2. classification)
    :param users: np.array, observations used for constructing preferences
    :param graphs: list of list of tuples, preferences graphs for each user (a graph is a list of 2d-tuples/edges)
    :param classes: np.array, true classes for multi-classifcation or best preferred item (root of the graph)
    :return: - train: users, graphs and classes used for training
             - test: users, graphs and classes used for testing
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    idx = np.random.choice(range(users.shape[0]), users.shape[0], replace=False)
    X = pd.DataFrame(min_max_scaler.fit_transform(users), columns=users.columns, index=users.index)
    test_idx, train_idx = idx[0:int(0.6*len(idx))], idx[(int(0.6*len(idx))):]
    users_train, users_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    graphs_train, graphs_test = [graphs[i] for i in train_idx], [graphs[i] for i in test_idx]
    classes_train, classes_test = classes[train_idx], classes[test_idx]
    train, test = [np.array(users_train), graphs_train, classes_train], [np.array(users_test), graphs_test, classes_test]
    return train, test


def to_preference(data, user):
    """ Label Learning:
    From sushi datasets, transform input rows into
    :param data: pandas dataframe, it contains preferences for each user (string of type '5 10 4 3' which means
    that 5 is preferred to 10, preferred to 4, etc...)
    :param user: int, user/index in the dataframe data
    :return: np.array, preferences for user (np.array([5, 10, 4, 3]))
    """
    x = data.iloc[user, 0]
    return np.array(str.split(x)[2:]).astype('int').tolist()


def compute_all_edges(a):
    """ Label learning:
    From an array ([1, 2, 3]) construct the full graph ([(1, 2), (1, 3), (2, 3)])
    :param a: list, preferences of a user
    :return: list, all edges of the preferences graph of a user
    """
    edges = []
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            edges.append((a[j], a[i]))
    return edges


def letters_to_numbers(s):
    """ Label learning:
    Replace letter in numbers to format preferences of data sets german2005, german2009, algae, top7movies
    :param s: string
    :return: string
    """
    s = s.replace('a', '0').replace('b', '1').replace('c', '2')
    s = s.replace('d', '3').replace('e', '4').replace('f', '5')
    s = s.replace('g', '6')
    return s


def get_pref(s):
    """ Label learning:
    From a string ('a>bc>def') construct the full graph ([(a, b), (a, c), (a, d), (a, e), (a, f), (b, c), (b, d)...])
    This function is used to format preferences from data sets : german2005, german2009, algae, top7movies
    :param s: string
    :return: list of tuples, complete graph of preferences
    """
    s = letters_to_numbers(s)
    s = letters_to_numbers(s).split('>')
    b = [list(i) for i in s]
    b = [[int(float(j)) for j in i] for i in b]
    n = len(b)
    pref = []
    for j in range(0, n):
        for k in range(j + 1, n):
            pref.append([i for i in itertools.product(b[j], b[k])])
    return list(itertools.chain.from_iterable(pref))


def pipeline_graph(data, title):
    """ Label learning:
    :param data: list of tuples, list of edges
    :param title: string
    :return: None, print graph
    """
    print(data)
    G = nx.DiGraph()
    a = np.unique(data)
    G.add_nodes_from(a)
    G.add_edges_from(data)
    nx.draw(G, pos=None, with_labels=True, font_weight='bold', node_size=1e3)
    plt.title(title, fontsize=10)


