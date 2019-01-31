import numpy as np
import copy
import utils


class pref_generator():
    """ Instance Learning:
    Generate training and testing sets for real data sets.
    """
    def __init__(self, dataset, n=-1, d=-1):
        """
        Initialize the data frame and all possible preference pairs (pairs_index)
        :param dataset: string, name of the data set
        :param n: int, number of instances to consider
        :param d: int, number of attributes to consider
        """
        self.X = utils.read_data_IL(dataset, n, d)
        self.nmax, self.dmax = self.X.shape
        self.n = self.nmax if n == -1 else n
        self.d = self.dmax if d == -1 else d
        self.pairs_index = utils.combinations(self.n)

    def train_generator(self, m):
        """
        Initialize training pairs (list of indices)
        :param m: int, number of preferences
        :return: list of tuples, training pairs
        """
        replace = True if m > len(self.pairs_index) else False
        idx = np.random.choice(len(self.pairs_index), m, replace=replace)
        pairs = self.pairs_index[idx]
        self.training_pairs = tuple(map(tuple, pairs))
        return self.training_pairs

    def test_generator(self, m):
        """
        Initialize testing pairs (list of indices)
        :param m: int, number of preferences
        :return: list of tuples, testing pairs which are different from training pairs
        """
        n1 = 0 if self.n == self.nmax else self.n+1
        self.pairs_index = np.array([(i, j) for (i, j) in utils.combinations(self.nmax, n1)])
        replace = True if m > len(self.pairs_index) else False
        idx = np.random.choice(len(self.pairs_index), m, replace=replace)
        pairs = self.pairs_index[idx]
        self.testing_pairs = tuple(map(tuple, pairs))
        return self.testing_pairs

    def draw_preference(self, pairs):
        """
        Compute preference relations for each tuple of indices (a, b) in comparing (y_a, y_b) with y the target variable
        :param pairs: list of tuples, list of index
        :return: list of tuples, preferences
        """
        pref = []
        for p in pairs:
            a, b = p
            if self.X.iloc[a, -1] > self.X.iloc[b, -1]:
                pref.append((a, b))
            else:
                pref.append((b, a))
        return pref

    def get_true_pref(self, data):
        """
        Compute preference relations for each tuple of indices (a, b) in comparing (y_a, y_b) with y the target variable
        It takes data as an input and no more pairs. This function is redundant with draw_preference but avoids
        errors with instance_pref_generator class when running experiments.
        :param data: list of tuples, list of index
        :return: list of tuples, preferences
        """
        p = data.shape[0]
        real_pref = np.zeros((p, p))
        X = np.array(self.X)
        for i in range(p):
            for j in range(p):
                if X[self.training_indices[i], -1] > X[self.training_indices[j], -1]:
                    real_pref[i, j] = 1
        return real_pref

    def get_input_train(self, m):
        """
        Construct training data to use for modelling
        :param m: int, number of preferences
        :return: list, - data: np.array, n instances with their attributes
                       - pref: list of tuples, training preferences
        """
        pairs = self.train_generator(m)
        pref, self.training_indices = utils.reshape_pref(self.draw_preference(pairs))
        data = self.X.iloc[self.training_indices, [i for i in range(self.d-1)] + [-1]]
        return [np.array(data), pref]

    def get_input_test(self, m):
        """
        Construct testing data to use for modelling
        :param m: int, number of preferences
        :return: list, - data: np.array, n instances with their attributes
                       - pref: list of tuples, testing preferences
        """
        pairs = self.test_generator(m)
        pref, self.testing_indices = utils.reshape_pref(self.draw_preference(pairs))
        data = self.X.iloc[self.testing_indices, [i for i in range(self.d-1)] + [-1]]
        return [np.array(data), pref]


class instance_pref_generator:
    """ Instance Learning:
    Generate training and testing sets for simulated data sets.
    """
    def __init__(self, func, func_param):
        """
        Initialize utility function (np.array, e.g. Cobb-Douglas) and parameters (np.array, e.g. alpha parameters)
        :param func: np.array
        :param func_param: np.array
        """
        self.real_f = func
        self.f_param = func_param

    def generate_X(self, n, d):
        """
        Generate the dataset
        :param n: int, number of instances
        :param d: int, number of attributes
        :return: np.array, matrix of size nxd
        """
        return np.random.uniform(size=(n, d))

    @staticmethod
    def draw_preference(n):
        """
        Draw a random index to be used for computing preference relation (add_a_pref)
        :param n: int, number of instances
        :return: tuple, (a,b)
        """
        a = np.random.randint(n)
        b = np.random.randint(n)
        while b == a:
            b = np.random.randint(n)
        return a, b

    def add_a_pref(self, X, existing_pref, iter_max=10):
        """
        Compute 1 preference relation based on utility function real_f
        :param X: np.array, data
        :param existing_pref: list, list of pairs of indices
        :param iter_max: int, stopping criteria
        :return: tuple, (a,b) if a is preferred to b and vice versa
        """
        n_iter, n = 0, X.shape[0]
        a, b = self.draw_preference(n)
        while (a, b) in existing_pref or (b, a) in existing_pref and n_iter < iter_max:
            a, b = self.draw_preference(n)
            n_iter += 1
        f_a, f_b = self.real_f(X[a], self.f_param), self.real_f(X[b], self.f_param)
        if f_a > f_b:
            return a, b
        else:
            return b, a

    def set_m_preference(self, X, m):
        """
        Compute m preferences relations
        :param X: np.array, data
        :param m: int, number of preferences
        :return: list, list of preferences
        """
        pref = []
        for i in range(m):
            pref.append(self.add_a_pref(X, pref))
        return pref

    def generate_X_pref(self, n, m, d):
        """
        Construct data + preferences to use for modelling
        :param n: int, number of instances
        :param m: int, number of preferences
        :param d: int, number of attributes
        :return: tuple, - X: np.array, data
                        - D: np.array, list of preferences
        """
        X = self.generate_X(n, d)
        D = self.set_m_preference(X, m)
        return X, D

    def sample_datasets(self, n, d, m, mp):
        """
        Construct training and testing inputs for modelling (data + preferences)
        :param n: int, number of training instances
        :param d: int, number of attributes
        :param m: int, number of preferences
        :param mp: int, number of testing preferences
        :return: tuple, - train: X_train, D_train
                        - test: X_test, D_test
        """
        train = self.generate_X_pref(n, m, d)
        test = (train[0], self.set_m_preference(train[0], mp))
        return train, test

    def get_true_pref(self, X):
        """
        Construct training data to use for modelling
        :param X: np.array, data
        :return: list of tuples
        """
        n = X.shape[0]
        pref = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if self.real_f(X[i], self.f_param) > self.real_f(X[j], self.f_param):
                    pref[i, j] = 1
        return pref


class label_pref_generator(instance_pref_generator):
    """ Label Learning:
    Generate training and testing sets for real data sets.
    """
    def __init__(self, func, func_param):
        super().__init__(func, func_param)
        self.n_label = len(self.real_f)

    def add_a_pref(self, x, existing_pref, iter_max=10):
        """
        Compute 1 preference relation based on utility function real_f
        :param x: np.array
        :param existing_pref: list, list of pairs of indices
        :param iter_max: int, stopping criteria
        :return: tuple, (a,b) if a is preferred to b and vice versa
        """
        a, b = self.draw_preference(self.n_label)
        n_iter = 0
        while (a, b) in existing_pref or (b, a) in existing_pref and n_iter < iter_max:
            a, b = self.draw_preference(self.n_label)
            n_iter += 1
        f_a, f_b = self.real_f[a](x, self.f_param[a]), self.real_f[b](x, self.f_param[b])
        if f_a > f_b:
            return a, b
        else:
            return b, a

    def set_m_preference(self, X, m):
        """
        :param X: np.array, data
        :param m: int, number of preferences. It becomes the maximum number of edges to draw for each vector in X
        """
        pref = []
        n = X.shape[0]
        n_observed = np.random.randint(low=1, high=m+1, size=n)
        for i in range(n):
            pref_i = []
            for j in range(n_observed[i]):
                pref_i.append(self.add_a_pref(X[i], pref_i))
            pref.append(copy.copy(pref_i))
        return pref


def cobb_douglas(x, alpha):
    """
    :param x: np.array
    :param alpha: np.array
    :return: np.array, utility functions
    """
    x_alpha = x**alpha
    return np.prod(x_alpha)