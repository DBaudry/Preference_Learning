import numpy as np
import copy
import utils
import itertools

class pref_generator():
    def __init__(self, dataset):
        self.X = utils.read_data(dataset)
        self.n, self.d = self.X.shape
        self.pairs_index = np.array([p for p in itertools.combinations(range(self.n), 2)])

    def train_test_generator(self, m):
        replace = True if (m > self.n * (self.n - 1) / 2) else False
        idx = np.random.choice(len(self.pairs_index), m, replace=replace)
        pairs = self.pairs_index[idx]
        return tuple(map(tuple, pairs))

    def draw_preference(self, pairs):
        pref = []
        for p in pairs:
            a, b = p
            if self.X.iloc[a, -1] > self.X.iloc[b, -1]:
                pref.append((a, b))
            else:
                pref.append((b, a))
        return pref

    def get_input(self, m):
        pairs = self.train_test_generator(m)
        pref = self.draw_preference(pairs)
        data = self.X.iloc[np.unique(pairs), :]
        return [np.array(data), pref]



class instance_pref_generator:
    def __init__(self, func, func_param, d, n, m):
        self.n = n
        self.m = m
        self.d = d
        self.real_f = func
        self.f_param = func_param

    def generate_X(self):
        return np.random.uniform(size=(self.n, self.d))

    @staticmethod
    def draw_preference(n):
        a = np.random.randint(n)
        b = np.random.randint(n)
        while b == a:
            b = np.random.randint(n)
        return a, b

    def add_a_pref(self, X, existing_pref, iter_max=10):
        a, b = self.draw_preference(self.n)
        n_iter = 0
        while (a, b) in existing_pref or (b, a) in existing_pref and n_iter < iter_max:
            a, b = self.draw_preference(self.n)
            n_iter += 1
        f_a, f_b = self.real_f(X[a], self.f_param), self.real_f(X[b], self.f_param)
        if f_a > f_b:
            return a, b
        else:
            return b, a

    def set_m_preference(self, X):
        pref = []
        for i in range(self.m):
            pref.append(self.add_a_pref(X, pref))
        return pref

    def generate_X_pref(self):
        X = self.generate_X()
        D = self.set_m_preference(X)
        return X, D

    def get_true_pref(self, X):
        pref = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.real_f(X[i], self.f_param) > self.real_f(X[j], self.f_param):
                    pref[i, j] = 1
        return pref


class label_pref_generator(instance_pref_generator):
    def __init__(self, func, func_param, d, n, m):
        super().__init__(func,func_param, d, n, m)
        self.n_label = len(self.real_f)

    def add_a_pref(self, x, existing_pref, iter_max=10):
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

    def set_m_preference(self, X):
        """
        m becomes the maximum number of edges to draw for each vector in X
        """
        pref = []
        n_observed = np.random.randint(low=1, high=self.m+1, size=self.n)
        for i in range(self.n):
            pref_i = []
            for j in range(n_observed[i]):
                pref_i.append(self.add_a_pref(X[i], pref_i))
            pref.append(copy.copy(pref_i))
        return pref


def cobb_douglas(x, alpha):
    x_alpha = x**alpha
    return np.prod(x_alpha)