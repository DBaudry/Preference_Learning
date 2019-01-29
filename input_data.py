import numpy as np
import copy
import utils

class pref_generator():
    def __init__(self, dataset, n=-1, d=-1):
        self.X = utils.read_data_IL(dataset, n, d)
        self.nmax, self.dmax = self.X.shape
        self.n = self.nmax if n == -1 else n
        self.d = self.dmax if d == -1 else d
        self.pairs_index = utils.combinations(self.n)  # pkl.load(open('./Data/combinations_'+dataset+'.pkl', 'rb'))

    def train_generator(self, m):
        replace = True if m > len(self.pairs_index) else False
        idx = np.random.choice(len(self.pairs_index), m, replace=replace)
        pairs = self.pairs_index[idx]
        self.training_pairs = tuple(map(tuple, pairs))
        return self.training_pairs

    def test_generator(self, m):
        if self.n == self.nmax:
            n1 = 0
        else:
            n1 = self.n+1
        self.pairs_index = np.array([(i, j) for (i, j) in utils.combinations(self.nmax, n1)]) #if (i, j) not in self.training_pairs])
        replace = True if m > len(self.pairs_index) else False
        idx = np.random.choice(len(self.pairs_index), m, replace=replace)
        pairs = self.pairs_index[idx]
        self.testing_pairs = tuple(map(tuple, pairs))
        return self.testing_pairs

    def draw_preference(self, pairs):
        pref = []
        for p in pairs:
            a, b = p
            if self.X.iloc[a, -1] > self.X.iloc[b, -1]:
                pref.append((a, b))
            else:
                pref.append((b, a))
        return pref

    def get_input_train(self, m):
        pairs = self.train_generator(m)
        pref, self.training_indices = utils.reshape_pref(self.draw_preference(pairs))
        data = self.X.iloc[self.training_indices, [i for i in range(self.d-1)] + [-1]]
        return [np.array(data), pref]

    def get_input_test(self, m):
        pairs = self.test_generator(m)
        pref, self.testing_indices = utils.reshape_pref(self.draw_preference(pairs))
        data = self.X.iloc[self.testing_indices, [i for i in range(self.d-1)] + [-1]]
        return [np.array(data), pref]

    def get_true_pref(self, data):
        p = data.shape[0]
        real_pref = np.zeros((p, p))
        X = np.array(self.X)
        for i in range(p):
            for j in range(p):
                if X[self.training_indices[i], -1] > X[self.training_indices[j], -1]:
                    real_pref[i, j] = 1
        return real_pref


class instance_pref_generator:
    def __init__(self, func, func_param):
        self.real_f = func
        self.f_param = func_param

    def generate_X(self, n, d):
        return np.random.uniform(size=(n, d))

    @staticmethod
    def draw_preference(n):
        a = np.random.randint(n)
        b = np.random.randint(n)
        while b == a:
            b = np.random.randint(n)
        return a, b

    def add_a_pref(self, X, existing_pref, iter_max=10):
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
        pref = []
        for i in range(m):
            pref.append(self.add_a_pref(X, pref))
        return pref

    def generate_X_pref(self, n, m, d):
        X = self.generate_X(n, d)
        D = self.set_m_preference(X, m)
        return X, D

    def sample_datasets(self, n, d, m, mp):
        train = self.generate_X_pref(n, m, d)
        test = self.set_m_preference(train[0], mp)
        return train, test

    def get_true_pref(self, X):
        n = X.shape[0]
        pref = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if self.real_f(X[i], self.f_param) > self.real_f(X[j], self.f_param):
                    pref[i, j] = 1
        return pref


class label_pref_generator(instance_pref_generator):
    def __init__(self, func, func_param):
        super().__init__(func, func_param)
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

    def set_m_preference(self, X, m):
        """
        m becomes the maximum number of edges to draw for each vector in X
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
    x_alpha = x**alpha
    return np.prod(x_alpha)