import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.stats import norm
import copy
from scipy.optimize import minimize
from utils import distance, n_pdf, gaussian_kernel
from scipy.linalg import block_diag

Nfeval = 1
S = 0


class learning_label_preference:
    def __init__(self, inputs, K, sigma):
        self.init_param(inputs, K, sigma)

    def init_param(self, inputs, all_K, sigma):
        """
        :param inputs: tuple->(subset of X, list of preferences (u,v)=> u>v
        :param K: coefficient for the gaussian Kernels
        :param sigma: variance of the noise
        :return: Set the attributes of the model with the given inputs
        """
        self.X = inputs[0]
        self.D = inputs[1]
        self.n, self.d = self.X.shape
        self.n_labels = len(all_K)
        self.all_cov = [self.compute_cov(K) for K in all_K]
        self.all_inv_cov = [np.linalg.inv(cov) for cov in self.all_cov]
        self.sigma = sigma
        self.K = all_K
        self.corresp = self.get_corresp()
        self.current_y = None
        self.z = None
        self.phi = None

    def compute_cov(self, K):
        cov = np.eye(self.n)
        for i in range(self.n):
            for j in range(i):
                cov_ij = gaussian_kernel(self.X[i], self.X[j], K)
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij
        return cov

    def get_corresp(self):
        """
        :return: indices of a preference in the preference list in a flattened array. The
        returned object is the list of length n where each component is a list with length equal to the number
        of edges in the preference graph of x_i
        """
        D_index = []
        count = 0
        for i in range(self.n):
            indices = []
            for k in range(len(self.D[i])):
                indices.append(count)
                count += 1
            D_index.append(copy.copy(indices))
        return D_index

    def compute_prior(self, y):
        """
        :param y: array of shape n_labels x n
        :return: prior probability of y
        """
        y = y.reshape((self.n_labels, self.n))
        prod = 1.
        for a in range(self.n_labels):
            prod *= mn.pdf(y[a], mean=np.zeros(self.n), cov=self.all_cov[a])
        return prod

    def compute_z_phi(self, y):
        """
        :param y:
        :return:
        """
        y = y.reshape((self.n_labels, self.n))
        z = []
        for i in range(self.n):
            for pref in self.D[i]:
                current_z = (y[pref[0], i]-y[pref[1], i])/np.sqrt(2)/self.sigma
                z.append(current_z)
        self.current_y = y.flatten()
        self.z = z
        self.phi = norm.cdf(z)

    def compute_S(self, y):
        if distance(y, self.current_y) != 0:
            self.compute_z_phi(y)
        prior = 0
        y = y.reshape((self.n_labels, self.n))
        for a in range(self.n_labels):
            prior += 0.5*np.inner(y[a], np.dot(self.all_inv_cov[a], y[a]))
        return -(np.log(self.phi)).sum()+prior

    def compute_grad_S(self, y):
        """
        :param y: array of shape n_labels x n
        :return: grad of S according to a vector of size(n_labels x n) by reshaping y
        """
        if distance(y, self.current_y) != 0:
            self.compute_z_phi(y)
        grad = np.zeros((self.n_labels, self.n))
        y = y.reshape((self.n_labels, self.n))
        for i in range(self.n):
            for j, pref in enumerate(self.D[i]):
                index = self.corresp[i][j]
                grad[pref[0], i] -= n_pdf(self.z[index])/self.phi[index]/np.sqrt(2)/self.sigma
                grad[pref[1], i] += n_pdf(self.z[index])/self.phi[index]/np.sqrt(2)/self.sigma
        for a in range(self.n_labels):
            grad[a] += np.dot(self.all_inv_cov[a], y[a])
        return grad.reshape(self.n_labels*self.n)

    def compute_Hessian_S(self, y):
        if distance(y, self.current_y) != 0:
            self.compute_z_phi(y)
        Hess_phi = [np.zeros((self.n_labels, self.n_labels)) for _ in range(self.n)]
        for i in range(self.n):
            for j, pref in enumerate(self.D[i]):
                index = self.corresp[i][j]
                nk = n_pdf(self.z[index])/self.phi[index]
                Hess_phi[i][pref[0], pref[1]] += -1/2/self.sigma**2*(nk**2+self.z[index]*nk)
                Hess_phi[i][pref[1], pref[0]] += -1/2/self.sigma**2*(nk**2+self.z[index]*nk)
        Hess_phi_all = block_diag(*Hess_phi)
        Hess_cov = block_diag(*self.all_inv_cov)
        return Hess_phi_all + Hess_cov

    def callbackF(self, Xi):
        global Nfeval
        global S
        if Nfeval == 1:
            S = self.compute_S(Xi)
            print('Iteration {0:2.0f} : S(y)={1:3.6f}'.format(Nfeval, S))
        else:
            s_next = self.compute_S(Xi)
            print('Iteration {0:2.0f} : S(y)={1:3.6f}, tol={2:0.6f}'.format(Nfeval, s_next, abs(S-s_next)))
            S = s_next
        Nfeval += 1

    def compute_MAP(self, y):
        """
        :param y: Starting vector for the minimization program
        :return: A scipy OptimizeResult dict with results after the minimization
        (convergence, last value, jacobian,...)
        """
        print('Starting gradient descent:\n')
        return minimize(self.compute_S, y, method='Newton-CG', jac=self.compute_grad_S,
                        hess=self.compute_Hessian_S, tol=1e-4, callback=self.callbackF)

    def evidence_approx(self, y):
        """
        :param y: a vector with n values of f(x) (for x in self.X)
        :return: Laplace approximation of the evidence of the model with y
        """
        S, H = self.compute_S(y), self.compute_Hessian_S(y)
        cov = block_diag(*self.all_cov)
        denom = np.linalg.det(np.dot(cov, H))
        return np.log(np.exp(-S)/np.sqrt(np.abs(denom)))

    def get_kernel(self, data, a):
        K = self.K[a]
        kt = np.empty((data.shape[0], self.X.shape[0]))
        for i in range(data.shape[0]):
            for j in range(self.X.shape[0]):
                kt[i, j] = gaussian_kernel(data[i], self.X[j], K)
        return kt

    def predict(self, data, map):
        beta = np.dot(block_diag(*self.all_inv_cov), map).reshape((self.n_labels, self.n))
        E = np.empty((self.n_labels, data.shape[0]))
        for a in range(self.n_labels):
            Ka = self.get_kernel(data, a)
            E[a] = np.dot(Ka, beta[a])
        return E.T

    def label_pref_rate(self, pref, map):
        count_glob, count_correct = 0, 0
        for i, p in enumerate(map):
            n = len(p)
            p = np.array(p).argsort()[::-1]
            pref_map = []
            for j in range(0, n):
                for k in range(j + 1, n):
                    pref_map.append((p[j], p[k]))
            for p_true in pref[i]:
                count_glob += 1
                if p_true in pref_map:
                    count_correct += 1
        return count_correct / count_glob

    def label_score_rate(self, pref, map):
        """
        :param pref: Boolean matrix with input preferences
        :param map: Boolean matrix with output preferences
        :return: confusion matrix (the result is symmetric)
        """
        class_pred = np.argsort(map, axis=1)[:, -1]
        return np.mean(pref == class_pred)

