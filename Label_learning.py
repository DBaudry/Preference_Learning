import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.stats import norm
import copy
from tqdm import tqdm
from scipy.optimize import minimize


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

    @staticmethod
    def gaussian_kernel(x, y, K):
        return np.exp(-K/2.*np.sum((x-y)**2))

    def n_pdf(self, x):
        return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

    def compute_cov(self, K):
        cov = np.eye(self.n)
        for i in range(self.n):
            for j in range(i):
                cov_ij = self.gaussian_kernel(self.X[i], self.X[j], K)
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij
        return cov

    def get_corresp(self):
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
        '''
        :param y: array of shape n_labels x n
        :return: prior probability of y
        '''
        prod = 1.
        for a in range(self.n_labels):
            prod *= mn.pdf(y[a], mean=np.zeros(self.n), cov=self.all_cov[a])
        return prod

    def compute_z_phi(self, y):
        z = []
        for i in range(self.n):
            for pref in self.D[i]:
                current_z = (y[pref[0], i]-y[pref[1], i])/np.sqrt(2)/self.sigma
                z.append(current_z)
        return z, norm.cdf(z)

    def compute_S(self, y):
        phi = self.compute_z_phi(y)[1]
        prior = 0
        for a in range(self.n_labels):
            prior += 0.5*np.inner(y[a], np.dot(self.all_inv_cov[a], y[a]))
        return -(np.log(phi)).sum()+prior

    def compute_grad_S(self, y):
        """
        :param y: array of shape n_labels x n
        :return: grad of S according to a vector of size(n_labels x n) by reshaping y
        """
        z, phi = self.compute_z_phi(y)
        grad = np.zeros((self.n_labels, self.n))
        for i in range(self.n):
            for j, pref in enumerate(self.D[i]):
                index = self.corresp[i][j]
                grad[pref[0], i] -= self.n_pdf(z[index])/phi[index]/np.sqrt(2)/self.sigma
                grad[pref[1], i] += self.n_pdf(z[index])/phi[index]/np.sqrt(2)/self.sigma
        for a in range(self.n_labels):
            grad[a] += np.dot(self.all_inv_cov[a], y[a])
        return grad.reshape(self.n_labels*self.n)

    def compute_Hessian_S(self, y):
        z, phi = self.compute_z_phi(y)
        Hess_phi = [np.zeros((self.n_labels, self.n_labels)) for _ in range(self.n)]
        for i in range(self.n):
            for j, pref in enumerate(self.D[i]):
                index = self.corresp[i][j]
                nk = self.n_pdf(z[index])/phi[index]
                Hess_phi[i][pref[0], pref[1]] += -1/2/self.sigma**2*(nk**2+z[index]*nk)
                Hess_phi[i][pref[1], pref[0]] += -1/2/self.sigma**2*(nk**2+z[index]*nk)
        Hess_phi_all = np.zeros((self.n_labels*self.n, self.n_labels*self.n))
        for i in range(self.n):
            idx = i*self.n
            Hess_phi_all[idx:idx+self.n_labels, idx:idx+self.n_labels] = Hess_phi[i]
        Hess_cov = np.zeros((self.n_labels*self.n, self.n_labels*self.n))
        for a in range(self.n_labels):
            idx = a*self.n_labels
            Hess_cov[idx:idx+self.n, idx:idx+self.n] = self.all_inv_cov[a]
        return Hess_phi_all + Hess_cov