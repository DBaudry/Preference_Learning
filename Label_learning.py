import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.stats import norm
import copy
from scipy.optimize import minimize
from utils import distance, n_pdf, gaussian_kernel
from scipy.linalg import block_diag


class learning_label_preference:
    """
    Implement GP for Label Preference Learning
    """
    def __init__(self, inputs, K, sigma, print_callback=True):
        self.init_param(inputs, K, sigma)
        # Parameters for gradient descent
        self.Nfeval = 1
        self.S = 0
        self.print_callback = print_callback

    def init_param(self, inputs, all_K, sigma):
        """
        :param inputs: tuple, - np.array, subset of X
                              - list of preferences where (u,v) means that u is preferred to v and vice versa
        :param K: float, coefficient for Gaussian Kernels
        :param sigma: float, noise variance
        """
        self.X = inputs[0]
        self.D = inputs[1]
        self.n, self.d = self.X.shape
        if not isinstance(all_K, list):
            self.n_labels = len(all_K)
            self.all_cov = [self.compute_cov(K) for K in all_K]
            self.all_inv_cov = [np.linalg.inv(cov) for cov in self.all_cov]
            self.sigma = sigma
            self.K = all_K
        else:
            self.n_labels = len(all_K[0])
        self.corresp = self.get_corresp()
        self.current_y = None
        self.z = None
        self.phi = None

    def compute_cov(self, K):
        """
        :param K: float, parameter of the Gaussian kernel
        :return: np.array, covariance matrix for training
        """
        cov = np.eye(self.n)
        for i in range(self.n):
            for j in range(i):
                cov_ij = gaussian_kernel(self.X[i], self.X[j], K)
                cov[i, j], cov[j, i] = cov_ij, cov_ij
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
        :param y: np.array, array of shape n_labels x n
        :return: float, prior probability of y
        """
        y = y.reshape((self.n_labels, self.n))
        prod = 1.
        for a in range(self.n_labels):
            prod *= mn.pdf(y[a], mean=np.zeros(self.n), cov=self.all_cov[a])
        return prod

    def compute_z_phi(self, y):
        """
        :param y: np.array, array of shape n_labels x n
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
        """
        :param y: np.array, array of shape n_labels x n
        :return: np.array, S(y)
        """
        if distance(y, self.current_y) != 0:
            self.compute_z_phi(y)
        prior = 0
        y = y.reshape((self.n_labels, self.n))
        for a in range(self.n_labels):
            prior += 0.5*np.inner(y[a], np.dot(self.all_inv_cov[a], y[a]))
        return -(np.log(self.phi)).sum()+prior

    def compute_grad_S(self, y):
        """
        :param y: np.array, array of shape n_labels x n
        :return: np.array, grad of S according to a vector of size(n_labels x n) by reshaping y
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
        """
        :param y: np.array, array of shape n_labels x n
        :return: np.array, Hessian of S evaluated in y
        """
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
        """
        :param Xi: np.array, values returned by scipy.minimize at each iteration
        :return: None, update print
        """
        if self.Nfeval == 1:
            self.S = self.compute_S(Xi)
            print('Iteration {0:2.0f} : S(y)={1:3.6f}'.format(self.Nfeval, self.S))
        else:
            s_next = self.compute_S(Xi)
            print('Iteration {0:2.0f} : S(y)={1:3.6f}, tol={2:0.6f}'.format(self.Nfeval, s_next, abs(self.S-s_next)))
            self.S = s_next
        self.Nfeval += 1

    def compute_MAP(self, y):
        """
        :param y: np.array, starting vector for the minimization program
        :return: scipy.optimize.minimize object, i.e. a scipy OptimizeResult dict with results after the minimization
        (convergence, last value, Jacobian matrix,...)
        """
        if self.print_callback:
            print('Starting gradient descent:')
            m = minimize(self.compute_S, y, method='Newton-CG', jac=self.compute_grad_S,
                         hess=self.compute_Hessian_S, tol=1e-4, callback=self.callbackF)
        else:
            m = minimize(self.compute_S, y, method='Newton-CG', jac=self.compute_grad_S,
                         hess=self.compute_Hessian_S, tol=1e-4)
        return m

    def compute_MAP_with_gridsearch(self, y0, grid_K, grid_sigma):
        """
        :param y0: np.array, starting point for optimization
        :param grid_K: np.array, grid for the kernel parameter K
        :param grid_sigma: np.array, grid for the noise variance parameter sigma
        :return: np.array, Maximum a posteriori for the given value of x for the parameters
        with the largest evidence
        """
        best_K, best_sigma, best_evidence = grid_K[0], grid_sigma[0], -np.inf
        for K in grid_K:
            for sigma in grid_sigma:
                self.K = K
                self.sigma = sigma
                self.all_cov = [self.compute_cov(K) for K in self.K]
                try:
                    self.all_inv_cov = [np.linalg.inv(cov) for cov in self.all_cov]
                    MAP = self.compute_MAP(y0)
                    evidence = self.evidence_approx(MAP['x'])
                    print('K={}, sigma={:0.4f} : evidence={:0.4f}'.format(self.K, self.sigma, evidence))
                except:
                    print('K={}, sigma={:0.4f} : singular matrix'.format(self.K, self.sigma))
                    continue
                if 0 > evidence > best_evidence:
                    best_evidence = evidence
                    best_K, best_sigma = self.K, self.sigma
                    best_MAP = MAP
        print('Best K and Sigma in grid : {},{}'.format(best_K, best_sigma))
        self.K, self.sigma = best_K, best_sigma
        self.all_cov = [self.compute_cov(K) for K in self.K]
        self.all_inv_cov = [np.linalg.inv(cov) for cov in self.all_cov]
        return best_MAP

    def evidence_approx(self, y):
        """
        :param y: np.array, a vector with n values of f(x) (for x in self.X)
        :return: np.array, Laplace approximation of the evidence of the model evaluated in y
        """
        S, H = self.compute_S(y), self.compute_Hessian_S(y)
        cov = block_diag(*self.all_cov)
        denom = np.linalg.det(np.dot(cov, H))
        return -(S+0.5*np.log(np.abs(denom)))

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
        """
        Compute Pref Error Rate as detailed in the report
        """
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
        Compute Label Error Rate as detailed in the report
        """
        class_pred = np.argsort(map, axis=1)[:, -1]
        return np.mean(pref == class_pred)

