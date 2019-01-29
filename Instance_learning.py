import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.stats import norm
from tqdm import tqdm
from scipy.optimize import minimize
from utils import distance, n_pdf, gaussian_kernel



class learning_instance_preference:
    def __init__(self, inputs, K, sigma, print_callback=True):
        self.init_param(inputs, K, sigma)
        self.Nfeval = 1
        self.S = 0
        self.print_callback = print_callback
        self.tol = 1e-4

    def init_param(self, inputs, K, sigma):
        """
        :param inputs: tuple->(subset of X, list of preferences (u,v)=> u>v
        :param K: coefficient for the gaussian Kernels
        :param sigma: variance of the noise
        :return: Set the attributes of the model with the given inputs
        """
        self.X = inputs[0]
        self.D = inputs[1]
        self.n, self.d = self.X.shape
        self.m = len(self.D)
        if not isinstance(K, list):
            self.cov = self.compute_cov(K)
            self.inv_cov = np.linalg.inv(self.cov)
        self.sigma = sigma
        self.K = K
        self.current_y = None
        self.z = None
        self.phi = None

    def compute_cov(self, K):
        """
        :param K: parameter of the gaussian kernel
        :return: Covariance matrix for the training dataset
        """
        cov = np.eye(self.n)
        for i in range(self.n):
            for j in range(i):
                cov_ij = gaussian_kernel(self.X[i], self.X[j], K)
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij
        return cov

    def compute_prior(self, y):
        """
        :param y: vector of size self.n
        :return: prior density in y
        """
        return mn.pdf(y, mean=np.zeros(self.n), cov=self.cov)

    def compute_z_phi(self, y):
        """
        :param y: vector of size n
        :return: list of z_k and list of phi(z_k) as defined in (5) in the paper
        """
        z = np.apply_along_axis(lambda x: (y[x[0]]-y[x[1]])/(np.sqrt(2)*self.sigma), 1, self.D)
        self.current_y = y
        self.z = z
        self.phi = norm.cdf(z)

    def compute_S(self, y):
        """
        :param y: vector of size n
        :return: S(y)
        """
        if distance(y, self.current_y) != 0:
            self.compute_z_phi(y)
        prior = 0.5*np.inner(y, np.dot(self.inv_cov, y))
        return -np.sum(np.log(self.phi))+prior

    def s(self, k, i):
        check_0 = 1 if self.D[k][0] == i else 0
        check_1 = 1 if self.D[k][1] == i else 0
        return check_0 - check_1

    def partial_df(self, k, i):  # for the log_phi part
        return -self.s(k, i) / np.sqrt(2) / self.sigma * n_pdf(self.z[k]) / self.phi[k]

    def compute_grad_S(self, y):
        """
        :param y: vector of size n
        :return: Gradient of S in y
        """
        if distance(y, self.current_y) != 0:
            self.compute_z_phi(y)
        phi_grad = np.array([sum([self.partial_df(k, i) for k in range(self.m)]) for i in range(self.n)])
        return phi_grad+np.dot(self.inv_cov, y)

    def compute_Hessian_S(self, y):
        """
        :param y: vector of size n
        :return: Hessian of S in y
        """
        if distance(y, self.current_y) != 0:
            self.compute_z_phi(y)
        def partial_d2f(k, i, j):
            s_ij = self.s(k, i)*self.s(k, j)/2/self.sigma**2
            nk = n_pdf(self.z[k])/self.phi[k]
            return s_ij*(nk**2+self.z[k]*nk)
        phi_Hess = np.array([[sum([partial_d2f(k, i, j) for k in range(self.m)])
                              for j in range(self.n)] for i in range(self.n)])
        self.nu = phi_Hess
        return phi_Hess+self.inv_cov

    def callbackF(self, Xi):
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
        :param y: Starting vector for the minimization program
        :return: A scipy OptimizeResult dict with results after the minimization
        (convergence, last value, jacobian,...)
        """
        if self.print_callback:
            print('Starting gradient descent:')
            m = minimize(self.compute_S, y, method='Newton-CG', jac=self.compute_grad_S,
                        hess=self.compute_Hessian_S, tol=self.tol, callback=self.callbackF)
        else:
            m = minimize(self.compute_S, y, method='Newton-CG', jac=self.compute_grad_S,
                     hess=self.compute_Hessian_S, tol=self.tol)
        return m

    def evidence_approx(self, y):
        """
        :param y: a vector with n values of f(x) (for x in self.X)
        :return: Laplace approximation of the evidence of the model with y
        """
        S, H = self.compute_S(y), self.compute_Hessian_S(y)
        denom = np.linalg.det(np.dot(self.cov, H))
        return np.log(np.exp(-S)/np.sqrt(np.abs(denom)))

    def compute_MAP_with_gridsearch(self, y0, grid_K, grid_sigma):
        """
        :param y0: Starting point for optimization
        :param grid_K: grid for the kernel parameter K
        :param grid_sigma: grid for the noise variance parameter sigma
        :return: Maximum a posteriori for the given value of x for the parameters
        with the largest evidence
        """
        best_K, best_sigma, best_evidence = grid_K[0], grid_sigma[0], -np.inf
        for K in tqdm(grid_K):
            for sigma in grid_sigma:
                self.K = K
                self.sigma = sigma
                self.cov = self.compute_cov(K)
                self.inv_cov = np.linalg.inv(self.cov)
                MAP = self.compute_MAP(y0)
                evidence = self.evidence_approx(MAP['x'])
                print('\nK={}, sigma={} : evidence={}'.format(K, sigma, evidence))
                if evidence > best_evidence:
                    best_evidence = evidence
                    best_K, best_sigma = K, sigma
                    best_MAP = MAP
        print('Best K and Sigma in grid : {},{}'.format(best_K, best_sigma))
        self.K, self.sigma = best_K, best_sigma
        self.cov = self.compute_cov(self.K)
        self.inv_cov = np.linalg.inv(self.cov)
        return best_MAP

    def predict_single_pref(self, r, s, f_map, M):
        """
        :param r: vector of dimension d
        :param s: vector of dimension d
        :param f_map: Maximum a posteriori for n vectors in R^d
        :param M: covariance matrix computed for the maximum a posteriori
        :return: Probability that r is preferred over s
        """
        sigma_t = np.array([[1., gaussian_kernel(r, s, self.K)], [gaussian_kernel(r, s, self.K), 1.]])
        kt = np.zeros((self.n, 2))
        kt[:, 0] = [gaussian_kernel(r, self.X[i], self.K) for i in range(self.n)]
        kt[:, 1] = [gaussian_kernel(s, self.X[i], self.K) for i in range(self.n)]
        mu_star = np.dot(kt.T, np.dot(self.inv_cov, f_map))
        s_star = sigma_t-np.dot(np.dot(kt.T, M), kt)
        new_sigma = np.sqrt(2*self.sigma**2+s_star[0, 0]+s_star[1, 1]-2*s_star[0, 1])
        return norm.cdf((mu_star[0]-mu_star[1])/new_sigma, loc=0, scale=1)

    def predict(self, test_set, f_map):
        """
        :param test_set: tuple of length 2:
        * an array for the test X from which preferences are drawn
        * a list of tuples (i,j) for the m' new preferences to predict, where i is
        preferred to j (helps for the score)
        :param f_map: Maximum a Posteriori obtained with the Gaussian Kernels method
        :return: Score of the prediction, list of the probabilities P(i preferred over j)
        """
        X = test_set[0]
        pref = test_set[1]
        score = 0
        proba_pref = []
        self.compute_Hessian_S(f_map)
        M = np.linalg.inv(self.cov+np.linalg.inv(self.nu))
        for p in pref:
            proba = self.predict_single_pref(X[p[0]], X[p[1]], f_map, M)
            proba_pref.append(proba)
            if proba > 0.5:
                score += 1
        return score/len(pref), proba_pref

    def get_train_pref(self, mp):
        """
        :param mp: Maximum a posteriori for n values of x
        :return: A matrix with booleans: 1 if map[i]>map[j] else 0
        """
        pref = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if mp[i] > mp[j]:
                    pref[i, j] = 1
        return pref

    def get_confusion_matrix(self, pref, map):
        """
        :param pref: Boolean matrix with input preferences
        :param map: Boolean matrix with output preferences
        :return: confusion matrix (the result is symmetric)
        """
        matrix = np.zeros((2, 2))
        matrix[0, 0] = ((1-pref)*(1-map)).sum()
        matrix[0, 1] = (pref*(1-map)).sum()
        matrix[1, 0] = ((1-pref)*map).sum()
        matrix[1, 1] = (pref*map).sum()
        return matrix

    def score(self, pref, map):
        """
        :param pref: Boolean matrix with input preferences
        :param map: Boolean matrix with output preferences
        :return: confusion matrix (the result is symmetric)
        """
        tot = self.n*(self.n-1)/2
        err = ((pref-map) != 0).sum()/2
        return (tot-err)/tot
