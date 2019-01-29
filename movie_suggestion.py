import Instance_learning as IL
import numpy as np
from tqdm import tqdm
from utils import gaussian_kernel

np.random.seed(211)

class Movielens():
    def __init__(self, n_actions=207, eta=1, sigma=10, K=0.1, sigma_IL=0.01):
        self.eta = eta
        self.sigma = sigma
        self.features = np.loadtxt('Data/Vt.csv', delimiter=',').T
        self.n_movies, self.n_features = self.features.shape
        self.real_theta = np.empty(self.n_features)
        self.K = K
        self.sigma_IL = sigma_IL
        self.MAP = None

    def get_new_user(self):
        """
        :return: draw a new theta for an experiment
        """
        self.real_theta = np.random.normal(loc=0., scale=self.sigma, size=self.n_features)

    def update_pref(self, t, pref, ratings):
        if t == 0:
            return pref
        for i in range(t-1):
            if ratings[i] > ratings[t]:
                pref.append([i, t])
            else:
                pref.append([t, i])
        return pref

    def predict_movie(self, model, instances, data, MAP):
        """
        :param model: Instance_Learning instance
        :param instances: Training dataset (features of movies already watched)
        :param data: Test dataset (feature of movies not watched)
        :param MAP: Maximum a posteriori estimate on watched movies
        :return: indice of the best instance in the test dataset
        """
        beta = np.dot(model.inv_cov, MAP)
        kt = np.array([[gaussian_kernel(data[i], instances[j], self.K) for j in range(instances.shape[0])]
                       for i in range(data.shape[0])])
        return int(np.argmax(np.dot(kt, beta)))

    def movie_suggestion(self, t, v, m, to_watch, burnin):
        """
        :param t: current time
        :param v: features dataset, in order of movies watched
        :param m: current known preferences
        :param to_watch: movies that have not been tested yet
        :param burnin: time limit of the training phase
        :return: indice of the suggestion in the original feature dataset
        """
        if t < 3:
            return int(np.random.choice(to_watch))
        else:
            if t < burnin:
                self.instances = v[:t]
                self.model = IL.learning_instance_preference(inputs=[self.instances, m],
                                                             K=self.K, sigma=self.sigma_IL)
                self.model.tol = 1e-2
                self.MAP = self.model.compute_MAP(np.zeros(t))['x']
            test_instance = self.features[to_watch]
            return to_watch[int(self.predict_movie(self.model, self.instances, test_instance, self.MAP))]

    def get_regret(self, r, to_watch):
        best_rating = np.dot(self.features[to_watch], self.real_theta)
        return np.max(best_rating)-r

    # Preference learning with Gaussian process (article)
    def GP_suggestion(self, T, burnin=np.inf):
        """
        :param T: Time Horizon
        :param burnin: Length of the adaptation phase
        :return: Rewards of the sequential suggestion
        """
        self.get_new_user()
        ratings = np.zeros(T)
        cum_regret = np.zeros(T)
        movies = np.zeros(T, dtype='int64')
        to_watch = list(np.arange(self.n_movies, dtype='int64'))
        V = np.zeros((T, self.n_features))
        pref = []
        for t in tqdm(range(T)):
            movies[t] = self.movie_suggestion(t, V, pref, to_watch, burnin)
            ratings[t] = np.inner(self.real_theta, self.features[movies[t]]) + \
                         np.random.normal(scale=self.eta)
            cum_regret[t] = self.get_regret(ratings[t], to_watch)
            to_watch.remove(movies[t])
            V[t] = self.features[movies[t]]
            pref = self.update_pref(t, pref, ratings)
        cum_regret = cum_regret.cumsum()
        return {'movies_list': movies, 'all_ratings': ratings,
                'observed_preferences': pref, 'cumulative_regret': cum_regret}

    def gridsearch(self, K_list, sigma_list, n_expe, T, burnin):
        times = [int(burnin/2), burnin, T-1]
        results = {str(times[0]): {}, str(times[1]): {}, str(times[2]): {}}
        for K in tqdm(K_list):
            for sigma in tqdm(sigma_list):
                print(K, sigma)
                regret = np.zeros(T)
                for _ in range(n_expe):
                    self.K, self.sigma_IL = K, sigma
                    regret += self.GP_suggestion(T, burnin)['cumulative_regret']/n_expe
                for tx in times:
                    results[str(tx)][str((K, sigma))] = regret[tx]
        return results

    # Random suggestion
    def random_choice(self, T):
        ratings = np.zeros(T)
        cum_regret = np.zeros(T)
        to_watch = list(np.arange(self.n_movies))
        for t in range(T):
            a = np.random.choice(to_watch)
            ratings[t] = np.inner(self.real_theta, self.features[a])+ \
                       np.random.normal(scale=self.eta)
            cum_regret[t] = self.get_regret(ratings[t], to_watch)
            to_watch.remove(a)
        cum_regret = cum_regret.cumsum()
        return {'all_ratings': ratings, 'cumulative_regret': cum_regret}

    # Thompson Sampling
    def initPrior(self, a0=1, s0=10):
        mu_0 = a0 * np.ones(self.n_features)
        sigma_0 = s0 * np.eye(self.n_features)  # to adapt according to the true distribution of theta
        return mu_0, sigma_0

    def TS(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward, regret = np.zeros(T), np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        to_watch = list(np.arange(self.n_movies))
        for t in range(T):
            theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            a_t = to_watch[np.argmax(np.dot(self.features[to_watch], theta_t))]
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
            regret[t] = np.max(np.dot(self.features, self.real_theta))-reward[t]
            to_watch.remove(a_t)
        regret = regret.cumsum()
        return {'reward': reward, 'sequence of arm': arm_sequence, 'cumulative_regret': regret}

    def updatePosterior(self, a, mu, sigma):
        """
        Update posterior mean and covariance matrix
        :param arm: int, arm chose
        :param mu: np.array, posterior mean vector
        :param sigma: np.array, posterior covariance matrix
        :return: float and np.arrays, reward obtained with arm a, updated means and covariance matrix
        """
        f = self.features[a]
        r = np.inner(f, self.real_theta) + self.eta*np.random.normal()
        s_inv = np.linalg.inv(sigma)
        ffT = np.outer(f, f)
        mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.eta**2), np.dot(s_inv, mu) + r * f / self.eta**2)
        sigma_ = np.linalg.inv(s_inv + ffT/self.eta**2)
        return r, mu_, sigma_

    def LinUCB(self, T, lbda=1e-3, alpha=1e-1):
        """
        Implementation of Linear UCB algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param lbda: float, regression regularization parameter
        :param alpha: float, tunable parameter to control between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        to_watch = list(np.arange(self.n_movies))
        arm_sequence, reward, regret = np.zeros(T), np.zeros(T), np.zeros(T)
        a_t, b_t = np.random.randint(self.n_movies), np.zeros(self.n_features)
        to_watch.remove(a_t)
        A_t = lbda*np.eye(self.n_features)
        r_t = np.inner(self.features[a_t], self.real_theta) + self.eta*np.random.normal()
        for t in range(T):
            A_t += np.outer(self.features[a_t], self.features[a_t])
            b_t += r_t*self.features[a_t]
            inv_A = np.linalg.inv(A_t)
            theta_t = np.dot(inv_A, b_t)
            beta_t = alpha*np.sqrt(np.diagonal(np.dot(np.dot(self.features, inv_A), self.features.T)))
            a_t = to_watch[np.argmax((np.dot(self.features, theta_t)+beta_t)[to_watch])]
            r_t = np.inner(self.real_theta, self.features[a_t])+self.eta*np.random.normal()
            arm_sequence[t], reward[t] = a_t, r_t
            regret[t] = np.max(np.inner(self.features, self.real_theta)[to_watch])-r_t
            to_watch.remove(a_t)
        regret = regret.cumsum()
        return {'reward': reward, 'sequence of movies': arm_sequence, 'cumulative_regret': regret}


import matplotlib.pyplot as plt
if __name__ == '__main__':
    """
    TODO: * implement an epsilon-greedy policy which would choose the movie 
    with the closest feature vector to the best with probability 1-epsilon, else random choice
    * Check to introduce a covariance matrix in Thompson sampling or remove it
    """
    model = Movielens(K=1e-3, sigma_IL=1e-3)
    model.eta = 1.
    T = 60
    xp = model.GP_suggestion(T, burnin=30)['cumulative_regret']
    regret_random_policy = model.random_choice(T)['cumulative_regret']
    regret_TS = model.TS(T)['cumulative_regret']
    regret_LinUCB = model.LinUCB(T, lbda=1e-3, alpha=1e-1)['cumulative_regret']
    plt.plot(xp)
    plt.plot(regret_random_policy)
    plt.plot(regret_TS)
    plt.plot(regret_LinUCB)
    plt.show()
    # K_list = [1e-4, 1e-3, 1e-2, 0.1, 1.]
    # sigma_list = [1e-4, 1e-3, 1e-2, 0.1, 1.]
    # gs = model.gridsearch(K_list, sigma_list, 10, 50, 20)
    # print(gs)