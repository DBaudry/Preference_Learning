import Instance_learning as IL
import numpy as np
from tqdm import tqdm
from utils import gaussian_kernel
import matplotlib.pyplot as plt

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
            if t > burnin:
                to_watch.append(np.random.randint(self.n_movies))
        cum_regret = cum_regret.cumsum()
        return {'movies_list': movies, 'all_ratings': ratings,
                'observed_preferences': pref, 'cumulative_regret': cum_regret}

    def gridsearch(self, K_list, sigma_list, n_expe, T, burnin):
        """
        :param K_list: list of K parameters
        :param sigma_list: list of sigma_IL parameters
        :param n_expe: number of experiment to average results for each set of parameters
        :param T: time horizon
        :param burnin: length of the adaptive phase
        :return: dict with average cumulative regret for each set and different times
        """
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

    def random_choice(self, T):
        """
        Propose a Random movie at each time step
        :param T: Time Horizon
        :return: Dict of results (cumulative regret,...)
        """
        ratings = np.zeros(T)
        movies = np.zeros(T)
        cum_regret = np.zeros(T)
        to_watch = list(np.arange(self.n_movies))
        for t in tqdm(range(T)):
            a = np.random.choice(to_watch)
            ratings[t] = np.inner(self.real_theta, self.features[a])+ \
                       np.random.normal(scale=self.eta)
            cum_regret[t] = self.get_regret(ratings[t], to_watch)
            movies[t] = a
            to_watch.remove(a)
            to_watch.append(np.random.randint(self.n_movies))
        cum_regret = cum_regret.cumsum()
        return {'all_ratings': ratings, 'cumulative_regret': cum_regret}

    def epsilon_greedy(self, T, eps, t_explore):
        """
        Epsilon-greedy algorithm: the greedy policy is here to pick the movie in the list which is the closest
        to the user's best rated movie (in the sense of the distance between the features). This movie is
        chosen with probability 1-epsilon, and a random movie is selected with probability epsilon.
        Before t_explore a random movie is always selected
        :param T: Time Horizon
        :param eps: probability of picking a random movie after exploration step
        :return: Dict with all results
        """
        ratings = np.zeros(T)
        movies = np.zeros(T)
        cum_regret = np.zeros(T)
        to_watch = list(np.arange(self.n_movies))
        for t in tqdm(range(T)):
            u = np.random.uniform()
            if t >= t_explore and u < 1-eps:
                best_feature = self.features[int(movies[np.argmax(ratings[:t])])]
                uniques = np.unique(to_watch)
                feature_set = self.features[uniques]
                idx = np.argmax((feature_set-best_feature).sum(axis=1)**2)
                a = uniques[idx]
            else:
                a = np.random.choice(to_watch)
            ratings[t] = np.inner(self.real_theta, self.features[a]) + \
                       np.random.normal(scale=self.eta)
            cum_regret[t] = self.get_regret(ratings[t], to_watch)
            movies[t] = a
            to_watch.remove(a)
            to_watch.append(np.random.randint(self.n_movies))
        cum_regret = cum_regret.cumsum()
        return {'all_ratings': ratings, 'cumulative_regret': cum_regret}

    def LinUCB(self, T, lbda=1e-3, alpha=1e-1):
        """
        Implementation of Linear UCB algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param lbda: float, regression regularization parameter
        :param alpha: float, tunable parameter to control between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        to_watch = list(np.arange(self.n_movies))
        movies, reward, regret = np.zeros(T), np.zeros(T), np.zeros(T)
        a_t, b_t = np.random.randint(self.n_movies), np.zeros(self.n_features)
        to_watch.remove(a_t)
        A_t = lbda*np.eye(self.n_features)
        r_t = np.inner(self.features[a_t], self.real_theta) + self.eta*np.random.normal()
        for t in tqdm(range(T)):
            A_t += np.outer(self.features[a_t], self.features[a_t])
            b_t += r_t*self.features[a_t]
            inv_A = np.linalg.inv(A_t)
            theta_t = np.dot(inv_A, b_t)
            beta_t = alpha*np.sqrt(np.diagonal(np.dot(np.dot(self.features, inv_A), self.features.T)))
            a_t = to_watch[np.argmax((np.dot(self.features, theta_t)+beta_t)[to_watch])]
            r_t = np.inner(self.real_theta, self.features[a_t])+self.eta*np.random.normal()
            movies[t], reward[t] = a_t, r_t
            regret[t] = np.max(np.inner(self.features, self.real_theta)[to_watch])-r_t
            to_watch.remove(a_t)
            to_watch.append(np.random.randint(self.n_movies))
        regret = regret.cumsum()
        return {'reward': reward, 'sequence of movies': movies, 'cumulative_regret': regret}

    def xp(self, T, n_algo, n_expe, burnin=30, eps=(0.4, 0.7, 0.9), t_exp=20,
           ucb_param=(1e-3, 0.1), q=(0.025, 0.975), plot=True):
        results = np.zeros((n_algo, n_expe, T))
        for n in range(n_expe):
            self.get_new_user()
            results[2, n] = self.GP_suggestion(T, burnin=burnin)['cumulative_regret']
            results[0, n] = self.random_choice(T)['cumulative_regret']
            results[1, n] = model.LinUCB(T, lbda=ucb_param[0], alpha=ucb_param[1])['cumulative_regret']
            for i, e in enumerate(eps):
                results[3+i, n] = self.epsilon_greedy(T, e, t_exp)['cumulative_regret']
        means = results.mean(axis=1)
        q = np.quantile(results, q, axis=1)
        if plot:
            names = ['Random Policy', 'Linear UCB', 'GP model'] + ['Epsilon Greedy '+str(e) for e in eps]
            color = ['y', 'tomato', 'steelblue']
            color_fill = ['palegoldenrod', 'peachpuff', 'lightblue']
            for i in range(n_algo):
                if i < len(color):
                    plt.plot(means[i], label=names[i], color=color[i])
                    plt.fill_between(np.arange(T), q[0, i], q[1, i], color=color_fill[i])
                else:
                    plt.plot(means[i], label=names[i])
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Time-Horizon')
            plt.ylabel('Cumulative regret (log-scale')
            plt.show()
            for i in range(n_algo):
                if i < len(color):
                    plt.plot(means[i], label=names[i], color=color[i])
                    plt.fill_between(np.arange(T), q[0, i], q[1, i], color=color_fill[i])
                else:
                    plt.plot(means[i], label=names[i])
            plt.legend()
            plt.xlabel('Time-Horizon')
            plt.ylabel('Cumulative regret')
            plt.show()


if __name__ == '__main__':
    model = Movielens(K=1e-5, sigma_IL=1e-2)
    model.eta = 1.
    T = 100
    n_algo = 5
    n_expe = 1000
    model.xp(T, n_algo, n_expe, burnin=20, eps=(0.4, 0.8), t_exp=20,
             q=(0.05, 0.95), ucb_param=(1e-3, 0.1), plot=True)
    # K_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.]
    # sigma_list = [1e-4, 1e-3, 1e-2, 0.1, 1.]
    # gs = model.gridsearch(K_list, sigma_list, 10, 50, 20)
    # print(gs)