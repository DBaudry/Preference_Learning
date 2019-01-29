import Instance_learning as IL
import numpy as np
from tqdm import tqdm
from utils import gaussian_kernel

np.random.seed(211)

class Movielens():
    def __init__(self, n_actions=207, eta=1, sigma=10, K=1/30, sigma_IL=1.):
        self.eta = eta
        self.sigma = sigma
        self.features = np.loadtxt('Data/Vt.csv', delimiter=',').T
        self.n_movies, self.n_features = self.features.shape
        self.real_theta = np.empty(self.n_features)
        self.K = K
        self.sigma_IL = sigma_IL

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

    def movie_suggestion(self, t, v, m, to_watch):
        """
        :param t: current time
        :param v: features dataset, in order of movies watched
        :param m: current known preferences
        :param to_watch: movies that have not been tested yet
        :return: indice of the suggestion in the original feature dataset
        """
        if t < 3:
            return int(np.random.choice(to_watch))
        else:
            instances = v[:t]
            model = IL.learning_instance_preference(inputs=[instances, m], K=self.K, sigma=self.sigma_IL)
            model.tol = 1e-2
            MAP = model.compute_MAP(np.zeros(t))['x']
            test_instance = self.features[to_watch]
            return to_watch[int(self.predict_movie(model, instances, test_instance, MAP))]

    def sequential_suggestion(self, T):
        """
        :param T: Time Horizon
        :return: Rewards of the sequential suggestion
        """
        self.get_new_user()
        ratings = np.zeros(T)
        movies = np.zeros(T, dtype='int64')
        to_watch = list(np.arange(self.n_movies, dtype='int64'))
        V = np.zeros((T, self.n_features))
        pref = []
        for t in tqdm(range(T)):
            movies[t] = self.movie_suggestion(t, V, pref, to_watch)
            to_watch.remove(movies[t])
            V[t] = self.features[movies[t]]
            ratings[t] = np.inner(self.real_theta, self.features[movies[t]]) + \
                         np.random.normal(scale=self.eta)
            pref = self.update_pref(t, pref, ratings)
        return {'movies_list': movies, 'all_ratings': ratings,
                'observed_preferences': pref}


if __name__ == '__main__':
    model = Movielens()
    print(model.sequential_suggestion(50))