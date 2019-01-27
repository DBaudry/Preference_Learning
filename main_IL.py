import Instance_learning as IL
import numpy as np
import input_data as data
import expe as xp
import pickle as pkl
from tqdm import tqdm
from utils import ratio_n_obs

dataset_shapes = {'abalone': (4177, 9), 'diabetes': (43, 3), 'housing': (506, 14),
                  'machine': (209, 7), 'pyrim': (74, 28), 'r_wpbc': (194, 33), 'triazines': (186, 61)}

authors_n_pref = {'housing': 700} #'pyrim': 100, 'triazines': 300 , 'machine':500, 'housing': 700, 'abalone': 1000}

best_parameters = {'pyrim': (0.005, 0.007), 'triazines': (0.007, 0.006), 'machine': 500, 'housing': 700, 'abalone': 1000}

ratios = {'pyrim': 5.23, 'triazines': 7.59, 'machine': 6.60, 'housing': 13.52, 'abalone': 93.40}

np.random.seed(42311)

check_random = False
check_real_data = True
check_authors_expe, n_expe = False, 2

if __name__ == '__main__':
    if check_random:
        alpha = np.array([1/10.]*10)
        n, d, m, mp = 20, 10, 100, 200
        generator = data.instance_pref_generator(func=data.cobb_douglas, func_param=alpha)
        train, test = generator.sample_datasets(n, d, m, mp)
        K, sigma = 10., 0.1
        model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma)
        xp.run_instance_xp(generator, model, train, test, K, sigma, gridsearch=False)

    if check_real_data:
        n_obs, n_features = 20, -1  # put to -1, -1 if you want the whole data set
        n_pref_train, n_pref_test = 400, 200
        generator = data.pref_generator('housing', n_obs, n_features)
        train = generator.get_input_train(n_pref_train)
        test = generator.get_input_test(n_pref_test)
        K, sigma = 10., 0.1  # Best parameters with the grid below
        # K, sigma = [0.1, 1., 5., 10.], [0.01, 0.1, 1.]
        model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma)
        xp.run_instance_xp(generator, model, train, test, K, sigma, gridsearch=False, show_results=True)
    if check_authors_expe:
        K, sigma = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        results = {}
        for i, m in tqdm(enumerate(authors_n_pref.keys()), desc='Running experiments on 5 datasets', total=5):
            #K, sigma = best_parameters[m]
            n, d = dataset_shapes[m]
            n_obs, n_features = ratio_n_obs(authors_n_pref[m]), -1
            n_pref_train, n_pref_test = authors_n_pref[m], 20000
            score_train, score_test = [], []
            for expe in range(n_expe):
                generator = data.pref_generator(m, n_obs, n_features)
                train, test = generator.get_input_train(n_pref_train), generator.get_input_test(n_pref_test)
                model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma)
                results = xp.xp_random_dataset(generator, model, train, test, K, sigma, gridsearch=True, show_results=False)
                score_train.append(1-results['score_train'])
                score_test.append(1-results['score_test'])
            m_train, std_train, m_test, std_test = np.mean(score_train), np.std(score_train), np.mean(score_test), np.std(score_test)
            print('Data set '+m+' : Mean error on train {:0.4f} ± {:0.3f}, mean error on test {:0.4f} ± {:0.3f}'.
                  format(m_train, std_train, m_test, std_test))
            results[m] = m_train, std_train, m_test, std_test
        pkl.dump(results, open('results.pkl', 'wb'))
