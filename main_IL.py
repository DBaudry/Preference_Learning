import numpy as np
import input_data as data
import Instance_learning as IL
import expe as xp


np.random.seed(42311)

check_random = False
check_real_data = False
check_authors_expe = True

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
        xp.run_instance_xp(generator, model, train, test, K, sigma,
                           gridsearch=False, show_results=True)
    if check_authors_expe:
        datasets = ['pyrim', 'triazines', 'machine', 'housing']
        n_expe = 1
        best_param = True
        K, sigma = (None, None) if best_param else (1, 1)
        xp.run_instance_xp_authors(best_param=True, K=K, sigma=sigma, n_expe=n_expe, datasets=datasets, gridsearch=False, show_results=False)