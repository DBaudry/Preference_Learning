import Label_learning as LL
import numpy as np
import input_data as data
import expe as xp
import utils

check_random = False
check_label = False
check_authors_expe = True


if __name__ == '__main__':
    if check_random:
        n, d, m, n_label = 20, 5, 6, 4
        label_func = [data.cobb_douglas for l in range(n_label)]
        rho = 0.9
        alpha = [utils.get_alpha(d) for _ in range(n_label)]
        generator = data.label_pref_generator(func=label_func, func_param=alpha)
        train = generator.generate_X_pref(n, m, d)
        generator.n, generator.m = 150, 200
        test = generator.generate_X_pref(n, m, d)
        K, sigma = np.arange(n_label)+1., 0.1
        model = LL.learning_label_preference(inputs=train, K=K, sigma=sigma)
        xp.run_label_xp(generator, model, train, test, K, sigma, gridsearch=False)

    if check_label:
        n = 100
        dataset = 'waveform'
        users, graphs, classes = utils.read_data_LL(dataset, n)
        train, test = utils.train_test_split(users, graphs, classes)
        K0, sigma0 = 1, 1
        K, sigma = np.ones(utils.mapping_n_labels[dataset])*K0, sigma0
        model = LL.learning_label_preference(inputs=train, K=K, sigma=sigma, print_callback=True)
        xp.run_label_xp(model, train, test, K, sigma, show_results=False, gridsearch=False, showgraph=True, user=(5, 6))

    if check_authors_expe:
        datasets = ['dna']
        n_expe = 1
        xp.run_label_xp_authors(n_expe, datasets, param=[[0.01, 0.1, 1], [0.01, 0.1, 1]], show_results=False,
                                showgraph=False, print_callback=False)

