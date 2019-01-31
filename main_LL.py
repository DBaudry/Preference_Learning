import Label_learning as LL
import input_data as data
import expe as xp
import utils
import numpy as np
import itertools

check_random = False
check_label = False
check_authors_expe = True
check_SVM = False

np.random.seed(42312)

if __name__ == '__main__':
    if check_random:
        # Check_random aims at running Label Preference model on a simulated data set.
        # - n          : int, number of observations from each preferences are generated
        # - d          : int, number of variables
        # - m          : int, number of preferences possible for training
        # - n_label    : int, number of labels
        n, d, m, n_label = 20, 5, 6, 4
        alpha = [utils.get_alpha(d) for _ in range(n_label)]
        label_func = [data.cobb_douglas for l in range(n_label)]
        generator = data.label_pref_generator(func=label_func, func_param=alpha)
        train = generator.generate_X_pref(n, m, d)
        # Set number of observations and preferences for testing
        generator.n, generator.m = 150, 200
        test = generator.generate_X_pref(n, m, d)
        K, sigma = np.arange(n_label)+1., 0.1
        model = LL.learning_label_preference(inputs=train, K=K, sigma=sigma)
        xp.run_label_xp(generator, model, train, test, K, sigma, gridsearch=False, random=True)

    if check_label:
        # Check_real_data does the same thing as before but with real data sets, namely data sets in Data folder.
        # - n : int, number of observations or users
        # - dataset : string, which dataset to use in 'dna', 'waveform', 'satimage', 'segment', 'usps', 'sushia',
        # 'sushib', 'movies', 'algae', 'german2005', 'german2009'
        n = 100
        dataset = 'segment'
        users, graphs, classes = utils.read_data_LL(dataset, n)
        train, test = utils.train_test_split(users, graphs, classes)
        # Set K0 and sigma0 (hyper parameters)
        K0, sigma0 = 0.01, 1
        K, sigma = np.ones(utils.mapping_n_labels[dataset])*K0, sigma0
        model = LL.learning_label_preference(inputs=train, K=K, sigma=sigma, print_callback=True)
        # showgraph shows real and predicted preferences graphs for 2 users (5, 6) by default
        xp.run_label_xp(None, model, train, test, K, sigma, show_results=False, gridsearch=False, showgraph=True, user=(5, 6))

    if check_authors_expe:
        datasets = ['waveform']  # ['dna', 'waveform', 'satimage', 'segment', 'usps']
        n_expe = 3
        # For gridsearch method you can start with
        #param = [[j*10**i for (i,j) in itertools.product(range(-5, 2), [1, 2, 5, 7])],
        #         [j * 10 ** i for (i, j) in itertools.product(range(-4, 2), [1, 2, 5, 7])]]
        # If you want to use best parameters set param = 'best'
        param = 'best'
        xp.run_label_xp_authors(n_expe, datasets, param=param, show_results=False, showgraph=True,
                                print_callback=True)

    if check_SVM:
        # for the SVM based methods
        datasets = ['waveform']  # ['dna', 'waveform', 'satimage', 'segment', 'usps']
        n_expe = 3
        C, K = np.exp(np.arange(-2, 5)*np.log(10)), np.exp(np.arange(-3, 4)*np.log(10))
        xp.run_label_xp_authors_SVM(datasets, n_expe=n_expe, K=K, C=C)

