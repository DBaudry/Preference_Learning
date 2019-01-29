import Label_learning as LL
import numpy as np
import input_data as data
import expe as xp
import utils

check_random = False
check_label = True

mapping_multiclf = {'sushia': True, 'sushib': True, 'movies': False, 'german2005': False, 'german2009': False,
                    'algae': False, 'dna': True, 'letter': True, 'mnist': True, 'satimage': True, 'segment': True,
                    'usps': True, 'waveform': True}

mapping_n_labels = {'sushia': 10, 'sushib': 100, 'movies': 7, 'german2005': 5, 'german2009': 5,
                    'algae': 7, 'dna': 3, 'letter': 26, 'mnist': 10, 'satimage': 6, 'segment': 7,
                    'usps': 10, 'waveform': 3}

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
        n = 60
        dataset = 'dna'
        m = mapping_multiclf[dataset]
        users, graphs, classes = utils.read_data_LL(dataset, n, mutliclf=m)
        train, test = utils.train_test_split(users, graphs, classes)
        K, sigma = np.ones(mapping_n_labels[dataset])*1, 0.1
        model = LL.learning_label_preference(inputs=train, K=K, sigma=sigma)
        xp.run_label_xp(0, model, train, test, K, sigma, gridsearch=False)
