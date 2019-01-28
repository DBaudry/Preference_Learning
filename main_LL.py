import Label_learning as LL
import numpy as np
import input_data as data
import expe as xp
import utils

check_random = False
check_label = True

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
        users, pref, graphs = utils.read_sushi('b')
        train, test = utils.train_test_split_sushi(users, graphs)
        print(train[0])