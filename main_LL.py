import Label_learning as LL
import numpy as np
import input_data as data
import expe as xp
import utils

if __name__ == '__main__':
    n, d, m, n_label = 10, 5, 12, 8
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
