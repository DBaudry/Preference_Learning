import Instance_learning as IL
import numpy as np
import input_data as data
import expe as xp

np.random.seed(42311)

if __name__ == '__main__':
    alpha = np.array([1/10.]*10)
    n, d, m, mp = 20, 10, 100, 200
    generator = data.instance_pref_generator(func=data.cobb_douglas, func_param=alpha)
    train, test = generator.sample_datasets(n, d, m, mp)
    K, sigma = 10., 0.1
    model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma)
    xp.xp_random_dataset(generator, model, train, test, K, sigma, gridsearch=False)
