import numpy as np
import input_data as data
import Instance_learning as IL
import expe as xp
from SVM_IL import *
import utils

#np.random.seed(42311)

check_random = False
check_real_data = True
check_authors_expe = False
ckeck_label = False


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
        n_obs, n_features = 200, -1  # put to -1, -1 if you want the whole data set
        n_pref_train, n_pref_test = 700, 20000
        generator = data.pref_generator('housing', n_obs, n_features)
        #train = generator.get_input_train(n_pref_train)
        #test = generator.get_input_test(n_pref_test)
        #K, sigma = 10., 0.1  # Best parameters with the grid below
        # K, sigma = [0.1, 1., 5., 10.], [0.01, 0.1, 1.]
        SVMHerb, SVMHar = [], []
        for _ in range(20):
            train = generator.get_input_train(n_pref_train)
            test = generator.get_input_test(n_pref_test)
            K, C = [10 ** i for i in range(-2, 5)], [10 ** i for i in range(-3, 4)]
            # K, C = 10, 0.01
            learnerHerb = SVM_InstancePref(train, K, C)
            learnerHar = CCSVM(train, K, C)
            learnerHerb.fit()
            learnerHar.fit()
            SVMHerb.append(learnerHerb.score(test[0], test[1], train=False))
            SVMHar.append(learnerHar.score(test[0], test[1], train=False))
        print(np.mean(SVMHar), np.std(SVMHar))
        print(np.mean(SVMHerb), np.std(SVMHerb))
        # model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma, print_callback=True)
        # xp.run_instance_xp(generator, model, train, test, K, sigma, gridsearch=False, show_results=True)
        
    if check_authors_expe:
        datasets = ['pyrim']  # ['pyrim', 'triazines', 'machine', 'housing']
        n_expe = 1
        # If you want to use the best parameters computed using gridsearching you should write param='best'.
        # If not, rather to use your own values or list of values for gridsearching, write param = [., .]
        # or param = [[., ., .], [., ., .]] (list of 2 lists)
        # For gridsearch method you can start with
        # param=[[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]] and then refine according
        # to the best values returned
        param = [[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                 [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]]
        xp.run_instance_xp_authors(param=param, n_expe=n_expe, datasets=datasets, show_results=True, print_callback=False)

    if check_label:
        users, pref, graphs = utils.read_sushi('b')
        train, test = utils.train_test_split_sushi(users, graphs)
        utils.pipeline_graph(data=pref, user=1, mode='compute_linear_edges')
        utils.pipeline_graph(data=pref, user=1, mode='compute_all_edges')
