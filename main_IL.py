import numpy as np
import input_data as data
import Instance_learning as IL
import expe as xp

np.random.seed(42311)

check_random = False
check_real_data = False
check_authors_expe = True
check_SVM = False

if __name__ == '__main__':
    if check_random:
        # Check_random aims at running Instance Preference model on a simulated data set.
        # - alpha : np.array, corresponds to the coefficient for the Cobb Douglas utility function
        # - n     : int, number of observations from each preferences are generated
        # - d     : int, number of variables
        # - m     : int, number of preferences for training
        # - mp    : int, number of preferences for testing
        alpha = np.array([1 / 10.] * 10)
        n, d, m, mp = 20, 10, 100, 200
        generator = data.instance_pref_generator(func=data.cobb_douglas, func_param=alpha)
        train, test = generator.sample_datasets(n, d, m, mp)
        # If you want to do gridsearching, please set K = [., ., .] and sigma = [., ., .] with values you want
        # to explore. Add gridsearch = True
        gridsearch = False
        K, sigma = 10., 0.1
        model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma)
        xp.run_instance_xp(generator, model, train, test, K, sigma, gridsearch=gridsearch)

    if check_real_data:
        # Check_real_data does the same thing as before but with real data sets, namely data sets in Data folder.
        # Data is not simulated. Parameters keep their previous meaning.
        n, d = 10, -1  # put to -1, -1 if you want the whole data set
        m, mp = 20, 500
        generator = data.pref_generator('pyrim', n, d)
        train, test = generator.get_input_train(m), generator.get_input_test(mp)
        K, sigma = 10., 0.1
        # K, sigma = [0.1, 1., 5., 10.], [0.01, 0.1, 1.]  # if you want to make a gridsearch
        # Show_results controls if you want to print various quantities for better understanding of the model output.
        # print_callback controls if you want to print gradient descent evolution (can be messy but useful to have
        # an estimation of time remaining).
        gridsearch, show_results, print_callback = False, True, True
        model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma, print_callback=print_callback)
        xp.run_instance_xp(generator, model, train, test, K, sigma, gridsearch=gridsearch, show_results=show_results)

    if check_authors_expe:
        # We reproduce here the experiments of the authors. Namely, we make 20 independent experiments of Instance
        # Learning Model on 5 datasets : 'machine', 'pyrim', 'triazines', 'housing', 'abalone'. We use the same number
        # of preferences for training and testing.
        datasets = ['pyrim']  # ['pyrim', 'triazines', 'machine', 'housing', 'abalone']
        n_expe = 3
        # If you want to use the best parameters computed using gridsearching you should write param='best'.
        # If not, rather to use your own values or list of values for gridsearching, write param = [., .]
        # or param = [[., ., .], [., ., .]] (list of 2 lists)
        # For gridsearch method you can start with
        # param=[[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]] and then refine according
        # to the best values returned
        param = 'best'
        xp.run_instance_xp_authors(param=param, n_expe=n_expe, datasets=datasets, show_results=False,
                                   print_callback=True)

    if check_SVM:
        # for the SVM-based experiments
        datasets = ['pyrim']  # ['pyrim', 'triazines', 'machine', 'housing', 'abalone']
        n_expe = 3
        C, K = np.exp(np.arange(-2, 5)*np.log(10)), np.exp(np.arange(-3, 4)*np.log(10))
        xp.run_instance_xp_authors_SVM(datasets, n_expe=n_expe, K=K, C=C)
