import numpy as np
from tqdm import tqdm
import pickle as pkl
import input_data as data
import Instance_learning as IL
import Label_learning as LL
import utils
import matplotlib.pyplot as plt
from SVM_IL import *
from SVM_LL import CCSVM_LL


def run_instance_xp(gen, model, train, test, K, sigma, gridsearch=False, show_results=True):
    """ Instance Learning:
    :param gen: instance_pref_generator object
    :param model: learning_instance_preference object
    :param train: list, training set : data (numpy.array) + preferences (list of tuples)
    :param test: list, testing set : data (numpy.array) + preferences (list of tuples)
    :param K: float, kernel parameter (list of floats if gridsearch)
    :param sigma: float, error variance (list of floats if gridsearch)
    :param gridsearch: Bool, do gridsearch to find the best parameters for K and sigma based on the model evidence
    :param show_results: Bool, display the results of interest
    :return: dict: - MAP obtained with the training set
                   - Accuracy on preferences predictions on the training set
                   - Accuracy on preferences predictions on the testing set
                   - Evidence
                   - Predictions on the testing set
    """
    train_pref = gen.get_true_pref(train[0])
    y0 = np.zeros(model.n)
    if not gridsearch:
        MAP = model.compute_MAP(y0)
    else:
        MAP = model.compute_MAP_with_gridsearch(y0, K, sigma)
    evidence = model.evidence_approx(MAP['x'])
    pref_ap = model.get_train_pref(MAP['x'])
    score_train = model.score(pref_ap, train_pref)
    score, proba = model.predict(test, MAP['x'])
    if show_results:
        print('Convergence of the minimizer of S : {}'.format(MAP['success']))
        print('Maximum a Posteriori : {}'.format(MAP['x']))
        print('Score on train: {}'.format(score_train))
        print('Evidence Approximation (p(D|f_MAP)) : {}'.format(evidence))
        print('Probabilities : {}'.format(proba))
        print('Score on test: {}'.format(score))
    else:
        print('Error on train: {:0.4f}, error on test: {:0.4f}'.format(1-score_train, 1-score))
    return {'MAP': MAP, 'score_train': score_train, 'evidence': evidence, 'proba_test': proba, 'score_test': score}


def run_instance_xp_authors(n_expe, datasets, param='best', show_results=False, print_callback=False):
    """ Instance Learning:
    Reproduce here the experiments of the authors. Namely, we make 20 (or more) independent experiments of Instance
    Learning Model on 5 datasets : 'machine', 'pyrim', 'triazines', 'housing', 'abalone'. We use the same number
    of preferences for training and testing.
    :param n_expe: int, number of experiments
    :param datasets: list of string, list of data sets
    :param param: tuple, list of 2 lists (for gridsearch) or 'best' (to use best parameters)
    :param show_results: Bool, display the results of interest
    :param print_callback: Bool, display the evolution of gradient descent
    :return: dict with mean errors and standard deviations computed on the n_expe experiments.
    """
    results = {}
    l = len(datasets)
    gridsearch = utils.gridsearchBool(param)
    for i, m in tqdm(enumerate(datasets), desc='Running experiments on ' + str(l) +' datasets', total=l):
        b = utils.best_parameters[m]
        K0 = b[0] if param == 'best' else param[0]
        sigma0 = b[1] if param == 'best' else param[1]
        n_obs, n_features = utils.ratio_n_obs(utils.authors_n_pref[m]), -1
        n_pref_train, n_pref_test = utils.authors_n_pref[m], 20000
        score_train, score_test = [], []
        for expe in range(n_expe):
            generator = data.pref_generator(m, n_obs, n_features)
            train, test = generator.get_input_train(n_pref_train), generator.get_input_test(n_pref_test)
            model = IL.learning_instance_preference(inputs=train, K=K0, sigma=sigma0, print_callback=print_callback)
            results = run_instance_xp(generator, model, train, test, K0, sigma0, gridsearch=gridsearch,
                                      show_results=show_results)
            score_train.append(1 - results['score_train'])
            score_test.append(1 - results['score_test'])
        m_train, std_train = np.nanmean(score_train), np.nanstd(score_train)
        m_test, std_test = np.nanmean(score_test), np.nanstd(score_test)
        print('Data set ' + m + ' : Mean error on train {:0.4f} ± {:0.3f}, mean error on test {:0.4f} ± {:0.3f}'.
              format(m_train, std_train, m_test, std_test))
        results[m] = m_train, std_train, m_test, std_test
    pkl.dump(results, open('results_IL.pkl', 'wb'))  # To save results
    return results


def run_instance_xp_authors_SVM(datasets, n_expe=20, K=10, C=1):
    """
    :param datasets: list of strings which are the name of the datasets studied.
    :param n_expe: The number of repeated experience on the datasets.
    :param K: Gamma parameter for the SVM
    :param C: Complexity parameter for the SVM
    This function applies repeatedly the algorithm inspired from the methodology in the articles from Herbrich et al.
    and Har-Peled et al.on a set of datasets, in an instance learning problem.
    """
    results = {}
    l = len(datasets)
    for i, m in tqdm(enumerate(datasets), desc='Running experiments on '+ str(l) +' datasets', total=l):
        n_obs, n_features = utils.ratio_n_obs(utils.authors_n_pref[m]), -1
        n_pref_train, n_pref_test = utils.authors_n_pref[m], 20000
        score_trainHerb, score_testHerb = [], []
        score_trainHar, score_testHar = [], []
        for expe in range(n_expe):
            generator = data.pref_generator(m, n_obs, n_features)
            train, test = generator.get_input_train(n_pref_train), generator.get_input_test(n_pref_test)
            learnerHerb = SVM_InstancePref(train, K, C)
            learnerHar = CCSVM_IL(train, K, C)
            learnerHerb.fit()
            learnerHar.fit()
            score_testHerb.append(1 -learnerHerb.score(test[0], test[1], train=False))
            score_testHar.append(1 - learnerHar.score(test[0], test[1], train=False))
            score_trainHerb.append(1 -learnerHerb.score())
            score_trainHar.append(1 - learnerHar.score())
            print('Har-Peled: Error on train: {:0.4f}, error on test: {:0.4f}'.
                  format(score_trainHar[-1], score_testHar[-1]))
            print('Herbrich: Error on train: {:0.4f}, error on test: {:0.4f}'.
                  format(score_trainHerb[-1], score_testHerb[-1]))

        m_trainHerb, std_trainHerb, m_testHerb, std_testHerb = np.mean(score_trainHerb), np.std(score_trainHerb), \
                                                               np.mean(score_testHerb), np.std(score_testHerb)
        m_trainHar, std_trainHar, m_testHar, std_testHar = np.mean(score_trainHar), np.std(score_trainHar),\
                                                           np.mean(score_testHar), np.std(score_testHar)
        print('Data set ' + m +
              ' (Herbrich): Mean error on train {:0.4f} ± {:0.3f}, mean error on test {:0.4f} ± {:0.3f}'.
              format(m_trainHerb, std_trainHerb, m_testHerb, std_testHerb))
        print('Data set ' + m +
              ' (Har-Peled): Mean error on train {:0.4f} ± {:0.3f}, mean error on test {:0.4f} ± {:0.3f}'.
              format(m_trainHar, std_trainHar, m_testHar, std_testHar))
        results[m] = {'Herbrich': [m_trainHerb, std_trainHerb, m_testHerb, std_testHerb],
                      'Herbrich': [m_trainHar, std_trainHar, m_testHar, std_testHar]}
    pkl.dump(results, open('resultsSVM.pkl', 'wb'))


def run_label_xp(gen, model, train, test, K, sigma, show_results=True, gridsearch=False, showgraph=False, user=(0,1), random=False):
    """ Label Learning:
    :param gen: label_pref_generator object
    :param model: learning_label_preference object
    :param train: list, training set : data (array) + preferences graphs (list of list of tuples) + classes (array)
    :param test:  list, testing set : data (array) + preferences graphs (list of list of tuples) + classes (array)
    :param K: float, kernel parameter (list of floats if gridsearch)
    :param sigma: float, error variance (list of floats if gridsearch)
    :param show_results: Bool, display the results of interest
    :param gridsearch: Bool, do gridsearch to find the best parameters for K and sigma based on the model evidence
    :param showgraph: Bool, showgraph shows real and predicted preferences graphs for 2 users (0, 1) by default
    :param user: tuple, 2 users
    :return: dict: - MAP obtained with the training set
                   - Evidence
                   - Predictions on training set
                   - Predictions on testing set
                   - Label error on train (number of predicted 'best' item/classes different from true ones)
                   - Label error on test
                   - Pref error on train (number of edges from true graph which belong to predicted graph)
                   - Pref error on test
    """
    y0 = np.zeros(model.n * model.n_labels)
    if not gridsearch:
        MAP = model.compute_MAP(y0)
    else:
        MAP = model.compute_MAP_with_gridsearch(y0, K, sigma)
    evidence = model.evidence_approx(MAP['x'])
    pred_train, pred_test = model.predict(train[0], MAP['x']), model.predict(test[0], MAP['x'])
    label_score_train = model.label_score_rate(train[2], pred_train) if not random else np.nan
    label_score_test = model.label_score_rate(test[2], pred_test) if not random else np.nan
    pref_score_train = model.label_pref_rate(train[1], pred_train)
    pref_score_test = model.label_pref_rate(test[1], pred_test)
    if show_results:
        print('Convergence of the minimizer of S : {}'.format(MAP['success']))
        print('Maximum a Posteriori : {}'.format(MAP['x']))
        print('Evidence Approximation (p(D|f_MAP)) : {}'.format(evidence))
        if not random:
            print('True best classes on train : {}'.format(train[2]))
            print('True best classes on test : {}'.format(test[2]))
        print('Best classes predicted on train : {}'.format(np.argsort(pred_train, axis=1)[:, -1]))
        print('Best classes predicted on test : {}'.format(np.argsort(pred_test, axis=1)[:, -1]))
    print('Mean label error on train: {:0.4f}, and test: {:0.4f}'.format(1 - label_score_train, 1 - label_score_test))
    print('Mean pref error on train: {:0.4f}, and test: {:0.4f}'.format(1 - pref_score_train, 1 - pref_score_test))
    if showgraph:
        graph1, graph2 = np.array(pred_train[user[0]]).argsort()[::-1], np.array(pred_train[user[1]]).argsort()[::-1]
        plt.subplot(2, 2, 1)
        utils.pipeline_graph(train[1][user[0]], 'True preferences of user ' + str(user[0]))
        plt.subplot(2, 2, 2)
        utils.pipeline_graph(utils.compute_all_edges(graph1), 'Predicted preferences of user ' + str(user[0]))
        plt.subplot(2, 2, 3)
        utils.pipeline_graph(train[1][user[1]], 'True preferences of user ' + str(user[1]))
        plt.subplot(2, 2, 4)
        utils.pipeline_graph(utils.compute_all_edges(graph2), 'Predicted preferences of user ' + str(user[1]))
        plt.show()
    return {'MAP': MAP, 'evidence': evidence, 'predictions train': pred_train, 'predictions test': pred_test,
            'Label error train': 1 - label_score_train, 'Label error test': 1 - label_score_test,
            'Pref error train': 1 - pref_score_train, 'Pref error test': 1 - pref_score_test}


def run_label_xp_authors(n_expe, datasets, param='best', show_results=False, showgraph=False, print_callback=False):
    """ Label Learning:
        Reproduce here the experiments of the authors. Namely, we make 20 (or more) independent experiments of Label
        Learning Model on 5 datasets : 'dna', 'waveform', 'satimage', 'segment', 'usps'. These data sets are based
        on a multi classification task. For example, 'dna' dataset contains 3 labels (0, 1, 2). If a line has a target
        equal to 0 then it means that 0 is the target to predict and we can construct a graph as in figure 1.a, namely
        0 is preferred to 1 and 0 is preferred to 2, which leads to graph (0,1), (0,2). We expect the algorithm
        to predict this graph and also the class.
        The way those graphs are drawn is not very relevant as the target variable is, for example with waveform, a
        class of waveform. You can also use more realistic data sets which have been obtained based on real data
        and surveys. This is the case of the data sets : 'sushia', 'sushib', 'movies', 'german2005', 'german2009',
        'algae'. Here the metric 'label_error_rate' is not as relevant as before as there are often items that
        are commonly preferred to others. Typically, movies 1 and 2 are preferred to 3 and 4 but we don't have any
        information about preference between 1 and 2.
        :param n_expe: int, number of experiments
        :param datasets: list of string, list of data sets
        :param param: tuple, list of 2 lists (for gridsearch) or 'best' (to use best parameters)
        :param show_results: Bool, display the results of interest
        :param showgraph: Bool, showgraph shows real and predicted preferences graphs for 2 users (5, 6) by default
        :param print_callback: Bool, display the evolution of gradient descent
        :return: dict with mean errors and standard deviations computed on the n_expe experiments (label error and
        pref error)
        """
    results = {}
    gridsearch = utils.gridsearchBool(param)
    l = len(datasets)
    for i, m in tqdm(enumerate(datasets), desc='Running experiments on ' + str(l) +' datasets', total=l):
        b = utils.best_parameters[m]
        n_obs = 800
        if param == 'best' and not gridsearch:
            K, sigma = np.ones(utils.mapping_n_labels[m]) * b[0], b[1]
        elif param != 'best' and not gridsearch:
            K, sigma = np.ones(utils.mapping_n_labels[m]) * param[0], param[1]
        else:
            K, sigma = [np.ones(utils.mapping_n_labels[m]) * param[0][i] for i in range(len(param[0]))], param[1]
        label_error_train, label_error_test, pref_error_train, pref_error_test = [], [], [], []
        for expe in range(n_expe):
            users, graphs, classes = utils.read_data_LL(m, n_obs)
            train, test = utils.train_test_split(users, graphs, classes)
            model = LL.learning_label_preference(inputs=train, K=K, sigma=sigma, print_callback=print_callback)
            r = run_label_xp(None, model, train, test, K, sigma, show_results=show_results, gridsearch=gridsearch,
                             showgraph=showgraph, user=(5, 6))
            label_error_train.append(r['Label error train'])
            pref_error_train.append(r['Pref error train'])
            label_error_test.append(r['Label error test'])
            pref_error_test.append(r['Pref error test'])
        l_train, l_std_train = np.mean(label_error_train), np.std(label_error_train)
        l_test, l_std_test = np.mean(label_error_test), np.std(label_error_test)
        p_train, p_std_train = np.mean(pref_error_train), np.std(pref_error_train)
        p_test, p_std_test = np.mean(pref_error_test), np.std(pref_error_test)
        print('Data set '+m+': Mean label error on train {:0.4f} ± {:0.3f}, mean label error on test {:0.4f} ± {:0.3f}'.
              format(l_train, l_std_train, l_test, l_std_test))
        print('Data set '+m+': Mean pref error on train {:0.4f} ± {:0.3f}, mean pref error on test {:0.4f} ± {:0.3f}'
              .format(p_train, p_std_train, p_test, p_std_test))
        results[m] = l_train, l_std_train, l_test, l_std_test, p_train, p_std_train, p_test, p_std_test
    pkl.dump(results, open('results_LL.pkl', 'wb'))


def run_label_xp_authors_SVM(datasets, n_expe=20, K=10, C=1):
    """
    :param datasets: list of strings which are the name of the datasets studied.
    :param n_expe: The number of repeated experience on the datasets.
    :param K: Gamma parameter for the SVM
    :param C: Complexity parameter for the SVM
    This function applies repeatedly the CCSVM algorithm on a set of datasets, in a label learning problem.
    """
    results = {}
    for i, m in enumerate(datasets):
        n_obs = 800
        label_error_train, label_error_test, pref_error_train, pref_error_test = [], [], [], []
        for _ in range(n_expe):
            users, graphs, classes = utils.read_data_LL(m, n_obs)
            train, test = utils.train_test_split(users, graphs, classes)
            print(max(test[2]), max(train[2]))
            model = CCSVM_LL(train, K, C)
            model.fit()
            train_lab_err, train_pref_err = model.score()
            test_lab_err, test_pref_err = model.score(*test, train=False)
            label_error_train.append(train_lab_err)
            pref_error_train.append(train_pref_err)
            label_error_test.append(test_lab_err)
            pref_error_test.append(test_pref_err)
        l_train, l_std_train, l_test, l_std_test = np.mean(label_error_train), np.std(label_error_train),\
                                               np.mean(label_error_test), np.std(label_error_test)
        p_train, p_std_train, p_test, p_std_test = np.mean(pref_error_train), np.std(pref_error_train), \
                                                   np.mean(pref_error_test), np.std(pref_error_test)
        print('\n______________________________________________\n'
              '(CCSVM) Data set ' + m + ' : Mean label error on train {:0.4f} ± {:0.3f}, mean label error on test {:0.4f} ± {:0.3f}'.
              format(l_train, l_std_train, l_test, l_std_test))
        print('(CCSVM) Data set ' + m + ' : Mean pref error on train {:0.4f} ± {:0.3f}, mean pref error on test {:0.4f} ± {:0.3f}'
                                '\n______________________________________________\n'. format(p_train, p_std_train, p_test, p_std_test))
        results[m] = l_train, l_std_train, l_test, l_std_test, p_train, p_std_train, p_test, p_std_test
    pkl.dump(results, open('results_LL_CCSVM.pkl', 'wb'))


