import numpy as np
from tqdm import tqdm
from utils import ratio_n_obs, gridsearchBool
import pickle as pkl
import input_data as data
import Instance_learning as IL
import utils
import matplotlib.pyplot as plt
from SVM_IL import *


dataset_shapes = {'abalone': (4177, 9), 'diabetes': (43, 3), 'housing': (506, 14),
                  'machine': (209, 7), 'pyrim': (74, 28), 'r_wpbc': (194, 33), 'triazines': (186, 61),
                  'algae': (316, 12), 'german2005': (412, 30), 'german2009': (412, 33), 'movies': (602, 9),
                  'sushia': (5000, 11), 'sushib': (5000, 11)}

authors_n_pref = {'pyrim': 100, 'triazines': 300, 'machine': 500, 'housing': 700, 'abalone': 1000}

best_parameters = {'pyrim': (0.005, 0.007), 'triazines': (0.007, 0.006), 'machine': (0.03, 0.0006),
                   'housing': (0.005, 0.001), 'abalone': (0.01, 0.005)}


def run_instance_xp(gen, model, train, test, K, sigma, gridsearch=False, show_results=True):
    """
    :param model: instance_pref_generator instance for a given function
    :param train: train set (data+pref)
    :param test: test set (pref)
    :param K: kernel parameter. Float, list of floats if gridsearch
    :param sigma: Error variance. Float, list of floats if gridsearch
    :param gridsearch: If true compute a gridsearch for the best parameters for K and sigma
    based on the model evidence
    :param show_results: display the results
    :return: Dict: MAP obtained with the train set, prediction with test set
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
        print('Error on train: {:0.4f}, error on test: {:0.4f}\n______________________________________________\n'
              ''.format(1-score_train, 1-score))
    return {'MAP': MAP, 'score_train': score_train, 'evidence': evidence,
            'proba_test': proba, 'score_test': score}


def run_instance_xp_authors(n_expe, datasets, param='best', show_results=False, print_callback=False):
    results = {}
    l = len(datasets)
    gridsearch = gridsearchBool(param)
    for i, m in tqdm(enumerate(datasets), desc='Running experiments on '+ str(l) +' datasets', total=l):
        b = best_parameters[m]
        K0 = b[0] if param == 'best' else param[0]
        sigma0 = b[1] if param == 'best' else param[1]
        n_obs, n_features = ratio_n_obs(authors_n_pref[m]), -1
        n_pref_train, n_pref_test = authors_n_pref[m], 20000
        score_train, score_test = [], []
        for expe in range(n_expe):
            generator = data.pref_generator(m, n_obs, n_features)
            train, test = generator.get_input_train(n_pref_train), generator.get_input_test(n_pref_test)
            model = IL.learning_instance_preference(inputs=train, K=K0, sigma=sigma0, print_callback=print_callback)
            results = run_instance_xp(generator, model, train, test, K0, sigma0, gridsearch=gridsearch, show_results=show_results)
            score_train.append(1 - results['score_train'])
            score_test.append(1 - results['score_test'])
        m_train, std_train, m_test, std_test = np.mean(score_train), np.std(score_train), np.mean(score_test), np.std(
            score_test)
        print('Data set ' + m + ' : Mean error on train {:0.4f} ± {:0.3f}, mean error on test {:0.4f} ± {:0.3f}'.
              format(m_train, std_train, m_test, std_test))
        results[m] = m_train, std_train, m_test, std_test
    pkl.dump(results, open('results.pkl', 'wb'))


def run_instance_xp_authors_SVM(n_expe, datasets, K, C, show_results=False, print_callback=False):
    results = {}
    l = len(datasets)
    for i, m in tqdm(enumerate(datasets), desc='Running experiments on '+ str(l) +' datasets', total=l):
        n_obs, n_features = ratio_n_obs(authors_n_pref[m]), -1
        n_pref_train, n_pref_test = authors_n_pref[m], 20000
        score_trainHerb, score_testHerb = [], []
        score_trainHar, score_testHar = [], []
        for expe in range(n_expe):
            generator = data.pref_generator(m, n_obs, n_features)
            train, test = generator.get_input_train(n_pref_train), generator.get_input_test(n_pref_test)
            learnerHerb = SVM_InstancePref(train, K, C)
            learnerHar = CCSVM(train, K, C)
            score_testHerb.append(1 -learnerHerb.score(test[0], test[1], train=False))
            score_testHar.append(1 - learnerHar.score(test[0], test[1], train=False))
            score_trainHerb.append(1 -learnerHerb.score())
            score_trainHar.append(1 - learnerHar.score())
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


def run_label_xp(gen, model, train, test, K, sigma, gridsearch=False, showgraph=False):
        """
        :param model: instance_pref_generator instance for a given function
        :param train: train set (data+pref)
        :param test: test set (pref)
        :param K: kernel parameter. list of floats, list of lists floats if gridsearch
        :param sigma: Error variance. Float, list of floats if gridsearch
        :param gridsearch: If true compute a gridsearch for the best parameters for K and sigma
        based on the model evidence
        :return: Dict: MAP obtained with the train set, prediction with test set
        """
        y0 = np.zeros(model.n*model.n_labels)
        if not gridsearch:
            MAP = model.compute_MAP(y0)
        else:
            MAP = model.compute_MAP_with_gridsearch(y0, K, sigma)
        evidence = model.evidence_approx(MAP['x'])
        pred_train = model.predict(train[0], MAP['x'])
        pred_test = model.predict(test[0], MAP['x'])
        score_train_classif = model.label_score_rate(train[2], pred_train)
        score_test_classif = model.label_score_rate(test[2], pred_test)
        score_train_pref = model.label_pref_rate(train[1], pred_train)
        score_test_pref = model.label_pref_rate(test[1], pred_test)
        print('Convergence of the minimizer of S : {}'.format(MAP['success']))
        print('Maximum a Posteriori : {}'.format(MAP['x']))
        print('Evidence Approximation (p(D|f_MAP)) : {}'.format(evidence))
        print('True best classes on train : {}'.format(train[2]))
        print('Best classes predicted on train : {}'.format(np.argsort(pred_train, axis=1)[:, -1]))
        print('True best classes on test : {}'.format(test[2]))
        print('Best classes predicted on test : {}'.format(np.argsort(pred_test, axis=1)[:, -1]))
        print('Mean label error on train: {}, and test: {}'.format(1-score_train_classif, 1-score_test_classif))
        print('Mean pref error on train: {}, and test: {}'.format(1 - score_train_pref, 1 - score_test_pref))
        if showgraph:
            plt.subplot(1,2,1)
            utils.pipeline_graph(train[1][0], 0, 'compute_all_edges')
            plt.subplot(1,2,2)
            utils.pipeline_graph(utils.compute_all_edges(np.array(pred_train[0]).argsort()[::-1]), 0,
                                 'compute_all_edges')
            plt.show()
        return {'MAP': MAP, 'evidence': evidence, 'predictions train': pred_train,
                'predictions test': pred_test}
