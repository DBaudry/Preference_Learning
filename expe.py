import numpy as np
from tqdm import tqdm
from utils import ratio_n_obs, gridsearchBool
import pickle as pkl
import input_data as data
import Instance_learning as IL

dataset_shapes = {'abalone': (4177, 9), 'diabetes': (43, 3), 'housing': (506, 14),
                  'machine': (209, 7), 'pyrim': (74, 28), 'r_wpbc': (194, 33), 'triazines': (186, 61)}

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

def run_instance_xp_authors(n_expe, datasets, param='best', show_results=False):
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
            model = IL.learning_instance_preference(inputs=train, K=K0, sigma=sigma0)
            results = run_instance_xp(generator, model, train, test, K0, sigma0, gridsearch=gridsearch, show_results=show_results)
            score_train.append(1 - results['score_train'])
            score_test.append(1 - results['score_test'])
        m_train, std_train, m_test, std_test = np.mean(score_train), np.std(score_train), np.mean(score_test), np.std(
            score_test)
        print('Data set ' + m + ' : Mean error on train {:0.4f} ± {:0.3f}, mean error on test {:0.4f} ± {:0.3f}'.
              format(m_train, std_train, m_test, std_test))
        results[m] = m_train, std_train, m_test, std_test
    pkl.dump(results, open('results.pkl', 'wb'))


def run_label_xp(gen, model, train, test, K, sigma, gridsearch=False):
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
        # train_pref = gen.get_true_pref(train[0])
        y0 = np.zeros(model.n*model.n_labels)
        if not gridsearch:
            MAP = model.compute_MAP(y0)
        else:
            MAP = model.compute_MAP_with_gridsearch(y0, K, sigma)
        # evidence = model.evidence_approx(MAP['x'])
        # pref_ap = model.get_train_pref(MAP['x'])
        # score_train = model.score(pref_ap, train_pref)
        predictions = model.predict(test, MAP['x'])
        print('Convergence of the minimizer of S : {}'.format(MAP['success']))
        print('Maximum a Posteriori : {}'.format(MAP['x']))
        # print('Score on train: {}'.format(score_train))
        # print('Evidence Approximation (p(D|f_MAP)) : {}'.format(evidence))
        # print('Probabilities : {}'.format(proba))
        # print('Score on test: {}'.format(score))
        # return {'MAP': MAP, 'score_train': score, 'evidence': evidence,
        #        'proba_test': proba, 'score_test': score}
