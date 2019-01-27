import numpy as np


def run_instance_xp(gen, model, train, test, K, sigma, gridsearch=False):
    """
    :param model: instance_pref_generator instance for a given function
    :param train: train set (data+pref)
    :param test: test set (pref)
    :param K: kernel parameter. Float, list of floats if gridsearch
    :param sigma: Error variance. Float, list of floats if gridsearch
    :param gridsearch: If true compute a gridsearch for the best parameters for K and sigma
    based on the model evidence
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
    print('Convergence of the minimizer of S : {}'.format(MAP['success']))
    print('Maximum a Posteriori : {}'.format(MAP['x']))
    print('Score on train: {}'.format(score_train))
    print('Evidence Approximation (p(D|f_MAP)) : {}'.format(evidence))
    print('Probabilities : {}'.format(proba))
    print('Score on test: {}'.format(score))
    return {'MAP': MAP, 'score_train': score, 'evidence': evidence,
            'proba_test': proba, 'score_test': score}


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
