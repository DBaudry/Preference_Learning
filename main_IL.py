import Instance_learning as IL
import numpy as np
import input_data as data

############### Parameters to simulate train/test samples #######################################

np.random.seed(42311)

# alpha=np.array([0.1,0.3,0.2,0.3,0.1])
# alpha = np.array([0.4, 0.6])
alpha = np.array([1/10.]*10)

n, d, m = 20, 10, 100
#n,d,m=30,10,80
generator = data.instance_pref_generator(func=data.cobb_douglas, func_param=alpha, d=d, n=n, m=m)
train = generator.generate_X_pref()
train_pref = generator.get_true_pref(train[0])
generator.m = 200
test = generator.set_m_preference(train[0])

################## Model ################################

K, sigma = 10., 0.1
model = IL.learning_instance_preference(inputs=train, K=K, sigma=sigma)

y0 = np.zeros(model.n)

MAP = model.compute_MAP(y0)

#K_range=[1.0,2.0,5.0,10.0]
#sigma_range=[0.001,0.01,0.05,0.1,1.0]
#MAP=model.compute_MAP_with_gridsearch(y0,K_range,sigma_range)

print('Convergence of the minimizer of S : {}'.format(MAP['success']))
print('Maximum a Posteriori : {}'.format(MAP['x']))

pref_ap = model.get_train_pref(MAP['x'])
print('Score on train: {}'.format(model.score(pref_ap, train_pref)))

print('Evidence Approximation (p(D|f_MAP)) : {}'.format(model.evidence_approx(MAP['x'])))

r, s = np.random.uniform(size=(2, d))
score, proba = model.predict((train[0], test), MAP['x'])

print('Probabilities : {}'.format(proba))
print('Score on test: {}'.format(score))
