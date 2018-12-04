import G_learning as GL
import numpy as np
import input_data as data
import matplotlib.pyplot as plt

############### Parameters to simulate train/test samples #######################################

#alpha=np.array([0.1,0.3,0.2,0.3,0.1])
alpha=np.array([0.4,0.6])
#alpha=np.array([1/10.]*10)

n,d,m=10,2,20
#n,d,m=30,10,80
generator=data.simulation_generator(func=data.cobb_douglas,func_param=alpha,d=d,n=n,m=m)
train=generator.generate_X_pref()
generator.n,generator.m=150,200
test=generator.generate_X_pref()

################## Model ################################

K,sigma=5.,0.05
model=GL.learning_instance_preference(inputs=train,K=K,sigma=sigma)

y0=np.zeros(model.n)
MAP=model.compute_MAP(y0)

print('Convergence of the minimizer of S : {}'.format(MAP['success']))
print('Maximum a Posteriori : {}'.format(MAP['x']))

print('Evidence Approximation (p(D|f_MAP)) : {}'.format(model.evidence_approx(MAP['x'])))

r,s=np.random.uniform(size=(2,d))
score,proba=model.predict(test,MAP['x'],K,sigma)

print('Probabilities : {}'.format(proba))
print('Score : {}'.format(score))