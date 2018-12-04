import G_learning as GL
import numpy as np
import input_data as data
import matplotlib.pyplot as plt

#alpha=np.array([0.1,0.3,0.2,0.3,0.1])
alpha=np.array([0.4,0.6])
#alpha=np.array([1/10.]*10)

n,d,m=10,2,20
#n,d,m=30,10,80
K,sigma=5.,0.05
generator=data.simulation_generator(func=data.cobb_douglas,func_param=alpha,d=d,n=n,m=m)
train=generator.generate_X_pref()
generator.n,generator.m=150,200
test=generator.generate_X_pref()

model=GL.learning_instance_preference(inputs=train,K=K)

y=[data.cobb_douglas(model.X[i], alpha) for i in range(model.n)]
y0=np.zeros(model.n)
#MAP=model.MAP(y0,sigma)

print(y)
#print(MAP)
#print(model.evidence_approx(MAP,sigma))

r,s=np.random.uniform(size=(2,d))
#print(model.prediction(r,s,MAP,K=K,sigma=sigma))

score,proba=model.generalisation(test,K,sigma)

print('Probabilities : '+str(proba))
print('Score : '+str(score))