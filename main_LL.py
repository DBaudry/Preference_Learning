import Label_learning as LL
import numpy as np
import input_data as data

############### Parameters to simulate train/test samples #######################################

n,d,m,n_label=10,2,5,10
#n,d,m,n_label=30,10,80,10
label_func=[data.cobb_douglas for l in range(n_label)]
rho=0.9

def get_alpha(dim):
    rho=np.random.uniform()
    coeff=np.array([rho**(i+1) for i in range(dim)])
    return np.random.permutation(coeff/coeff.sum())

alpha=[get_alpha(d) for _ in range(n_label)]

generator=data.label_pref_generator(func=label_func,func_param=alpha,d=d,n=n,m=m)
train=generator.generate_X_pref()
generator.n,generator.m=150,200
test=generator.generate_X_pref()

################## Model ################################

K,sigma=np.arange(10)+1.,0.05
model=LL.learning_label_preference(inputs=train,K=K,sigma=sigma)

y0=np.zeros((n_label,n))
print(model.compute_Hessian_S(y0))