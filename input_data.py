import numpy as np

class simulation_generator:
    def __init__(self,func,func_param,d,n,m):
        self.n=n
        self.m=m
        self.d=d
        self.real_f=func
        self.f_param=func_param

    def generate_X(self):
        return np.random.uniform(size=(self.n,self.d))

    def draw_preference(self):
        a=np.random.randint(self.n)
        b=np.random.randint(self.n)
        while b==a:
            b=np.random.randint(self.n)
        return a,b

    def set_m_preference(self,X):
        pref=[]
        for i in range(self.m):
            a,b=self.draw_preference()
            while (a,b) in pref or (b,a) in pref:
                a, b=self.draw_preference()
            f_a,f_b=self.real_f(X[a],self.f_param),self.real_f(X[b],self.f_param)
            if f_a>f_b:
                pref.append((a,b))
            else:
                pref.append((b,a))
        return pref

    def generate_X_pref(self):
        X=self.generate_X()
        D=self.set_m_preference(X)
        return X,D

def cobb_douglas(x,alpha):
    x_alpha=x**alpha
    return np.prod(x_alpha)