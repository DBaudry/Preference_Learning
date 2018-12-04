import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.stats import norm
import copy
from tqdm import tqdm

class learning_instance_preference:
    def __init__(self,inputs,K):
        self.X=inputs[0]
        self.D=inputs[1]
        self.n,self.d=self.X.shape
        self.m=len(self.D)
        self.cov=self.compute_cov(K)
        self.inv_cov=np.linalg.inv(self.cov)

    @staticmethod
    def gaussian_kernel(x,y,K):
        return np.exp(-K/2.*np.sum((x-y)**2))

    def compute_cov(self,K):
        cov=np.eye(self.n)
        for i in range(self.n):
            for j in range(i):
                cov_ij=self.gaussian_kernel(self.X[i],self.X[j],K)
                cov[i,j]=cov_ij
                cov[j,i]=cov_ij
        return cov

    def compute_prior(self,y):
        return mn.pdf(y,mean=np.zeros(self.n),cov=self.cov)

    def compute_z_phi(self,y,sigma):
        z=np.apply_along_axis(lambda x:(y[x[0]]-y[x[1]])/(np.sqrt(2)*sigma),1,self.D)
        return z,np.apply_along_axis(lambda x:norm.cdf(x,loc=0,scale=1),0,z)

    def compute_S(self,phi,y):
        prior=0.5*np.inner(y,np.dot(self.inv_cov,y))
        return -np.sum(np.log(phi))+prior

    def s(self,k,i):
        return (i==self.D[k][0])-(i==self.D[k][1])

    def compute_grad_S(self,z,phi,y,sigma):
        def partial_df(k,i): #for the log_phi part
            return -self.s(k,i)/np.sqrt(2)/sigma*norm.pdf(z[k])/phi[k]
        phi_grad=np.array([sum([partial_df(k,i) for k in range(self.m)]) for i in range(self.n)])
        return phi_grad+np.dot(self.inv_cov,y)

    def compute_Hessian_S(self,z,phi,sigma):
        def partial_d2f(k,i,j):
            s_ij=self.s(k,i)*self.s(k,j)/2/sigma**2
            nk=norm.pdf(z[k])/phi[k]
            return s_ij*(nk**2+z[k]*nk)
        phi_Hess=np.array([[sum([partial_d2f(k,i,j) for k in range(self.m)]) for j in range(self.n)] for i in range(self.n)])
        return phi_Hess+self.inv_cov

    def compute_S_deriv(self,y,sigma):
        z,phi=self.compute_z_phi(y,sigma)
        S=self.compute_S(phi,y)
        grad=self.compute_grad_S(z,phi,y,sigma)
        Hessian=self.compute_Hessian_S(z,phi,sigma)
        return S, grad, Hessian

    def ALS(self,y,sigma,direction,grad,rho,c=0.3,alpha=1.,max_iter=20):
        t=-c*np.inner(direction,grad)
        n_iter=0
        phi=self.compute_z_phi(y,sigma)[1]
        S=self.compute_S(phi, y)
        diff=np.inf
        while diff>alpha*t and n_iter<=max_iter:
            n_iter+=1
            y-=alpha*direction
            print('y'+str(y))
            new_phi=self.compute_z_phi(y)[1]
            new_S=self.compute_S(new_phi, y)
            diff=S-new_S
            print(diff,S,new_S)
            alpha=rho*alpha
        return y

    def MAP(self,y,sigma,iter_max=10,eps=1e-2,rho=0.9):
        n_iter=0
        err=np.inf
        while n_iter<iter_max and err>eps:
            print(n_iter)
            prev_y=copy.copy(y)
            S,grad,H=self.compute_S_deriv(y,sigma)
            direction=np.dot(np.linalg.inv(H),grad)
            y-=direction

            err=np.sqrt(np.sum((y-prev_y)**2))
            n_iter+=1
        return y

    def evidence_approx(self,y,sigma):
        S,grad,H=self.compute_S_deriv(y,sigma)
        nu_map=grad-np.dot(self.inv_cov,y)
        denom=np.linalg.det(np.eye(self.n)+np.dot(self.cov,nu_map))
        return min(1,np.exp(-S)/np.sqrt(np.abs(denom)))

    def tuning_parameters(self): #gradient descent
        pass

    def prediction(self,r,s,f_map,K,sigma):
        sigma_t=np.array([[1.,self.gaussian_kernel(r,s,K)],
                          [self.gaussian_kernel(r,s,K),1.]])
        kt=np.zeros((self.n,2))
        kt[:,0]=[self.gaussian_kernel(r,self.X[i],K) for i in range(self.n)]
        kt[:,1]=[self.gaussian_kernel(s, self.X[i],K) for i in range(self.n)]

        z,phi=self.compute_z_phi(f_map,sigma)
        H=self.compute_Hessian_S(z,phi,sigma)

        mu_star=np.dot(kt.T,np.dot(self.inv_cov,f_map))
        s_star=sigma_t-np.dot(np.dot(kt.T,np.linalg.inv(self.cov+np.linalg.inv(H-self.cov))),kt)
        new_sigma=np.sqrt(2*sigma**2+s_star[0,0]+s_star[1,1]-s_star[0,1]-s_star[1,0])
        return norm.cdf((mu_star[0]-mu_star[1])/new_sigma,loc=0,scale=1)

    def generalisation(self,test_set,K,sigma):
        y=np.zeros(self.n)
        f_map=self.MAP(y,sigma)
        X=test_set[0]
        pref=test_set[1]
        score=0
        results=[]
        for p in tqdm(pref):
            proba=self.prediction(X[p[0]],X[p[1]],f_map,K,sigma)
            results.append(proba)
            if proba>0.5:
                score+=1
        return score/len(pref),results