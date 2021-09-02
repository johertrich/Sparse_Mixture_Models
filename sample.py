# This code belongs to the paper
#
# J. Hertrich, F. A. Ba and G. Steidl (2021).
# Sparse Mixture Models inspired by ANOVA.
# Arxiv preprint arXiv:2105.14893
#
# Please cite the paper if you use the code.
#
# This file contains the functions to sample from the different densities.
#
import numpy as np
import math

def sample_WGMM_unif(alphas,mus,Sigmas,us,d,n):
    # Sampling of a sparse mixture model (SPMM)
    # INPUTS:
    #   Parameters: alphas, mus, Sigmas, us
    #   d       - dimension of the SPMM
    #   n       - number of samples to generate
    # OUTPUTS:
    #   X       - generated samples
    #   classes - labels, which sample was generated from which class of the SPMM
    K=alphas.shape[0]
    X=np.zeros((n,d))
    cum_alpha=np.cumsum(alphas)
    zv=np.random.uniform(size=(n))
    classes=np.zeros(n)
    for i in range(0,K-1):
        classes=classes+(zv>cum_alpha[i])
    for i in range(0,K):
        mu=mus[i]
        Sigma=Sigmas[i]
        u=us[i]
        u_complement=[x for x in range(d) if x not in u]
        ni=np.sum(classes==i)
        gauss_rands=np.random.multivariate_normal(mu,Sigma,ni)
        gauss_rands-=np.floor(gauss_rands)
        unif_rands=np.random.uniform(size=(ni,len(u_complement)))
        rands=np.zeros((ni,d))
        rands[:,u]=gauss_rands
        rands[:,u_complement]=unif_rands
        X[classes==i,:]=rands
    return X,classes

def rejection_sampling_uniform(fun,M,d,n):
    # Rejection sampling.
    # INPUTS:
    #   fun     - target density function (up to a multiplicative constant)
    #   M       - constant, which is greater or equal the supremum of fun
    #   d       - input dimension of fun
    #   n       - number of samples to generate
    # OUTPUT:
    #   X       - generated samples
    X=np.zeros((0,d))
    while X.shape[0]<n:
        Xis=np.random.uniform(size=(n,d))
        u=np.random.uniform(size=n)
        fun_Xis=fun(Xis)
        X_take=Xis[fun_Xis>(M*u)]
        X=np.concatenate([X,X_take],0)
    return X[:n,:]

C=[np.sqrt(0.75),np.sqrt(315./604),np.sqrt(277200/655177)]

def b_spline_2(x):
    # Implementation of a b-spline of order 2.
    # Input: x, Output: f(x)
    C_2 = C[0]
    out=np.zeros_like(x)
    reg1=np.logical_and(x>=0.,x<0.5)
    out[reg1]=C_2*4*x[reg1]
    reg2=np.logical_and(x>=0.5,x<1.)
    out[reg2]=C_2*4*(1-x[reg2])
    return out

def b_spline_4(x):
    # Implementation of a b-spline of order 4
    # Input: x, Output: f(x)
    C_4 = C[1]
    out=np.zeros_like(x)
    reg1=np.logical_and(x>=0.,x<0.25)
    out[reg1]=C_4*128./3.*(x[reg1]**3)
    reg2=np.logical_and(x>=0.25,x<0.5)
    out[reg2]=C_4*(8./3.-32.*x[reg2]+128.*(x[reg2]**2)-128.*(x[reg2]**3))
    reg3=np.logical_and(x>=0.5,x<0.75)
    out[reg3]=C_4*(-88./3. -256.*(x[reg3]**2)+160.*x[reg3]+128.*(x[reg3]**3))
    reg4=np.logical_and(x>=0.75,x<1.)
    out[reg4]=C_4*(128./3.-128.*x[reg4]+128.*(x[reg4]**2)-(128./3.)*(x[reg4]**3))
    return out

def b_spline_6(x):
    # Implementation of a b-spline of order 6
    # Input: x, Output: f(x)
    C_6=C[2]
    out=np.zeros_like(x)
    reg1=np.logical_and(x>=0,x<1./6.)
    out[reg1]=C_6*1944./5.*(x[reg1]**5)
    reg2=np.logical_and(x>=1./6.,x<2./6.)
    out[reg2]=C_6*( 3./10.-9.*x[reg2]+108.*(x[reg2]**2)-648.*(x[reg2]**3)+1944.*(x[reg2]**4)-1944.*(x[reg2]**5))
    reg3=np.logical_and(x>=2./6.,x<3./6.)
    out[reg3]=C_6*(-237./10.+351*x[reg3]-2052.*(x[reg3]**2)+5832.*(x[reg3]**3)-7776.*(x[reg3]**4)+3888.*(x[reg3]**5))
    reg4=np.logical_and(x>=3./6.,x<4./6.)
    out[reg4]=C_6*(2193./10.+7668.*(x[reg4]**2)-2079.*x[reg4]+11664.*(x[reg4]**4)-13608.*(x[reg4]**3)-3888.*(x[reg4]**5))
    reg5=np.logical_and(x>=4./6.,x<5./6.)
    out[reg5]=C_6*(-5487./10.+3681.*x[reg5]-9612.*(x[reg5]**2)+12312.*(x[reg5]**3)-7776.*(x[reg5]**4)+1944.*(x[reg5]**5))
    reg6=np.logical_and(x>=5./6.,x<1.)
    out[reg6]=C_6*(1944./5.-1944.*x[reg6]+3888.*(x[reg6]**2)-3888.*(x[reg6]**3)+1944.*(x[reg6]**4)-1944./5.*(x[reg6]**5))
    return out

def prods_of_splines(x):
    # Implementation of the product of splines-function from Section 4.2 of the paper.
    # Input: x, Output: f(x)
    out1=b_spline_2(x[:,0])*b_spline_4(x[:,2])*b_spline_6(x[:,7])
    out2=b_spline_2(x[:,1])*b_spline_4(x[:,4])*b_spline_6(x[:,5])
    out3=b_spline_2(x[:,3])*b_spline_4(x[:,6])*b_spline_6(x[:,8])
    return out1+out2+out3

def Friedmann(x):
    # Friedmann-1 function.
    # Input: x, Output: f(x)
    out1=10*np.sin(math.pi*x[:,0]*x[:,1])
    out2=20*(x[:,2]-.5)**2
    out3=10*x[:,3]
    out4=5*x[:,4]
    return out1+out2+out3+out4
