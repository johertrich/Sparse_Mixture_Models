# This code belongs to the paper
#
# J. Hertrich, F. A. Ba and G. Steidl (2021).
# Sparse Mixture Models inspired by ANOVA.
# Arxiv preprint arXiv:2105.14893
#
# Please cite the paper if you use the code.
#
# This file contains diverse helper functions.
#
from EM_SPMM import *

l_max=1
num_l=2*l_max+1
log_2pi=tf.math.log(tf.constant(2*math.pi,dtype=tf.float64))

def ind2sub(sz,ind):
    # Converts linear indices to multi-indices in the range [0:sz[0],...,0:sz[-1]]
    # INPUTS:
    #   - sz        - defines the range
    #   - ind       - linear index
    # OUTPUT:
    #   - out       - muli-index
    out=[]
    sz.reverse()
    for s in sz:
        val=ind%s
        out.append(val)
        ind=(ind-val)/s
    out.reverse()
    return out

def argsort_list(my_list):
    # Arg-sorts a list of lists of integers by the first item
    return sorted(range(len(my_list)), key=my_list.__getitem__)

def my_prox(alphas,gamma):
    # Implements the proximal operator with respect to the function f(alpha)=||alpha||_0+iota_{\Delta}(alpha),
    # where iota_{\Delta}(alpha) is 0 on the probability simplex and infinity otherwise.
    # INPUTS:
    #   - alphas, gamma     - arguments of the prox
    # OUTPUT:
    #   - value of the proxy
    K=alphas.shape[0]
    sort_inds=np.argsort(alphas)
    alphas_sorted=alphas[sort_inds]
    alphas_cum=np.insert(np.cumsum(alphas_sorted)[:-1],0,0.)
    alphas_sq_cum=np.insert(np.cumsum(alphas_sorted**2)[:-1],0,0.)
    g_vals=1/(2*gamma)*((alphas_cum**2)/(K-np.arange(K))+alphas_sq_cum)-np.arange(K)
    my_n=np.argmin(g_vals)
    alphas_sorted[my_n:]+=alphas_cum[my_n]/(K-my_n)
    alphas_sorted[:my_n]=0.
    alphas_out=alphas_sorted[np.argsort(sort_inds)]
    return alphas_sorted[np.argsort(sort_inds)]

@tf.function
def compute_betas_inner(alpha,mu,Sigma,u,inputs):
    # Computes the numerator of the betas from the EM algorithm for wrapped-Normal SPMMs
    # INPUTS:
    #   - alpha, mu, Sigma, u   - parameters of the corresponding component k
    #   - inputs                - samples given in the EM alg.
    # OUTPUT:
    #   - list of length num_l where the l-th entry (l=-l_max,...,l_max) contains the values beta_{i,k,l}, i=1,...,N
    n_inp=inputs.shape[0]
    Sigma_inv=tf.linalg.inv(Sigma)
    logdet_Sigma=tf.linalg.logdet(Sigma)
    log_fun_vals_k=[]
    inputs_u=tf.gather(inputs,u,axis=1)
    for l_ind in range(num_l**len(u)):
        l=np.array(ind2sub([num_l]*len(u),l_ind))-l_max
        cent_inp=inputs_u+l-tf.tile(tf.expand_dims(mu,0),(n_inp,1))
        deltas=tf.reduce_sum(tf.matmul(cent_inp,Sigma_inv)*cent_inp,1)
        factor=-0.5*logdet_Sigma-0.5*deltas-0.5*len(u)*log_2pi
        line=factor+tf.math.log(alpha)
        log_fun_vals_k.append(line)
    return tf.stack(log_fun_vals_k)

@tf.function
def compute_gammas_inner(alpha,mu,sigmas,u,inputs):
    # Computes the numerator of the gammas from the EM algorithm for diagonal wrapped-Normal SPMMs
    # INPUTS:
    #   - alpha, mu, sigmas, u  - parameters of the corresponding component k
    #   - inputs                - samples given in the EM alg.
    # OUTPUT:
    #   - list of length num_l where the l-th entry (l=-l_max,...,l_max) contains the values beta_{i,k,l}, i=1,...,N
    n_inp=inputs.shape[0]
    if len(u)==0:
        return tf.math.log(alpha)*tf.ones((1,1,n_inp),dtype=tf.float64)
    inputs_u=tf.gather(inputs,u,axis=1)
    my_factors_wN,my_factors=log_wN_density(mu,sigmas,inputs_u,return_normal_factors=True)
    log_fun_vals_k=[]
    for j in range(len(u)):
        log_fun_vals_j=[]
        j_val=tf.reduce_sum(my_factors_wN[:j,:],0)+tf.reduce_sum(my_factors_wN[(j+1):,:],0)
        for m_ind in range(num_l):
            line=my_factors[m_ind,:,j]+j_val+tf.math.log(alpha)
            log_fun_vals_j.append(line)
        log_fun_vals_k.append(tf.stack(log_fun_vals_j))
    log_fun_vals_k=tf.stack(log_fun_vals_k)
    return log_fun_vals_k

@tf.function
def log_wN_density(mu,sigmas,inputs,return_normal_factors=False):
    # Computes the log-density of the diagonal wrapped normal distribution distribution
    # INPUTS:
    #   - inputs     - points, where the log-density should be evaluated
    #   - mu, sigmas - parameters of the von Mises distribution
    # OUTPUTS:
    #   - fun_vals  - values of the log-density
    n_inp=inputs.shape[0]
    d_inp=inputs.shape[1]
    sigmas_inv=1./sigmas
    logdet_sigmas=tf.math.log(sigmas)
    log_fun_vals_k=[]
    my_factors=[]
    for m in range(-l_max,l_max+1):
        cent_inp=inputs+m-tf.tile(tf.expand_dims(mu,0),(n_inp,1))
        deltas=tf.tile(sigmas_inv[tf.newaxis,:],(n_inp,1))*cent_inp**2
        factor=-0.5*tf.tile(logdet_sigmas[tf.newaxis,:],(n_inp,1))-0.5*deltas-0.5*log_2pi
        my_factors.append(factor)
    my_factors=tf.stack(my_factors)
    my_factors_wN=[]
    for j in range(d_inp):
        my_factors_wN.append(log_exp_sum([my_factors[:,:,j]]))
    my_factors_wN=tf.stack(my_factors_wN)
    if return_normal_factors:
        return my_factors_wN,my_factors
    return tf.reduce_sum(my_factors_wN,0)


@tf.function
def compute_m_C(betas_k,u,inputs):
    # Computes m_k and C_k from the EM algorithm for wrapped normal SPMMs based on (beta_{i,k,l}), i=1,...,N, l=-l_max,...,l_max
    # INPUTS:
    #   - betas_k   - values of the betas
    #   - u         - coupled variables
    #   - inputs    - samples for the EM algorithm 
    # OUTPUTS:
    #   - m_k, C_k  - intermediate values in the M-step of the EM alg
    m_k=0
    C_k=0
    inputs_u=tf.gather(inputs,u,axis=1)
    for l_ind in range(num_l**len(u)):
        l=np.array(ind2sub([num_l]*len(u),l_ind))-l_max
        beta_inps=(inputs_u+l)*tf.tile(tf.expand_dims(tf.transpose(betas_k[l_ind,:]),1),(1,len(u)))
        m_k+=tf.reduce_sum(beta_inps,0)
        C_k+=tf.matmul(beta_inps,(inputs_u+l),transpose_a=True)
    return m_k,C_k

@tf.function
def compute_m_C_diag(gammas_k,u,inputs):
    # Computes m_k and C_k from the EM algorithm for diagonal wrapped normal SPMMs based on 
    # (gamma_{i,k,l}), i=1,...,N, l=-l_max,...,l_max
    # INPUTS:
    #   - gamma_k   - values of the betas
    #   - u         - coupled variables
    #   - inputs    - samples for the EM algorithm 
    # OUTPUTS:
    #   - m_k, C_k  - intermediate values in the M-step of the EM alg
    if len(u)==0:
        return tf.zeros(0,dtype=tf.float64),tf.zeros(0,dtype=tf.float64)
    m_k=0
    C_k=0
    inputs_u=tf.gather(inputs,u,axis=1)
    for m_ind in range(num_l):
        m=m_ind-l_max
        gamma_inps=(inputs_u+m)*tf.transpose(gammas_k[:,m_ind,:])
        m_k+=tf.reduce_sum(gamma_inps,0)
        C_k+=tf.reduce_sum(gamma_inps*(inputs_u+m),0)
    return m_k,C_k

def log_exp_sum(inputs):
    # inputs is a list of lenght K with rank 2 tensors of size |u_k| times n_inps
    # computes log(sum(exp())) along the first dimension of the tensors and then along the list.
    # the output is a rank 1 tensor of length n_inps
    # Numerical much more stable than the naive implementation.
    K=len(inputs)
    const=tf.reduce_max(inputs[0],0)
    for k in range(1,K):
        const=tf.maximum(const,tf.reduce_max(inputs[k],0))
    log_sum_exp=0
    for k in range(K):
        log_sum_exp_k=inputs[k]-tf.tile(tf.expand_dims(const,0),(inputs[k].shape[0],1))
        log_sum_exp+=tf.reduce_sum(tf.exp(log_sum_exp_k),0)
    log_sum_exp=tf.math.log(log_sum_exp)+const
    return log_sum_exp

def A(kappa):
    # Helper function for ML estimation of von Mises distributions.
    # INPUT:
    #   - kappa
    # OUTPUT:
    #   - A(kappa)
    return tf.math.bessel_i1e(kappa)/tf.math.bessel_i0e(kappa)
 
def A_derivative(kappa,A_kappa):
    # Derivative of A
    # INPUTS:
    #   - kappa   - point, where the derivative of A shall be computed
    #   - A_kappa - function value A(kappa)
    # OUTPUT:
    #   - A'(kappa)
    return -A_kappa/kappa-A_kappa**2+1.

@tf.function
def A_inverse(R):
    # Computes the inverse of A using a Newton iteration.
    # Input:
    #   - R     - point to compute the inverse
    # Outupt
    #   - kappa - A^{-1}(R)
    kappa=tf.ones_like(R,dtype=tf.float64)*0.1
    A_kappa=A(kappa)
    eps=tf.constant(1e-3,dtype=tf.float64)
    cond=lambda kappa,A_kappa: tf.less(eps,tf.reduce_max(tf.math.abs(A_kappa-R)))
    def body(kappa,A_kappa):
        kappa_new=kappa-(A_kappa-R)/A_derivative(kappa,A_kappa)
        kappa_new=tf.maximum(kappa_new,0.01)
        A_kappa_new=A(kappa_new)
        return kappa_new,A_kappa_new
    kappa,A_kappa=tf.while_loop(cond,body,(kappa,A_kappa),maximum_iterations=20)
    return kappa

@tf.function
def log_wN_density_full(samps,mu,Sigma):
    # log of the probabilty density function of a wrapped normal distribution with full covariance matrix.
    # INPUT:
    #   - samps         - points, where the pdf shall be computed
    #   
    u=list(range(samps.shape[1]))
    alpha=tf.constant(1.,dtype=tf.float64)
    log_fun_vals=compute_betas_inner(alpha,mu,Sigma,u,samps)
    log_beta_nenner=log_exp_sum([log_fun_vals])
    return log_beta_nenner

@tf.function
def log_vM_density(samps,mu,kappa):
    # Computes the log-density of the von Mises distribution
    # INPUTS:
    #   - samps     - points, where the log-density should be evaluated
    #   - mu, kappa - parameters of the von Mises distribution
    # OUTPUTS:
    #   - fun_vals  - values of the log-density
    n_inp=samps.shape[0]
    cent_inp=samps-mu
    bessel_kappa_log=tf.math.log(tf.math.bessel_i0e(kappa))+kappa
    fun_vals=kappa*tf.math.cos(2*math.pi*cent_inp)-bessel_kappa_log            
    return fun_vals

def MC_KL_wN(mu1,Sigma1,mu2,Sigma2):
    # KL-distance of two wrapped normal distributions computed by a Monte Carlo approximation of the integral
    # INPUTS:
    #   - mu1,Sigma1    - Parameters of the first wN distribution
    #   - mu2,Sigma2    - Parameters of the second wN distribution
    # OUTPUT:
    #   - approximated KL value
    n_KL=10000    
    if len(Sigma1.shape)==1:
        out=0.
        for d in range(mu1.shape[0]):
            out+=MC_KL_wN(mu1[d:d+1],Sigma1[d:d+1,tf.newaxis],mu2[d:d+1],Sigma2[d:d+1,tf.newaxis])
        return out
    else:
        integration_points=np.random.multivariate_normal(mu1,Sigma1,n_KL)
        integration_points-=np.floor(integration_points)
        log_density1=log_wN_density_full(integration_points,mu1,Sigma1)
        log_density2=log_wN_density_full(integration_points,mu2,Sigma2)
        return tf.reduce_sum(log_density1-log_density2)/n_KL

def MC_KL_vM(mu1,kappa1,mu2,kappa2):
    # KL-distance of two (products of) von Mises distributions computed by a Monte Carlo approximation of the integral
    # INPUTS:
    #   - mu1,kappa1    - Parameters of the first wN distribution
    #   - mu2,kappa2    - Parameters of the second wN distribution
    # OUTPUT:
    #   - approximated KL value
    out=0.
    n_KL=20000
    integration_points=np.random.uniform(size=n_KL)
    for d in range(mu1.shape[0]):
        log_density1=log_vM_density(integration_points,mu1[d],kappa1[d])
        log_density2=log_vM_density(integration_points,mu2[d],kappa2[d])
        out+=tf.reduce_sum(tf.exp(log_density1)*(log_density1-log_density2))
    return out
