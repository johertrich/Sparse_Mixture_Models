# This code belongs to the paper
#
# J. Hertrich, F. A. Ba and G. Steidl (2022).
# Sparse Mixture Models inspired by ANOVA Decompositions.
# Electronic Transactions on Numerical Analysis, vol. 55, pp. 142-168.
#
# Please cite the paper if you use the code.
#
# This file contains the implementations of the EM algorithms for Sparse Mixture Models as introduced in Section 3
# of the paper.
#
import tensorflow as tf
import numpy as np
import time
import math

log_2pi=tf.math.log(tf.constant(2*math.pi,dtype=tf.float64))

from utils import *

def opt_em_von_Mises(samps,alphas_init,mus_init,kappas_init,us,steps=100,batch_size=1000,stop_crit=1e-5,mute=False,alpha_prox_gamma=None,weights=None):
    # Implements the (proximal) EM algorithm for estimating the parameters of a Mixture Models where each component of 
    # the Mixture Model is a product of von Mises distributions and uniform distributions on [0,1].
    # INPUTS:
    #   samps             - N x n numpy array, where samps[i] contains the i-th data point
    #
    #   Initial parameters:    
    #   alphas_init       - numpy array of length K
    #   mus_init          - K x n numpy array
    #   kappas_init       - K x n numpy array
    #   us                - list of length k, where each entry indicates which dimensions are von Mises distributed and which
    #                       dimensions are uniformly distributed
    #
    #   steps             - number of steps. Default value: 100.
    #   batch_size        - Parameter for the computation order. Does not effect the results, but the
    #                       execution time. Default: 10000
    #   stop_crit         - algorithm stops, if the relative change of the objective function is smaller than stop_crit
    #   mute              - True: suppress all debugging prints
    #   alpha_prox_gamma  - if not None: apply a proximal step with gamma=alpha_prox_gamma after each EM step.
    #   weights           - weighting of the input samples. None for equal weighting.
    #
    # OUTPUTS:
    #   Resulting parameters
    #   alphas      - numpy array of length K
    #   mus         - K x n numpy array
    #   kappas      - K x n numpy array
    #   step        - number of steps until the stopping criteria was reached
    #
    K=alphas_init.shape[0]
    if alpha_prox_gamma is None:
        alpha_prox=False
    else:
        alpha_prox=True
        if not mute:
            print('Start with '+str(K)+' components.')
    K=alphas_init.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]
    if weights is None:
        weights=tf.ones(n,dtype=tf.float64)
        weight_sum=n
    else:
        weights=tf.constant(weights,dtype=tf.float64)
        weight_sum=tf.reduce_sum(weights)
    def compute_betas(inputs,weights_inp,alphas,mus,kappas,us):
        # compute the betas
        K=alphas.shape[0]
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            mu=mus[k]
            kappa=kappas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            inputs_u=tf.gather(inputs,u,axis=1)
            cent_inp=inputs_u-tf.tile(tf.expand_dims(mu,0),(n_inp,1))
            bessel_kappa_log=tf.math.log(tf.math.bessel_i0e(kappa))+kappa
            line=tf.tile(kappa[tf.newaxis,:],(n_inp,1))*tf.math.cos(2*math.pi*cent_inp)-bessel_kappa_log
            log_fun_vals.append(tf.reduce_sum(line,1)+tf.math.log(alpha))
        log_fun_vals=tf.stack(log_fun_vals)
        log_beta_nenner=log_exp_sum([log_fun_vals])
        log_betas=log_fun_vals-tf.tile(log_beta_nenner[tf.newaxis,:],(K,1))
        betas=tf.exp(log_betas)
        obj=-tf.reduce_sum(weights_inp*log_beta_nenner)
        # (partial) alpha-update
        alphas_new=[]
        for k in range(K):
            alphas_new.append(tf.reduce_sum(weights_inp*betas[k])/weight_sum)  
        alphas_new=tf.stack(alphas_new) 
        # compute S and C for the mu and kappa updates
        S=[]
        C=[]
        for k in range(K):
            betas_k=weights_inp*betas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            inputs_u=tf.gather(inputs,u,axis=1)
            C_k=tf.reduce_sum(tf.tile(betas_k[:,tf.newaxis],(1,len(u)))*tf.math.cos(2*math.pi*inputs_u),0)
            S_k=tf.reduce_sum(tf.tile(betas_k[:,tf.newaxis],(1,len(u)))*tf.math.sin(2*math.pi*inputs_u),0)
            S.append(S_k)
            C.append(C_k)
        return alphas_new,S,C,obj
   
    if alpha_prox:
        recompile_count=0
    else:
        compute_betas=tf.function(compute_betas)
    # init parameters
    alphas=tf.constant(alphas_init,dtype=tf.float64)
    mus=[]
    kappas=[]
    for k in range(K):
        mus.append(tf.constant(mus_init[k],dtype=tf.float64))
        kappas.append(tf.constant(kappas_init[k],dtype=tf.float64))
    ds=tf.data.Dataset.from_tensor_slices((samps,weights)).batch(batch_size)
    tic=time.time()
    old_obj=0
    # loop for the steps
    for step in range(steps):     
        objective=0
        alphas_new=0
        S=[0]*K
        C=[0]*K
        counter=0
        if not mute:
            print('E-Step')
        # beta computation
        for smps,wts in ds:
            counter+=1
            out=compute_betas(smps,wts,alphas,mus,kappas,us)
            alphas_new+=out[0]
            for k in range(K):
                S[k]+=out[1][k]
                C[k]+=out[2][k]
            objective+=out[3]
        mus_new=[]
        kappas_new=[]
        diff=objective.numpy()-old_obj
        if not mute:
            print('Step '+str(step)+' Objective '+str(objective.numpy())+' Diff: ' +str(diff))
        old_obj=objective.numpy()
        if not mute:
            print('M-Step')
        # mu and kappa update
        for k in range(K):
            mus_new_k=(tf.math.atan2(S[k],C[k]))/(2.*math.pi)
            mus_new.append(mus_new_k-tf.math.floor(mus_new_k))
            R_k=tf.math.sqrt(S[k]**2+C[k]**2)/(weight_sum*alphas_new[k])
            kappa_new_k=A_inverse(R_k)
            kappas_new.append(kappa_new_k)
        # compute eps for stopping criterion
        eps=tf.reduce_sum((alphas-alphas_new)**2)
        for k in range(K):
            mu_diff=tf.math.abs(mus[k]-mus_new[k])
            eps+=tf.reduce_sum(tf.math.minimum(mu_diff,1-mu_diff)**2)
            eps+=tf.reduce_sum((kappas[k]-kappas_new[k])**2)
        if not mute:
            print('Step '+str(step+1)+' Time: '+str(time.time()-tic)+' Change: '+str(eps.numpy()))
        # proximal step
        if alpha_prox:
            alphas_new=my_prox(alphas_new.numpy(),gamma=alpha_prox_gamma)
            mus_new_=[]
            kappas_new_=[]
            alphas_new_=[]
            us_=[]
            recompile=False
            for k in range(K):
                if alphas_new[k]!=0.:
                    alphas_new_.append(alphas_new[k])
                    mus_new_.append(mus_new[k])
                    kappas_new_.append(kappas_new[k])
                    us_.append(us[k])
                else:
                    recompile=True
            if not recompile:
                recompile_count+=1
                if recompile_count==2:
                    if not mute:
                        print('Compiling!')
                    compute_betas=tf.function(compute_betas)
            alphas_new=tf.constant(np.stack(alphas_new_),dtype=tf.float64)
            us=us_
            mus_new=mus_new_
            kappas_new=kappas_new_
            if alphas_new.shape[0]<K and not mute:
                print('Reduced number of components from '+str(K)+' to '+str(alphas_new.shape[0])+'.')
                if recompile_count>=2:
                    print('Recompile!')
            K=alphas_new.shape[0]
        # update alpha, mu, kappa
        alphas=alphas_new
        mus=mus_new
        kappas=kappas_new
        # stopping criterion
        if eps<stop_crit or np.abs(diff/old_obj)<stop_crit:
            if alpha_prox:
                return alphas,mus,kappas,us,step
            return alphas,mus,kappas,step
    if alpha_prox:
        return alphas,mus,kappas,us,step
    return alphas,mus,kappas,step

def opt_em(samps,alphas_init,mus_init,Sigmas_init,us,regularize=1e-5,steps=100,batch_size=1000,stop_crit=1e-5,mute=False,alpha_prox_gamma=None,weights=None):
    # Implements the (proximal) EM algorithm for estimating the parameters of a Mixture Models where each component of 
    # the Mixture Model is a product of a wrapped normal distribution with full covariance matrix and uniform 
    # distributions on [0,1].
    # INPUTS:
    #   samps             - N x n numpy array, where samps[i] contains the i-th data point
    #
    #   Initial parameters:    
    #   alphas_init       - numpy array of length K
    #   mus_init          - K x n numpy array
    #   Sigmas_init       - K x n x n numpy array
    #   us                - list of length k, where each entry indicates which dimensions are von Mises distributed and which
    #                       dimensions are uniformly distributed
    #
    #   steps             - number of steps. Default value: 100.
    #   batch_size        - Parameter for the computation order. Does not effect the results, but the
    #                       execution time. Default: 10000
    #   stop_crit         - algorithm stops, if the relative change of the objective function is smaller than stop_crit
    #   mute              - True: suppress all debugging prints
    #   alpha_prox_gamma  - if not None: apply a proximal step with gamma=alpha_prox_gamma after each EM step.
    #   weights           - weighting of the input samples. None for equal weighting.
    #
    # OUTPUTS:
    #   Resulting parameters
    #   alphas      - numpy array of length K
    #   mus         - K x n numpy array
    #   Sigmas      - K x n x n numpy array
    #   step        - number of steps until the stopping criteria was reached
    #
    if len(tf.constant(Sigmas_init[0],dtype=tf.float64).shape)==1:
        return opt_em_diagonal(samps,alphas_init,mus_init,Sigmas_init,us,regularize=regularize,steps=steps,batch_size=batch_size,stop_crit=stop_crit,mute=mute,alpha_prox_gamma=alpha_prox_gamma,weights=weights)
    K=alphas_init.shape[0]
    if alpha_prox_gamma is None:
        alpha_prox=False
    else:
        alpha_prox=True
        if not mute:
            print('Start with '+str(K)+' components.')
    K=alphas_init.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]
    if weights is None:
        weights=tf.ones(n,dtype=tf.float64)
        weight_sum=n
    else:
        weights=tf.constant(weights,dtype=tf.float64)
        weight_sum=tf.reduce_sum(weights)

    def compute_betas(inputs,weights_inp,alphas,mus,Sigmas,us):
        # compute the betas
        K=alphas.shape[0]
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            mu=mus[k]
            Sigma=Sigmas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            log_fun_vals.append(compute_betas_inner(alpha,mu,Sigma,u,inputs))
        log_beta_nenner=log_exp_sum(log_fun_vals)
        betas=[]
        for k in range(K):
            log_betas_k=log_fun_vals[k]-tf.tile(log_beta_nenner[tf.newaxis,:],(log_fun_vals[k].shape[0],1))
            betas.append(tf.exp(log_betas_k))
        obj=-tf.reduce_sum(weights_inp*log_beta_nenner)
        # compute the (partial) alpha-update
        alphas_new=[]
        for k in range(K):
            alphas_new.append(tf.reduce_sum(weights_inp*betas[k])/weight_sum)  
        alphas_new=tf.stack(alphas_new) 
        # compute the values m_k and C_k for the mu and Sigma update
        m=[]
        C=[]
        for k in range(K):
            betas_k=weights_inp*betas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            m_k,C_k=compute_m_C(betas_k,u,inputs)
            m.append(m_k)
            C.append(C_k)
        return alphas_new,m,C,obj
   
    if alpha_prox:
        recompile_count=0
    else:
        compute_betas=tf.function(compute_betas)
    # init parameters
    alphas=tf.constant(alphas_init,dtype=tf.float64)
    mus=[]
    Sigmas=[]
    for k in range(K):
        mus.append(tf.constant(mus_init[k],dtype=tf.float64))
        Sigmas.append(tf.constant(Sigmas_init[k],dtype=tf.float64))
    ds=tf.data.Dataset.from_tensor_slices((samps,weights)).batch(batch_size)
    tic=time.time()
    old_obj=0
    # loop for the steps
    for step in range(steps):     
        objective=0
        alphas_new=0
        m=[0]*K
        C=[0]*K
        counter=0
        if not mute:
            print('E-Step')
        # compute betas, m and C
        for smps,wts in ds:
            counter+=1
            out=compute_betas(smps,wts,alphas,mus,Sigmas,us)
            alphas_new+=out[0]
            for k in range(K):
                m[k]+=out[1][k]
                C[k]+=out[2][k]
            objective+=out[3]
        mus_new=[]
        Sigmas_new=[]
        diff=objective.numpy()-old_obj
        if not mute:
            print('Step '+str(step)+' Objective '+str(objective.numpy())+' Diff: ' +str(diff))
        old_obj=objective.numpy()
        if not mute:
            print('M-Step')
        # mu and Sigma update
        for k in range(K):
            mus_new_k=m[k]/(weight_sum*alphas_new[k])
            mus_new.append(mus_new_k-tf.math.floor(mus_new_k))
            Sigmas_new_k=C[k]/(weight_sum*alphas_new[k])-tf.matmul(mus_new_k[:,tf.newaxis],mus_new_k[:,tf.newaxis],transpose_b=True)
            Sigmas_new.append(Sigmas_new_k+regularize*tf.eye(C[k].shape[0],dtype=tf.float64)) 
        # compute eps for the stopping criterion
        eps=tf.reduce_sum((alphas-alphas_new)**2)
        for k in range(K):
            mu_diff=tf.math.abs(mus[k]-mus_new[k])
            eps+=tf.reduce_sum(tf.math.minimum(mu_diff,1-mu_diff)**2)
            eps+=tf.reduce_sum((Sigmas[k]-Sigmas_new[k])**2)
        if not mute:
            print('Step '+str(step+1)+' Time: '+str(time.time()-tic)+' Change: '+str(eps.numpy()))
        # proximal step
        if alpha_prox:
            alphas_new=my_prox(alphas_new.numpy(),gamma=alpha_prox_gamma)
            mus_new_=[]
            Sigmas_new_=[]
            alphas_new_=[]
            us_=[]
            recompile=False
            for k in range(K):
                if alphas_new[k]!=0.:
                    alphas_new_.append(alphas_new[k])
                    mus_new_.append(mus_new[k])
                    Sigmas_new_.append(Sigmas_new[k])
                    us_.append(us[k])
                else:
                    recompile=True
            if not recompile:
                recompile_count+=1
                if recompile_count==2:
                    if not mute:
                        print('Compiling!')
                    compute_betas=tf.function(compute_betas)
            alphas_new=tf.constant(np.stack(alphas_new_),dtype=tf.float64)
            us=us_
            mus_new=mus_new_
            Sigmas_new=Sigmas_new_
            if alphas_new.shape[0]<K and not mute:
                print('Reduced number of components from '+str(K)+' to '+str(alphas_new.shape[0])+'.')
                if recompile_count>=2:
                    print('Recompile!')
            K=alphas_new.shape[0]
        # update alphas, mus, Sigmas
        alphas=alphas_new
        mus=mus_new
        Sigmas=Sigmas_new
        # stopping criterion
        if eps<stop_crit or np.abs(diff/old_obj)<stop_crit:
            if alpha_prox:
                return alphas,mus,Sigmas,us,step
            return alphas,mus,Sigmas,step
    if alpha_prox:
        return alphas,mus,Sigmas,us,step
    return alphas,mus,Sigmas,step

def opt_em_diagonal(samps,alphas_init,mus_init,sigmas_init,us,regularize=1e-5,steps=100,batch_size=1000,stop_crit=1e-5,mute=False,alpha_prox_gamma=None,weights=None):
    # Implements the (proximal) EM algorithm for estimating the parameters of a Mixture Models where each component of 
    # the Mixture Model is a product of a wrapped normal distribution with diagonal covariance matrix and uniform 
    # distributions on [0,1].
    # INPUTS:
    #   samps             - N x n numpy array, where samps[i] contains the i-th data point
    #
    #   Initial parameters:    
    #   alphas_init       - numpy array of length K
    #   mus_init          - K x n numpy array
    #   sigmas_init       - K x n numpy array
    #   us                - list of length k, where each entry indicates which dimensions are von Mises distributed and which
    #                       dimensions are uniformly distributed
    #
    #   steps             - number of steps. Default value: 100.
    #   batch_size        - Parameter for the computation order. Does not effect the results, but the
    #                       execution time. Default: 10000
    #   stop_crit         - algorithm stops, if the relative change of the objective function is smaller than stop_crit
    #   mute              - True: suppress all debugging prints
    #   alpha_prox_gamma  - if not None: apply a proximal step with gamma=alpha_prox_gamma after each EM step.
    #   weights           - weighting of the input samples. None for equal weighting.
    #
    # OUTPUTS:
    #   Resulting parameters
    #   alphas      - numpy array of length K
    #   mus         - K x n numpy array
    #   sigmas      - K x n numpy array
    #   step        - number of steps until the stopping criteria was reached
    #
    K=alphas_init.shape[0]
    if alpha_prox_gamma is None:
        alpha_prox=False
    else:
        alpha_prox=True
        if not mute:
            print('Start with '+str(K)+' components.')
    K=alphas_init.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]
    if weights is None:
        weights=tf.ones(n,dtype=tf.float64)
        weight_sum=n
    else:
        weights=tf.constant(weights,dtype=tf.float64)
        weight_sum=tf.reduce_sum(weights)
    def compute_gammas(inputs,weights_inp,alphas,mus,sigmas,us):
        # compute gammas
        K=alphas.shape[0]
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            mu=mus[k]
            sigma=sigmas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            log_fun_vals.append(compute_gammas_inner(alpha,mu,sigma,u,inputs))
        log_gamma_nenner=log_exp_sum([log_fun_vals[k][0,:,:] for k in range(K)])
        gammas=[]
        for k in range(K):
            log_gammas_k=log_fun_vals[k]-tf.tile(log_gamma_nenner[tf.newaxis,tf.newaxis,:],(log_fun_vals[k].shape[0],log_fun_vals[k].shape[1],1))
            gammas.append(tf.exp(log_gammas_k))
        obj=-tf.reduce_sum(weights_inp*log_gamma_nenner)
        # (partial) alpha update
        alphas_new=[]
        for k in range(K):
            alphas_new.append(tf.reduce_sum(weights_inp*gammas[k][0,:,:])/weight_sum)  
        alphas_new=tf.stack(alphas_new) 
        # compute m and C for the mu and Sigma update
        m=[]
        C_diag=[]
        for k in range(K):
            gammas_k=weights_inp*gammas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            m_k,C_diag_k=compute_m_C_diag(gammas_k,u,inputs)
            m.append(m_k)
            C_diag.append(C_diag_k)
        return alphas_new,m,C_diag,obj
   
    if alpha_prox:
        recompile_count=0
    else:
        compute_gammas=tf.function(compute_gammas)
    # init parameters
    alphas=tf.constant(alphas_init,dtype=tf.float64)
    mus=[]
    sigmas=[]
    for k in range(K):
        mus.append(tf.constant(mus_init[k],dtype=tf.float64))
        sigmas.append(tf.constant(sigmas_init[k],dtype=tf.float64))
    ds=tf.data.Dataset.from_tensor_slices((samps,weights)).batch(batch_size)
    tic=time.time()
    old_obj=0
    # main loop for the steps
    for step in range(steps):     
        objective=0
        alphas_new=0
        m=[0]*K
        C_diag=[0]*K
        counter=0
        if not mute:
            print('E-Step')
        # compute gammas m and C
        for smps,wts in ds:
            counter+=1
            out=compute_gammas(smps,wts,alphas,mus,sigmas,us)
            alphas_new+=out[0]
            for k in range(K):
                m[k]+=out[1][k]
                C_diag[k]+=out[2][k]
            objective+=out[3]
        mus_new=[]
        sigmas_new=[]
        diff=objective.numpy()-old_obj
        if not mute:
            print('Step '+str(step)+' Objective '+str(objective.numpy())+' Diff: ' +str(diff))
        old_obj=objective.numpy()
        if not mute:
            print('M-Step')
        # mu and sigma update
        for k in range(K):
            mus_new_k=m[k]/(weight_sum*alphas_new[k])
            mus_new.append(mus_new_k-tf.math.floor(mus_new_k))
            sigmas_new_k=C_diag[k]/(weight_sum*alphas_new[k])-mus_new_k*mus_new_k
            sigmas_new.append(sigmas_new_k+regularize)
        # compute eps for stopping criterion
        eps=tf.reduce_sum((alphas-alphas_new)**2)
        for k in range(K):
            mu_diff=tf.math.abs(mus[k]-mus_new[k])
            eps+=tf.reduce_sum(tf.math.minimum(mu_diff,1-mu_diff)**2)
            eps+=tf.reduce_sum((sigmas[k]-sigmas_new[k])**2)
        if not mute:
            print('Step '+str(step+1)+' Time: '+str(time.time()-tic)+' Change: '+str(eps.numpy()))
        # proximal step
        if alpha_prox:
            alphas_new=my_prox(alphas_new.numpy(),gamma=alpha_prox_gamma)
            mus_new_=[]
            sigmas_new_=[]
            alphas_new_=[]
            us_=[]
            recompile=False
            for k in range(K):
                if alphas_new[k]!=0.:
                    alphas_new_.append(alphas_new[k])
                    mus_new_.append(mus_new[k])
                    sigmas_new_.append(sigmas_new[k])
                    us_.append(us[k])
                else:
                    recompile=True
            if not recompile:
                recompile_count+=1
                if recompile_count==2:
                    if not mute:
                        print('Compiling!')
                    compute_gammas=tf.function(compute_gammas)
            alphas_new=tf.constant(np.stack(alphas_new_),dtype=tf.float64)
            us=us_
            mus_new=mus_new_
            sigmas_new=sigmas_new_
            if alphas_new.shape[0]<K and not mute:
                print('Reduced number of components from '+str(K)+' to '+str(alphas_new.shape[0])+'.')
                if recompile_count>=2:
                    print('Recompile!')
            K=alphas_new.shape[0]
        # apply updates for alphas, mus, sigmas
        alphas=alphas_new
        mus=mus_new
        sigmas=sigmas_new
        # stopping criterion
        if eps<stop_crit or np.abs(diff/old_obj)<stop_crit:
            if alpha_prox:
                return alphas,mus,sigmas,us,step
            return alphas,mus,sigmas,step
    if alpha_prox:
        return alphas,mus,sigmas,us,step
    return alphas,mus,sigmas,step


def log_likelihood_SPMM(samps,alphas,mus_,Sigmas_,us,batch_size=1000,return_probs=False,von_Mises=False,weights=None,compile_inner=True):
    # evaluates the negative log likelihood function of a SPMM
    # INPUTS:
    #   - samps         - points where the neg log likelihood shell be evaluated
    #   - parameters of the SPMM: alphas, mus_, Sigmas_, us
    #   - batch_size    - batch size of the beta computation
    #   - return_probs  - True for returning also the density values of the SPMM at the points samps, False otherwise.
    #   - von_Mises     - True, if we consider a von Mises SPAMM. In this case the kappas are specified by Sigmas_
    #                     False otherwise
    #   - weights       - weighting of the samples, None for equal weighting
    #   - compile_inner - specifies if the subroutine should be compiled into a tensorflow graph. Note that this parameter
    #                     does not influences the result of the function. Only the computation time is effected.
    # OUTPUTS:
    #   - log likelihood function evaluated for the samples samps
    #   - if return_probs is True: values of the pdf of the SPMM at the points samps
    K=alphas.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]
    if weights is None:
        weights=tf.ones(n,dtype=tf.float64)
    
    def compute_betas(inputs,alphas,mus,Sigmas,inp_weights):
        # compute betas (probability that sample i belongs to component k) for wrapped normal distributions with full cov-matrices
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            mu=mus[k]
            Sigma=Sigmas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            log_fun_vals.append(compute_betas_inner(alpha,mu,Sigma,u,inputs))
        log_beta_nenner=log_exp_sum(log_fun_vals)
        obj=-tf.reduce_sum(inp_weights*log_beta_nenner)
        if return_probs:
            return obj,tf.exp(log_beta_nenner)
        return obj

    def compute_betas_von_Mises(inputs,alphas,mus,kappas,inp_weights):
        # compute betas (probability that sample i belongs to component k) for von Mises distributions with full cov-matrices
        K=alphas.shape[0]
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            mu=mus[k]
            kappa=kappas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            inputs_u=tf.gather(inputs,u,axis=1)
            cent_inp=inputs_u-tf.tile(tf.expand_dims(mu,0),(n_inp,1))
            bessel_kappa_log=tf.math.log(tf.math.bessel_i0(kappa))
            line=tf.tile(kappa[tf.newaxis,:],(n_inp,1))*tf.math.cos(2*math.pi*cent_inp)-bessel_kappa_log
            log_fun_vals.append(tf.reduce_sum(line,1)+tf.math.log(alpha))
        log_fun_vals=tf.stack(log_fun_vals)
        log_beta_nenner=log_exp_sum([log_fun_vals])
        obj=-tf.reduce_sum(inp_weights*log_beta_nenner)
        if return_probs:
            return obj,tf.exp(log_beta_nenner)
        return obj

    def compute_weights_diag(inputs,alphas,mus,Sigmas,inp_weights):
        # compute betas (probability that sample i belongs to component k) for wrapped normal distributions with diag cov-matrices
        if len(Sigmas[0].shape)==2:
            full_matrices=True
        else:
            full_matrices=False
        prob_ks=[]
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            u=us[k]
            mu=mus[k]
            sigmas=Sigmas[k]
            u=tf.constant(u,dtype=tf.int64)
            if len(u)==0:
                prob_ks.append(tf.math.log(alpha)*tf.ones((1,n_inp),dtype=tf.float64))
            else:
                inputs_u=tf.gather(inputs,u,axis=1)
                prob_ks.append(tf.math.log(alpha)+tf.expand_dims(log_wN_density(mu,sigmas,inputs_u),0))
        prob_ks_nenner=log_exp_sum(prob_ks)
        obj=-tf.reduce_sum(inp_weights*prob_ks_nenner)
        if return_probs:
            return obj,tf.exp(prob_ks_nenner)
        return obj

    # set parameters
    alphas=tf.constant(alphas,dtype=tf.float64)
    mus=[]
    Sigmas=[]
    for k in range(K):
        mus.append(tf.constant(mus_[k],dtype=tf.float64))
        Sigmas.append(tf.constant(Sigmas_[k],dtype=tf.float64))
    ds=tf.data.Dataset.from_tensor_slices((samps,weights)).batch(batch_size)
    objective=0
    objective2=0
    betas=[]

    if compile_inner:
        if von_Mises:
            compute_betas_von_Mises=tf.function(compute_betas_von_Mises)
        elif len(Sigmas[0].shape)==1:
            compute_weights_diag=tf.function(compute_weights_diag)
        else:
            compute_betas=tf.function(compute_betas)
    # compute weights
    for smps,wts in ds:
        if von_Mises:
            out=compute_betas_von_Mises(smps,alphas,mus,Sigmas,wts)
        elif len(Sigmas[0].shape)==1:
            out=compute_weights_diag(smps,alphas,mus,Sigmas,wts)
        else:
            out=compute_betas(smps,alphas,mus,Sigmas,wts)
        if return_probs:
            objective+=out[0]
            betas.append(out[1])
        else:
            objective+=out
    # return result
    if return_probs:
        probs=tf.concat(betas,0)
        return objective,probs
    return objective

def classify(samps,alphas,mus_,Sigmas_,us,num_classes,im_classes,batch_size=1000):
    # Classification using a SPMM. For a detailed description see Section 4.3 in the paper.
    # INPUTS:
    #   - samps             - samples, which should be classified
    #   - Parameters of the SPMM: alphas, mus_, Sigmas_, us
    #   - num_classes       - number of classes
    #   - im_classes        - list, which specifies, which component of the SPMM belongs to which class
    #   - batch_size        - batch size for the weight computation
    # OUTPUTS:
    #   - classification    - list of classes assigned to the samples
    K=alphas.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]

    def compute_betas(inputs,alphas,mus,Sigmas):
        # compute betas (probability that sample i belongs to component k)
        log_fun_vals=[]
        beta_nenner=0
        n_inp=inputs.shape[0]
        for k in range(K):
            alpha=alphas[k]
            mu=mus[k]
            Sigma=Sigmas[k]
            u=us[k]
            u=tf.constant(u,dtype=tf.int64)
            log_fun_vals.append(compute_betas_inner(alpha,mu,Sigma,u,inputs))
        log_beta_nenner=log_exp_sum(log_fun_vals)
        log_betas=[]
        for k in range(K):
            log_betas_k=log_fun_vals[k]-tf.tile(log_beta_nenner[tf.newaxis,:],(log_fun_vals[k].shape[0],1))
            log_betas.append(tf.reduce_sum(tf.exp(log_betas_k),axis=0))
        return tf.stack(log_betas)

    alphas=tf.constant(alphas,dtype=tf.float64)
    mus=[]
    Sigmas=[]
    for k in range(K):
        mus.append(tf.constant(mus_[k],dtype=tf.float64))
        Sigmas.append(tf.constant(Sigmas_[k],dtype=tf.float64))
    ds=tf.data.Dataset.from_tensor_slices(samps).batch(batch_size)
    classification=[]
    # classification
    for smps in ds:
        out=compute_betas(smps,alphas,mus,Sigmas)
        c_out=[]
        for c in range(num_classes):
            g_inds=np.array(range(K))[np.array(im_classes)==c]
            out_c=tf.gather(out,g_inds)
            out_c=tf.reduce_sum(out_c,axis=0)
            if len(g_inds)==0:
                c_out.append(-np.inf*tf.ones_like(out_c,dtype=tf.float64))
            else:
                c_out.append(out_c)
        classification.append(tf.math.argmax(c_out,0).numpy())
    classification=np.concatenate(classification,0)
    return classification
    
