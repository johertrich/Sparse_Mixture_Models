# This code belongs to the paper
#
# J. Hertrich, F. A. Ba and G. Steidl (2021).
# Sparse Mixture Models inspired by ANOVA.
# Arxiv preprint arXiv:2105.14893
#
# Please cite the paper if you use the code.
#
# This file contains the implementation of the heuristic from Section 3.3 of the paper and several helper functions
#
from EM_SPMM import *
from utils import *

def test_uniform(samps,alphas_init,mus_init,Sigmas_init,us,my_k,batch_size=1000,M=1,mode=0,mute=True,sample_weights=None,ks_thresh=2.5):
    # Implements the Kolmogorov Smirnov test for testing which dimensions should be added to component with number my_k
    # as in the heuristic of Section 3.3
    # INPUTS:
    #   - samps             - input data points from the heuristic
    #   - current parameters alphas_init, mus_init, Sigmas_init,us
    #   - my_k              - component for which each dimension should be testet
    #   - batch_size        - batch size of the weight computation
    #   - M                 - number of mixture components which shell be added if a dimension is added
    #   - mode              - 0 for wrapped normal with full covariance matrices
    #                         1 for wrapped normal with diagonal covariance matrices
    #                         2 for products of von Mises distributions
    #   - mute              - False for enabling debug prints, True for disabling debug prints
    #   - sample_weights    - weights for weighted samples, None for equal weights
    # OUTPUTS:
    #   - parameters of the added components: weighting of the added components, new_us,new_mus,new_Sigmas
    K=alphas_init.shape[0]
    n=samps.shape[0]
    d=samps.shape[1]
    von_Mises=mode==2
    if sample_weights is None:
        sample_weights=tf.ones(n,dtype=tf.float64)

    @tf.function
    def compute_betas(inputs,alphas,mus,Sigmas,input_weights):
        # Computation of the weights beta in Sec. 3.3 for wrapped normal distributions with full covariance matrices
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
        weights=input_weights*tf.reduce_sum(tf.exp(log_fun_vals[my_k]-tf.tile(log_beta_nenner[tf.newaxis,:],(num_l**len(us[my_k]),1))),0)
        betas_sum=tf.reduce_sum(weights)
        beta_inps=inputs*tf.tile(tf.expand_dims(weights,1),(1,d))
        m_k=tf.reduce_sum(beta_inps,0)
        C_k=tf.matmul(beta_inps,(inputs),transpose_a=True)
        return weights,betas_sum,m_k,C_k

    @tf.function
    def compute_betas_von_Mises(inputs,alphas,mus,kappas,input_weights):
        # Computation of the weights beta in Sec. 3.3 for products of von Mises distributions
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
        log_betas_my_k=log_fun_vals[my_k]-log_beta_nenner
        weights=input_weights*tf.exp(log_betas_my_k)
        betas_sum=tf.reduce_sum(weights)
        beta_inps=inputs*tf.tile(tf.expand_dims(weights,1),(1,d))
        m_k=tf.reduce_sum(beta_inps,0)
        C_k=tf.matmul(beta_inps,(inputs),transpose_a=True)
        return weights,betas_sum,m_k,C_k

    @tf.function
    def compute_weights_diag(inputs,alphas,mus,Sigmas,input_weights):
        # Computation of the weights beta in Sec. 3.3 for wrapped normal distributions with diagonal covariance matrices
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
        weights=input_weights*tf.squeeze(tf.exp(prob_ks[my_k]-prob_ks_nenner))
        weight_sum=tf.reduce_sum(weights)
        weight_inps=inputs*tf.tile(tf.expand_dims(weights,1),(1,d))
        m_k=tf.reduce_sum(weight_inps,0)
        C_k=tf.matmul(weight_inps,(inputs),transpose_a=True)
        return weights,weight_sum,m_k,C_k
        
        

    alphas=tf.constant(alphas_init,dtype=tf.float64)
    mus=[]
    Sigmas=[]
    for k in range(K):
        mus.append(tf.constant(mus_init[k],dtype=tf.float64))
        Sigmas.append(tf.constant(Sigmas_init[k],dtype=tf.float64))
    ds=tf.data.Dataset.from_tensor_slices((samps,sample_weights)).batch(batch_size)
    my_betas=[]
    m=0.
    C=0.
    my_betas_sum=0.
    # Computation of the betas
    for smps,smps_w in ds:
        if von_Mises:
            out=compute_betas_von_Mises(smps,alphas,mus,Sigmas,smps_w)
        elif len(Sigmas[0].shape)==1:
            out=compute_weights_diag(smps,alphas,mus,Sigmas,smps_w)
        else:
            out=compute_betas(smps,alphas,mus,Sigmas,smps_w)
        my_betas.append(out[0])
        my_betas_sum+=out[1]
        m+=out[2]
        C+=out[3]
    my_betas_sum=my_betas_sum.numpy()
    m/=my_betas_sum    
    C=C/my_betas_sum-tf.matmul(m[:,tf.newaxis],m[tf.newaxis,:])
    C=C.numpy()+1e-8*np.eye(C.shape[0])
    my_betas=tf.concat(my_betas,0).numpy()
    my_betas/=my_betas_sum
    s_2=np.sum(my_betas**2)
    u=us[my_k].copy()
    thresh=ks_thresh
    thresh_remove=thresh
    correlation_threshold=0.1
    u_old=u.copy()
    removed=[]
    # remove uncorrelated and uniform distributed dimensions
    for i in range(len(u_old)):
        dim=u_old[i]
        dim_ind=u.index(dim)
        # KS test
        samps_dim=samps[:,dim]
        permute=np.argsort(samps_dim)
        my_betas_dim=my_betas[permute]
        my_betas_dim_cumsum=np.cumsum(my_betas_dim)
        samps_dim=samps_dim[permute]
        ks_statistic1=np.sqrt(1/s_2)*np.max(my_betas_dim_cumsum-samps_dim)
        ks_statistic2=np.sqrt(1/s_2)*np.max(samps_dim-my_betas_dim_cumsum)
        ks_statistic=ks_statistic1+ks_statistic2
        # Correlation test
        correlations=C[u,dim]/np.sqrt(np.diag(C)[u]*C[dim,dim])
        correlations=np.concatenate([correlations[:dim_ind],correlations[dim_ind+1:]],0)
        if not mute:
            print(correlations)
            print(ks_statistic,ks_statistic1,ks_statistic2)
        if ks_statistic<thresh_remove and (correlations.shape[0]>0 and np.max(np.abs(correlations))<correlation_threshold):
            # adapt mu and Sigma
            u.remove(dim)
            mu_new=mus[my_k].numpy()
            mu_new=np.concatenate([mu_new[:dim_ind],mu_new[dim_ind+1:]],0)
            mus[my_k]=tf.constant(mu_new,dtype=tf.float64)
            Sigma_new=Sigmas[my_k].numpy()
            if len(Sigmas[0].shape)==1:
                Sigma_new=np.concatenate([Sigma_new[:dim_ind],Sigma_new[dim_ind+1:]],0)
            else:
                Sigma_new=np.concatenate([Sigma_new[:,:dim_ind],Sigma_new[:,dim_ind+1:]],1)
                Sigma_new=np.concatenate([Sigma_new[:dim_ind,:],Sigma_new[dim_ind+1:,:]],0)
            Sigmas[my_k]=tf.constant(Sigma_new,dtype=tf.float64)
            removed.append(dim)
    new_us=[u]
    new_mus=[mus[my_k]]
    new_Sigmas=[Sigmas[my_k]]
    # add correlated or not uniform distributed dimensions
    for dim in range(d):
        if dim in u or dim in removed:
            continue
        # KS test
        samps_dim=samps[:,dim]
        permute=np.argsort(samps_dim)
        my_betas_dim=my_betas[permute]
        my_betas_dim_cumsum=np.cumsum(my_betas_dim)
        samps_dim=samps_dim[permute]
        ks_statistic1=np.sqrt(1/s_2)*np.max(my_betas_dim_cumsum-samps_dim)
        ks_statistic2=np.sqrt(1/s_2)*np.max(samps_dim-my_betas_dim_cumsum)
        ks_statistic=ks_statistic1+ks_statistic2
        # Correlation test
        correlations=C[u,dim]/np.sqrt(np.diag(C)[u]*C[dim,dim])
        if not mute:
            print(correlations)
            print(ks_statistic,ks_statistic1,ks_statistic2)
        if ks_statistic>thresh or (correlations.shape[0]>0 and np.max(np.abs(correlations))>correlation_threshold):
            new_u=u+[dim]
            new_u.sort()
            centers=np.random.choice(samps_dim,M,p=my_betas_dim)
            if von_Mises:
                alphas_out,mu_out,Sigma_out,us_out,_=opt_em_von_Mises(samps_dim[:,np.newaxis],tf.ones(shape=(M+1,),dtype=tf.float64)/(M+1),[[m] for m in centers]+[[]],[[3.]]*M+[np.zeros((0))],[[0]]*M+[[]],mute=True,weights=my_betas_dim,alpha_prox_gamma=1e-3,steps=100)
            else:
                alphas_out,mu_out,Sigma_out,us_out,_=opt_em(samps_dim[:,np.newaxis],tf.ones(shape=(M+1,),dtype=tf.float64)/(M+1),[[m] for m in centers]+[[]],[[[0.1]]]*M+[np.zeros((0,0))],[[0]]*M+[[]],mute=True,weights=my_betas_dim,alpha_prox_gamma=1e-3,steps=100)
            for i in range(len(mu_out)):
                # compute initialization of the new components
                if us_out[i]==[]:
                    continue
                new_us.append(new_u.copy())
                mu_dim=mu_out[i].numpy()
                Sigma_dim=Sigma_out[i].numpy()
                mu_new=mus[my_k].numpy()
                ind=new_u.index(dim)
                mu_new=np.concatenate([mu_new[:ind],mu_dim,mu_new[ind:]],0)
                Sigma_new=Sigmas[my_k].numpy()
                if len(Sigmas[0].shape)==1:
                    if von_Mises:
                        Sigma_new=np.concatenate([Sigma_new[:ind],Sigma_dim,Sigma_new[ind:]],0)           
                    else:
                        Sigma_new=np.concatenate([Sigma_new[:ind],np.diag(Sigma_dim),Sigma_new[ind:]],0)
                else:
                    Sigma_new=np.concatenate([Sigma_new[:,:ind],np.zeros((Sigma_new.shape[0],1)),Sigma_new[:,ind:]],axis=1)            
                    Sigma_new=np.concatenate([Sigma_new[:ind,:],np.zeros((1,Sigma_new.shape[1])),Sigma_new[ind:,:]],axis=0)
                    Sigma_new[ind,ind]=Sigma_dim
                new_mus.append(mu_new)
                new_Sigmas.append(Sigma_new)
    if not mute:
        print('Component '+str(u_old)+' adds components '+str(new_us))
    weighting=tf.ones(len(new_us),dtype=tf.float64)/len(new_us)
    return weighting,new_us,new_mus,new_Sigmas

def remove_same(alphas,mus,Sigmas,us,mode=0):
    # Measure the distance of the components and unite components which are close together in terms of the KL divergence
    # INPUTS:
    #   - parameters alphas, mus, Sigmas, us
    #   - mode          - 0 for wrapped normal with full covariance matrices
    #                     1 for wrapped normal with diagonal covariance matrices
    #                     2 for products of von Mises distributions
    # OUTPUTS:
    #   - updated parameters alphas, mus, Sigmas, us
    i=0
    alphas=alphas.numpy()
    while i<len(us):
        j=i+1
        while j<len(us) and us[i]==us[j]:
            if mode==0 or mode==1:
                dist=MC_KL_wN(mus[i],Sigmas[i],mus[j],Sigmas[j])
                thresh=0.2
            else:
                dist=MC_KL_vM(mus[i],Sigmas[i],mus[j],Sigmas[j])
                thresh=0.2
            if dist>thresh:
                j+=1
            else:
                alphas[i]+=alphas[j]
                alphas=np.concatenate([alphas[:j],alphas[(j+1):]],0)
                us.pop(j)
                mus.pop(j)
                Sigmas.pop(j)
        i+=1
    return alphas,mus,Sigmas,us

def adaptive_selection(samples,d_s,M,weights=None,mode=0,mute=True,gamma_prox=1e-4,ks_thresh=2.5):
    # Applies the heuristic from Section 3.3 for d_s iterations.
    # INPUTS:
    #   - samples       - input data points
    #   - d_s           - number of repeats within the heuristic
    #   - M             - number of mixture components per dimension
    #   - weights       - weights for weighted samples, None for equal weights
    #   - mode          - 0 for wrapped normal with full covariance matrices
    #                     1 for wrapped normal with diagonal covariance matrices
    #                     2 for products of von Mises distributions
    #   - mute          - False for enabling debug prints, True for disabling debug prints
    # OUTPUTS:
    #   - resulting parameters alphas, mus, Sigmas, us
    n=samples.shape[0]
    d=samples.shape[1]
    alphas=tf.ones((1,),dtype=tf.float64)
    K=1
    # Initialization
    mus=[np.zeros(0)]
    if mode in [1,2]:
        Sigmas=[np.zeros((0))]
    else:
        Sigmas=[np.zeros((0,0))]
    us=[[]]
    # Loop over the number of runs of the heuristic
    for my_d_s in range(0,d_s):
        next_alphas=[]
        next_us=[]
        next_mus=[]
        next_Sigmas=[]
        # add components
        for k in range(K):
            w,u,m,S=test_uniform(samples,alphas,mus,Sigmas,us,k,M=M,mode=mode,mute=mute,sample_weights=weights,ks_thresh=ks_thresh)
            next_alphas.append(alphas[k]*w)
            next_us=next_us+u
            next_mus=next_mus+m
            next_Sigmas=next_Sigmas+S
        next_alphas=tf.concat(next_alphas,0)
        next_alphas/=tf.reduce_sum(next_alphas)
        inds=argsort_list(next_us)
        next_us=[next_us[i] for i in inds]
        us=next_us
        next_mus=[next_mus[i] for i in inds]
        next_Sigmas=[next_Sigmas[i] for i in inds]
        next_alphas=tf.constant(next_alphas.numpy()[inds],dtype=tf.float64)
        K=next_alphas.shape[0]
        # run proximal EM algorithm
        if mode==2:
            alphas,mus,Sigmas,step1=opt_em_von_Mises(samples,next_alphas,next_mus,next_Sigmas,us,steps=20,mute=mute,weights=weights)
            alphas,mus,Sigmas,us,step2=opt_em_von_Mises(samples,alphas,mus,Sigmas,us,steps=980,alpha_prox_gamma=gamma_prox,stop_crit=1e-7,mute=mute,weights=weights)
        else:
            alphas,mus,Sigmas,step1=opt_em(samples,next_alphas,next_mus,next_Sigmas,us,steps=20,mute=mute,weights=weights)
            alphas,mus,Sigmas,us,step2=opt_em(samples,alphas,mus,Sigmas,us,steps=980,alpha_prox_gamma=gamma_prox,stop_crit=1e-7,mute=mute,weights=weights)
        before_rms=alphas.shape[0]
        if K==before_rms and step1+step2<=2:
            break
        # remove doubled components
        alphas,mus,Sigmas,us=remove_same(alphas,mus,Sigmas,us,mode=mode)
        K=alphas.shape[0]
        if not mute:
            print('Removed '+str(before_rms-K)+' components because of similarity!')
            for k in range(K):
                print(str(us[k])+': '+str(alphas[k]))
                print(mus[k])
                print(Sigmas[k])
    return alphas,mus,Sigmas,us
