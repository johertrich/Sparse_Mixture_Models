# This code belongs to the paper
#
# J. Hertrich, F. A. Ba and G. Steidl (2021).
# Sparse Mixture Models inspired by ANOVA.
# Arxiv preprint arXiv:2105.14893
#
# Please cite the paper if you use the code.
#
# This script reproduces the numerical examples from Section 4.1 and Section 4.2 of the paper.
#
from find_coupled_components import *
from EM_SPMM import *
from utils import *
from sample import *
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt

mm_model_list=[1,2]
n_test=100000
def generate_samples(model_nr, n=10000):
    # Generates samples from the different density functions considered in the paper.
    # INPUT:
    #   - model_nr                  - 1 and 2 for the functions considered in Sec 4.1
    #                                 3 and 4 for the functions considered in Sec 4.2
    #   - n                         - number of samples to generate
    # OUTPUTS:
    #   - samples                   - n samples from the corresponding model
    # If model_nr is 1 or 2: 
    #   - [alphas, mus,Sigmas,us]   - parameters of the corresponding mixture model
    # else:
    #   - fun_l1                    - approximated L1-norm of the corresponding function
    sig_sq=0.1**2
    if model_nr==1:
        d=10
        alphas=tf.constant([.2,.2,.2,.2,.1,.1],dtype=tf.float64)
        mus=[[.5,.5],[.5,.5],[.5,.5,.5],[.5,.5],[.5,.5],[.5]]
        Sigmas=[[[sig_sq,0.],[0.,sig_sq]],[[sig_sq,0.],[0.,sig_sq]],[[sig_sq,0.,0.],[0.,sig_sq,0.],[0.,0.,sig_sq]],[[sig_sq,0.],[0.,sig_sq]],[[sig_sq,0.],[0.,sig_sq]],[[sig_sq]]]
        us=[[0,1],[2,3],[4,5,6],[6,7],[8,9],[2]]
        samples,_=sample_WGMM_unif(alphas,mus,Sigmas,us,d,n)    
    elif model_nr==3:
        d=9
        samples=rejection_sampling_uniform(prods_of_splines,25.,d,n)
        test_points=np.random.uniform(size=(n_test,d))
        fun_vals1=prods_of_splines(test_points)
        fun_l1=np.sum(np.abs(fun_vals1))/n_test 
    elif model_nr==2:
        d=10
        alphas=tf.constant([.2,.2,.2,.2,.1,.1],dtype=tf.float64)
        mus=[[.5,.5],[.5,.5],[.5,.5,.5],[.5,.5],[.5,.5],[.5]]
        Sigmas=[sig_sq*np.array([[1.,.5],[.5,1.]]),sig_sq*np.array([[1.,.5],[.5,1.]]),sig_sq*np.array([[1.,.3,.2],[.3,1.,.1],[.2,.1,1.]]),sig_sq*np.array([[1.,-.6],[-.6,1.]]),sig_sq*np.array([[1.,.1],[.1,1.]]),sig_sq*np.array([[1.]])]
        us=[[0,1],[2,3],[4,5,6],[6,7],[8,9],[2]]
        samples,_=sample_WGMM_unif(alphas,mus,Sigmas,us,d,n)    
    elif model_nr==4:
        d=10
        samples=rejection_sampling_uniform(Friedmann,35.,d,n)
        test_points=np.random.uniform(size=(n_test,d))
        fun_vals1=Friedmann(test_points)
        fun_l1=np.sum(np.abs(fun_vals1))/n_test
    if model_nr in mm_model_list:
        return samples,[alphas,mus,Sigmas,us]
    return samples,fun_l1
  
def run_example(samples,model_nr,mode=0):
    # Estimates a SPMM using the heuristic of Section 3.3 of the paper.
    # INPUTs:
    #   - samples       - data points, for which the SPMM should be estimated
    #   - mode          - 0 for wrapped normal with full covariance matrices
    #                     1 for wrapped normal with diagonal covariance matrices
    #                     2 for products of von Mises distributions
    # OUTPUTs:
    #   - estimated parameters [alphas_out, mus_out,Sigmas_out,us_out]
    #   - estimation time
    d=samples.shape[1]
    tic=time.time()
    if model_nr==4:
        alphas_out,mus_out,Sigmas_out,us_out=adaptive_selection(samples,2,2,mode=mode,mute=True,gamma_prox=1e-4,ks_thresh=3.)
    else:
        alphas_out,mus_out,Sigmas_out,us_out=adaptive_selection(samples,3,1,mode=mode,mute=True,gamma_prox=1e-4,ks_thresh=4.)
    toc=time.time()-tic  
    return [alphas_out,mus_out,Sigmas_out,us_out],toc

def evaluate_example(model_nr,samples,results,ground_truth=None,l1_norm=None,von_Mises=False):
    # Evaluation of the quality measures for the estimated SPMM.
    # INPUTs:
    #   - model_nr      - number of the ground truth model
    #   - samples       - data points for estimation
    #   - results       - estimated parameters [alphas_out, mus_out, Sigmas_out, us_out]
    #   - ground_truth  - ground truth parameters [alphas, mus, Sigmas, us], None for the examples from Section 4.2
    #   - l1_norm       - l1_norm for the ground truth function, None for the examples from Section 4.1
    #   - von_Mises     - True, if the estimated model is a von Mises SPMM. False otherwise.
    # OUTPUTs:
    #   - ll_f          - log likelihood of the ground truth function
    #   - new_ll        - log likelihood of the estimated SPMM
    #   - rel_l1_err    - relative l1 error of the estimated SPMM
    #   - rel_l2_err    - relative l2 error of the estimated SPMM
    #   - mse           - mean square error of the estimated SPMM
    alphas_out,mus_out,Sigmas_out,us_out=results
    d=samples.shape[1]
    if model_nr in mm_model_list:
        alphas,mus,Sigmas,us=ground_truth
    n_test=100000
    test_points=np.random.uniform(size=(n_test,d))
    if model_nr==3:
        fun_vals1=prods_of_splines(test_points)
    elif model_nr==4:
        fun_vals1=Friedmann(test_points)
    elif model_nr in mm_model_list:
        _,fun_vals1=log_likelihood_SPMM(test_points,alphas,mus,Sigmas,us,return_probs=True)
    _,probs=log_likelihood_SPMM(test_points,alphas_out,mus_out,Sigmas_out,us_out,return_probs=True,von_Mises=von_Mises)
    new_ll,probs_samps=log_likelihood_SPMM(samples,alphas_out,mus_out,Sigmas_out,us_out,return_probs=True,von_Mises=von_Mises)
    fun_l2=np.sqrt(np.sum(fun_vals1**2)/n_test)
    fun_l1=np.sum(np.abs(fun_vals1))/n_test
    av_fac=1/fun_l1
    print('Average factor: ' +str(av_fac))
    probs_l1=np.sum(np.abs(probs))/n_test
    print(probs_l1)
    print('Approx L1-Norm of original function: ' +str(fun_l1))
    print('Approx L2-Norm of original function: ' +str(fun_l2))
    l1_err=np.sum(np.abs((probs.numpy()/av_fac-fun_vals1)))/n_test
    l2_err=np.sqrt(np.sum((probs.numpy()/av_fac-fun_vals1)**2)/n_test)
    rel_l1_err=l1_err/fun_l1
    rel_l2_err=l2_err/fun_l2
    mse=l2_err**2
    print('L1-Error: ' +str(l1_err))
    print('L2-Error: ' +str(l2_err))
    print('MSE: '+str(mse))
    print('Relative L1-Error: ' +str(rel_l1_err))
    print('Relative L2-Error: ' +str(rel_l2_err))
    if model_nr in mm_model_list:
        ll_f=log_likelihood_SPMM(samples,alphas,mus,Sigmas,us)
        print('Negative Log-Likelihood of original parameters: '+str(ll_f.numpy()))
    else:
        if model_nr==3:
            fun_samps=prods_of_splines(samples)
        elif model_nr==4:
            fun_samps=Friedmann(samples)
        ll_f=-np.sum(np.log(fun_samps/l1_norm))
        print('Negative Log-Likelihood of original function: '+str(ll_f))
    print('Negative Log-Likelihood of estimated parameters: '+str(new_ll.numpy()))
    return ll_f,new_ll.numpy(),rel_l1_err,rel_l2_err,mse


def simulation_study(model_nr=1,repeats=10):
    # Runs and evaluates the examples from Section 4.1 and 4.2 several times.
    # Saves the results in the directory "simulation_results"
    # INPUTS:
    #   - model_nr      - number of the ground truth model
    #   - repeats       - number of repeats

    # set seed (note that the results in the paper were produced without seed)
    np.random.seed(20)
    tf.random.set_seed(21)
    metrics_full=[]
    metrics_diagonal=[]
    metrics_vM=[]
    times_full=[]
    times_diagonal=[]
    times_vM=[]
    coupled_components_full=[]
    coupling_weights_full=[]
    coupled_components_diag=[]
    coupling_weights_diag=[]
    coupled_components_vM=[]
    coupling_weights_vM=[]
    # main loop
    for i in range(repeats):
        print('Run '+str(i+1)+':')
        # sample
        if model_nr in mm_model_list:
            samples,ground_truth=generate_samples(model_nr)
            fun_l1=1.
        else:
            samples,fun_l1=generate_samples(model_nr)
            ground_truth=None
        # run heuristic
        print('full covariances')
        results_full,time_full=run_example(samples,model_nr,mode=0)
        print('diagonal covariances')
        results,time_diag=run_example(samples,model_nr,mode=1)
        print('von Mises')
        results_vM,time_vM=run_example(samples,model_nr,mode=2)
        times_full.append(time_full)
        times_diagonal.append(time_diag)
        times_vM.append(time_vM)
        print("Time full: "+str(time_full)+" Time diagonal "+str(time_diag)+ " Time von Mises: "+str(time_vM))
        # evaluate results
        for i,[alphas,mus,Sigmas,us] in enumerate([results_full,results,results_vM]):
            for k in range(len(us)):
                print(str(us[k])+': '+str(alphas[k]))
                print(mus[k])
                if i==2:
                    print(Sigmas[k])
                    print((-2*tf.math.log(A(Sigmas[k]))/((2*math.pi)**2)))
                else:
                    print(Sigmas[k])
        print("Time full: "+str(time_full)+" Time diagonal "+str(time_diag)+ " Time von Mises: "+str(time_vM))

        alphas,mus,Sigmas,us=results_full
        for u,alpha in zip(us,alphas):
            found=False
            for i in range(len(coupled_components_full)):
                if u==coupled_components_full[i]:
                    coupling_weights_full[i]+=alpha
                    found=True
                    break
            if not found:
                coupled_components_full.append(u)
                coupling_weights_full.append(alpha)

        alphas,mus,Sigmas,us=results
        for u,alpha in zip(us,alphas):
            found=False
            for i in range(len(coupled_components_diag)):
                if u==coupled_components_diag[i]:
                    coupling_weights_diag[i]+=alpha
                    found=True
                    break
            if not found:
                coupled_components_diag.append(u)
                coupling_weights_diag.append(alpha)

        alphas,mus,Sigmas,us=results_vM
        for u,alpha in zip(us,alphas):
            found=False
            for i in range(len(coupled_components_vM)):
                if u==coupled_components_vM[i]:
                    coupling_weights_vM[i]+=alpha
                    found=True
                    break
            if not found:
                coupled_components_vM.append(u)
                coupling_weights_vM.append(alpha)


        met_full=evaluate_example(model_nr,samples,results_full,ground_truth=ground_truth,l1_norm=fun_l1)
        met_diag=evaluate_example(model_nr,samples,results,ground_truth=ground_truth,l1_norm=fun_l1)
        met_vM=evaluate_example(model_nr,samples,results_vM,ground_truth=ground_truth,l1_norm=fun_l1,von_Mises=True)
        metrics_full.append(met_full)
        metrics_diagonal.append(met_diag)
        metrics_vM.append(met_vM)

    # compute couplings
    perm_full=argsort_list(coupled_components_full)
    perm_diag=argsort_list(coupled_components_diag)
    perm_vM=argsort_list(coupled_components_vM)
    coupled_components_full=[coupled_components_full[i] for i in perm_full]
    coupled_components_diag=[coupled_components_diag[i] for i in perm_diag]
    coupled_components_vM=[coupled_components_vM[i] for i in perm_vM]
    coupling_weights_full=[tf.identity(cw).numpy()/repeats for cw in [coupling_weights_full[i] for i in perm_full]]
    coupling_weights_diag=[tf.identity(cw).numpy()/repeats for cw in [coupling_weights_diag[i] for i in perm_diag]]    
    coupling_weights_vM=[tf.identity(cw).numpy()/repeats for cw in [coupling_weights_vM[i] for i in perm_vM]]

    # generate print
    my_print="\nEvaluation full Sigma:\n"
    my_print+="Average Time: "+str(np.mean(times_full))+"+-"+str(np.std(times_full))+"\n"
    my_print+="Average likelihood ground truth: "+str(np.mean([x[0] for x in metrics_full]))+"+-"+str(np.std([x[0] for x in metrics_full]))+"\n"
    my_print+="Average likelihood estimate: "+str(np.mean([x[1] for x in metrics_full]))+"+-"+str(np.std([x[1] for x in metrics_full]))+"\n"
    my_print+="Average rel L1-Error: "+str(np.mean([x[2] for x in metrics_full]))+"+-"+str(np.std([x[2] for x in metrics_full]))+"\n"
    my_print+="Average rel L2-Error: "+str(np.mean([x[3] for x in metrics_full]))+"+-"+str(np.std([x[3] for x in metrics_full]))+"\n"
    my_print+="Average MSE: "+str(np.mean([x[4] for x in metrics_full]))+"+-"+str(np.std([x[4] for x in metrics_full]))+"\n"
    my_print+="Coupled components: "+str(coupled_components_full)+"\n"
    my_print+="Coupling weights: "+str(coupling_weights_full)+"\n"


    my_print+="\nEvaluation diagonal Sigma:\n"
    my_print+="Average Time: "+str(np.mean(times_diagonal))+"+-"+str(np.std(times_diagonal))+"\n"
    my_print+="Average likelihood ground truth: "+str(np.mean([x[0] for x in metrics_diagonal]))+"+-"+str(np.std([x[0] for x in metrics_diagonal]))+"\n"
    my_print+="Average likelihood estimate: "+str(np.mean([x[1] for x in metrics_diagonal]))+"+-"+str(np.std([x[1] for x in metrics_diagonal]))+"\n"
    my_print+="Average rel L1-Error: "+str(np.mean([x[2] for x in metrics_diagonal]))+"+-"+str(np.std([x[2] for x in metrics_diagonal]))+"\n"
    my_print+="Average rel L2-Error: "+str(np.mean([x[3] for x in metrics_diagonal]))+"+-"+str(np.std([x[3] for x in metrics_diagonal]))+"\n"
    my_print+="Average MSE: "+str(np.mean([x[4] for x in metrics_diagonal]))+"+-"+str(np.std([x[4] for x in metrics_diagonal]))+"\n"
    my_print+="Coupled components: "+str(coupled_components_diag)+"\n"
    my_print+="Coupling weights: "+str(coupling_weights_diag)+"\n"

    my_print+="\nEvaluation von Mises:\n"
    my_print+="Average Time: "+str(np.mean(times_vM))+"+-"+str(np.std(times_vM))+"\n"
    my_print+="Average likelihood ground truth: "+str(np.mean([x[0] for x in metrics_vM]))+"+-"+str(np.std([x[0] for x in metrics_vM]))+"\n"
    my_print+="Average likelihood estimate: "+str(np.mean([x[1] for x in metrics_vM]))+"+-"+str(np.std([x[1] for x in metrics_vM]))+"\n"
    my_print+="Average rel L1-Error: "+str(np.mean([x[2] for x in metrics_vM]))+"+-"+str(np.std([x[2] for x in metrics_vM]))+"\n"
    my_print+="Average rel L2-Error: "+str(np.mean([x[3] for x in metrics_vM]))+"+-"+str(np.std([x[3] for x in metrics_vM]))+"\n"
    my_print+="Average MSE: "+str(np.mean([x[4] for x in metrics_vM]))+"+-"+str(np.std([x[4] for x in metrics_vM]))+"\n"
    my_print+="Coupled components: "+str(coupled_components_vM)+"\n"
    my_print+="Coupling weights: "+str(coupling_weights_vM)+"\n"

    # save results
    if not os.path.isdir('simulation_results'):
        os.mkdir('simulation_results')
    timestamp=int(time.time())
    myfile=open('simulation_results/sim_stud_model'+str(model_nr)+'_'+str(timestamp)+'.txt',"w")
    myfile.write(my_print)
    myfile.close()
    print(my_print)
    fig=plt.figure()
    plt.bar([str(u) for u in coupled_components_full],coupling_weights_full)
    fig.savefig('simulation_results/coupled_components_model'+str(model_nr)+'_full_'+str(timestamp)+'.eps', bbox_inches='tight',pad_inches = 0.05)
    plt.close(fig)
    fig=plt.figure()
    plt.bar([str(u) for u in coupled_components_diag],coupling_weights_diag)
    fig.savefig('simulation_results/coupled_components_model'+str(model_nr)+'_diag_'+str(timestamp)+'.eps', bbox_inches='tight',pad_inches = 0.05)
    plt.close(fig)
    fig=plt.figure()
    plt.bar([str(u) for u in coupled_components_vM],coupling_weights_vM)
    fig.savefig('simulation_results/coupled_components_model'+str(model_nr)+'_vM_'+str(timestamp)+'.eps', bbox_inches='tight',pad_inches = 0.05)
    plt.close(fig)

# Call the simulation study for all models from Section 4.1 and 4.2
if __name__=='__main__':    
    model_nrs=[1,2,3,4]
    for model_nr in model_nrs:
        simulation_study(model_nr,repeats=10)
