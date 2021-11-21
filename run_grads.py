# This code belongs to the paper
#
# J. Hertrich, F. A. Ba and G. Steidl (2022).
# Sparse Mixture Models inspired by ANOVA Decompositions.
# Electronic Transactions on Numerical Analysis, vol. 55, pp. 142-168.
#
# Please cite the paper if you use the code.
#
# This script reproduces the example from Section 4.3 of the paper.
#
import numpy as np
import math
from find_coupled_components import *
from EM_SPMM import *
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt

im_size=[8,10]

def generate_image(im_class=0):
    # Generates a image from one of the classes
    # INPUT:
    #   - im_class      - number of the class
    # OUTPUT:
    #   - generated image
    val1=np.random.normal()
    val2=np.random.normal()
    im=np.zeros(im_size)
    if im_class==0:
        im[:,:2]=.1+.05*val1
        im[:,2:]=.9+.1*val2
    elif im_class==1:
        im[:,:4]=.9+.1*val1
        im[:,4:]=.1+.05*val2
    elif im_class==2:
        im[:,:6]=.2+.025*val1
        im[:,6:]=.6+.05*val2
    elif im_class==3:
        im[:,:8]=.7+.1*val1
        im[:,8:]=.1+.05*val2
    elif im_class==4:
        im[:4]=.2+.1*val1
        im[4:]=.9+.025*val2
    elif im_class==5:
        im=val1
    return im+.2*np.random.normal(size=im_size)


def grad_directions(im):
    # Computes the orientations of the gradients for a given image
    # INPUT:
    #   - im    - input image
    # OUTPUT:
    #   - orientations of the symmetric gradients
    diffs1=im[1:-1,2:]-im[1:-1,:-2]
    diffs2=im[2:,1:-1]-im[:-2,1:-1]
    angle=np.angle(diffs1+diffs2*1j)
    angle[angle<0]+=2*math.pi
    angle/=2*math.pi
    return angle

def special_cut(im):
    # Selects the pixels from an image, where the orientation of the gradient is known.
    # INPUT:
    #   - im        - matrix which contains the orientation of the gradients
    # OUTPUT:
    #   - given data in the Example from Section 4.3
    cb1=list(range(0,im.shape[1],4))+list(range(1,im.shape[1],4))
    cb1.sort()
    cb2=list(range(2,im.shape[1],4))+list(range(3,im.shape[1],4))
    cb2.sort()
    out=[]
    for i in range(im.shape[0]):
        if i%4==0:
            out.append(im[i][cb1])
        elif i%4==2:
            out.append(im[i][cb2])
    out=np.concatenate(out,0)
    return out

def generate_data(alphas,N,return_labels=False):
    # Generates the data points for the Example from Section 4.3
    # INPUT:
    #   - alpha         - weights of the classes
    #   - N             - number of images to generate
    #   - return_labels - if this parameter is true, then the labels of the images will be returned, otherwise not.
    # OUTPUT:
    #   - data          - generated images
    #   - Optionally: labels of the data
    data=[]
    labels=[]
    for _ in range(N):
        im_class=np.random.choice(range(len(alphas)),p=alphas)
        labels.append(im_class)
        im=generate_image(im_class=im_class)
        gd=grad_directions(im)
        data.append(special_cut(gd))
    data=np.stack(data)
    if return_labels:
        return data,np.array(labels)
    return data

# visualize data
true_alphas=[0.,1.,0.,0.,0.]
data=generate_data(true_alphas,10000)

fig, axes = plt.subplots(2, 6, figsize=(24,8))
for i in range(6):
    axes[0,i].clear()
    axes[0,i].hist(data[:,i],bins=20)
    axes[0,i].title.set_text('Component '+str(i+1))
for i in range(6):
    axes[1,i].clear()
    axes[1,i].hist(data[:,6+i],bins=20)
    axes[1,i].title.set_text('Component '+str(7+i))
fig.savefig('histogram_class_2.eps', bbox_inches='tight',pad_inches = 0.05)
plt.close(fig)

# set seed for reproducibility
np.random.seed(30)
tf.random.set_seed(31)

# specify weights and generate data
true_alphas=[.2,.3,.1,.2,.2]
data=generate_data(true_alphas,10000)

# Learn the SPMM
alphas_out,mus_out,Sigmas_out,us_out=adaptive_selection(data,4,1,mode=0,mute=False,ks_thresh=3.)

# Evaluate couplings
coupled_components=[]
coupling_weights=[]
for u,alpha in zip(us_out,alphas_out):
    found=False
    for i in range(len(coupled_components)):
        if u==coupled_components[i]:
            coupling_weights[i]+=alpha
            found=True
            break
    if not found:
        coupled_components.append(u)
        coupling_weights.append(alpha)

perm=argsort_list(coupled_components)
coupled_components=[coupled_components[i] for i in perm]
coupling_weights=[tf.identity(cw).numpy() for cw in [coupling_weights[i] for i in perm]]

# Coupling plot
fig=plt.figure()
plt.bar([str(u) for u in coupled_components],coupling_weights)
fig.savefig('coupled_components_grads.eps', bbox_inches='tight',pad_inches = 0.05)
plt.close(fig)

# get supervised data points
labled_data=[]
for i in range(len(true_alphas)):
    my_alphas=np.zeros(len(true_alphas))
    my_alphas[i]=1.
    my_alphas=list(my_alphas)
    labled_data.append(generate_data(my_alphas,3))

im_classes=[]
# connect components of the SPMM with the image classes
for k in range(len(us_out)):
    my_alphas=np.ones(1)
    i_max=-1
    ll_min=np.inf
    for i in range(len(true_alphas)):
        ll_i=log_likelihood_SPMM(labled_data[i],my_alphas,mus_out[k:k+1],Sigmas_out[k:k+1],us_out[k:k+1],return_probs=False,von_Mises=False,compile_inner=False).numpy()
        if ll_i<=ll_min:
            i_max=i
            ll_min=ll_i
    im_classes.append(i_max)
print(us_out,im_classes)

# Test the model
n_test=1000
data_test,labels_test=generate_data(true_alphas,n_test,return_labels=True)
predicted_labels=classify(data_test,alphas_out,mus_out,Sigmas_out,us_out,5,im_classes)
# print results
print(predicted_labels[:10])
print(labels_test[:10])
correct=np.sum(predicted_labels==labels_test)
print(correct)
print(correct*1.0/n_test)

