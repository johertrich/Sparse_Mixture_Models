# Sparse Mixture Models inspired by ANOVA

This repository contains the numerical examples of the paper [1]. Please cite the paper, if you use this code.

It is available at  
https://arxiv.org/abs/2105.14893

For questions and bug reports, please contact Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

## Requirements and Usage

The code is written in Python using Tensorflow 2.3.0.
Usually the code is also compatible with some other versions of Tensorflow 2.x.
A rough specification of each function/script is given in the corresponding header.

The script `run_example.py` reproduces the examples from Section 4.1 and 4.2 and the script 
`run_grads.py` reproduces the examples from Section 4.3 of [1].

Note that the code is highly experimental, far from optimized and only sparsely commented, so use it with care.

## Reference

[1] J. Hertrich, F. A. Ba and G. Steidl (2021).  
Sparse Mixture Models inspired by ANOVA.  
Arxiv preprint arXiv:2105.14893
