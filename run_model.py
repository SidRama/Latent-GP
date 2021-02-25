######## Source code for running the model described in the manuscript:    ########
## Latent Gaussian process with composite likelihoods and numerical quadrature   ##
##																				 ##
## This file is the main file for executing the model.                           ##
##                                          									 ##
## This code makes use of the Theano Python library.                             ##
################################################################################### 

import sys; sys.path.append("./")
from model_opt import LGP_opt
import numpy as np
import theano
import pandas as pd
import time
import datetime
import sys


def execute_model(obs, gaussian_idx, binomial_idx, beta_idx, poisson_idx, gamma_idx, 
	mask=None, N = 100, M = 4, Q = 2, K = 1, D = 6, n_hidden_m=40,  
	max_iteration = 20, minibatch_size=500, sx2 = 1, kl_weight=1):
	"""
    Builds and exectues the model

    :param obs: training data
    :param gaussian_idx:  list of indices for the gaussian distributed data
    :param binomial_idx:  list of indices for the binomial distributed data
    :param beta_idx:  list of indices for beta distributed data
    :param poisson_idx:  list of indices for poisson distributed data
    :param gamma_idx:  ist of indices for gamma distributed data
    :param mask: mask for missing values
    :param N: number of instances
    :param M: number of inducing points
    :param Q: dimensionality of the latent space
    :param K: number of samples
    :param D: number of attributes
    :param n_hidden_m: number of hidden nodes
    :param max_iteration: number of iterations
    :param minibatch_size: size of the minibatch
    :param sx2: optional variance term
    :param kl_weight: weight for the KL term

    :return: trained model and trace for KL terms
    """

    # Random initialisation 
    Z = np.random.randn(M,Q)

    # mean nnet
    # Glorot & Bengio
    W_hidden_1 = np.random.uniform(low=-np.sqrt(6. / (D + n_hidden_m)), 
    							   high=np.sqrt(6. / (D + n_hidden_m)), 
    							   size = (D, n_hidden_m))
    b_hidden_1 = np.zeros((1, n_hidden_m))
    W_lin_1 = np.random.uniform(low=-np.sqrt(6. / (n_hidden_m + Q)), 
    							high=np.sqrt(6. / (n_hidden_m + Q)), 
    							size = (n_hidden_m, Q))
    b_lin_1 = np.zeros((1,Q))

    # cov net
    W_hidden_2 = np.random.uniform(low=-np.sqrt(6. / (D + n_hidden_m)), 
                                   high=np.sqrt(6. / (D + n_hidden_m)), 
                                   size = (D, n_hidden_m))
    b_hidden_2 = np.zeros((1, n_hidden_m))
    W_lin_2 = np.random.uniform(low=-np.sqrt(6. / (n_hidden_m + Q)), 
                                high=np.sqrt(6. / (n_hidden_m + Q)), 
                                size = (n_hidden_m, Q))
    b_lin_2 = np.zeros((1,Q))
    mu = np.random.randn(D,M,K) * 1e-2
    lLs = np.zeros((D,M,M))
    lhyp = np.zeros((Q + 2))
    sigma = np.exp(np.random.randn() * 1e-1)
    nu = np.exp(np.random.randn() * 1e-1)
    params = {'Z': Z, 'W_hidden_1':W_hidden_1, 'b_hidden_1':b_hidden_1,
              'W_lin_1': W_lin_1, 'b_lin_1':b_lin_1,'W_hidden_2':W_hidden_2, 
              'b_hidden_2':b_hidden_2, 'W_lin_2': W_lin_2, 'b_lin_2':b_lin_2,
              'lL': lLs, 'lhyp': lhyp, 'sigma':sigma, 'nu':nu}

    for i in xrange(D):
        params['mu' + str(i)] = mu[i]

    lgp = LGP_opt(params, obs,gaussian_idx, binomial_idx, beta_idx, 
                    poisson_idx, gamma_idx, samples=10, kl_weight=kl_weight)

    lgp.Y_orig = obs
    print 'data shape: ' + str(obs.shape)
    print 'Optimising...'
    iteration = 0
    elbo_arr = np.array([])
    KL_X_arr = np.array([])
    KL_U_arr = np.array([])
    LS_arr = np.array([])
    best = -1e100
    chk_pt = 1
    X = np.array([])
    TIME0 = time.time()
    while iteration < max_iteration:
        lgp.opt_one_step(params.keys(), iteration, opt='adam')
        [current_ELBO, LS_val, KL_U_val, std_sum, KL_X_val]=lgp.ELBO(obs,
                                                            mask=lgp.mask_orig)
        elbo_arr = np.append(elbo_arr, current_ELBO)
	KL_X_arr = np.append(KL_X_arr, KL_X_val)
        KL_U_arr = np.append(KL_U_arr, KL_U_val)
        LS_arr = np.append(LS_arr, LS_val)
        print str(datetime.timedelta(seconds=time.time() - TIME0)) + 
                    ' Iter '+str(iteration)+': '+str(current_ELBO) +
                    ' +- '+str(std_sum)
        if current_ELBO > best:
            best = current_ELBO
            X = lgp.estimate(lgp.f['X'], obs)[0]
            np.save(fname_out, [X, elbo_arr, lgp.params, KL_X_arr, 
                    KL_U_arr, LS_arr])
        if np.isnan(current_ELBO):
            break
        iteration += 1
    X = lgp.estimate(lgp.f['X'], obs)[0]
    return (X, elbo_arr, lgp.params, lgp, KL_X_arr, KL_U_arr, LS_arr)


##### Load data ####
# 3 dimensional: (D, N, K). K is 1 in our use-case so, it should be 
# (D, N, 1) where D is the total number of features/covariates, N is 
# the number of patients.

########## SET PARAMETERS HERE ###############
##############################################

Q = int(sys.argv[-3])
fname_in = sys.argv[-2]
fname_out_original = sys.argv[-1]

# data in 
DATA, LIKELIHOODS, _ = np.load(fname_in)
'''
idx = np.where(np.array(LIKELIHOODS)!='beta')[0]
LIKELIHOODS=[x for x in LIKELIHOODS if x!='beta']
DATA = DATA.iloc[:,idx]
DATA=DATA.dropna()
LIKELIHOODS=LIKELIHOODS[:DATA.shape[1]]
'''
cols = DATA.columns
DATA = np.array(DATA)
LIKELIHOODS = np.array(LIKELIHOODS)
gaussian_idx = np.where(LIKELIHOODS=='gaussian')[0]
binomial_idx = np.where(LIKELIHOODS=='binomial')[0]
beta_idx = np.where(LIKELIHOODS=='beta')[0]
poisson_idx = np.where(LIKELIHOODS=='poisson')[0]
gamma_idx = np.where(LIKELIHOODS=='gamma')[0]

# normalize gaussian variables
m = np.nanmean(DATA[:, gaussian_idx], axis=0)
s = np.nanstd(DATA[:, gaussian_idx], axis=0)
DATA[:,gaussian_idx] = (DATA[:, gaussian_idx]-m)/s
DATA[:, beta_idx] = DATA[:, beta_idx] / 100

#### Set number of test samples ###
test_size = 10

test_indices = np.random.choice(np.arange(DATA.shape[0]), test_size, replace=False)
test_DATA = DATA[test_indices, :]
DATA = DATA[~np.isin(np.arange(DATA.shape[0]), test_indices), :]

mask = np.isnan(DATA) # this is of dimension N x D
DATA = np.nan_to_num(DATA)
DATA = DATA.transpose()[:,:,np.newaxis]
D, N, _ = DATA.shape

test_DATA = np.nan_to_num(test_DATA)
test_DATA = test_DATA.transpose()[:,:,np.newaxis]

kl_weight = 1 # ENTER KL_X weight over here

M = 50 # number of indcuing points
#Q = 2  # dimensionality of latent space
K = 1
max_iteration = 1000 # number of iterations
minibatch_size = N # size of mini-batch
n_hidden_m = 40 # number of neurons in the first hidden layer

##############################################
ftmp = fname_out_original.split('.')
for ro in range(5):
    try:
        fname_out = ftmp[0]+'_r'+str(ro)+'.'+ftmp[1]
        fname_out_end = ftmp[0]+'_r'+str(ro)+'_end.'+ftmp[1]
        (X, elbo_arr, para, lgp, KL_X_arr, KL_U_arr, LS_arr) = execute_model(obs=DATA, 
            gaussian_idx=gaussian_idx, binomial_idx=binomial_idx,beta_idx=beta_idx, 
            poisson_idx=poisson_idx, gamma_idx=gamma_idx, N=N, M=M, Q=Q, K=K, D=D, 
            n_hidden_m=n_hidden_m, max_iteration = max_iteration, minibatch_size=minibatch_size, 
            kl_weight=kl_weight)
	
	test_obj = LGP_opt(para, test_DATA, gaussian_idx, binomial_idx, beta_idx, 
                            poisson_idx, gamma_idx)
	[test_lik, test_std] = test_obj.pred_lik(test_DATA, mask=test_obj.mask_orig)
	print('Predictive log likelihood: ' + str(test_lik))
        np.save(fname_out_end, [X, elbo_arr, lgp.params, test_lik, KL_X_arr, KL_U_arr, LS_arr])
    except:
        continue
    print('Output file: ' + fname_out)
