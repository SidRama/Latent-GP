import model_def
from model_def import LGP_model
import numpy as np


class LGP_opt:
    def __init__(self, params, Y, gaussian_idx, binomial_idx, beta_idx, 
                 poisson_idx, gamma_idx, mask=None, sx2=1, linear_model=False, 
                 samples = 20, use_hat = False, kl_weight= 1):
        """
        Builds and exectues the model

        :param params: parameters to be learnt
        :param Y:  training data
        :param binomial_idx:  list of indices for the Gaussian distributed data
        :param binomial_idx:  list of indices for the binomial distributed data
        :param beta_idx:  list of indices for beta distributed data
        :param poisson_idx:  list of indices for poisson distributed data
        :param gamma_idx:  ist of indices for gamma distributed data
        :param mask: mask for missing values
        :param sx2: optional variance term
        :param linear_model: boolean for the use of a linear model
        :param samples: number of Monte Carlo samples
        :param use_hat: boolean for save file
        :param kl_weight: weight for the KL term

        """

        self.Y = Y
        self.Y_orig = Y
        self.Y_epoch = Y
        self.gaussian_idx = gaussian_idx
        self.binomial_idx = binomial_idx
        self.beta_idx = beta_idx
        self.poisson_idx = poisson_idx
        self.gamma_idx = gamma_idx
	    self.kl_weight = kl_weight
        if mask is None:
            self.mask = np.zeros((Y[0].shape[0], len(Y)), dtype=bool)
        else:
            self.mask = mask
        self.mask_orig = self.mask
        self.mask_orig_epoch = self.mask
        self.lgp = LGP_model(params, gaussian_idx, binomial_idx, beta_idx, 
                               poisson_idx, gamma_idx, sx2, 
                               linear_model=linear_model, samples=samples, 
                               use_hat=use_hat, kl_weight=kl_weight)
        self.ELBO, self.f = self.lgp.ELBO, self.lgp.f
        self.params, self.KmmInv = self.lgp.params, self.lgp.KmmInv
        self.exec_f, self.estimate, self.pred_lik = self.lgp.exec_f, self.lgp.estimate, self.lgp.pred_lik
        self.param_updates = {n: np.zeros_like(v) for n, v in params.iteritems()}
        self.moving_mean_squared = {n: np.zeros_like(v) for n, v in params.iteritems()}
        self.moving_mean = {n: np.zeros_like(v) for n, v in params.iteritems()}
        self.learning_rates = {n: 1e-2*np.ones_like(v) for n, v in params.iteritems()}

    def get_KmmInv_grad(self, Y, mask, (M, Q)):
         """
        Compute Kmm^{-1} gradient 

        :param Y: data
        :param mask: mask for unobserved values
        :param M: number of inducing points:
        :param Q: dimensionality of the latent space

        :return: computed gradient
        """
        dKL_U_dKmmInv, dLS_dKmmInv = {}, {}
        for modality in xrange(len(Y)):
                
            if modality in self.binomial_idx:
                dKL_U_dKmmInv[modality] = self.exec_f(self.lgp.g['KmmInv']['KL_U'], 
                                                       Y, modality, mask[:, modality])
                dLS_dKmmInv[modality] = self.estimate(self.lgp.g['KmmInv']['LS'], 
                                                       Y, modality, mask[:, modality])[0]
            elif modality in self.gaussian_idx:
                dKL_U_dKmmInv[modality] = self.exec_f(self.lgp.g2['KmmInv']['KL_U'], 
                                                       Y, modality, mask[:, modality])
                dLS_dKmmInv[modality] = self.estimate(self.lgp.g2['KmmInv']['LS'], 
                                                       Y, modality, mask[:, modality])[0]
            elif modality in self.beta_idx:
                dKL_U_dKmmInv[modality] = self.exec_f(self.lgp.g3['KmmInv']['KL_U'], 
                                                       Y, modality, mask[:, modality])
                dLS_dKmmInv[modality] = self.estimate(self.lgp.g3['KmmInv']['LS'], 
                                                       Y, modality, mask[:, modality])[0]
            elif modality in self.poisson_idx:
                dKL_U_dKmmInv[modality] = self.exec_f(self.lgp.g4['KmmInv']['KL_U'], 
                                                      Y, modality, mask[:, modality])
                dLS_dKmmInv[modality] = self.estimate(self.lgp.g4['KmmInv']['LS'], 
                                                      Y, modality, mask[:, modality])[0]
            elif modality in self.gamma_idx:
                dKL_U_dKmmInv[modality] = self.exec_f(self.lgp.g5['KmmInv']['KL_U'], 
                                                      Y, modality, mask[:, modality])
                dLS_dKmmInv[modality] = self.estimate(self.lgp.g5['KmmInv']['LS'], 
                                                      Y, modality, mask[:, modality])[0]
        
        df_dn_i = {'Z': {'KL_U': {}, 'LS': {}}, 'lhyp': {'KL_U': {}, 'LS': {}}}
        dKmm_dlhyp = self.lgp.dKmm_d['lhyp'](self.params['Z'], 
                                              self.params['lhyp']).reshape(M, M, -1)
        dKmm_dn_KmmInv = np.dot(dKmm_dlhyp.transpose((2, 0, 1)), self.lgp.KmmInv)
        KmmInv_dKmm_dn_KmmInv = np.dot(dKmm_dn_KmmInv.transpose((0,2,1)),
                                      self.lgp.KmmInv.T).transpose((0,2,1))
        dKmmInv_dlhyp = -1.0 * KmmInv_dKmm_dn_KmmInv.transpose((1,2,0))

        dKmm_dZ = self.lgp.dKmm_d['Z'](self.params['Z'], 
                                        self.params['lhyp']).reshape(M, M, M, Q)
        dKmm_dn_KmmInv = np.dot(dKmm_dZ.transpose((2,3,0,1)), self.lgp.KmmInv)
        KmmInv_dKmm_dn_KmmInv = np.dot(dKmm_dn_KmmInv.transpose((0,1,3,2)),
                                       self.lgp.KmmInv.T).transpose((0,1,3,2))
        dKmmInv_dZ = -1.0 * KmmInv_dKmm_dn_KmmInv.transpose((2,3,0,1))

        for modality in xrange(len(Y)):
            df_dn_i['lhyp']['KL_U'][modality] = (dKL_U_dKmmInv[modality][:,:,None] * dKmmInv_dlhyp).sum(0).sum(0)
            df_dn_i['lhyp']['LS'][modality] = (dLS_dKmmInv[modality][:,:,None] * dKmmInv_dlhyp).sum(0).sum(0)
            df_dn_i['Z']['KL_U'][modality] = (dKL_U_dKmmInv[modality][:,:,None,None] * dKmmInv_dZ).sum(0).sum(0)
            df_dn_i['Z']['LS'][modality] = (dLS_dKmmInv[modality][:,:,None,None] * dKmmInv_dZ).sum(0).sum(0)
        return df_dn_i

    def get_grad(self, param_name, Y, KmmInv_grad, mask = None):
        """
        Compute gradients

        :param param_name: gradient with respect to parameter
        :param Y: data
        :KmmInv_grad: gradient with respect to Kmm^{-1} 
        :param mask: mask for unobserved values

        :return: computed gradient
        """
        if 'mu' in param_name:
            modality = int(param_name[2:])
            temp = []
            if modality in self.binomial_idx:
                temp = self.exec_f(self.lgp.g['mu']['KL_U'], Y, modality, mask[:, modality]) 
                        + self.estimate(self.lgp.g['mu']['LS'], Y, modality, mask[:, modality])[0]
            elif modality in self.gaussian_idx:
                temp = self.exec_f(self.lgp.g2['mu']['KL_U'], Y, modality, mask[:, modality])
                        + self.estimate(self.lgp.g2['mu']['LS'], Y, modality, mask[:, modality])[0]
            elif modality in self.beta_idx:
                temp = self.exec_f(self.lgp.g3['mu']['KL_U'], Y, modality, mask[:, modality])
                        + self.estimate(self.lgp.g3['mu']['LS'], Y, modality, mask[:, modality])[0]
            elif modality in self.poisson_idx:
                temp = self.exec_f(self.lgp.g4['mu']['KL_U'], Y, modality, mask[:, modality])
                        + self.estimate(self.lgp.g4['mu']['LS'], Y, modality, mask[:, modality])[0]
            elif modality in self.gamma_idx:
                temp = self.exec_f(self.lgp.g5['mu']['KL_U'], Y, modality, mask[:, modality])
                        + self.estimate(self.lgp.g5['mu']['LS'], Y, modality, mask[:, modality])[0]
            
            return temp

        if 'sigma' in param_name:
            g = 0
            temp=[]
            for modality in xrange(len(Y)):
                if modality in self.gaussian_idx:
                    temp =  self.exec_f(self.lgp.g2['sigma']['KL_U'], Y, 
                                        modality, mask[:, modality])
                            + self.estimate(self.lgp.g2['sigma']['LS'], 
                                            Y, modality, mask[:, modality])[0]
                    g = g + temp

            return g

        if 'nu' in param_name:
            g = 0
            temp=[]
            for modality in xrange(len(Y)):
                if modality in self.beta_idx:
                    temp = self.exec_f(self.lgp.g3['nu']['KL_U'], Y, modality, 
                                       mask[:, modality]) + 
                           self.estimate(self.lgp.g3['nu']['LS'], Y, modality, 
                                       mask[:, modality])[0]
                    g = g + temp

            return g
        
        grad = []
        for modality in xrange(len(Y)):
            # 'W_hidden_1':W_hidden_1, 'W_hidden_2':W_hidden_2,'W_lin_1':W_lin_1, 'W_lin_2':W_lin_2, 
         # 'b_hidden_1':b_hidden_1, 'b_hidden_2':b_hidden_2, 'b_lin_1':b_lin_1, 'b_lin_2':b_lin_2
            m = mask[:, modality]
            if param_name in ['W_hidden_1', 'W_lin_1', 'b_hidden_1',  'b_lin_1', 
                              'W_hidden_2', 'W_lin_2', 'b_hidden_2',  'b_lin_2']:
            	g = np.zeros_like(self.lgp.params[param_name])
                if modality in self.binomial_idx:
                    g = (self.exec_f(self.lgp.gparams[param_name]['KL_U'], Y, modality, m)+ 
                         self.exec_f(self.lgp.gparams[param_name]['LS'], Y, modality, m))
                    grad += [g]
                elif modality in self.gaussian_idx:
                    g = (self.exec_f(self.lgp.gparams2[param_name]['KL_U'], Y, modality, m)+ 
                         self.exec_f(self.lgp.gparams2[param_name]['LS'], Y, modality, m))
                    grad += [g]
                elif modality in self.beta_idx:
                    g = (self.exec_f(self.lgp.gparams3[param_name]['KL_U'], Y, modality, m)+ 
                         self.exec_f(self.lgp.gparams3[param_name]['LS'], Y, modality, m))
                    grad += [g]
                elif modality in self.poisson_idx:
                    g = (self.exec_f(self.lgp.gparams4[param_name]['KL_U'], Y, modality, m)+ 
                         self.exec_f(self.lgp.gparams4[param_name]['LS'], Y, modality, m))
                    grad += [g]
                elif modality in self.gamma_idx:
                    g = (self.exec_f(self.lgp.gparams5[param_name]['KL_U'], Y, modality, m)+ 
                         self.exec_f(self.lgp.gparams5[param_name]['LS'], Y, modality, m))
                    grad += [g]
            else:

                if modality in self.binomial_idx:
                    grad_ls, grad_std = self.estimate(self.lgp.g[param_name]['LS'], Y, modality, m)
                    grad += [self.exec_f(self.lgp.g[param_name]['KL_U'], Y, modality, m) + grad_ls]
                elif modality in self.gaussian_idx:
                    grad_ls, grad_std = self.estimate(self.lgp.g2[param_name]['LS'], Y, modality, m)
                    grad += [self.exec_f(self.lgp.g2[param_name]['KL_U'], Y, modality, m) + grad_ls]
                elif modality in self.beta_idx:
                    grad_ls, grad_std = self.estimate(self.lgp.g3[param_name]['LS'], Y, modality, m)
                    grad += [self.exec_f(self.lgp.g3[param_name]['KL_U'], Y, modality, m) + grad_ls]
                elif modality in self.poisson_idx:
                    grad_ls, grad_std = self.estimate(self.lgp.g4[param_name]['LS'], Y, modality, m)
                    grad += [self.exec_f(self.lgp.g4[param_name]['KL_U'], Y, modality, m) + grad_ls]
                elif modality in self.gamma_idx:
                    grad_ls, grad_std = self.estimate(self.lgp.g5[param_name]['LS'], Y, modality, m)
                    grad += [self.exec_f(self.lgp.g5[param_name]['KL_U'], Y, modality, m) + grad_ls]
                    #grad += [grad_ls]
                    
                if param_name in ['Z', 'lhyp']:
                    grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality]
                        + KmmInv_grad[param_name]['LS'][modality])

                    
        if param_name in ['Z', 'lhyp','W_hidden_1', 'W_lin_1',  'b_hidden_1', 'b_lin_1', 
                          'W_hidden_2', 'W_lin_2',  'b_hidden_2', 'b_lin_2']:
            grad = np.sum(grad, 0)

        if param_name in ['W_hidden_1', 'W_lin_1', 'b_hidden_1',  'b_lin_1', 'W_hidden_2', 
                          'W_lin_2', 'b_hidden_2',  'b_lin_2']:
            m = ~np.any(~mask, axis=1)
            grad += self.exec_f(self.lgp.gparams[param_name]['KL_X'], Y, mask=m)

        # DEBUG
        if param_name == 'lhyp' and np.any(np.abs(grad) < grad_std / np.sqrt(self.lgp.samples)):

                samples = self.lgp.samples * 10
                grad = []
                for modality in xrange(len(Y)):
                    m = mask[:, modality]
                    if modality in self.binomial_idx:
                        grad_ls, grad_std = self.estimate(self.lgp.g[param_name]['LS'], Y, 
                                                          modality, m, samples=samples)
                        grad += [self.exec_f(self.lgp.g[param_name]['KL_U'], Y, modality, m) + grad_ls]
                        grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality] 
                                          + KmmInv_grad[param_name]['LS'][modality])
                    elif modality in self.gaussian_idx:
                        grad_ls, grad_std = self.estimate(self.lgp.g2[param_name]['LS'], Y,
                                                          modality, m, samples=samples)
                        grad += [self.exec_f(self.lgp.g2[param_name]['KL_U'], Y, modality, m) + grad_ls]
                        grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality] 
                                          + KmmInv_grad[param_name]['LS'][modality])
                    elif modality in self.beta_idx:
                        grad_ls, grad_std = self.estimate(self.lgp.g3[param_name]['LS'], Y,
                                                          modality, m, samples=samples)
                        grad += [self.exec_f(self.lgp.g3[param_name]['KL_U'], Y, modality, m) + grad_ls]
                        grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality] 
                                          + KmmInv_grad[param_name]['LS'][modality])
                    elif modality in self.poisson_idx:
                        grad_ls, grad_std = self.estimate(self.lgp.g4[param_name]['LS'], Y, 
                                                          modality, m, samples=samples)
                        grad += [self.exec_f(self.lgp.g4[param_name]['KL_U'], Y, modality, m) + grad_ls]
                        grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality] 
                                         + KmmInv_grad[param_name]['LS'][modality])
                    elif modality in self.gamma_idx:
                        grad_ls, grad_std = self.estimate(self.lgp.g5[param_name]['LS'], Y, 
                                                          modality, m, samples=samples)
                        grad += [self.exec_f(self.lgp.g5[param_name]['KL_U'], Y, modality, m) + grad_ls]
                        grad[modality] += (KmmInv_grad[param_name]['KL_U'][modality] 
                                         + KmmInv_grad[param_name]['LS'][modality])

                grad = np.sum(grad, 0)
                self.grad_std = grad_std

        return np.array(grad)

    def opt_one_step(self, params, iteration, opt = 'rmsprop', learning_rate_adapt = 0.2, 
                     use_einsum = True, minibatch_size=500):
        """
        Perform optimisation step

        :param params: parameters to be optimised
        :param iteration: number of iterations
        :param opt: optimisation technique to use: ['grad_ascent', 'rmsprop', 'adam']
        :param learning_rate_adapt: rate for adapting learning rate
        :param use_einsum: boolean to use einsum
        :param minibatch_size: size of minibatch
        """
        
        # mini-bactching	
        shuff_indices = np.arange(self.Y_orig.shape[1])
        np.random.shuffle(shuff_indices)
        self.Y_epoch = self.Y_orig[:, shuff_indices, :]
        self.mask_orig_epoch = self.mask_orig[shuff_indices,:]
        for batch_idx in range(0, self.Y_orig.shape[1], minibatch_size):
        	self.Y = np.array(self.Y_epoch[:,batch_idx:batch_idx + minibatch_size,:])
        	self.mask = self.mask_orig_epoch[batch_idx:batch_idx + minibatch_size, :]
	        KmmInv_grad = self.get_KmmInv_grad(self.Y, self.mask, self.params['Z'].shape)
	        
	        for param_name in params:
	            # DEBUG
	            if opt == 'grad_ascent' or param_name in ['ls']:
	                self.grad_ascent_one_step(param_name, 
                                              [param_name, self.Y, KmmInv_grad, self.mask], 
	                    learning_rate_decay = learning_rate_adapt * 100 / (iteration + 100.0))
	            elif opt == 'rmsprop':
	                self.rmsprop_one_step(param_name, 
                                          [param_name, self.Y, KmmInv_grad, self.mask], 
                                          learning_rate_adapt = learning_rate_adapt)
                    #, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))
                elif opt == 'adam':
                    self.adam(param_name, [param_name, self.Y, KmmInv_grad, self.mask], 
                              iter_num=iteration)

	            if param_name in ['lhyp']:
	                self.params[param_name] = np.clip(self.params[param_name], -8, 8)
	            if param_name in ['lhyp', 'Z']:
	                self.lgp.update_KmmInv_cache()

    def grad_ascent_one_step(self, param_name, grad_args, momentum = 0.9, learning_rate_decay = 1):
        """
        Perform gradient ascent step

        :param param_name: parameter to be optimised
        :param grad_args: gradient arguments
        :param momentum: momentum value
        :param learning_rate_decay: learning rate decay rate
        """
        self.lgp.params[param_name] += (learning_rate_decay*self.learning_rates[param_name]
                                           * self.param_updates[param_name])
        grad = self.get_grad(*grad_args)
        if param_name in ['lhyp']:
            self.param_updates[param_name] = momentum*self.param_updates[param_name] 
                                                + (1. - momentum)*grad
        else:
            self.param_updates[param_name] = grad

    def rmsprop_one_step(self, param_name, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
                         learning_rate_min = 1e-6, learning_rate_max = 10):
        """
        Perform RMSPROP step
        RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py
        
        We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.

        :param param_name: parameter to be optimised
        :param grad_args: gradient arguments
        :param decay: learning rate decay rate
        :param momentum: momentum value
        :param learning_rate_adapt: rate for adapting learning rate
        :param learning_rate_min: lower bound for learning rate
        :param learning_rate_max: upper bound for learning rate
        """

        step1 = self.param_updates[param_name] * momentum
        self.params[param_name] += step1
        grad = self.get_grad(*grad_args)

        self.moving_mean_squared[param_name] = (decay * self.moving_mean_squared[param_name] 
                                               + (1 - decay) * grad ** 2)
        step2 = self.learning_rates[param_name] 
                * grad / (self.moving_mean_squared[param_name] + 1e-8)**0.5

        # DEBUG
        if param_name == 'lhyp':
            step2 = np.clip(step2, -0.1, 0.1)

        self.params[param_name] += step2

        step = step1 + step2

        if param_name == 'sigma':
            if self.params[param_name] < 0:
                self.params[param_name] = 1e-2
                
        if param_name == 'nu':
            if self.params[param_name] < 0:
                self.params[param_name] = 1e-2

        # Step rate adaption. If the current step and the momentum agree,
        # we slightly increase the step rate for that dimension.
        if learning_rate_adapt:
            # This code might look weird, but it makes it work with both numpy and gnumpy.
            step_non_negative = step > 0
            step_before_non_negative = self.param_updates[param_name] > 0
            agree = (step_non_negative == step_before_non_negative) * 1.
            adapt = 1 + agree * learning_rate_adapt * 2 - learning_rate_adapt
            self.learning_rates[param_name] *= adapt
            self.learning_rates[param_name] = np.clip(self.learning_rates[param_name],
                                                      learning_rate_min, learning_rate_max)

        self.param_updates[param_name] = step

    def adam(self, param_name, grad_args, iter_num, beta_1=0.9, beta_2=0.999):
        """
        Perform ADAM step

        :param param_name: parameter to be optimised
        :param grad_args: gradient arguments
        :param iter_num: current iteration number
        :param beta_1: exponential decay rate for the first moment estimates
        :param beta_2: exponential decay rate for the second moment estimates
        """
        grad = self.get_grad(*grad_args)
        
        self.moving_mean[param_name] = ((beta_1 * self.moving_mean[param_name]) + (1 - beta_1) * grad)
        self.moving_mean_squared[param_name] = ((beta_2 * self.moving_mean_squared[param_name]) 
                                                + (1 - beta_2) * grad ** 2)

        # bias correction
        m_hat = self.moving_mean[param_name]/(1 - np.power(beta_1, iter_num+1))
        v_hat = self.moving_mean_squared[param_name]/(1 - np.power(beta_2, iter_num+1))

        # update
        step = self.learning_rates[param_name] * m_hat/((v_hat ** 0.5) + 1e-8)

        self.params[param_name] = self.params[param_name] + step

        # clipping
        if param_name == 'sigma':
            if self.params[param_name] < 0:
                self.params[param_name] = 1e-2
                
        if param_name == 'nu':
            if self.params[param_name] < 0:
                self.params[param_name] = 1e-2  
