# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
from theano.ifelse import ifelse
import numpy as np
import cPickle

print 'Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False
eps = 1e-4


class kernel:
    def RBF(self, sf2, l, X1, X2 = None):
        """
        Radial basis function kernel definition

        :param sf2: kernel variance
        :param l: kernel length scale
        :param X1: data
        :param X2: data

        :return RBF kernel evaluation
        """
        _X2 = X1 if X2 is None else X2
        dist = ((X1 / l)**2).sum(1)[:, None] + ((_X2 / l)**2).sum(1)[None, :] - 2
                *(X1 / l).dot((_X2 / l).T)
        RBF = sf2 * T.exp(-dist / 2.0)
        return (RBF + eps * T.eye(X1.shape[0])) if X2 is None else RBF
    def RBFnn(self, sf2, l, X):
        return sf2 + eps
    def LIN(self, sl2, X1, X2 = None):
        """
        Linear kernel definition
        
        :param sf2: kernel variance
        :param X1: data
        :param X2: data

        :return Linear kernel evaluation
        """
        _X2 = X1 if X2 is None else X2
        LIN = sl2 * (X1.dot(_X2.T) + 1)
        return (LIN + eps * T.eye(X1.shape[0])) if X2 is None else LIN
    def LINnn(self, sl2, X):
        return sl2 * (T.sum(X**2, 1) + 1) + eps


class LGP_model:
    def __init__(self, params, gaussian_idx, binomial_idx, beta_idx, poisson_idx, 
                 gamma_idx, sx2 = 1, linear_model = False, samples=10, use_hat=False, 
                 kl_weight=1):
        """
        Model definition
        
        :param params: parameters to be optimised
        :param gaussian_idx:  list of indices for the gaussian distributed data
        :param binomial_idx:  list of indices for the binomial distributed data
        :param beta_idx:  list of indices for beta distributed data
        :param poisson_idx:  list of indices for poisson distributed data
        :param gamma_idx:  ist of indices for gamma distributed data
        :param sx2: optional variance term'
        :param linear_model: boolean for the use of a linear model
        :param samples: number of Monte Carlo samples
        :param use_hat: boolean for save file
        :param kl_weight: weight for the KL term

        """

        ker, self.samples, self.params, self.KmmInv  = kernel(), samples, params, {}
        self.use_hat = use_hat
        self.gaussian_idx = gaussian_idx
        self.binomial_idx = binomial_idx
        self.beta_idx = beta_idx
        self.poisson_idx = poisson_idx
        self.gamma_idx = gamma_idx
	    self.kl_weight = kl_weight

        model_file_name = 'model' + ('_hat' if use_hat else '') + 
                          ('_linear' if linear_model else '') + '.save'

        try:
            print 'Trying to load model...'
            with open(model_file_name, 'rb') as file_handle:
                obj = cPickle.load(file_handle)
                self.f, self.g, self.g2, self.g3, slef.g4, slef.gparams, self.params2, self.gparams3, self.gparams4,
                self.f_Kmm, self.f_KmmInv, self.dKmm_d = obj
                self.update_KmmInv_cache()
                print 'Loaded!'
            return
        except:
            print 'Failed. Creating a new model...'

        Y, Z, mu, lL, eps_MK, eps_NQ, eps_NK, KmmInv, eps = T.dmatrices('Y', 
                                        'Z', 'mu', 'lL', 'eps_MK','eps_NQ', 'eps_NK', 'KmmInv', 'eps')

        W_hidden_1, W_lin_1 = T.dmatrices('W_hidden_1', 'W_lin_1')
        b_hidden_1 = T.drow('b_hidden_1')
        b_lin_1 = T.drow('b_lin_1')
	    W_hidden_2, W_lin_2 = T.dmatrices('W_hidden_2', 'W_lin_2')    
        b_hidden_2 = T.drow('b_hidden_2')
        b_lin_2 = T.drow('b_lin_2')
        Y_orig = T.dmatrix('Y_orig')
        #### Neural Net (MLP)
        ### Compute all gradients
        ### grads computed for all modality, then updated
        lhyp = T.dvector('lhyp')
        sigma = T.dscalar('sigma')
        nu = T.dscalar('nu')
    	kl_weight_t = T.dscalar('kl_weight_t')
        (M, K), N, Q = mu.shape, Y.shape[0], Z.shape[1]
        sl2, sf2, l = T.exp(lhyp[0]), T.exp(lhyp[1]), T.exp(lhyp[2:2+Q])
        L = T.tril(lL - T.diag(T.diag(lL)) + T.diag(T.exp(T.diag(lL))))        
        print 'Setting up cache...'
        Kmm = ker.RBF(sf2, l, Z) if not linear_model else ker.LIN(sl2, Z)
        KmmInv_cache = sT.matrix_inverse(Kmm)
        self.f_Kmm = theano.function([Z, lhyp], Kmm, name='Kmm')      
        self.f_KmmInv = theano.function([Z, lhyp], KmmInv_cache, name='KmmInv_cache')
        self.update_KmmInv_cache()
        self.dKmm_d = {'Z': theano.function([Z, lhyp], T.jacobian(Kmm.flatten(), Z), name='dKmm_dZ'),
                       'lhyp': theano.function([Z, lhyp], T.jacobian(Kmm.flatten(), lhyp), name='dKmm_dlhyp')}
        print 'Setting up model...'
        mu_scaled, L_scaled = sf2**0.5 * mu, sf2**0.5 * L

        ### nnet for m 
        out_hidden_1 = T.tanh(T.dot(Y_orig, W_hidden_1) + b_hidden_1)
        out_1 = T.dot(out_hidden_1, W_lin_1) + b_lin_1

        ### net net for s
        out_hidden_2 = T.tanh(T.dot(Y_orig, W_hidden_2) + b_hidden_2)
        out_2 = T.nnet.sigmoid(T.dot(out_hidden_2, W_lin_2) + b_lin_2)

	    m_nnet = out_1
	    v_nnet = out_2
        X = m_nnet + v_nnet * eps_NQ
        U = mu_scaled + L_scaled.dot(eps_MK)
        Kmn = ker.RBF(sf2, l, Z, X) if not linear_model else ker.LIN(sl2, Z, X)
        Knn = ker.RBFnn(sf2, l, X) if not linear_model else ker.LINnn(sl2, X)
        
        A = KmmInv.dot(Kmn)
        B = Knn - T.sum(Kmn * KmmInv.dot(Kmn), 0)
        mean = A.T.dot(mu_scaled)
        var = Knn + T.sum(Kmn * KmmInv.dot(L_scaled.T.dot(L_scaled) - Kmm).dot(KmmInv).dot(Kmn), 0)
        F = mean + T.maximum(var, 1e-16)[:, None]**0.5 * eps_NK
        F = T.concatenate((T.zeros((N,1)), F), axis=1)
        S = T.nnet.softmax(F)

        Y_hot = T.extra_ops.to_one_hot(T.cast(Y[:,0], 'int32'), 2)
        LS_cat = T.sum(T.log(T.maximum(T.sum(Y_hot * S, 1), 1e-16)))
        KL_U_cat = -0.5 * (T.sum(KmmInv.T * T.sum(mu_scaled[:,None,:]*mu_scaled[None,:,:], 2))
                           + K * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M - 2.0*T.sum(T.log(T.diag(L_scaled)))
                           + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))))
           
        F_gauss = mean + T.maximum(var, 1e-16)[:, None]**0.5 * eps_NK    
        LS_gauss = (-N/2)*T.log(2*np.pi) - (N/2)*T.log(sigma**2) - (1/(2*sigma**2))*T.sum((Y - F_gauss)**2)
        F_poiss =mean + T.maximum(var, 1e-16)[:, None]**0.5 * eps_NK
        lamb = T.exp(F_poiss)
        LS_poiss = T.sum(-lamb + Y * T.log(lamb))
        F_beta = mean + T.maximum(var, 1e-16)[:, None]**0.5 * eps_NK
        mu_beta = (1 + T.erf(F_beta/T.sqrt(2)))/2
        alpha = T.maximum(nu * mu_beta, 1e-6)
        beta = T.maximum(nu * (1 - mu_beta),1e-6)
        B_func = (T.gamma(alpha) * T.gamma(beta))/T.gamma(alpha + beta)
        LS_beta = T.sum((alpha - 1) * T.log(T.maximum(Y,1e-16)) +  (beta - 1) * T.log(T.maximum(1 - Y,1e-16)) - T.log(B_func))
        F_gamma =  mean + T.maximum(var, 1e-16)[:, None]**0.5 * eps_NK
        gamma_alpha = T.nnet.softplus(F_gamma)
        LS_gamma = T.sum(-T.log(T.maximum(T.gamma(gamma_alpha), 1e-16)) + (gamma_alpha - 1) * T.log(T.maximum(Y,1e-16)) - Y)
    
        KL_U_gauss = -0.5 * (T.sum(KmmInv.T * T.sum(mu_scaled[:,None,:]*mu_scaled[None,:,:], 2))
                            + K * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M - 2.0*T.sum(T.log(T.diag(L_scaled)))
                            + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))))
    
        KL_U_beta = -0.5 * (T.sum(KmmInv.T * T.sum(mu_scaled[:,None,:]*mu_scaled[None,:,:], 2))
                            + K * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M 
                            - 2.0*T.sum(T.log(T.diag(L_scaled))) + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))))

        KL_U_poiss = -0.5 * (T.sum(KmmInv.T * T.sum(mu_scaled[:,None,:]*mu_scaled[None,:,:], 2))
                            + K * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M 
                            - 2.0*T.sum(T.log(T.diag(L_scaled))) + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))))
        
        KL_U_gamma = -0.5 * (T.sum(KmmInv.T * T.sum(mu_scaled[:,None,:]*mu_scaled[None,:,:], 2))
                            + K * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M 
                            - 2.0*T.sum(T.log(T.diag(L_scaled))) + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm))))))
	
        KL_X = kl_weight_t * T.sum(KL_X_all)
 	    KL_X_all = -0.5 * T.sum((m_nnet**2.0 + v_nnet**2.0)/sx2 - 1.0 - 2.0*T.log(v_nnet) + T.log(sx2), 1)

        print 'Compiling...'
        inputs = {'Y': Y, 'Y_orig': Y_orig, 'Z': Z, 'mu': mu, 'lL': lL, 'lhyp': lhyp, 'KmmInv': KmmInv, 
            'eps_MK': eps_MK, 'eps_NK': eps_NK, 'eps' : eps, 'W_hidden_1':W_hidden_1, 
            'W_lin_1':W_lin_1,  'b_hidden_1':b_hidden_1, 'b_lin_1':b_lin_1, 'W_hidden_2':W_hidden_2, 
            'W_lin_2':W_lin_2,  'b_hidden_2':b_hidden_2, 'b_lin_2':b_lin_2,'sigma':sigma, 'nu':nu, 
            'eps_NQ':eps_NQ, 'kl_weight_t':kl_weight_t}
        z = 0.0*sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        f = zip(['X', 'U', 'S', 'LS_cat', 'KL_U_cat', 'F_gauss', 'LS_gauss', 'KL_U_gauss','F_beta','LS_beta',
            'KL_U_beta', 'F_poiss', 'LS_poiss', 'KL_U_poiss', 'F_gamma', 'LS_gamma','KL_U_gamma', 'Knn', 
            'latent','KL_X', 'KL_X_all'], [X, U, S, LS_cat,KL_U_cat,
            F_gauss, LS_gauss, KL_U_gauss, F_beta,LS_beta, KL_U_beta, F_poiss, LS_poiss, KL_U_poiss, 
            F_gamma, LS_gamma, KL_U_gamma, Knn, m_nnet, KL_X, KL_X_all])
        self.f = {n: theano.function(inputs.values(), f+z, name=n, on_unused_input='ignore') for n,f in f}

        # categorical gradients
        g = zip(['LS', 'KL_U', 'KL_X'], [LS_cat, KL_U_cat, KL_X])
        wrt = {'Z': Z, 'mu': mu, 'lL': lL, 'lhyp': lhyp, 'KmmInv': KmmInv, 'sigma':sigma, 'nu':nu}
        self.g = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in g} for vn, vv in wrt.iteritems()}
        
        # gaussian gradients
        g2 = zip(['LS', 'KL_U', 'KL_X'], [LS_gauss, KL_U_gauss, KL_X])
        self.g2 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in g2} for vn, vv in wrt.iteritems()}

        # beta gradients
        g3 = zip(['LS', 'KL_U', 'KL_X'], [LS_beta, KL_U_beta, KL_X])
        self.g3 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in g3} for vn, vv in wrt.iteritems()}

        # poisson gradients
        g4 = zip(['LS', 'KL_U', 'KL_X'], [LS_poiss, KL_U_poiss, KL_X])
        self.g4 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in g4} for vn, vv in wrt.iteritems()}
        
        # gamma gradients
        g5 = zip(['LS', 'KL_U', 'KL_X'], [LS_gamma, KL_U_gamma, KL_X])
        self.g5 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in g5} for vn, vv in wrt.iteritems()}
        
        # nnet gradients
        # categorical gradients
        gparams = zip(['LS', 'KL_U', 'KL_X'], [LS_cat, KL_U_cat,KL_X])
        nnet_wrt = {'W_hidden_1':W_hidden_1, 'W_lin_1':W_lin_1,
        'b_hidden_1':b_hidden_1, 'b_lin_1':b_lin_1, 'W_hidden_2':W_hidden_2, 'W_lin_2':W_lin_2,
        'b_hidden_2':b_hidden_2, 'b_lin_2':b_lin_2}
        self.gparams = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
        	on_unused_input='ignore') for gn,gv in gparams} for vn, vv in nnet_wrt.iteritems()}

        # gaussian gradients
        gparams2 = zip(['LS', 'KL_U', 'KL_X'], [LS_gauss, KL_U_gauss,KL_X])
        self.gparams2 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
        	on_unused_input='ignore') for gn,gv in gparams2} for vn, vv in nnet_wrt.iteritems()}

        # beta gradients
        gparams3 = zip(['LS', 'KL_U', 'KL_X'], [LS_beta, KL_U_beta, KL_X])
        self.gparams3 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
        	on_unused_input='ignore') for gn,gv in gparams3} for vn, vv in nnet_wrt.iteritems()}

        # poisson gradients
        gparams4 = zip(['LS', 'KL_U', 'KL_X'], [LS_poiss, KL_U_poiss, KL_X])
        self.gparams4 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
        	on_unused_input='ignore') for gn,gv in gparams4} for vn, vv in nnet_wrt.iteritems()}
        
        # gamma gradients
        gparams5 = zip(['LS', 'KL_U', 'KL_X'], [LS_gamma, KL_U_gamma, KL_X])
        self.gparams5 = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
        	on_unused_input='ignore') for gn,gv in gparams5} for vn, vv in nnet_wrt.iteritems()}

    def update_KmmInv_cache(self):
        self.KmmInv = self.f_KmmInv(self.params['Z'], self.params['lhyp']).astype(theano.config.floatX)

    def exec_f(self, f, Y = np.array([[[0]]]), modality = 0, mask = None, s = None):
        """
        Compute value of node
        
        :param f: function to evaluate
        :param Y: data
        :param modality: current attribute to perform operation
        :param mask: mask for unobserved values
        :param s: sample number

        :return: evaluated function
        """
        inputs, (M, K) = {}, self.params['mu' + str(modality)].shape
        N = Y.shape[1]
        D = Y.shape[0]
        Q = self.params['Z'].shape[1]

        inputs['Z'] = self.params['Z']
        inputs['mu'], inputs['lL'] = self.params['mu' + str(modality)], self.params['lL'][modality]
        inputs['lhyp'], inputs['KmmInv'] = self.params['lhyp'], self.KmmInv
        inputs['eps_MK']= np.random.randn(M,K)
	    inputs['eps_NQ']= np.random.randn(N,Q)
        inputs['eps'] = np.random.randn(N,1)
        inputs['Y'] = Y[modality] if len(Y) > 1 else Y[0]
        nodes = [-1.224744871391589, 0, 1.224744871391589]
        inputs['sigma'] = self.params['sigma']
        inputs['nu'] = self.params['nu']
        inputs['Y_orig'] = np.transpose(np.reshape(Y, (D, N)))
        inputs['W_hidden_1'] = self.params['W_hidden_1']
        inputs['W_lin_1'] = self.params['W_lin_1']
        inputs['b_hidden_1'] = self.params['b_hidden_1']
        inputs['b_lin_1'] = self.params['b_lin_1']
        inputs['W_hidden_2'] = self.params['W_hidden_2']
        inputs['W_lin_2'] = self.params['W_lin_2']
        inputs['b_hidden_2'] = self.params['b_hidden_2']
        inputs['b_lin_2'] = self.params['b_lin_2']
	    inputs['kl_weight_t'] = self.kl_weight
        
        if s is not None:
            inputs['eps_NK'] = np.full((N,K), nodes[s])
        else:
            inputs['eps_NK']  = np.random.randn(N,K)  

        if mask is not None:
            inputs['Y'] = inputs['Y'][~mask]
            inputs['eps_NK'] = inputs['eps_NK'][~mask]
        return f(**inputs)

    def estimate(self, f, Y = np.array([[[0]]]), modality = 0, mask = None, samples = None):
       	"""
        Perform 3 point Gauss-Hermite Quadrature
        
        :param f: function to evaluate
        :param Y: data
        :param modality: current attribute to perform operation
        :param mask: mask for unobserved values
        :param samples: number of repitions

        :return: result
        """
        weights = np.array([0.2954089751509, 1.181635900603, 0.2954089751509])
        f_acc = np.array([self.exec_f(f, Y, modality, mask, s) for s in xrange(3)])
        val = (1/np.sqrt(np.pi)) * (f_acc[0]*weights[0] + f_acc[1]*weights[1] + f_acc[2]*weights[2])
        return val, np.nanstd(f_acc, 0)

    def ELBO(self, Y, mask = None):
        """
        Evaluate evidence lower bound
        
        :param Y: data
        :param mask: mask for unobserved values

        :return: ELBO value
        """

        if mask is None:
            mask = np.zeros((Y.shape[1], len(Y)), dtype=bool)
	    ELBO, std_sum = self.exec_f(self.f['KL_X'], Y, mask=np.all(mask, 1)), 0
        KL_X_val = ELBO
        LS_val = 0
        KL_U_val = 0
        for modality in xrange(len(Y)):
            if modality in self.binomial_idx:
                LS, std = self.estimate(self.f['LS_cat'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS
                temp = self.exec_f(self.f['KL_U_cat'], Y, modality, mask[:, modality])
                KL_U_val = KL_U_val + temp
                ELBO = ELBO + temp + LS
            elif modality in self.gaussian_idx:
                LS, std = self.estimate(self.f['LS_gauss'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS
                temp = self.exec_f(self.f['KL_U_gauss'], Y, modality, mask[:, modality]) 
                KL_U_val = KL_U_val + temp
                ELBO = ELBO + temp + LS
            elif modality in self.beta_idx:
                LS, std = self.estimate(self.f['LS_beta'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS
                temp = self.exec_f(self.f['KL_U_beta'], Y, modality, mask[:, modality])
                KL_U_val = KL_U_val + temp
                ELBO = ELBO + temp + LS
            elif modality in self.poisson_idx:
                LS, std = self.estimate(self.f['LS_poiss'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS
                temp = self.exec_f(self.f['KL_U_poiss'], Y, modality, mask[:, modality])
                KL_U_val = KL_U_val + temp
                ELBO = ELBO + temp + LS  
            elif modality in self.gamma_idx:
                LS, std = self.estimate(self.f['LS_gamma'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS
                temp = self.exec_f(self.f['KL_U_gamma'], Y, modality, mask[:, modality])
                KL_U_val = KL_U_val + temp
                ELBO = ELBO + temp + LS
                
            std_sum += std**2
        return ELBO, LS_val, KL_U_val, std_sum**0.5, KL_X_val

    def pred_lik(self, Y, mask=None):
        """
        Evaluate perdictive likelihood
        
        :param Y: data
        :param mask: mask for unobserved values

        :return: predictive likelihood value and standard deviation
        """
        if mask is None:
            mask = np.zeros((Y.shape[1], len(Y)), dtype=bool)
	    std_sum = 0
        LS_val = 0
        for modality in xrange(len(Y)):
            if modality in self.binomial_idx:
                LS, std = self.estimate(self.f['LS_cat'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS

            elif modality in self.gaussian_idx:
                LS, std = self.estimate(self.f['LS_gauss'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS

            elif modality in self.beta_idx:
                LS, std = self.estimate(self.f['LS_beta'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS

            elif modality in self.poisson_idx:
                LS, std = self.estimate(self.f['LS_poiss'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS 
            elif modality in self.gamma_idx:
                LS, std = self.estimate(self.f['LS_gamma'], Y, modality, mask[:, modality])
                LS_val = LS_val + LS 
                
            std_sum += std**2
        return LS_val, std_sum**0.5
