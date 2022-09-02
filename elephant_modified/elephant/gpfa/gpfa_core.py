# -*- coding: utf-8 -*-
"""
GPFA core functionality.

:copyright: Copyright 2014-2020 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import time
import warnings

import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.sparse as sparse
from sklearn.decomposition import FactorAnalysis
from tqdm import trange

from . import gpfa_util
from . import bayesian_optimisation



def fit(seqs_train, x_dim=3, bin_width=20.0, min_var_frac=0.01, em_tol=1.0E-8,
        em_max_iters=500, tau_init=100.0, eps_init=1.0E-3, freq_ll=5,
        verbose=False,covType='rbf',bo=0):
    """
    Fit the GPFA model with the given training data.

    Parameters
    ----------
    seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields
        T : int
            number of bins
        y : (#units, T) np.ndarray
            neural data
    x_dim : int, optional
        state dimensionality
        Default: 3
    bin_width : float, optional
        spike bin width in msec
        Default: 20.0
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    em_tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    em_max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tau_init : float, optional
        GP timescale initialization in msec
        Default: 100
    eps_init : float, optional
        GP noise variance initialization
        Default: 1e-3
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations. freq_ll = 1
        means that data likelihood is computed at every iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False
    covType: {'rbf', 'tri', 'exp', 'rq', 'matern', 'sm',
        'tri_times_rq', 'exp_times_rq', 'exp_times_tri'}
        type of GP covariance
        Default : 'rbf'
    bo: int
        number of iterations to tune kernel parameters using Bayesian Optimisation
        using Scipy Powell method instead when set to False
        Default: False

    Returns
    -------
    parameter_estimates : dict
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
            covType: {'rbf', 'tri', 'exp', 'rq', 'matern', 'sm',
                'tri_times_rq', 'exp_times_rq', 'exp_times_tri'}
                type of GP covariance
            gamma: np.ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_width / sqrt(gamma)'
            eps: np.ndarray of shape (1, #latent_vars)
                GP noise variances
            d: np.ndarray of shape (#units, 1)
                observation mean
            C: np.ndarray of shape (#units, #latent_vars)
                mapping between the neuronal data space and the latent variable
                space
            R: np.ndarray of shape (#units, #latent_vars)
                observation noise covariance
        
        New developed GP kernels have other kernel parameters:
            gamma: rbf, rbf try
            sigma: tri, exp
            alpha, ell: rq
            nu, ell: matern
            Q, w, mu, vs: sm
                Q: int
                number of spectral mixtures, cann be adjusted.
                Default: 2
            sigma, alpha, ell: tri_times_rq, exp_times_rq
            sigma_e, sigma_t: exp_times_tri

    fit_info : dict
        Information of the fitting process and the parameters used there
        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
    """
    # For compute efficiency, train on equal-length segments of trials
    seqs_train_cut = gpfa_util.cut_trials(seqs_train)
    if len(seqs_train_cut) == 0:
        warnings.warn('No segments extracted for training. Defaulting to '
                      'segLength=Inf.')
        seqs_train_cut = gpfa_util.cut_trials(seqs_train, seg_length=np.inf)

    # ==================================
    # Initialize state model parameters
    # ==================================
    params_init = dict()
    params_init['covType'] = covType
    # GP timescale
    # Assume binWidth is the time step size.
    params_init['gamma'] = (bin_width / tau_init) ** 2 * np.ones(x_dim)
    # GP noise variance
    params_init['eps'] = eps_init * np.ones(x_dim)

    # ========================================
    # Initialize observation model parameters
    # ========================================
    print('Initializing parameters using factor analysis...')

    y_all = np.hstack(seqs_train_cut['y'])
    fa = FactorAnalysis(n_components=x_dim, copy=True,
                        noise_variance_init=np.diag(np.cov(y_all, bias=True)))
    fa.fit(y_all.T)
    params_init['d'] = y_all.mean(axis=1)
    params_init['C'] = fa.components_.T
    params_init['R'] = np.diag(fa.noise_variance_)

    # Define parameter constraints
    params_init['notes'] = {
        'learnKernelParams': True,
        'learnGPNoise': False,
        'RforceDiagonal': True,
        'learnKernelParams_with_bo':bo
    }
    # initialise kernel parameters 
    if params_init['covType'] in ['exp','tri']:

        if params_init['notes']['learnKernelParams']:
            # params_init['sigma'] = np.full(x_dim, 0.04)
            params_init['sigma'] = np.linspace(0.002,0.006,x_dim)

    if params_init['covType'] in ['rq']:
        if params_init['notes']['learnKernelParams']:
            params_init['alpha'] = np.linspace(0.002,0.006,x_dim)
            params_init['ell'] = np.linspace(0.002,0.006,x_dim)

    if params_init['covType'] in ['matern']:
        if params_init['notes']['learnKernelParams']:
            params_init['nu'] = np.linspace(0.002, 0.006, x_dim)
            params_init['ell'] = np.linspace(0.002, 0.006, x_dim)

    if params_init['covType'] in ['tri_times_rq']:
        if params_init['notes']['learnKernelParams']:
            params_init['sigma'] = np.linspace(0.002, 0.006, x_dim)
            params_init['alpha'] = np.linspace(0.002, 0.006, x_dim)
            params_init['ell'] = np.linspace(0.002, 0.006, x_dim)

    if params_init['covType'] in ['exp_times_rq']:
        if params_init['notes']['learnKernelParams']:
            params_init['sigma'] = np.linspace(0.002, 0.006, x_dim)
            params_init['alpha'] = np.linspace(0.002, 0.006, x_dim)
            params_init['ell'] = np.linspace(0.002, 0.006, x_dim)

    if params_init['covType'] in ['exp_times_tri']:
        if params_init['notes']['learnKernelParams']:
            params_init['sigma_exp'] = np.linspace(0.002, 0.006, x_dim)
            params_init['sigma_tri'] = np.linspace(0.002, 0.006, x_dim)

    if params_init['covType'] in ['sm']:
        if params_init['notes']['learnKernelParams']:
            #create spectral mixtures
            params_init['Q'] = 2
            params_init['w'] = np.zeros((x_dim,params_init['Q']))
            params_init['mu'] = np.zeros((x_dim,params_init['Q']))
            params_init['vs'] = np.zeros((x_dim,params_init['Q']))
            # initialise spectral mixture parameters
            for i in range(x_dim):
                weights = np.ones(params_init['Q'])
                params_init['w'][i] = weights / np.sum(weights)
                params_init['mu'][i] = np.random.uniform(1e-5, 1, params_init['Q'])
                params_init['vs'][i] = np.random.uniform(1e-5, 1, params_init['Q'])

    # =====================
    # Fit model parameters
    # =====================
    print('\nFitting GPFA model...')


    params_est, seqs_train_cut, ll_cut, iter_time = em(
        params_init, seqs_train_cut, min_var_frac=min_var_frac,
        max_iters=em_max_iters, tol=em_tol, freq_ll=freq_ll, verbose=verbose)


    fit_info = {'iteration_time': iter_time, 'log_likelihoods': ll_cut}

    # print optimised kernel parameters  
    # if params_init['covType'] == 'rbf':
    #     print('gamma: '+str(params_est['gamma']))
    # elif params_init['covType'] in ['exp','tri']:
    #     print('sigma: '+str(params_est['sigma']))
    # elif params_init['covType'] in ['rq']:
    #     print('alpha: ' + str(params_est['alpha']))
    #     print('ell: ' + str(params_est['ell']))
    # elif params_init['covType'] in ['matern']:
    #     print('nu: ' + str(params_est['nu']))
    #     print('ell: ' + str(params_est['ell']))
    # elif params_init['covType'] in ['tri_times_rq']:
    #     print('sigma: ' + str(params_est['sigma']))
    #     print('alpha: ' + str(params_est['alpha']))
    #     print('ell: ' + str(params_est['ell']))
    # elif params_init['covType'] in ['exp_times_rq']:
    #     print('sigma: ' + str(params_est['sigma']))
    #     print('alpha: ' + str(params_est['alpha']))
    #     print('ell: ' + str(params_est['ell']))
    # elif params_init['covType'] in ['exp_times_tri']:
    #     print('sigma_e: ' + str(params_est['sigma_exp']))
    #     print('sigma_t: ' + str(params_est['sigma_tri']))
    # if params_init['covType'] in ['sm']:
    #     for i in range(x_dim):
    #         print('w[{}]: '.format(i) + str(params_est['w'][i]))
    #         print('mu[{}]: '.format(i) + str(params_est['mu'][i]))
    #         print('vs[{}]: '.format(i) + str(params_est['vs'][i]))
    # else:
    #     ValueError('kernel is not supported')

    return params_est, fit_info

def learn_sigma(params,seqs_train):
    """
    find optimal sigma for exponential / triangular kernel with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP rbf kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    
    Returns
    -------
    param_opt : np.ndarray
        updated exponential / triangular kernel parameter sigma
    """
    param_name = 'sigma'
    param_init = params[param_name]
    param_opt = {param_name: np.empty_like(param_init)}

    x_dim = param_init.shape[-1]

    def kernel_cost(sigma_i, params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting
        
        Parameters
        ----------
        sigma_i : dict
            current kernel parameter sigma at GP state dimension i
        params : dict
            GPFA model parameters
        seqs_train : np.recarray
            training data structure
        params_idx : the state dimension performing kernel paramater optimisation

        Returns
        -------
        ll : float
            data log likelihood, returned by exact_inference_with_ll
        """

        params['sigma'][params_idx] = sigma_i
        sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        res_opt = optimize.minimize(kernel_cost, param_init[i],
                                    args=(params, seqs_train, i),
                                    # emprical safety bound for searching hyperparameter 
                                    # <1e-5 may cause numberical underflow 
                                    # >3e10 may cause infinity in computation
                                    bounds = [(1e-5,3e10)],
                                    method='Powell')
        param_opt['sigma'][i] = res_opt.x

    return param_opt

def learn_sigma_with_bo(params,seqs_train):
    """
    find optimal sigma for exponential / triangular kernel with Bayesian Optimisation
    Parameters
    ----------
        sigma_i : float
            current kernel parameter sigma at GP state dimension i
        params : dict
            GPFA model parameters
        seqs_train : np.recarray
            training data structure
        params_idx : the state dimension performing kernel paramater optimisation
    
    Returns
    -------
    param_opt : np.ndarray
        updated exponential / triangular kernel parameter sigma
    """
    param_name = 'sigma'
    param_init = params[param_name]
    param_opt = {param_name: np.empty_like(param_init)}

    x_dim = param_init.shape[-1]
    
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
       
       # innder function catching parameters in outer function
        def kernel_cost(sigma_i, seqs_train = seqs_train,params = params,params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting
            
            Parameters
            ----------
            sigma_i : float
                current kernel parameter sigma at GP state dimension i
            params : dict
                GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension that performing kernel paramater optimisation

            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
            """

            params['sigma'][params_idx] = sigma_i
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp,yp = bayesian_optimisation.bayesian_optimisation(
                                    n_iters=params['notes']['learnKernelParams_with_bo'],
                                    sample_loss=kernel_cost,
                                    # kernel parameters learned by powell method 
                                    # are usually distributed in the scale
                                    bounds=np.array([[1e-5,3]]),
                                    n_pre_samples=10,
                                    random_search=100000,
                                    greater_is_better=False,
                                    acquisition_func=bayesian_optimisation.probability_improvement)

        param_opt['sigma'][i] = xp[yp.argmin()]

    return param_opt

def learn_rq_params(params, seqs_train):
    """
    find optimal alpha and ell for Rational Quadratic kernel with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Rational Quadratic kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields
    
    Returns
    -------
    param_opt : np.ndarray
        updated Rational Quadratic kernel parameter alpha ell
    """
    # horizentally join kernel parameters 
    param_init = np.concatenate((params['alpha'],params['ell']),axis=0)
    param_opt = {'alpha': np.empty_like(params['alpha']),'ell':np.empty_like(params['ell'])}
     
    x_dim = int((param_init.shape[-1])/2)

    def kernel_cost(alpha_ell_i, params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting
        
        Parameters
        ----------
        alpha_ell_i : array
            current kernel parameter [alpha,ell] at GP state dimension i
        params : dict
            GPFA model parameters
        seqs_train : np.recarray
            training data structure
        params_idx : the state dimension performing kernel paramater optimisation

        
        Returns
        -------
        ll : float
            data log likelihood, returned by exact_inference_with_ll
        """
        params['alpha'][params_idx] = alpha_ell_i[0]
        params['ell'][params_idx] = alpha_ell_i[1]
        sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        #optimise the specific dimension's kernel parameter 
        res_opt = optimize.minimize(kernel_cost, np.array([param_init[i],param_init[i+x_dim]]), 
                                    args=(params, seqs_train, i),
                                    method='Powell',
                                    # emprical safety bound for searching hyperparameter 
                                    bounds=[(1e-5,3e10),(1e-5,3e10)])
        param_opt['alpha'][i] = res_opt.x[0]
        param_opt['ell'][i] = res_opt.x[1]

    return param_opt

def learn_rq_params_with_bo(params, seqs_train):
    """
    find optimal alpha and ell for Rational Quadratic kernel with Bayesian Optimisation
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Rational Quadratic kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Rational Quadratic kernel parameter alpha and ell
    """
    param_init = np.concatenate((params['alpha'],params['ell']),axis=0)
    param_opt = {'alpha': np.empty_like(params['alpha']),'ell':np.empty_like(params['ell'])}

    x_dim = int((param_init.shape[-1])/2)

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        def kernel_cost(alpha_ell_i, seqs_train = seqs_train,params = params,params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting
            
            Parameters
            ----------
            alpha_ell_i : array
                current kernel parameter [alpha,ell] at GP state dimension i
            params : dict
            GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension performing kernel paramater optimisation

            
            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
            """
            alpha_ell_i = np.array(alpha_ell_i)
            # optimal scale for searching hyperparameter under bayesian optimisation
            # kernel parameters learned by powell method are usually distributed in the scale 
            params['alpha'][params_idx] = alpha_ell_i[0]*100
            params['ell'][params_idx] = alpha_ell_i[1]/1000
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp,yp = bayesian_optimisation.bayesian_optimisation(n_iters=params['notes']['learnKernelParams_with_bo'],
                               sample_loss=kernel_cost,
                                # bounds=np.array([1e-5,1000],[1e-5,1e-2])
                               bounds=np.array([[1e-7,10],[1e-2,10]]),
                               
                               n_pre_samples=10,
                               random_search=100000,
                                greater_is_better=False,
                                acquisition_func=bayesian_optimisation.probability_improvement)

        optimized_x = xp[yp.argmin()]
        # multiply
        param_opt['alpha'][i] = optimized_x[0]*100
        param_opt['ell'][i] = optimized_x[1]/1000

    return param_opt

def learn_matern_params(params, seqs_train):
    """
    find optimal nu and ell for Matern kernel with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Matern kernel parameter nu and ell
    """
    param_init = np.concatenate((params['nu'],params['ell']),axis=0)
    param_opt = {'nu': np.empty_like(params['nu']),'ell':np.empty_like(params['ell'])}

    x_dim = int((param_init.shape[-1])/2)

    def kernel_cost(nu_ell_i, params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting
        
        Parameters
        ----------
        nu_ell_i : array
            current kernel parameter [nu,ell] at GP state dimension i
        params : dict
            GPFA model parameters
        seqs_train : np.recarray
            training data structure
        params_idx : the state dimension performing kernel paramater optimisation

        
        Returns
        -------
        ll : float
            data log likelihood, returned by exact_inference_with_ll
        """
        # Powell method have a small chance to sample out of bound parameters
        # 1e-5 < nu < 30, out of bound may cause value error
        params['nu'][params_idx] = np.abs(nu_ell_i[0])
        params['ell'][params_idx] = nu_ell_i[1]
        if params['nu'][params_idx] > 30:
            ll = np.inf
        else:
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):

        res_opt = optimize.minimize(kernel_cost, np.array([param_init[i],param_init[i+x_dim]]),
                                    args=(params, seqs_train, i),
                                    method='Powell',
                                    bounds=[(1e-05,30),(1e-05,5)])
        param_opt['nu'][i] = res_opt.x[0]
        param_opt['ell'][i] = res_opt.x[1]

    return param_opt

def learn_matern_params_with_bo(params, seqs_train):
    """
    find optimal nu and ell for Matern kernel with Bayesian Optimisation
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Matern kernel parameter nu and ell
    """
    param_init = np.concatenate((params['nu'],params['ell']),axis=0)
    param_opt = {'nu': np.empty_like(params['nu']),'ell':np.empty_like(params['ell'])}

    x_dim = int((param_init.shape[-1])/2)

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        def kernel_cost(nu_ell_i, seqs_train = seqs_train, params = params, params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting
            
            Parameters
            ----------
            nu_ell_i : array
                current kernel parameter [nu,ell] at GP state dimension i
            params : dict
                GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension performing kernel paramater optimisation

            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
            """
            params['nu'][params_idx] = nu_ell_i[0]
            params['ell'][params_idx] = nu_ell_i[1]
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp,yp = bayesian_optimisation.bayesian_optimisation(n_iters=params['notes']['learnKernelParams_with_bo'],
                               sample_loss=kernel_cost,
                                # <1e-5 may cause numberical underflow; 
                                # >3e10 may cause infinity in computation
                               bounds=np.array([[1e-05,30],[1e-05,5.0]]),
                               n_pre_samples=10,
                               random_search=100000,
                                greater_is_better=False,
                                acquisition_func=bayesian_optimisation.probability_improvement)

        optimized_x = xp[yp.argmin()]
        param_opt['nu'][i] = optimized_x[0]
        param_opt['ell'][i] = optimized_x[1]

    return param_opt

def learn_tri_times_rq_params(params, seqs_train):
    """
    find optimal sigma alpha ell for Triangular x Rational Quadratic kernel 
    with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Triangular x Rational kernel parameter sigma, alpha, ell
    """
    param_init = np.concatenate((params['sigma'],params['alpha'],params['ell']),axis=0)
    param_opt = {'sigma':np.empty_like(params['sigma']),
                'alpha': np.empty_like(params['alpha']),
                 'ell':np.empty_like(params['ell'])}

    x_dim = int((param_init.shape[-1])/3)

    def kernel_cost(sigma_alpha_ell_i, params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting
        
        Parameters
        ----------
        sigma_alpha_ell_i : array
            current kernel parameter [sigma,alpha,ell] at GP state dimension i
        params : dict
            GPFA model parameters
        seqs_train : np.recarray
            training data structure
        params_idx : the state dimension performing kernel paramater optimisation

        Returns
        -------
        ll : float
            data log likelihood, returned by exact_inference_with_ll
        """
        params['sigma'][params_idx] = sigma_alpha_ell_i[0]
        params['alpha'][params_idx] = sigma_alpha_ell_i[1]
        params['ell'][params_idx] = sigma_alpha_ell_i[2]
        sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):

        res_opt = optimize.minimize(kernel_cost, np.array([param_init[i],param_init[i+x_dim],param_init[i+2*x_dim]]),
                                    args=(params, seqs_train, i),
                                    method='Powell',
                                    bounds=[(1e-5,3e10),(1e-5,3e10),(1e-5,3e10)])
        param_opt['sigma'][i] = res_opt.x[0]
        param_opt['alpha'][i] = res_opt.x[1]
        param_opt['ell'][i] = res_opt.x[2]
    return param_opt

def learn_tri_times_rq_params_with_bo(params, seqs_train):
    """
    find optimal sigma alpha ell for Triangular x Rational Quadratic kernel 
    with Bayesian Optimisation
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
    seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Triangular x Rational kernel parameter sigma, alpha, ell
    """
    param_init = np.concatenate((params['sigma'],params['alpha'],params['ell']),axis=0)
    param_opt = {'sigma':np.empty_like(params['sigma']),
                'alpha': np.empty_like(params['alpha']),
                 'ell':np.empty_like(params['ell'])}

    x_dim = int((param_init.shape[-1])/3)

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):

        def kernel_cost(sigma_alpha_ell_i, seqs_train = seqs_train, params = params, params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting
            
            Parameters
            ----------
            sigma_alpha_ell_i : array
                current kernel parameter [sigma,alpha,ell] at GP state dimension i
            params : dict
                GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension performing kernel paramater optimisation

            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
            """
            # kernel parameters learned by powell method are usually distributed in the scale 
            params['sigma'][params_idx] = sigma_alpha_ell_i[0]/5
            params['alpha'][params_idx] = sigma_alpha_ell_i[1]/100
            params['ell'][params_idx] = sigma_alpha_ell_i[2]/100
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp, yp = bayesian_optimisation.bayesian_optimisation(n_iters=params['notes']['learnKernelParams_with_bo'],
                                                             sample_loss=kernel_cost,
                                                            #   bounds=np.array([[1e-5, 2], [1e-5, 1e-1],[1e-5,1e-1]]),
                                                             bounds=np.array([[2e-4, 10], [1e-3, 10],[1e-3,10]]),
                                                             n_pre_samples=10,
                                                             random_search=100000,
                                                             greater_is_better=False,
                                                             acquisition_func=bayesian_optimisation.probability_improvement)

        optimized_x = xp[yp.argmin()]
        param_opt['sigma'][i] = optimized_x[0]/5
        param_opt['alpha'][i] = optimized_x[1]/100
        param_opt['ell'][i] = optimized_x[1]/100

    return param_opt


def learn_exp_times_rq_params(params, seqs_train):
    """
    find optimal sigma alpha ell for Exponential x Rational Quadratic kernel 
    with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Exponential x Rational kernel parameter sigma, alpha, ell
    """
    param_init = np.concatenate((params['sigma'],params['alpha'],params['ell']),axis=0)
    param_opt = {'sigma':np.empty_like(params['sigma']),
                'alpha': np.empty_like(params['alpha']),
                 'ell':np.empty_like(params['ell'])}

    x_dim = int((param_init.shape[-1])/3)

    def kernel_cost(sigma_alpha_ell_i, params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting
        
        Parameters
        ----------
        sigma_alpha_ell_i : array
            current kernel parameter [sigma,alpha,ell] at GP state dimension i
        params : dict
            GPFA model parameters
        seqs_train : np.recarray
            training data structure
        params_idx : the state dimension performing kernel paramater optimisation

        Returns
        -------
        ll : float
            data log likelihood, returned by exact_inference_with_ll
        """
        params['sigma'][params_idx] = sigma_alpha_ell_i[0]
        params['alpha'][params_idx] = sigma_alpha_ell_i[1]
        params['ell'][params_idx] = sigma_alpha_ell_i[2]
        sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        res_opt = optimize.minimize(kernel_cost, np.array([param_init[i],param_init[i+x_dim],param_init[i+2*x_dim]]),
                                    args=(params, seqs_train, i),
                                    method='Powell',
                                    bounds=[(1e-5,3e10),(1e-5,3e10),(1e-5,3e10)])
        param_opt['sigma'][i] = res_opt.x[0]
        param_opt['alpha'][i] = res_opt.x[1]
        param_opt['ell'][i] = res_opt.x[2]

    return param_opt

def learn_exp_times_rq_params_with_bo(params,seqs_train):
    """
    find optimal sigma alpha ell for Exponential x Rational Quadratic kernel 
    with Bayesian Optimisation
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
    seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Exponential x Rational kernel parameter sigma, alpha, ell
    """
    param_init = np.concatenate((params['sigma'],params['alpha'],params['ell']),axis=0)
    param_opt = {'sigma':np.empty_like(params['sigma']),
                'alpha': np.empty_like(params['alpha']),
                 'ell':np.empty_like(params['ell'])}

    x_dim = int((param_init.shape[-1])/3)


    # Loop once for each state dimension (each GP)
    for i in range(x_dim):

        def kernel_cost(sigma_alpha_ell_i, seqs_train = seqs_train, params = params, params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting
            
            Parameters
            ----------
            sigma_alpha_ell_i : array
                current kernel parameter [sigma,alpha,ell] at GP state dimension i
            params : dict
                GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension performing kernel paramater optimisation

            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
            """
            sigma_alpha_ell_i = np.array(sigma_alpha_ell_i)
            # kernel parameters learned by powell method are usually distributed in the scale 
            params['sigma'][params_idx] = sigma_alpha_ell_i[0]
            params['alpha'][params_idx] = sigma_alpha_ell_i[1]*30
            params['ell'][params_idx] = sigma_alpha_ell_i[2]/1000
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp,yp = bayesian_optimisation.bayesian_optimisation(n_iters=params['notes']['learnKernelParams_with_bo'],
                                                            sample_loss=kernel_cost,
                                                            # bounds=np.array([[1e-5,10],[3e-04,300],[1e-8,1e-2]]),
                                                            bounds=np.array([[1e-5,10],[1e-05,10],[1e-5,10]]),
                                                            n_pre_samples=10,
                                                            random_search=100000,
                                                            greater_is_better=False,
                                                            acquisition_func=bayesian_optimisation.expected_improvement)

        optimized_x = xp[yp.argmin()]
        param_opt['sigma'][i] = optimized_x[0]
        param_opt['alpha'][i] = optimized_x[1]*30
        param_opt['ell'][i] = optimized_x[2]/1000

    return param_opt

def learn_exp_times_tri_params(params, seqs_train):
    """
    find optimal sigma alpha ell for Exponential x Rational Quadratic kernel 
    with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Exponential x Rational kernel parameter sigma, alpha, ell
    """
    param_init = np.concatenate((params['sigma_exp'],params['sigma_tri']),axis=0)
    param_opt = {'sigma_exp':np.empty_like(params['sigma_exp']),
                'sigma_tri': np.empty_like(params['sigma_tri'])}

    x_dim = int((param_init.shape[-1])/2)

    def kernel_cost(sigma_e_sigma_t_i, params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting
        
        Parameters
        ----------
        sigma_e_sigma_t_i : array
            current kernel parameter [sigma_exp, sigma_tri] at GP state dimension i
        params : dict
            GPFA model parameters
        seqs_train : np.recarray
            training data structure
        params_idx : the state dimension performing kernel paramater optimisation

        Returns
        -------
        ll : float
            data log likelihood, returned by exact_inference_with_ll
        """
        params['sigma_exp'][params_idx] = sigma_e_sigma_t_i[0]
        params['sigma_tri'][params_idx] = sigma_e_sigma_t_i[1]

        sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        const = {'eps': params['eps'][i]}
        # initp = np.log(param_init[i])

        res_opt = optimize.minimize(kernel_cost, np.array([param_init[i],param_init[i+x_dim]]),
                                    args=(params, seqs_train, i),
                                    bounds = [(1e-5,3e10),(1e-5,3e10)],
                                    method='Powell')
        param_opt['sigma_exp'][i] = res_opt.x[0]
        param_opt['sigma_tri'][i] = res_opt.x[1]

    return param_opt

def learn_exp_times_tri_params_with_bo(params, seqs_train):
    """
    find optimal sigma alpha ell for Exponential x Triangular kernel 
    with Bayesian Optimisation
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP Matern kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated Exponential x Triangular kernel parameter sigma_exp, sigma_tri
    """
    param_init = np.concatenate((params['sigma_exp'],params['sigma_tri']),axis=0)
    param_opt = {'sigma_exp':np.empty_like(params['sigma_exp']),
                'sigma_tri': np.empty_like(params['sigma_tri'])}

    x_dim = int((param_init.shape[-1])/2)

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):

        def kernel_cost(sigma_e_sigma_t_i, seqs_train = seqs_train, params = params, params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting
            
            Parameters
            ----------
            sigma_e_sigma_t_i : array
                current kernel parameter [sigma_exp, sigma_tri] at GP state dimension i
            params : dict
                GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension performing kernel paramater optimisation

            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
            """
            # kernel parameters learned by powell method are usually distributed in the scale 
            params['sigma_exp'][params_idx] = sigma_e_sigma_t_i[0]*10000000
            params['sigma_tri'][params_idx] = sigma_e_sigma_t_i[1]/5
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp, yp = bayesian_optimisation.bayesian_optimisation(n_iters=params['notes']['learnKernelParams_with_bo'],
                                                             sample_loss=kernel_cost,
                                                            #  bounds=np.array([[1e+2, 1e+8], [2e-6, 2]]),
                                                             bounds=np.array([[1e-5, 10], [1e-5, 10]]),
                                                             n_pre_samples=10,
                                                             random_search=100000,
                                                             greater_is_better=False,
                                                             acquisition_func=bayesian_optimisation.probability_improvement)

        optimized_x = xp[yp.argmin()]
        param_opt['sigma_exp'][i] = optimized_x[0]*10000000
        param_opt['sigma_tri'][i] = optimized_x[1]/5

    return param_opt

def learn_rbf_params(params, seqs_train):
    """
    find optimal gamma for rbf kernel with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP rbf kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    
    Returns
    -------
    param_opt : np.ndarray
        updated rbf kernel parameter gamma
    """
    param_name = 'gamma'
    param_init = params[param_name]
    param_opt = {param_name: np.empty_like(param_init)}

    x_dim = param_init.shape[-1]

    def kernel_cost(gamma_i,  params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting

        Parameters
            ----------
            gamma_i : array
                current kernel parameter gamma at GP state dimension i
            params : dict
                GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension performing kernel paramater optimisation

            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
        """
        params['gamma'][params_idx] = gamma_i
        sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    # Loop once for each state dimension (each GP)
    for i in range(x_dim):

        res_opt = optimize.minimize(kernel_cost, param_init[i],
                                    args=(params, seqs_train, i),
                                    method='Powell',
                                    bounds=[(0.1,100)])
        param_opt['gamma'][i] = res_opt.x
    return param_opt

def learn_rbf_params_with_bo(params,seqs_train):
    """
    find optimal gamma for rbf kernel with Bayesian Optimisation
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP rbf kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    
    Returns
    -------
    param_opt : np.ndarray
        updated rbf kernel parameter gamma
    """
    param_name = 'gamma'
    param_init = params[param_name]
    param_opt = {param_name: np.empty_like(param_init)}

    x_dim = param_init.shape[-1]
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):

        def kernel_cost(gamma_i,seqs_train = seqs_train, params = params, params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting

            Parameters
                ----------
                gamma_i : array
                    current kernel parameter gamma at GP state dimension i
                params : dict
                    GPFA model parameters
                seqs_train : np.recarray
                    training data structure
                params_idx : the state dimension performing kernel paramater optimisation

                Returns
                -------
                ll : float
                    data log likelihood, returned by exact_inference_with_ll
            """

            params['gamma'][params_idx] = gamma_i
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp,yp = bayesian_optimisation.bayesian_optimisation(n_iters=params['notes']['learnKernelParams_with_bo'],
                               sample_loss=kernel_cost,
                               bounds=np.array([[0.1,10]]),
                               n_pre_samples=10,
                               random_search=100000,
                                greater_is_better=False,
                                acquisition_func=bayesian_optimisation.probability_improvement)

        param_opt['gamma'][i] = xp[yp.argmin()]

    return param_opt

def learn_sm_params(params, seqs_train):
    """
    find optimal w mu vs for rbf kernel with Scipy Powell Minimizer
    
    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP rbf kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated rbf kernel parameter gamma
    """

    x_dim = params['C'].shape[1]

    def kernel_cost(w_mu_vs_i,  params, seqs_train, params_idx):
        """
        return Negative Loglikelihood cost for current kernel parameter setting

        Parameters
            ----------
            w_mu_vs_i : array
                current kernel parameter [w,mu,vs] at GP state dimension i
                where w,mu,vs are params['Q'] dimensional
            params : dict
                GPFA model parameters
            seqs_train : np.recarray
                training data structure
            params_idx : the state dimension performing kernel paramater optimisation

            Returns
            -------
            ll : float
                data log likelihood, returned by exact_inference_with_ll
        """

        params['w'][params_idx] = w_mu_vs_i[:params['Q']]
        params['mu'][params_idx] = w_mu_vs_i[params['Q']:params['Q']*2]
        params['vs'][params_idx] = w_mu_vs_i[params['Q']*2:params['Q']*3]
        sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                     get_ll=True)
        return -ll

    param_opt = {'w':np.empty_like(params['w']),
                'mu': np.empty_like(params['mu']),
                'vs':np.empty_like(params['vs'])}
    bounds = [(1e-5,1)]*(3*params['Q'])
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        param_init = np.concatenate((params['w'][i],params['mu'][i],params['vs'][i]),axis=0)


        res_opt = optimize.minimize(kernel_cost, param_init,
                                    args=(params, seqs_train, i),
                                    method='Powell',
                                    bounds= bounds)
        param_opt['w'][i] = res_opt.x[:params['Q']]
        param_opt['mu'][i] = res_opt.x[params['Q']:params['Q']*2]
        param_opt['vs'][i] = res_opt.x[params['Q']*2:params['Q']*3]

    return param_opt

def learn_sm_params_with_bo(params, seqs_train):
    """
    find optimal w mu vs for rbf kernel with Scipy Powell Minimizer

    Parameters
    ----------
    params : dict
        current GP state model parameters, which gives starting point
        for GP rbf kernel parameter optimization;
        seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    Returns
    -------
    param_opt : np.ndarray
        updated rbf kernel parameter gamma
    """

    x_dim = params['C'].shape[1]


    param_opt = {'w':np.empty_like(params['w']),
                'mu': np.empty_like(params['mu']),
                'vs':np.empty_like(params['vs'])}
    bounds = [(1e-5,1)]*(3*params['Q'])
    # Loop once for each state dimension (each GP)
    for i in range(x_dim):
        # param_init = np.concatenate((params['w'][i],params['mu'][i],params['vs'][i]),axis=0)

        def kernel_cost(w_mu_vs_i, seqs_train = seqs_train, params = params, params_idx = i):
            """
            return Negative Loglikelihood cost for current kernel parameter setting

            Parameters
                ----------
                w_mu_vs_i : array
                    current kernel parameter [w,mu,vs] at GP state dimension i
                    where w,mu,vs are params['Q'] dimensional
                params : dict
                    GPFA model parameters
                seqs_train : np.recarray
                    training data structure
                params_idx : the state dimension performing kernel paramater optimisation

                Returns
                -------
                ll : float
                    data log likelihood, returned by exact_inference_with_ll
            """
            w_mu_vs_i = np.array(w_mu_vs_i)
            params['w'][params_idx] = w_mu_vs_i[:params['Q']]
            params['mu'][params_idx] = w_mu_vs_i[params['Q']:params['Q']*2]
            params['vs'][params_idx] = w_mu_vs_i[params['Q']*2:params['Q']*3]
            sequece_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                         get_ll=True)
            return -ll

        xp,yp = bayesian_optimisation.bayesian_optimisation(n_iters=params['notes']['learnKernelParams_with_bo'],
                                                            sample_loss=kernel_cost,
                                                            bounds=np.array(bounds),
                                                            n_pre_samples=10,
                                                            random_search=100000,
                                                            greater_is_better=False,
                                                            acquisition_func=bayesian_optimisation.probability_improvement)
        optimized_x = xp[yp.argmin()]
        param_opt['w'][i] = optimized_x[:params['Q']]
        param_opt['mu'][i] = optimized_x[params['Q']:params['Q']*2]
        param_opt['vs'][i] = optimized_x[params['Q']*2:params['Q']*3]

    return param_opt


def em(params_init, seqs_train, max_iters=500, tol=1.0E-8, min_var_frac=0.01,
       freq_ll=5, verbose=False):
    """
    Fits GPFA model parameters using expectation-maximization (EM) algorithm.

    Parameters
    ----------
    params_init : dict
        GPFA model parameters at which EM algorithm is initialized
          covType: {'rbf', 'tri', 'exp', 'rq', 'matern', 'sm',
                'tri_times_rq', 'exp_times_rq', 'exp_times_tri'}
                type of GP covariance
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by
            'bin_width / sqrt(gamma)'
            new GP kernels have different kernel parameters
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the neuronal data space and the
            latent variable space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance
        
        New developed GP kernels have other kernel parameters:
            gamma: rbf, rbf try
            sigma: tri, exp
            alpha, ell: rq
            nu, ell: matern
            Q, w, mu, vs: sm
                Q: int
                number of spectral mixtures, cann be adjusted.
                Default: 2
            sigma, alpha, ell: tri_times_rq, exp_times_rq
            sigma_e, sigma_t: exp_times_tri

    seqs_train : np.recarray
        training data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
        T : int
            number of bins
        y : np.ndarray (yDim x T)
            neural data
    max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations.
        freq_ll = 1 means that data likelihood is computed at every
        iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False

    Returns
    -------
    params_est : dict
        GPFA model parameter estimates, returned by EM algorithm (same
        format as params_init)
    seqs_latent : np.recarray
        a copy of the training data structure, augmented with the new
        fields:
        latent_variable : np.ndarray of shape (#latent_vars x #bins)
            posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
            posterior covariance between latent variables at each
            timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
            posterior covariance over time for each latent
            variable
    ll : list
        list of log likelihoods after each EM iteration
    iter_time : list
        lisf of computation times (in seconds) for each EM iteration
    """
    params = params_init
    t = seqs_train['T']
    y_dim, x_dim = params['C'].shape
    lls = []
    ll_old = ll_base = ll = 0.0
    iter_time = []
    var_floor = min_var_frac * np.diag(np.cov(np.hstack(seqs_train['y'])))
    seqs_latent = None

    # Loop once for each iteration of EM algorithm
    for iter_id in trange(1, max_iters + 1, desc='EM iteration',
                          disable=not verbose):
        if verbose:
            print()
        tic = time.time()
        get_ll = (np.fmod(iter_id, freq_ll) == 0) or (iter_id <= 2)

        # ==== E STEP =====
        if not np.isnan(ll):
            ll_old = ll
        seqs_latent, ll = exact_inference_with_ll(seqs_train, params,
                                                  get_ll=get_ll)
        lls.append(ll)

        # ==== M STEP ====
        sum_p_auto = np.zeros((x_dim, x_dim))
        for seq_latent in seqs_latent:
            sum_p_auto += seq_latent['Vsm'].sum(axis=2) \
                + seq_latent['latent_variable'].dot(
                seq_latent['latent_variable'].T)
        y = np.hstack(seqs_train['y'])
        latent_variable = np.hstack(seqs_latent['latent_variable'])
        sum_yxtrans = y.dot(latent_variable.T)
        sum_xall = latent_variable.sum(axis=1)[:, np.newaxis]
        sum_yall = y.sum(axis=1)[:, np.newaxis]

        # term is (xDim+1) x (xDim+1)
        term = np.vstack([np.hstack([sum_p_auto, sum_xall]),
                          np.hstack([sum_xall.T, t.sum().reshape((1, 1))])])
        # yDim x (xDim+1)
        cd = gpfa_util.rdiv(np.hstack([sum_yxtrans, sum_yall]), term)

        params['C'] = cd[:, :x_dim]
        params['d'] = cd[:, -1]

        # yCent must be based on the new d
        # yCent = bsxfun(@minus, [seq.y], currentParams.d);
        # R = (yCent * yCent' - (yCent * [seq.latent_variable]') * \
        #     currentParams.C') / sum(T);
        c = params['C']
        d = params['d'][:, np.newaxis]
        if params['notes']['RforceDiagonal']:
            sum_yytrans = (y * y).sum(axis=1)[:, np.newaxis]
            yd = sum_yall * d
            term = ((sum_yxtrans - d.dot(sum_xall.T)) * c).sum(axis=1)
            term = term[:, np.newaxis]
            r = d ** 2 + (sum_yytrans - 2 * yd - term) / t.sum()

            # Set minimum private variance
            r = np.maximum(var_floor, r)
            params['R'] = np.diag(r[:, 0])
        else:
            sum_yytrans = y.dot(y.T)
            yd = sum_yall.dot(d.T)
            term = (sum_yxtrans - d.dot(sum_xall.T)).dot(c.T)
            r = d.dot(d.T) + (sum_yytrans - yd - yd.T - term) / t.sum()

            params['R'] = (r + r.T) / 2  # ensure symmetry

        if params['notes']['learnKernelParams']:
            # res = learn_gp_params(seqs_latent, params, verbose=verbose)
            if params['covType']=='rbf':
                res = learn_gp_params(seqs_train, seqs_latent, params, verbose=verbose)
                params['gamma'] = res['gamma']

            if params_init['covType'] in ['rbf_try']:
                if params_init['notes']['learnKernelParams_with_bo']:
                    opt = learn_rbf_params_with_bo(params_init, seqs_train)
                    params_init['gamma'] = opt['gamma']
                else:
                    opt = learn_rbf_params(params_init, seqs_train)
                    params_init['gamma'] = opt['gamma']

        if params_init['covType'] in ['exp', 'tri']:
            if params_init['notes']['learnKernelParams_with_bo']:
                opt = learn_sigma_with_bo(params_init, seqs_train)
                params_init['sigma'] = opt['sigma']
            else:
                opt = learn_sigma(params_init, seqs_train)
                params_init['sigma'] = opt['sigma']

        if params_init['covType'] in ['rq']:
            if params_init['notes']['learnKernelParams_with_bo']:
                opt = learn_rq_params_with_bo(params_init, seqs_train)
                params_init['alpha'] = opt['alpha']
                params_init['ell'] = opt['ell']
            else:
                opt = learn_rq_params(params_init, seqs_train)
                params_init['alpha'] = opt['alpha']
                params_init['ell'] = opt['ell']
        if params_init['covType'] in ['matern']:
            if params_init['notes']['learnKernelParams_with_bo']:
                opt = learn_matern_params_with_bo(params_init, seqs_train)
                params_init['nu'] = opt['nu']
                params_init['ell'] = opt['ell']
            else:
                opt = learn_matern_params(params_init, seqs_train)
                params_init['nu'] = opt['nu']
                params_init['ell'] = opt['ell']

        if params_init['covType'] in ['tri_times_rq']:
            if params_init['notes']['learnKernelParams_with_bo']:
                opt = learn_tri_times_rq_params_with_bo(params_init, seqs_train)
                params_init['sigma'] = opt['sigma']
                params_init['alpha'] = opt['alpha']
                params_init['ell'] = opt['ell']
            else:
                opt = learn_tri_times_rq_params(params_init, seqs_train)
                params_init['sigma'] = opt['sigma']
                params_init['alpha'] = opt['alpha']
                params_init['ell'] = opt['ell']

        if params_init['covType'] in ['exp_times_rq']:
            if params_init['notes']['learnKernelParams_with_bo']:
                opt = learn_exp_times_rq_params_with_bo(params_init, seqs_train)
                params_init['sigma'] = opt['sigma']
                params_init['alpha'] = opt['alpha']
                params_init['ell'] = opt['ell']
            else:
                opt = learn_exp_times_rq_params(params_init, seqs_train)
                params_init['sigma'] = opt['sigma']
                params_init['alpha'] = opt['alpha']
                params_init['ell'] = opt['ell']

        if params_init['covType'] in ['exp_times_tri']:
            if params_init['notes']['learnKernelParams_with_bo']:
                opt = learn_exp_times_tri_params_with_bo(params_init, seqs_train)
                params_init['sigma_exp'] = opt['sigma_exp']
                params_init['sigma_tri'] = opt['sigma_tri']
            else:
                opt = learn_exp_times_tri_params(params_init, seqs_train)
                params_init['sigma_exp'] = opt['sigma_exp']
                params_init['sigma_tri'] = opt['sigma_tri']
        
        if params_init['covType'] in ['sm']:
            if params_init['notes']['learnKernelParams_with_bo']:
                opt = learn_sm_params_with_bo(params_init, seqs_train)
                params_init['w'] = opt['w']
                params_init['mu'] = opt['mu']
                params_init['vs'] = opt['vs']
            else:
                opt = learn_sm_params(params_init, seqs_train)
                params_init['w'] = opt['w']
                params_init['mu'] = opt['mu']
                params_init['vs'] = opt['vs']

        t_end = time.time() - tic
        iter_time.append(t_end)

        # Verify that likelihood is growing monotonically
        if iter_id <= 2:
            ll_base = ll
        elif verbose and ll < ll_old:
            print('\nError: Data likelihood has decreased ',
                  'from {0} to {1}'.format(ll_old, ll))
        elif (ll - ll_base) < (1 + tol) * (ll_old - ll_base):
            break

    if len(lls) < max_iters:
        print('Fitting has converged after {0} EM iterations.)'.format(
            len(lls)))

    if np.any(np.diag(params['R']) == var_floor):
        warnings.warn('Private variance floor used for one or more observed '
                      'dimensions in GPFA.')

    return params, seqs_latent, lls, iter_time


def exact_inference_with_ll(seqs, params, get_ll=True):
    """
    Extracts latent trajectories from neural data, given GPFA model parameters.

    Parameters
    ----------
    seqs : np.recarray
        Input data structure, whose n-th element (corresponding to the n-th
        experimental trial) has fields:
        y : np.ndarray of shape (#units, #bins)
            neural data
        T : int
            number of bins
    params : dict
        GPFA model parameters whe the following fields:
        C : np.ndarray
            FA factor loadings matrix
        d : np.ndarray
            FA mean vector
        R : np.ndarray
            FA noise covariance matrix
        gamma : np.ndarray
            GP timescale
            new GP kernels have different kernel parameters
        eps : np.ndarray
            GP noise variance
        
        New developed GP kernels have other kernel parameters:
            gamma: rbf, rbf try
            sigma: tri, exp
            alpha, ell: rq
            nu, ell: matern
            Q, w, mu, vs: sm
                Q: int
                number of spectral mixtures, cann be adjusted.
                Default: 2
            sigma, alpha, ell: tri_times_rq, exp_times_rq
            sigma_e, sigma_t: exp_times_tri

    get_ll : bool, optional
          specifies whether to compute data log likelihood (default: True)

    Returns
    -------
    seqs_latent : np.recarray
        a copy of the input data structure, augmented with the new
        fields:
        latent_variable :  (#latent_vars, #bins) np.ndarray
              posterior mean of latent variables at each time bin
        Vsm :  (#latent_vars, #latent_vars, #bins) np.ndarray
              posterior covariance between latent variables at each
              timepoint
        VsmGP :  (#bins, #bins, #latent_vars) np.ndarray
                posterior covariance over time for each latent
                variable
    ll : float
        data log likelihood, np.nan is returned when `get_ll` is set False
    """
    y_dim, x_dim = params['C'].shape

    # copy the contents of the input data structure to output structure
    dtype_out = [(x, seqs[x].dtype) for x in seqs.dtype.names]
    dtype_out.extend([('latent_variable', np.object), ('Vsm', np.object),
                      ('VsmGP', np.object)])
    seqs_latent = np.empty(len(seqs), dtype=dtype_out)
    for dtype_name in seqs.dtype.names:
        seqs_latent[dtype_name] = seqs[dtype_name]

    # Precomputations
    if params['notes']['RforceDiagonal']:
        rinv = np.diag(1.0 / np.diag(params['R']))
        logdet_r = (np.log(np.diag(params['R']))).sum()
    else:
        rinv = linalg.inv(params['R'])
        rinv = (rinv + rinv.T) / 2  # ensure symmetry
        logdet_r = gpfa_util.logdet(params['R'])

    c_rinv = params['C'].T.dot(rinv)
    c_rinv_c = c_rinv.dot(params['C'])

    t_all = seqs_latent['T']
    t_uniq = np.unique(t_all)
    ll = 0.

    # Overview:
    # - Outer loop on each element of Tu.
    # - For each element of Tu, find all trials with that length.
    # - Do inference and LL computation for all those trials together.
    for t in t_uniq:
        k_big, k_big_inv, logdet_k_big = gpfa_util.make_k_big(params, t)
        k_big = sparse.csr_matrix(k_big)

        blah = [c_rinv_c for _ in range(t)]
        c_rinv_c_big = linalg.block_diag(*blah)  # (xDim*T) x (xDim*T)
        try:
            minv, logdet_m = gpfa_util.inv_persymm(k_big_inv + c_rinv_c_big, x_dim)
        except:
            minv = np.linalg.inv(k_big_inv + c_rinv_c_big)
            # det_m = np.log(np.linalg.det(k_big_inv + c_rinv_c_big))

            # new built kernels may cause np.linalg.det(k_big_inv + c_rinv_c_big) being negative
            # which will cause value error when getting its log
            # use slogdet instead, which returns the natural log of the ABSOLUTE value of the determinant.
            # The same implementation with https://github.com/brian-lau/NeuralTraj
            sign,logdet_m = np.linalg.slogdet(k_big_inv + c_rinv_c_big)

        # Note that posterior covariance does not depend on observations,
        # so can compute once for all trials with same T.
        # xDim x xDim posterior covariance for each timepoint
        vsm = np.full((x_dim, x_dim, t), np.nan)
        idx = np.arange(0, x_dim * t + 1, x_dim)
        for i in range(t):
            vsm[:, :, i] = minv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

        # T x T posterior covariance for each GP
        vsm_gp = np.full((t, t, x_dim), np.nan)
        for i in range(x_dim):
            vsm_gp[:, :, i] = minv[i::x_dim, i::x_dim]

        # Process all trials with length T
        n_list = np.where(t_all == t)[0]
        # dif is yDim x sum(T)
        dif = np.hstack(seqs_latent[n_list]['y']) - params['d'][:, np.newaxis]
        # term1Mat is (xDim*T) x length(nList)
        term1_mat = c_rinv.dot(dif).reshape((x_dim * t, -1), order='F')

        # Compute blkProd = CRinvC_big * invM efficiently
        # blkProd is block persymmetric, so just compute top half
        t_half = np.int(np.ceil(t / 2.0))
        blk_prod = np.zeros((x_dim * t_half, x_dim * t))
        idx = range(0, x_dim * t_half + 1, x_dim)
        for i in range(t_half):
            blk_prod[idx[i]:idx[i + 1], :] = c_rinv_c.dot(
                minv[idx[i]:idx[i + 1], :])
        blk_prod = k_big[:x_dim * t_half, :].dot(
            gpfa_util.fill_persymm(np.eye(x_dim * t_half, x_dim * t) -
                                   blk_prod, x_dim, t))
        # latent_variableMat is (xDim*T) x length(nList)
        latent_variable_mat = gpfa_util.fill_persymm(
            blk_prod, x_dim, t).dot(term1_mat)

        for i, n in enumerate(n_list):
            seqs_latent[n]['latent_variable'] = \
                latent_variable_mat[:, i].reshape((x_dim, t), order='F')
            seqs_latent[n]['Vsm'] = vsm
            seqs_latent[n]['VsmGP'] = vsm_gp

        if get_ll:
            # Compute data likelihood
            val = -t * logdet_r - logdet_k_big - logdet_m \
                  - y_dim * t * np.log(2 * np.pi)
            ll = ll + len(n_list) * val - (rinv.dot(dif) * dif).sum() \
                + (term1_mat.T.dot(minv) * term1_mat.T).sum()

    if get_ll:
        ll /= 2
    else:
        ll = np.nan

    return seqs_latent, ll


def learn_gp_params(seqs_train, seqs_latent, params, verbose=False):
    """Updates parameters of GP state model, given neural trajectories.

    Parameters
    ----------
    seqs_train : np.recarray
        training data structure, whose n-th element (corresponding to
        the n-th experimental trial) has fields

    seqs_latent : np.recarray
        data structure containing neural trajectories;
    params : dict
        current GP state model parameters, which gives starting point
        for gradient optimization;
    verbose : bool, optional
        specifies whether to display status messages (default: False)

    Returns
    -------
    param_opt : np.ndarray
        updated GP state model parameter

    Raises
    ------
    ValueError
        If `params['covType'] != 'rbf'`.
        If `params['notes']['learnGPNoise']` set to True.

    """
    # if params['covType'] != 'rbf':
    #     raise ValueError("Only 'rbf' GP covariance type is supported.")
    if params['covType'] == 'rbf':
        if params['notes']['learnGPNoise']:
            raise ValueError("learnGPNoise is not supported.")
        param_name = 'gamma'
        param_init = params[param_name]
        param_opt = {param_name: np.empty_like(param_init)}

        x_dim = param_init.shape[-1]
        precomp = gpfa_util.make_precomp(seqs_latent, x_dim)

        # Loop once for each state dimension (each GP)
        for i in range(x_dim):
            const = {'eps': params['eps'][i]}
            initp = np.log(param_init[i])
            res_opt = optimize.minimize(gpfa_util.grad_betgam, initp,
                                        args=(precomp[i], const),
                                        method='L-BFGS-B', jac=True)
            param_opt['gamma'][i] = np.exp(res_opt.x)
            if verbose:
                print('\n Converged p; xDim:{}, p:{}'.format(i, res_opt.x))

    else:
        ValueError('kernel not supported')
        param_opt = params['gamma']
    return param_opt


def orthonormalize(params_est, seqs):
    """
    Orthonormalize the columns of the loading matrix C and apply the
    corresponding linear transform to the latent variables.

    Parameters
    ----------
    params_est : dict
        First return value of extract_trajectory() on the training data set.
        Estimated model parameters.
        When the GPFA method is used, following parameters are contained
        covType: {'rbf', 'tri', 'exp', 'rq', 'matern', 'sm',
                'tri_times_rq', 'exp_times_rq', 'exp_times_tri'}
                type of GP covariance
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by 'bin_width / sqrt(gamma)'
            new GP kernels have different kernel parameters
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the neuronal data space and the latent variable
            space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance
        
        New developed GP kernels have other kernel parameters:
            gamma: rbf, rbf try
            sigma: tri, exp
            alpha, ell: rq
            nu, ell: matern
            Q, w, mu, vs: sm
                Q: int
                number of spectral mixtures, cann be adjusted.
                Default: 2
            sigma, alpha, ell: tri_times_rq, exp_times_rq
            sigma_e, sigma_t: exp_times_tri

    seqs : np.recarray
        Contains the embedding of the training data into the latent variable
        space.
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has fields
        T : int
          number of timesteps
        y : np.ndarray of shape (#units, #bins)
          neural data
        latent_variable : np.ndarray of shape (#latent_vars, #bins)
          posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
          posterior covariance between latent variables at each
          timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
          posterior covariance over time for each latent variable

    Returns
    -------
    params_est : dict
        Estimated model parameters, including `Corth`, obtained by
        orthonormalizing the columns of C.
    seqs : np.recarray
        Training data structure that contains the new field
        `latent_variable_orth`, the orthonormalized neural trajectories.
    """
    C = params_est['C']
    X = np.hstack(seqs['latent_variable'])
    latent_variable_orth, Corth, _ = gpfa_util.orthonormalize(X, C)
    seqs = gpfa_util.segment_by_trial(
        seqs, latent_variable_orth, 'latent_variable_orth')

    params_est['Corth'] = Corth

    return Corth, seqs
