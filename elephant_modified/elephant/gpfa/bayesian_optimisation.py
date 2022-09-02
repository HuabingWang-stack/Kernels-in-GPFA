

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize

import os



def probability_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)


    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        probability_improvement = norm.cdf(Z)
        probability_improvement[sigma == 0.0] = 0.0

    return -1 * probability_improvement


def Mprobability_improvement(x, gaussian_process, evaluated_loss, xp, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
        x0_index=np.argmax(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)
        x0_index=np.argmin(evaluated_loss)

    mo_x = np.reshape(xp[x0_index],(-1,n_params))
    x_to_predict=np.array(x_to_predict)
    mo_x=np.array(mo_x)
    X=np.concatenate((mo_x,x_to_predict))
    Mu, Kappa = gaussian_process.predict(X,return_cov=True)
    Kappa=np.array(Kappa)
    mu0=Mu[0]
    mu=Mu[1:]
    sigma=np.sqrt(np.diag(Kappa[1:,1:]))
    new_sigma=(sigma**2+Kappa[0,0]-2*Kappa[0,1:])
    new_sigma=np.sqrt(new_sigma)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z =  scaling_factor * (mu - mu0) / new_sigma
        Mprobability_improvement = norm.cdf(Z)
        Mprobability_improvement[new_sigma == 0.0] = 0.0

    return -1 * Mprobability_improvement

def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)
    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    #print("x:",x)
    #print("x___:",x_to_predict)
    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)
    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        # print(mu,loss_optimum,mu-loss_optimum)
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] = 0.0

    return -1 * expected_improvement


def Mexpected_improvement(x, gaussian_process, evaluated_loss, xp,greater_is_better=False, n_params=1):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """
    kernel = gp.kernels.Matern()
    x_to_predict = x.reshape(-1, n_params)
  #  mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
        x0_index=np.argmax(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)
        x0_index=np.argmin(evaluated_loss)

    #print("x_to_predict:",x_to_predict)
#    print("loss_optimum: ",loss_optimum)
#    print("corrspond x0: ",xp[x0_index])
#    print("mu: ",mu)
#    print("loss_optimum", loss_optimum)
    mo_x=np.reshape(xp[x0_index],(-1,n_params))
    #print("mo_x:", mo_x)
    x_to_predict=np.array(x_to_predict)
    mo_x=np.array(mo_x)
    X=np.concatenate((mo_x,x_to_predict))
    Mu, Kappa = gaussian_process.predict(X,return_cov=True)
    Kappa=np.array(Kappa)
    mu0=Mu[0]
    mu=Mu[1:]
    sigma=np.sqrt(np.diag(Kappa[1:,1:]))
    new_sigma=(sigma**2+Kappa[0,0]-2*Kappa[0,1:])
    new_sigma=np.sqrt(new_sigma)
    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - mu0) / new_sigma
        # print(mu,mu0,mu-mu0)
        Mexpected_improvement = scaling_factor * (mu - mu0) * norm.cdf(Z) + new_sigma * norm.pdf(Z)
        Mexpected_improvement[new_sigma == 0.0] = 0.0

    return -1 * Mexpected_improvement



def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, bounds, x0=None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7,greater_is_better = False,
                          acquisition_func = expected_improvement):
    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    n_params = bounds.shape[0]
    if x0 is None:
        for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
            x_list.append(params)
            y_list.append(sample_loss(params))
    else:
        for params in x0:
            x_list.append(params)
            y_list.append(sample_loss(params))
    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    if acquisition_func == Mexpected_improvement or acquisition_func == Mprobability_improvement:

        for n in range(n_iters):
            model.fit(xp, yp)
            if random_search:
                x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
                ei = -1 * acquisition_func(x_random, model, yp,xp, greater_is_better=greater_is_better, n_params=n_params)
                next_sample = x_random[np.argmax(ei), :]
            else:
                next_sample = sample_next_hyperparameter(acquisition_func, model, yp, greater_is_better=greater_is_better, bounds=bounds, n_restarts=100)

            if np.any(np.abs(next_sample - xp) <= epsilon):
                next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (1,bounds.shape[0]))
                next_sample = next_sample[0]
            cv_score = sample_loss(next_sample)

            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)

        return xp,yp
    else:

        for n in range(n_iters):

            model.fit(xp, yp)

            # Sample next hyperparameter
            # x_random里面的size也需要改,第二个if里面的next_sample里面的size也需要改
            if random_search:
                x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))  #random_search是参数，赋值为100000
                #ei = -1 * probability_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
                ei = -1 * acquisition_func(x_random, model, yp, greater_is_better=greater_is_better, n_params=n_params)
                #ei = -1 * Mexpected_improvement(x_random, model, yp,xp,greater_is_better=True, n_params=n_params)
                next_sample = x_random[np.argmax(ei), :]  #np.argmax(ei)是一个数
                #print("x_random",x_random)
                #print("n:",next_sample)
                #print("x_random",x_random.shape)
                #print("np.argmax(ei)",np.argmax(ei))
            else:
                #next_sample = sample_next_hyperparameter(probability_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)
                next_sample = sample_next_hyperparameter(acquisition_func, model, yp, greater_is_better=greater_is_better, bounds=bounds, n_restarts=100)
                #next_sample = sample_next_hyperparameter(Mexpected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=100)

            #n_restarts=50
            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            if np.any(np.abs(next_sample - xp) <= epsilon):
                next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (1,bounds.shape[0]))
                next_sample = next_sample[0]
            ######print("next",next_sample)出现的结果是ValueError: array must not contain infs or NaNs（感觉是127行的问题），160行原来是bounds.shape[0]
            # Sample loss for new set of parameters
            cv_score = sample_loss(next_sample)

            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)

    return xp, yp