#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pyapprox as pya
from scipy.stats import uniform
from sklearn.model_selection import KFold

def least_squares(basis_matrix_function,samples,values):
    """
    Help function of least square regression used for PCE.
    Parameters:
    ===========
    basis_matrix_function: 
    samples: ndarray of inputs, 
             shape of (P, N) where P is the number of inputs and N is the size of samples
    values: ndarray of outputs
    Returns:
    ========
    coef: ndarray, solution of the regression.
    """
    basis_matrix = basis_matrix_function(samples)
    coef = np.linalg.lstsq(basis_matrix,values,rcond=None)[0]
    return coef

def fun(variable, samples, values, degree=2, nboot=500, I=None, ntrain_samples=None):
    """
    Function to train PCE and conduct boostrap.
    Parameters:
    ===========
    fpath: str, file path for inputs and outputs
    nboot: int, the number of bootstraps
    variables: pyapprox.variables.IndependentMultivariateRandomVariable
    samples: ndarray of inputs, 
             shape of (P, N) where P is the number of inputs and N is the size of samples
    values: ndarray of outputs
    ntrain_samples: int, the size of training set, 
                    default is None which then will adopt 3 times the number of poly terms

    Returns:
    ========
    errors_cv: list, Errors computed from the cross validation.
    errors_bt: list, errors calculated from bootstrap
    sensitivity_indices: list, total effects for all the bootstraps
    """
    poly = pya.get_polynomial_from_variable(variable)
    poly.set_indices(pya.compute_hyperbolic_indices(
        variable.num_vars(),degree))

    # nsamples = samples.shape[1]
    if ntrain_samples == None:
        ntrain_samples = poly.get_indices().shape[1]*3

    errors_cv, main_sensitivity, total_indices = [], [], []
    index_cover = []
    # Cross-validation
    kf = KFold(n_splits=10)
    if nboot > 1:
        for ii in range(nboot):
        
            index_cover.append(np.unique(I[ii]).size / ntrain_samples)

            train_samples = samples[:, I[ii]]
            train_values = values[I[ii]]
            x_cv = train_samples
            y_cv = train_values        
            for train_index, test_index in kf.split(x_cv.T):
                x_train, x_test = x_cv[:,train_index], x_cv[:,test_index]
                y_train, y_test = y_cv[train_index], y_cv[test_index]
                coef = least_squares(poly.basis_matrix,x_train,y_train)
                poly.set_coefficients(coef)
                approx_values = poly(x_test)
                cv = np.linalg.norm(approx_values-y_test)/np.linalg.norm(train_values)
                errors_cv.append(cv)


            coef = least_squares(poly.basis_matrix,train_samples,train_values)
            poly.set_coefficients(coef)
            main_effect, total_effect = pya.get_main_and_total_effect_indices_from_pce(poly.get_coefficients(),poly.get_indices())
            total_indices.append(total_effect[:])
            main_sensitivity.append(main_effect[:])
        return errors_cv, main_sensitivity, total_indices, index_cover
    else:
        I = np.arange(ntrain_samples)
        train_samples = samples[:, I]
        train_values = values[I]
        coef = least_squares(poly.basis_matrix,train_samples,train_values)
        poly.set_coefficients(coef)
        return poly

    
def fun_new(variable, samples, values, nboot=10):

    poly = pya.get_polynomial_from_variable(variable)

    # Find best PCE basis
    options = {'basis_type': 'expanding_basis', 'variable': variable,
               'verbosity': 0, 'options': {'max_num_terms': 66,
               'linear_solver_options': solver_options}}
    approx_res = approximate(samples, values, 'polynomial_chaos', options)

    # Compute PCE on each fold using best PCE basis and least squares
    nfolds = min(nboot, train_samples.shape[1])
    linear_solver_options = [
        {'alpha':approx_res.reg_params[ii]}
        for ii in range(len(approx_res.reg_params))]
    indices = [approx_res.approx.indices[:, np.where(np.absolute(c)>0)[0]]
               for c in approx_res.approx.coefficients.T]
    options = {'basis_type': 'fixed', 'variable': variable,
               'options': {'linear_solver_options': linear_solver_options,
                           'indices': indices}}
    
    from pyapprox.approximate import cross_validate_approximation
    approx_list, residues_list, cv_score = cross_validate_approximation(
        train_samples, train_vals_normalized, options, nfolds,
        'polynomial_chaos', random_folds='sklearn')
    pce_cv_total_effects = []
    pce_cv_main_effects = []
    for ii in range(nfolds):
        approx_list[ii].coefficients *= vals_std
        pce_sa_res_ii = pya.analyze_sensitivity_polynomial_chaos(
            approx_list[ii])
        pce_cv_main_effects.append(pce_sa_res_ii.main_effects)
        pce_cv_total_effects.append(pce_sa_res_ii.total_effects)
    return cv_score, main_sensitivity, total_indices, index_cover
