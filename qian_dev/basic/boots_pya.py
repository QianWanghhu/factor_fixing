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

def pce_fun(variable, samples, values, ntrain_samples, degree=2, boot_ind=None):
    """
    Help function for only fitting and returning PCE object.
    """
    poly = pya.get_polynomial_from_variable(variable)
    poly.set_indices(pya.compute_hyperbolic_indices(
        variable.num_vars(),degree))

    # nsamples = samples.shape[1]
    if ntrain_samples == None:
        ntrain_samples = poly.get_indices().shape[1]*3
    
    if boot_ind is None:
        train_samples = samples[:,:ntrain_samples]
        train_values = values[:ntrain_samples]
    else:
        train_samples = samples[:,boot_ind]
        train_values = values[boot_ind]
    coef = least_squares(poly.basis_matrix,train_samples,train_values)
    poly.set_coefficients(coef)

    validation_samples = samples[:,ntrain_samples:]
    validation_values = values[ntrain_samples:]
    approx_values = poly(validation_samples)
    error = np.linalg.norm(approx_values-validation_values)/np.linalg.norm(values)
    return poly, error


def fun(variable, samples, values, degree=2, nboot=500, ntrain_samples=None):
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

    for _ in range(nboot):
        if nboot == 1:
            I = np.arange(ntrain_samples)
        else:
            I = np.random.randint(0, ntrain_samples, ntrain_samples)
        index_cover.append(np.unique(I).size / ntrain_samples)

        train_samples = samples[:,I]
        train_values = values[I]
        x_cv = train_samples
        y_cv = train_values

        # start_validation = ntrain_samples
        
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


        # validation_samples = samples[:,start_validation:]
        # validation_values = values[start_validation:]
        
        # approx_values = poly(validation_samples)        
        # error = np.linalg.norm(approx_values-validation_values)/np.linalg.norm(values)

        main_effect, total_effect = pya.get_main_and_total_effect_indices_from_pce(poly.get_coefficients(),poly.get_indices())

        # errors_bt.append(error)
        # condition_numbers.append(cond)
        total_indices.append(total_effect[:])
        main_sensitivity.append(main_effect[:])
        if nboot == 1:
            return poly, errors_cv, main_sensitivity, total_indices
        else:
            return errors_cv, main_sensitivity, total_indices, index_cover