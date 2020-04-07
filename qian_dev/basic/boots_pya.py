#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pyapprox as pya
from scipy.stats import uniform

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

def fun(fpath, variable, samples, values, ntrials=500, ntrain_samples=None):
    """
    Function to train PCE and conduct boostrap.
    Parameters:
    ===========
    fpath: str, file path for inputs and outputs
    ntrials: int, the number of bootstraps
    variables: pyapprox.variables.IndependentMultivariateRandomVariable
    samples: ndarray of inputs, 
             shape of (P, N) where P is the number of inputs and N is the size of samples
    values: ndarray of outputs
    ntrain_samples: int, the size of training set, 
                    default is None which then will adopt 3 times the number of poly terms

    Returns:
    ========
    None. 
    But with a .npz file containing errors, condition_numbers, and total effects written to the fpath.
    
    """
    # fpath = '../../../Research/pce_fixing/pyfile/John/'
    filename = f'{fpath}parameter-ranges.csv'
    ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
    univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(ranges.shape[0]//2)]
    variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

    filename = f'{fpath}test_qian.csv'
    data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
    samples = data[:,:22].T
    values = data[:,22:]
    values = values[:,:1]# focus on first qoi

    degree=2
    poly = pya.get_polynomial_from_variable(variable)
    poly.set_indices(pya.compute_hyperbolic_indices(
        variable.num_vars(),degree))

    nsamples = samples.shape[1]
    if ntrain_samples == None:
        ntrain_samples = poly.get_indices().shape[1]*3

    errors,condition_numbers,sensitivity_indices = [],[],[]
    for ii in range(ntrials):
        I = np.random.permutation(np.arange(nsamples))
        train_samples = samples[:,I[:ntrain_samples]]
        train_values = values[I[:ntrain_samples]]
        coef = least_squares(poly.basis_matrix,train_samples,train_values)

        # compute condition number of training set
        cond = np.linalg.cond(poly.basis_matrix(train_samples))

        validation_samples = samples[:,I[ntrain_samples:]]
        validation_values = values[I[ntrain_samples:]]

        poly.set_coefficients(coef)
        approx_values = poly(validation_samples)
        
        error = np.linalg.norm(approx_values-validation_values)/np.linalg.norm(values)
        # print('Error',error)

        _, total_effect = pya.get_main_and_total_effect_indices_from_pce(poly.get_coefficients(),poly.get_indices())
        
        errors.append(error)
        condition_numbers.append(cond)
        sensitivity_indices.append(total_effect)
    return errors, condition_numbers, sensitivity_indices
    filename = 'water-quality-cv-data.npz'
    np.savez(f'{fpath}{filename}',errors=errors,condition_numbers=condition_numbers,sensitivity_indices=sensitivity_indices)