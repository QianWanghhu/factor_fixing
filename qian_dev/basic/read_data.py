"""This script is used different data"""
import numpy as np
import pandas as pd
import json

from basic.utils import variables_prep  

def file_settings():
    model_dir = 'output/test/'
    input_dir = 'data/'
    model_ts_full = f'{input_dir}2000_2014_ave_annual.csv'
    model_ts_reduced = f'{model_dir}samples_adjust.csv'
    
    param_full = f'{input_dir}parameter.csv'
    param_reduced = f'{input_dir}parameter-adjust.csv'
    ranks_reduced_beta = f'{model_dir}partial_reduce_beta_552.json'
    ranks_reduced_uni = f'{model_dir}partial_reduce_params.json'
    return [model_dir, input_dir, model_ts_full, model_ts_reduced, param_full, \
        param_reduced, ranks_reduced_beta, ranks_reduced_uni]
# END file_settings()

def read_model_ts(filename, num_vars):
    """Read the model outputs used for building surrogates.
    Parameters:
    ===========
    filename: str, filename of the model output to read.

    Returns:
    samples: np.ndarray, of two dimension N * D 
        where N is the number of samples and D is the number of parameters
    values: np.ndarray, the Quantity of interest to simulate.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
    samples = data[:, :num_vars].T
    values = data[:, num_vars:]
    return samples, values
# END read_model_ts()

def read_parameters(filename, product_uniform):
    variable = variables_prep(filename, product_uniform=product_uniform)
    param_all = pd.read_csv(filename).loc[:, 'Veneer_name'].values
    return variable, param_all
# END read_parameters()

def read_ranks(filename):
    with open(f'{filename}', 'r') as fp:
        partial_order = json.load(fp)
    return partial_order
# END read_ranks()

def read_specify(data_type, param_type, product_uniform, num_vars=22):
    filenames = file_settings()
    assert (param_type in ['full', 'reduced']), 'param_type is not'
    if data_type == 'model':
        if param_type == 'full':
            return read_model_ts(filenames[2], num_vars)
        elif param_type == 'reduced':
            return read_model_ts(filenames[3], num_vars)
    elif data_type == 'parameter':
        if param_type == 'full':
            assert (product_uniform == False), 'product_uniform should be False when using full model.'
            assert (num_vars == 22), 'num_vars should be 22 when using full model.'
            return read_parameters(filenames[4], product_uniform)
        elif param_type == 'reduced':
            assert (num_vars == 11), 'num_vars should be 11 when using reduced model.'
            return read_parameters(filenames[5], product_uniform)
    else:
        if product_uniform == True:
            return read_ranks(filenames[6])
        else:
            return read_ranks(filenames[7])
            

