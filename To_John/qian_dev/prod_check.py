"""
Script used to run most of the results used in the paper.
Compring different PCEs, uncertainty measures due to factor fixing.
"""
import pandas as pd
import numpy as np
import pyapprox as pya

from pyapprox.multivariate_polynomials import AffineRandomVariableTransformation
from pyapprox.variables import IndependentMultivariateRandomVariable

##==============================##============================##
##===============RUN test with a analytic function=============##
def fun_prod(x1, x2, x3, x4, x5, x6, x7):
    """
    x: array-like
    """
    y = x1*x2 + x3 + x4 * x5 * x6 - np.log(x7)
    return y

def samples_produce(fun_prod, data_path, out_path, nsample=10000):
    from scipy.stats import uniform
    univariate_variables = [uniform(0, 1)] * 7
    # the dist. of the first and last vars are not correct but will be overwriten using the product
    re_univariable = [uniform(0, 1.5), uniform(0, 1), uniform(0.2, 1), uniform(0, 1)] 
    re_variable = IndependentMultivariateRandomVariable(re_univariable)
    re_var_trans = AffineRandomVariableTransformation(re_variable)

    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)

    # generate samples and values for training and comparison
    samples = pya.generate_independent_random_samples(variable, nsample)
    values = np.array([[fun_prod(*ii)] for ii in (samples.T)])
    index_product = [[0, 1], [3, 4, 5]]
    np.save(f'{file_settings()[1]}index_product.npy', index_product)
    samples_adjust = np.zeros((4, samples.shape[1]))
    samples_adjust[0, :] = np.prod(samples[index_product[0], :], axis=0)
    samples_adjust[1, :] = samples[2, :]
    samples_adjust[2, :] = np.prod(samples[index_product[1], :], axis=0)
    samples_adjust[3, :] = samples[-1, :]

    samples_df = pd.DataFrame(data=np.append(samples, values.T, axis=0).T, 
        columns=[*([f'x{i}' for i in range(7)]), 'y'], index=np.arange(nsample))
    samples_df.to_csv(f'{data_path}/2000_2014_ave_annual.csv')

    samples_adjust_df = pd.DataFrame(data=np.append(samples_adjust, values.T, axis=0).T, 
        columns=[*([f'x{i}' for i in range(4)]), 'y'], index=np.arange(nsample))
    samples_adjust_df.to_csv(f'{out_path}/samples_adjust.csv')


# the following use the product of uniforms to define basis
"""
Script used to conduct adaptive PCE.
"""
import os
from basic.read_data import file_settings

# apply_pya to produce the sensitivities of parameters for different PCEs
from apply_pya import run_pya
out_path, data_path = file_settings()[0:2]
num_pce = 10; seed=222
if not os.path.exists(f'{out_path}/samples_adjust.csv'): 
    samples_produce(fun_prod, data_path, out_path, nsample=10000)

print('--------PCE-E with increasing samples--------')
run_pya(out_path, num_pce, seed, product_uniform='exact')

# PCE with uniform distributions
print('--------PCE-U with increasing samples--------')
run_pya(out_path, num_pce, seed, product_uniform='uniform')

# PCE with 7 parameters
print('----------------PCE-7----------------')
run_pya(out_path, num_pce, seed, product_uniform=False)
