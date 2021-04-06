"""
Script used to run most of the results used in the paper.
Compring different PCEs, uncertainty measures due to factor fixing.
"""

import pandas as pd
import numpy as np
import pyapprox as pya
from pyapprox.approximate import approximate
from pyapprox.indexing import compute_hyperbolic_indices
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
from pyapprox.multivariate_polynomials import \
        define_poly_options_from_variable_transformation, AffineRandomVariableTransformation
from pyapprox.variables import IndependentMultivariateRandomVariable

##==============================##============================##
def fun_prod(x1, x2, x3, x4, x5, x6):
    """
    x: array-like
    """
    y = x1*x2 + x3 + x4 * x5 * x6
    return y

from scipy.stats import uniform, norm
univariate_variables = [uniform(0, 1)] * 6
# the dist. of the first and last vars are not correct but will be overwriten using the product
re_univariable = [uniform(0, 1.5), uniform(0, 1), uniform(0.2, 1)] 
re_variable = IndependentMultivariateRandomVariable(re_univariable)
re_var_trans = AffineRandomVariableTransformation(re_variable)

variable = IndependentMultivariateRandomVariable(univariate_variables)
var_trans = AffineRandomVariableTransformation(variable)

# generate samples and values for training and comparison
samples = pya.generate_independent_random_samples(variable, 10000)
values = np.array([[fun_prod(*ii)] for ii in samples.T])
index_product = [[0, 1], [3, 4, 5]]
samples_adjust = np.zeros((3, samples.shape[1]))
samples_adjust[0, :] = np.prod(samples[index_product[0], :], axis=0)
samples_adjust[1, :] = samples[2, :]
samples_adjust[2, :] = np.prod(samples[index_product[1], :], axis=0)


# the following use the product of uniforms to define basis
from pyapprox.variables import get_distribution_info
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.utilities import total_degree_space_dimension

def identity_fun(x):
    return x

degree = 3
poly = PolynomialChaosExpansion()
basis_opts = dict()
identity_map_indices = []
cnt = 0
for ii in range(re_variable.nunique_vars):
    rv = re_variable.unique_variables[ii]
    name, scales, shapes = get_distribution_info(rv)
    if ii not in [0, 2]:
        opts = {'rv_type': name, 'shapes': shapes,
                'var_nums': re_variable.unique_variable_indices[ii]}
        basis_opts['basis%d' % ii] = opts
        continue

    #identity_map_indices += re_variable.unique_variable_indices[ii] # wrong
    identity_map_indices += list(re_variable.unique_variable_indices[ii]) # right
    
    quad_rules = []    
    inds = index_product[cnt]
    nquad_samples_1d = 50

    for jj in inds:
        a, b = variable.all_variables()[jj].interval(1)
        x, w = gauss_jacobi_pts_wts_1D(nquad_samples_1d, 0, 0)
        x = (x+1)/2 # map to [0, 1]
        x = (b-a)*x+a # map to [a,b]
        quad_rules.append((x, w))
    funs = [identity_fun]*len(inds)
    basis_opts['basis%d' % ii] = {'poly_type': 'product_indpnt_vars',
                                    'var_nums': [ii], 'funs': funs,
                                    'quad_rules': quad_rules}
    cnt += 1
        
poly_opts = {'var_trans': re_var_trans}
poly_opts['poly_types'] = basis_opts
#var_trans.set_identity_maps(identity_map_indices) #wrong
re_var_trans.set_identity_maps(identity_map_indices) #right

indices = compute_hyperbolic_indices(re_variable.num_vars(), degree)
nterms = total_degree_space_dimension(samples_adjust.shape[0], degree)
options = {'basis_type': 'fixed', 'variable': re_variable,
            'poly_opts': poly_opts,
            'options': {'linear_solver_options': dict(),
                        'indices': indices, 'solver_type': 'lstsq'}}
                        
approx_res = approximate(samples_adjust[:, 0:(2 * nterms)], values[0:(2 * nterms)], 'polynomial_chaos', options).approx
y_hat = approx_res(samples_adjust[:, 2 * nterms:])
print((y_hat - values[2 * nterms:]).mean())
print(f'Mean of samples: {values.mean()}')
print(f'Mean of pce: {approx_res.mean()}')
