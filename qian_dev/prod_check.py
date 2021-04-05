"""
Script used to run most of the results used in the paper.
Compring different PCEs, uncertainty measures due to factor fixing.
"""

import pandas as pd
import numpy as np
from SALib.util import read_param_file

from basic.boots_pya import fun_new
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from basic.read_data import file_settings, read_specify

##==============================##============================##
##==============================##============================##
# apply_pya to produce the sensitivities of parameters for different PCEs
from apply_pya import run_pya, pce_22
outpath = file_settings()[0]
seed=222
# check whether the mean and variance from 'exact' is correct
def pmf_check(product_uniform = False):    
    variable, _ = read_specify('parameter', 'full', product_uniform, num_vars=22)
    samples, values = read_specify('model', 'full', product_uniform, num_vars=22)

    approx_list_all = {}
    mean_list = {}
    variance_list = {}
    n_strat, n_end, n_step = [552, 552, 13]
    for i in range(n_strat, n_end+1, n_step):
    # for i in n_list:
        print(i)
        if (n_end - i)  < n_step:
            i = n_end
        np.random.seed(seed)                                                    
        approx_list_all[f'nsample_{i}'] = fun_new(variable, samples[:, :i], values[:i], product_uniform, nboot=1)
    for key, pce in approx_list_all.items():
        mean_list[key], variance_list[key] = pce.mean(), pce.variance()
    pmf_stat = pd.concat([pd.DataFrame.from_dict(mean_list).T, \
        pd.DataFrame.from_dict(variance_list).T], axis=1)
    return pmf_stat
pmf_stat = pmf_check(product_uniform = False)

def fun_prod(x1, x2, x3, x4, x5, x6):
    """
    x: array-like
    """
    y = x1*x2 + x3 + x4 * x5 * x6
    return np.prod(x)

from scipy.stats import uniform
univariate_variables = [uniform(0, 1)] * 6
degree = 10
poly = PolynomialChaosExpansion()
# the distribution and ranges of univariate variables is ignored
# when var_trans.set_identity_maps([0]) is used
variable = IndependentMultivariateRandomVariable(univariate_variables)
var_trans = AffineRandomVariableTransformation(variable)
# the following means do not map samples
var_trans.set_identity_maps([0])
quad_rules = [(x, w) for x, w in zip(x_1d, w_1d)]
poly.configure({'poly_types':
                {0: {'poly_type': 'function_indpnt_vars',
                        'var_nums': [0], 'fun': fun,
                        'quad_rules': quad_rules}},
                'var_trans': var_trans})