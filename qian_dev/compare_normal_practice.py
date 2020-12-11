import numpy as np
import pandas as pd
import json
import os
import pyapprox as pya
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
import time

from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal

def pce_dummy(filename, samples, dummy=False, numerical_sa = False, problem = None):
# Used for PCE fitting
    if (dummy & numerical_sa) | (not dummy):
        variable = variables_prep(filename, product_uniform=False, dummy=False)
    elif dummy & (not numerical_sa): 
        samples = np.append(samples, [np.random.random_sample(samples.shape[1])], axis=0)
        variable = variables_prep(filename, product_uniform=False, dummy=dummy)

    poly, _, total_effects, _ = fun(variable, samples, 
                                        values, degree=2, nboot=nboot, 
                                            ntrain_samples=ntrain_samples)
    if dummy:
        sa_df = pd.DataFrame(index = [*param_all, 'dummy'])
        if numerical_sa:
            total_effects = sa_salib(problem, poly)
            sa_df.loc[:, 'ST'] = total_effects
        else:
            sa_df.loc[:, 'ST'] = np.mean(total_effects, axis=0)                                                  
    else:
        sa_df = pd.DataFrame(index = param_all)
        sa_df.loc[:, 'ST'] = np.mean(total_effects, axis=0)   
    return variable, sa_df

def sa_salib(problem, poly):
    x_saltelli = saltelli.sample(problem, 2000)
    y_saltelli = poly(x_saltelli.T).flatten()
    sa = sobol.analyze(problem, y_saltelli, dummy=True)
    total_effects = sa['ST']
    return total_effects

def wrap_evaluate(poly, x_saltelli):
    y = poly(x_saltelli.T[0:problem['num_vars'] - 1, :])
    return y

# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
fpath_save = '../output/paper0915/'
filename = f'{fpath_save}2000_2014_ave_annual.csv'
data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
samples = data[:, :22].T
values = data[:, 22:]
fpath_input = '../data/'
filename = f'{fpath_input}parameter-implement.csv'
param_all = pd.read_csv(filename).loc[:, 'Veneer_name'].values
ntrain_samples = 552
nboot = 1


# Calculate the corresponding number of bootstrap with use of group_fix
input_path = '../data/'
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')

variable, sa_df = pce_dummy(filename,samples, True, True, problem)

sa_df.to_csv(fpath_save+'ST_552_dummy.csv')

###=============CALCULATE THE ERROR METRCIS FROM FACTOR FIXING============###
# Read input sensitivity results
sa_df = pd.read_csv(fpath_save + 'ST_552_dummy.csv')
sa_df.reset_index(inplace=True)
ind_fix = list(sa_df[sa_df.ST<sa_df.iloc[22].ST].index)
partial_order = {0: ind_fix}

x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
x_sample = np.loadtxt('../output/paper0915/metric_samples.txt')

conf_uncond, error_dict, pool_res, y_uncond = {}, {}, {}, {}
rand = np.random.randint(0, x_sample.shape[1], size=(1000, x_sample.shape[1]))
conf_level = [0.025, 0.975]
poly, error, _, _ = fun(variable, samples, values, 
                        ntrain_samples=ntrain_samples, degree=2, nboot=1)

start_time = time.time()
# add the calculation of y_uncond
y_uncond = poly(x_sample).flatten()
conf_uncond = uncond_cal(rand, y_uncond, conf_level)
conf_uncond['cv'] = np.sqrt(poly.variance())[0] / poly.mean()[0]
error_dict, pool_res = group_fix(partial_order, poly, x_sample, y_uncond, x_fix, rand, {}, file_exist=False)
# End for

error_df = pd.DataFrame.from_dict(error_dict)
error_df.to_csv(fpath_save+'fix_dummy.csv')
print("--- %s seconds ---" % (time.time() - start_time))
