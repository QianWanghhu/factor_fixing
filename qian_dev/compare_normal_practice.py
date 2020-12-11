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

def pce_dummy(filename, samples, values, param_names, dummy=False, product_uniform=False):
# Used for PCE fitting
    if dummy: samples = np.append(samples, [np.random.random_sample(samples.shape[1])], axis=0)
    
    variable = variables_prep(filename, dummy=dummy, product_uniform=product_uniform)
    poly, _, total_effects, _ = fun(variable, samples, 
                                        values, degree=2, nboot=nboot, 
                                            ntrain_samples=ntrain_samples)
    if dummy:
        sa_df = pd.DataFrame(index = [*param_names, 'dummy'])                                        
    else:
        sa_df = pd.DataFrame(index = param_names)
    sa_df.loc[:, 'ST'] = np.mean(total_effects, axis=0)   
    return variable, sa_df

# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
def samples_22(samples_file, param_file, ntrain_samples, dummy, product_uniform):
    fpath_save = '../output/paper0915/'
    filename = f'{fpath_save}{samples_file}'
    data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
    samples = data[:, :-1].T
    values = data[:, -1:]
    fpath_input = '../data/'
    filename = f'{fpath_input}{param_file}'
    param_all = pd.read_csv(filename).loc[:, 'Veneer_name'].values
    variable, sa_df = pce_dummy(filename, samples, values, param_all, dummy, product_uniform)
    # variablW and PCE for error analysis
    variable, _ = pce_dummy(filename, samples, values, param_all, False, product_uniform)
    samples = data[:, :-1].T
    poly, error, _, _ = fun(variable, samples, values, 
                    ntrain_samples=ntrain_samples, degree=2, nboot=1)
    return variable, sa_df, poly, error

# Calculate the corresponding number of bootstrap with use of group_fix
input_path = '../data/'
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')

ntrain_samples = 156
nboot = 500
samples_files = ['2000_2014_ave_annual.csv', 'samples_adjust.csv']
param_files = ['parameter-implement.csv', 'parameter-adjust.csv']

# Provide settings about whether to use dummy parameter and what distribution of parameters
dummy = False; product_uniform = True
variable, sa_df, poly, error = samples_22(samples_files[1], param_files[1], ntrain_samples, 
    dummy, product_uniform)

fpath_save = '../output/paper0915/'
# sa_df.to_csv(fpath_save+'ST_Beta_dummy.csv')

###=============CALCULATE THE ERROR METRCIS FROM FACTOR FIXING============###
# Read input sensitivity results
sa_df = pd.read_csv(fpath_save + 'sa_pce_beta.csv')
sa_df.reset_index(inplace=True)
ind_fix = list(sa_df[sa_df.ST<0.01].index) #sa_df.iloc[-1].ST
partial_order = {0: ind_fix}
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
x_sample = np.loadtxt('../output/paper0915/metric_samples.txt')
# Check the number of parameters in variables to decide whehter adjust the x_sample and x_fix
if (variable.num_vars()) == 11:
    index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
    x_fix_adjust = adjust_sampling(x_fix, index_product, x_fix)
    x_sample = adjust_sampling(x_sample, index_product, x_fix)

conf_uncond, error_dict, pool_res, y_uncond = {}, {}, {}, {}
rand = np.random.randint(0, x_sample.shape[1], size=(1000, x_sample.shape[1]))
conf_level = [0.025, 0.975]
start_time = time.time()
# add the calculation of y_uncond
y_uncond = poly(x_sample).flatten()
conf_uncond = uncond_cal(rand, y_uncond, conf_level)
conf_uncond['cv'] = np.sqrt(poly.variance())[0] / poly.mean()[0]
error_dict, pool_res = group_fix(partial_order, poly, x_sample, y_uncond, x_fix, rand, {}, file_exist=False)
# End for

error_df = pd.DataFrame.from_dict(error_dict)
error_df.to_csv(fpath_save+'fix_beta_threshold.csv')
print("--- %s seconds ---" % (time.time() - start_time))
