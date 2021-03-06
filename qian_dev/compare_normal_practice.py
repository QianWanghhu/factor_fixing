import numpy as np
import pandas as pd
import json
import os
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
import time

from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from SAFEpython.VBSA import vbsa_indices

# Calculate the corresponding number of bootstrap with use of group_fix
ntrain_samples = 552
nboot = 500
samples_files = ['2000_2014_ave_annual.csv', 'samples_adjust.csv']
param_files = ['parameter-implement.csv', 'parameter-adjust.csv']


fpath_save = '../output/paper0915/'
filename = f'{fpath_save}{samples_files[0]}'
data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
samples = data[:, :-1].T
values = data[:, -1:]

fpath_input = '../data/'
filename = f'{fpath_input}{param_files[0]}'
dummy = False; product_uniform = False
variable = variables_prep(filename, dummy=dummy, product_uniform=product_uniform)
poly, error, total_index, _ = fun(variable, samples, values, 
                    ntrain_samples=ntrain_samples, degree=2, nboot=1)

# Read problem.txt file and add a dummy parameter to it
input_path = '../data/'
problem_adjust = read_param_file(f'{input_path}problem_adjust.txt', delimiter=',')
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)

fpath_save = '../output/paper0915/'
# sa_df.to_csv(fpath_save+'ST_Beta_dummy.csv')

###=============CALCULATE THE ERROR METRCIS FROM FACTOR FIXING============###
# Read input sensitivity results
sa_df = pd.read_csv(fpath_save + 'ST_552.csv')
sa_df.reset_index(inplace=True)
ind_fix = list(sa_df[sa_df.ST<0.1].index) #sa_df.iloc[-1].ST
partial_order = {0: ind_fix}
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
x_sample = np.loadtxt('../output/paper0915/metric_samples.txt')
# Check the number of parameters in variables to decide whehter adjust the x_sample and x_fix
if (variable.num_vars()) == 11:
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
error_df.to_csv(fpath_save+'fix_threshold_10percent.csv')
print("--- %s seconds ---" % (time.time() - start_time))
