import numpy as np
import pandas as pd
import json
import os
from SALib.util import read_param_file
import time

from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from basic.read_data import file_settings, read_specify

# Calculate the corresponding number of bootstrap with use of group_fix
ntrain_samples = 552
nboot = 500
input_path = file_settings()[1]
variable, _ = read_specify('parameter', 'full', product_uniform=False, num_vars=22)

output_path = file_settings()[0]
samples, values = read_specify('model', 'full', product_uniform=False, num_vars=22)

# import index_prodcut which is a array defining the correlations between parameters
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))

poly = fun(variable, samples, values, 
                    ntrain_samples=ntrain_samples, degree=2, nboot=1)

###=============CALCULATE THE ERROR METRCIS FROM FACTOR FIXING============###
# Read input sensitivity results
sa_df = pd.read_csv(fpath_save + 'ST_552.csv')
sa_df.reset_index(inplace=True)
ind_fix = list(sa_df[sa_df.ST<0.1].index) #sa_df.iloc[-1].ST
partial_order = {0: ind_fix}
x_sample = np.loadtxt('../output/paper0915/metric_samples.txt')
# Check the number of parameters in variables to decide whehter adjust the x_sample and x_fix

conf_uncond, error_dict, pool_res, y_uncond = {}, {}, {}, {}
rand = np.random.randint(0, x_sample.shape[1], size=(1000, x_sample.shape[1]))
ci_bounds = [0.025, 0.975]
start_time = time.time()
# add the calculation of y_uncond
y_uncond = poly(x_sample).flatten()
conf_uncond = uncond_cal(rand, y_uncond, ci_bounds)
conf_uncond['cv'] = np.sqrt(poly.variance())[0] / poly.mean()[0]
error_dict, pool_res = group_fix(partial_order, poly, x_sample, y_uncond, x_fix, rand, {}, file_exist=False)
# End for

error_df = pd.DataFrame.from_dict(error_dict)
error_df.to_csv(fpath_save+'fix_threshold_10percent.csv')
print("--- %s seconds ---" % (time.time() - start_time))
