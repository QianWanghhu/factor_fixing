import numpy as np
import pandas as pd
import json
import os
from SALib.util import read_param_file
import time

from basic.boots_pya import fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from basic.read_data import file_settings, read_specify

# Calculate the corresponding number of bootstrap with use of group_fix
ntrain_samples = 552
num_pce = 500
input_path = file_settings()[1]
variable, _ = read_specify('parameter', 'full', product_uniform=False, num_vars=22)

output_path = file_settings()[0]
samples, values = read_specify('model', 'full', product_uniform=False, num_vars=22)

# import index_prodcut which is a array defining the correlations between parameters
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))

###=============CALCULATE THE ERROR METRCIS FROM FACTOR FIXING============###
x_sample = np.loadtxt(f'{output_path}metric_samples.txt')

pool_res = {}
rand = np.random.randint(0, x_sample.shape[1], size=(500, x_sample.shape[1]))
cv_temp = np.zeros(num_pce)
rand_pce = np.random.randint(0, ntrain_samples, size=(num_pce, ntrain_samples))
ci_bounds = [0.025, 0.975]
# add the calculation of y_uncond
y_temp = np.zeros(shape=(num_pce, x_sample.shape[1]))
pce_list = []
for i in range(rand_pce.shape[0]):
    poly = fun(variable, samples[:, rand_pce[i]], values[rand_pce[i]], 
        degree=2, nboot=1, ntrain_samples=ntrain_samples)
        
    pce_list.append(poly)
    y_temp[i, :] = poly(x_sample).flatten()
    cv_temp[i] = np.sqrt(poly.variance())[0] / poly.mean()[0]
# add the calculation of y_uncond

y_uncond = y_temp
conf_uncond= uncond_cal(y_uncond, ci_bounds, rand)
conf_uncond['cv'] = cv_temp.mean()
conf_uncond['cv_low'], conf_uncond['cv_up'] = \
            np.quantile(cv_temp, ci_bounds)
# End for

# sensitivity with PCE-22
sa_raw = pd.read_csv(f'{output_path}sa_pce_22.csv')
thresholds = [0.01, 0.1]
for thsd in thresholds:
    ind_fix = list(sa_raw[sa_raw.ST<thsd].index) 
    partial_order = {0: ind_fix}
    error_dict, pool_res = group_fix(partial_order, pce_list, 
        x_sample, y_uncond, x_fix, rand, {}, file_exist=False)

    error_df = pd.DataFrame.from_dict(error_dict)
    error_df.to_csv(f'{output_path}fix_threshold_{thsd}.csv')
