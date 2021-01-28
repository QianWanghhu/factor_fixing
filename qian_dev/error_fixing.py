import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import pyapprox as pya
import SALib.sample.latin as latin
from SALib.util import read_param_file
import time

from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from basic.read_data import file_settings, read_specify

# import variables and samples for PCE
input_path = file_settings()[1]
variable, _ = read_specify('parameter', 'reduced', product_uniform=True, num_vars=11)

output_path = file_settings()[0]
samples, values = read_specify('model', 'reduced', product_uniform=False, num_vars=11)
len_params = variable.num_vars()

# load partial order results
partial_order = read_specify('rank', 'reduced', product_uniform=True, num_vars=11)
key_use = [f'nsample_{ii}' for ii in np.arange(104, 182, 13)]
partial_order = dict((key, value) for key, value in partial_order.items() if key in key_use)

# import index_prodcut which is a array defining the correlations between parameters
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
# x_fix = np.ones(shape=(problem['num_vars'], 1))

x_sample = latin.sample(problem, 1000, seed=88)
x_sample = x_sample.T
# np.savetxt('../output/paper0915/metric_samples.txt',x_sample)
# if reduce parameters, change samples
if (variable.num_vars()) == 11:
    x_sample = adjust_sampling(x_sample, index_product, x_fix)
    x_fix_adjust = adjust_sampling(x_fix, index_product, x_fix)

# Calculate the corresponding number of bootstrap with use pf group_fix
conf_uncond, error_dict, pool_res, y_uncond = {}, {}, {}, {}
rand = np.random.randint(0, x_sample.shape[1], size=(500, x_sample.shape[1]))
conf_level = [0.025, 0.975]

start_time = time.time()
num_pce = 500
for key, value in partial_order.items():
    pce_list = []; cv_temp = np.zeros(num_pce)
    y_temp = np.zeros(shape=(num_pce, x_sample.shape[1]))
    _, sample_size = key.split('_')[0], int(key.split('_')[1])
    print(f'------------------Training samples: {sample_size}------------------------')
    rand_pce = np.random.randint(0, sample_size, size=(num_pce, sample_size))
    for i in range(rand_pce.shape[0]):
        poly, _, _, _ = fun(variable, samples[:, rand_pce[i]], values[rand_pce[i]], 
            degree=2, nboot=1, ntrain_samples=sample_size)
    # add the calculation of y_uncond
        pce_list.append(poly)
        y_temp[i, :] = poly(x_sample).flatten()
        cv_temp[i] = np.sqrt(poly.variance())[0] / poly.mean()[0]
    y_uncond[key] = y_temp
    conf_uncond[key] = uncond_cal(y_uncond[key], conf_level, rand)
    conf_uncond[key]['cv'] = cv_temp[i].mean()
    conf_uncond[key]['cv_low'], conf_uncond[key]['cv_up'] = \
        np.quantile(cv_temp, [0.025, 0.975])
    error_dict[key], pool_res = group_fix(value, pce_list, x_sample, y_uncond[key], x_fix_adjust, rand, {}, file_exist=True)
# End for

print("--- %s seconds ---" % (time.time() - start_time))

# # separate confidence intervals into separate dicts and write results
save_path = f'{output_path}error_measures/0112_cal/'
if not os.path.exists(save_path): os.mkdir(save_path)
# # convert the result into dataframe
key_outer = list(error_dict.keys())
f_names = list(error_dict[key_outer[0]].keys())
for ele in f_names:
    dict_measure = {key: error_dict[key][ele] for key in key_outer}
    df = to_df(partial_order, dict_measure)
    df.to_csv(f'{save_path}/{ele}.csv')

with open(f'{save_path}y_uncond_stats.json', 'w') as fp:
    json.dump(conf_uncond, fp, indent=2)

