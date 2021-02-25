# import packages
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import pyapprox as pya
import SALib.sample.latin as latin
from SALib.util import read_param_file
import time

from basic.boots_pya import fun
from basic.utils import adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from basic.read_data import file_settings, read_specify

# import variables and samples for PCE
# input_path = file_settings()[1]
# variable, _ = read_specify('parameter', 'reduced', product_uniform=True, num_vars=11)

# output_path = file_settings()[0]
# samples, values = read_specify('model', 'reduced', product_uniform=True, num_vars=11)

# # load partial order results
# partial_order = read_specify('rank', 'reduced', product_uniform=True, num_vars=11)
# key_use = [f'nsample_{ii}' for ii in [156]]
# partial_order = dict((key, value) for key, value in partial_order.items() if key in key_use)

# # import index_prodcut which is a array defining the correlations between parameters
# index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
# filename = f'{input_path}problem.txt'
# problem = read_param_file(filename, delimiter=',')
# x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
# if (variable.num_vars()) == 11:
#         x_fix_adjust = adjust_sampling(x_fix, index_product, x_fix)
# sample_range = [1000, 10000, 500]

def fix_increase_sample(input_path, variable, output_path, samples, values,
    partial_order, index_product, problem, x_fix, x_fix_adjust, sample_range):
# if reduce parameters, change samples
    key = list(partial_order.keys())[0]
    _, sample_size = key.split('_')
    value = partial_order[key]
    poly = fun(variable, samples, values, 
                degree=2, nboot=1, ntrain_samples=int(sample_size))
    start_time = time.time()
    conf_uncond, error_dict, pool_res, y_uncond = {'median': [], 'mean': []}, {}, {}, {}
    ci_bounds = [0.025, 0.975]
    nstart, nstop, nstep = sample_range
    for n in range(nstart, nstop + 1, nstep):
        print(n)
        x_sample = latin.sample(problem, n, seed=88)
        x_sample = x_sample.T
        # if reduce parameters, change samples
        if (variable.num_vars()) == 11:
            x_sample = adjust_sampling(x_sample, index_product, x_fix)
        
        # Calculate the corresponding number of bootstrap with use pf group_fix
        rand = np.random.randint(0, x_sample.shape[1], size=(500, x_sample.shape[1]))
        # add the calculation of y_uncond
        y_uncond_temp = poly(x_sample).flatten()
        conf_uncond[str(n)] = uncond_cal(y_uncond_temp, ci_bounds, rand)
        conf_uncond[str(n)]['median'].append(np.quantile(y_uncond_temp[rand], ci_bounds, axis=1).mean(axis=0).mean())
        conf_uncond[str(n)]['mean'].append(poly.mean()[0])
        error_dict[str(n)], pool_res = group_fix(value, poly, x_sample, y_uncond[str(n)], 
                                        x_fix_adjust, rand, {}, file_exist=True)
        # End for
    print("--- %s seconds ---" % (time.time() - start_time))

    # separate confidence intervals into separate dicts and write results
    save_path = f'{output_path}error_measures/'
    if not os.path.exists(save_path): os.mkdir(save_path)
    # convert the result into dataframe
    key_outer = list(error_dict.keys())
    f_names = list(error_dict[key_outer[0]].keys())
    for ele in f_names:
        dict_measure = {key: error_dict[key][ele] for key in key_outer}
        df = pd.DataFrame.from_dict(dict_measure)
        df.to_csv(f'{save_path}/{ele}_adaptive.csv')

    df_stats = pd.DataFrame(data=conf_uncond, index=np.arange(nstart, nstop + 1, nstep))
    df_stats.to_csv(f'{save_path}/stats_uncond_adaptive.csv')
