import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import pyapprox as pya
import SALib.sample.latin as latin
from SALib.util import read_param_file

from scipy.stats import uniform, median_absolute_deviation as mad
from basic.boots_pya import least_squares, fun, pce_fun
from basic.utils import variables_prep, to_df, cvg_check
from basic.group_fix import group_fix

reduce_params = True
if reduce_params:
    file_list = ['parameter-adjust', 'samples_adjust', 'partial_reduce_beta', 'partial_reduce_params']
else:
    file_list = ['parameter-ranges', 'test_qian', 'full_params', 'partial_pya']

input_path = '../data/'
filename = f'{input_path}{file_list[0]}.csv'
variable = variables_prep(filename, product_uniform=True)

output_path = '../output/paper/'
filename = f'{output_path}{file_list[1]}.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
len_params = variable.num_vars()
samples = data[:,:len_params].T
values = data[:,len_params:]
values = values[:,:1]# focus on first qoi
len_params = samples.shape[0]
# generate samples for error metric analysis
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_sample = latin.sample(problem, 1000, seed=88)
x_sample = x_sample.T

x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
# x_fix = np.ones(shape=(problem['num_vars'], 1))

# import index_prodcut which is a array defining the correlations between parameters
index_product = np.load(f'{input_path}index_prodcut.npy', allow_pickle=True)

# if reduce parameters, adapt samples
if (variable.num_vars()) == 11:
    samples_adjust = np.copy(x_sample)
    pars_delete = []
    for ii in range(list(index_product.shape)[0]):
        index_temp = index_product[ii]
        samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
        x_fix[index_temp[0]] = np.prod(x_fix[index_temp], axis=0)
        # samples_adjust[index_temp[1:], :] = 1
        pars_delete.extend(index_temp[1:])
    samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)
    x_fix = np.delete(x_fix, pars_delete, axis=0)
    x_sample = samples_adjust

# load partial order results
with open(f'{output_path}{file_list[2]}.json', 'r') as fp:
    partial_order = json.load(fp)
# Get the unique partial sort results and the corresponding number of bootstrap used for PCE

conf_uncond = {}
error_dict = {}
pool_res = {}
rand = np.random.randint(0, x_sample.shape[1], size=(1000, x_sample.shape[1]))
for key, value in partial_order.items():
    _, sample_size = key.split('_')
    poly, error = pce_fun(variable, samples, values, 
                        ntrain_samples=int(sample_size), degree=2)
    # add the calculation of y_uncond
    y_uncond_temp = poly(x_sample).flatten()
    conf_uncond[key] = np.quantile(y_uncond_temp, [0.025, 0.975])
    conf_uncond[key] = mad(y_uncond_temp) / np.median(y_uncond_temp)
    error_dict[key], pool_res = group_fix(value, poly, x_sample, y_uncond_temp, x_fix, rand, {}, file_exist=True)

# separate confidence intervals into separate dicts
save_path = f'{output_path}error_measures/'
os.mkdir(save_path)

if option_return=='conf':
# if the measure is confidence intervals
    low_conf, up_conf = {}, {}
    for k, v in measure2.items():
        low_conf[k] = {ii:conf[0] for ii, conf in v.items()}
        up_conf[k] = {ii:conf[1] for ii, conf in v.items()}
    # convert results in dictionary to 
    co_var = to_df(partial_order, measure1).round(4)
    low_conf_df = to_df(partial_order, low_conf).round(4)
    up_conf_df = to_df(partial_order, up_conf).round(4)

    # save results
    co_var.to_csv(f'{save_path}co_var.csv')
    low_conf_df.to_csv(f'{save_path}low_95.csv')
    up_conf_df.to_csv(f'{save_path}up_95.csv')

    uncond_df = pd.DataFrame.from_dict(conf_uncond)
    width_uncond = (uncond_df.loc[1, :] - uncond_df.loc[0, :]).values
    conf_width_relative = (up_conf_df - low_conf_df).apply(lambda x : x / width_uncond, axis=1)
    conf_width_relative.to_csv(f'{save_path}conf_width_relative_95.csv')
else:
    if option_return == 'ks': fnames_save = ['dvalue', 'pvalue']
    if option_return == 'median': fnames_save = ['median', 'normalized_median']
    # convert results in dictionary to 
    measure1 = to_df(partial_order, measure1).round(4)
    measure2 = to_df(partial_order, measure2).round(4)
    # save results
    measure1.to_csv(f'{save_path}{fnames_save[0]}.csv')
    measure2.to_csv(f'{save_path}{fnames_save[1]}.csv')