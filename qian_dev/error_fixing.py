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

# a list of filenames
file_list = ['parameter-adjust', 'samples_adjust', 'partial_reduce_beta', 'partial_reduce_params']

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

# Calculate the corresponding number of bootstrap with use pf group_fix
# the adaptive process requires PCE to be fitted with increasing sample size
# therefore the calculation is done without avoiding repeating analysis
conf_uncond, error_dict, pool_res, y_uncond = {}, {}, {}, {}
rand = np.random.randint(0, x_sample.shape[1], size=(1000, x_sample.shape[1]))

for key, value in partial_order.items():
    _, sample_size = key.split('_')
    poly, error = pce_fun(variable, samples, values, 
                        ntrain_samples=int(sample_size), degree=2)
    # add the calculation of y_uncond
    y_uncond[key] = poly(x_sample).flatten()
    conf_uncond[key] = np.quantile(y_uncond[key], [0.025, 0.975])
    # error_dict[key], pool_res = group_fix(value, poly, x_sample, y_uncond[key], x_fix, rand, {}, file_exist=True)
# End for

# separate confidence intervals into separate dicts and write results
save_path = f'{output_path}error_measures/'
if not os.path.exists(save_path): os.mkdir(save_path)
# convert the result into dataframe
key_outer = list(error_dict.keys())
f_names = list(error_dict[key_outer[0]].keys())
for ele in f_names:
    dict_measure = {key: error_dict[key][ele] for key in key_outer}
    df = to_df(partial_order, dict_measure)
    df.to_csv(f'{save_path}/{ele}.csv')

# calculate the statistics for unconditional model outputs
key = 'nsample_156'
y_uncond_stat = {'mean' : y_uncond[key].mean(),
                'q25': np.quantile(y_uncond[key], [0.25])[0],
                'q75': np.quantile(y_uncond[key], [0.75])[0]}
with open(f'{save_path}y_uncond_stats.json', 'w') as fp:
    json.dump(y_uncond_stat, fp, indent=2)

# calculate width of confidence intervals
uncond_df = pd.DataFrame.from_dict(conf_uncond)
width_uncond = (uncond_df.loc[1, :] - uncond_df.loc[0, :]).values

# import confidence intervals represented as upper and lower bounds
cf_upper = pd.read_csv(f'{save_path}cf_upper.csv', index_col='Unnamed: 0')
cf_lower = pd.read_csv(f'{save_path}cf_lower.csv', index_col='Unnamed: 0')
conf_width_relative = (cf_upper - cf_lower).apply(lambda x : x / width_uncond, axis=1)
conf_width_relative.to_csv(f'{save_path}conf_width_relative_95.csv')