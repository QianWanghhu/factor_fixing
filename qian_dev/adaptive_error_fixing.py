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

from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix
# from sa_product_dist import samples_df

# import variables and samples for PCE
file_list = ['parameter-adjust', 'samples_adjust', 'partial_reduce_uniform_552', 'partial_reduce_params']
input_path = '../data/'
filename = f'{input_path}{file_list[0]}.csv'
variable = variables_prep(filename, product_uniform=True)

output_path = '../output/paper0915/'
filename = f'{output_path}{file_list[1]}.csv'
data = np.loadtxt(filename, delimiter=",",skiprows=1)[:,1:]
len_params = variable.num_vars()
samples = data[:,:len_params].T
values = data[:,len_params:]
values = values[:,:1]
len_params = samples.shape[0]

# load partial order results
with open(f'{output_path}{file_list[2]}.json', 'r') as fp:
    partial_order = json.load(fp)

# import index_prodcut which is a array defining the correlations between parameters
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
if (variable.num_vars()) == 11:
    x_fix_adjust = adjust_sampling(x_fix, index_product, x_fix)
# x_fix = np.ones(shape=(problem['num_vars'], 1))

partial_order = dict((key, value) for key, value in partial_order.items() if str(156) in key)
key = list(partial_order.keys())[0]
_, sample_size = key.split('_')
value = partial_order[key]
poly, error = pce_fun(variable, samples, values, 
                    ntrain_samples=int(sample_size), degree=2)

start_time = time.time()
nstart, nstop, nstep = 1000, 5001, 500
conf_uncond, error_dict, pool_res, y_uncond = {'median': [], 'mean': []}, {}, {}, {}
for n in range(nstart, nstop + 1, nstep):
    print(n)
    x_sample = latin.sample(problem, n, seed=88)
    x_sample = x_sample.T
    
    # # if reduce parameters, change samples
    if (variable.num_vars()) == 11:
        x_sample = adjust_sampling(x_sample, index_product, x_fix)
    
    # Calculate the corresponding number of bootstrap with use pf group_fix
    rand = np.random.randint(0, x_sample.shape[1], size=(1000, x_sample.shape[1]))
    # add the calculation of y_uncond
    y_uncond[str(n)] = poly(x_sample).flatten()
    # conf_uncond['median'].append(np.quantile(y_uncond[str(n)][rand], [0.025, 0.975], axis=1).mean(axis=0).mean())
    # conf_uncond['mean'].append(poly.mean()[0])

    error_dict[str(n)], pool_res = group_fix(value, poly, x_sample, y_uncond[str(n)], 
                                    x_fix_adjust, rand, {}, file_exist=True)
    # End for
print("--- %s seconds ---" % (time.time() - start_time))

# separate confidence intervals into separate dicts and write results
save_path = f'{output_path}error_measures/1029_cal/'
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
