import numpy as np
import pandas as pd
from SALib.util import read_param_file
import pickle
import json

from basic.read_data import file_settings, read_specify
from basic.utils import read_total_effects, df_read
from basic.plot import short_name

output_path = file_settings()[0]
# read total effects calculated by different PCE settings.
nsample = 390
fileread = read_total_effects(output_path, product_uniform=False)
df_temp = pd.DataFrame.from_dict(fileread[fileread.files[-1]][()][f'nsample_{nsample}']).T
df_raw = pd.DataFrame(index=['ST', 'ST_lower', 'ST_upper'], columns=list(df_temp.index),
    data = [df_temp.mean(axis=1), df_temp.quantile(q=0.025, axis=1), df_temp.quantile(q=0.975, axis=1)]).T
df_raw = df_read(df_raw, 'PCE-Uniform', 2, False, 22)
df_raw.to_csv(f'{output_path}sa_pce_22.csv')

# change this to product_uniform='exact' to use new polynomials
product_uniform = False
input_path = file_settings()[1]
variable, _ = read_specify('parameter', 'full', 
    product_uniform=product_uniform, num_vars=22)
output_path = file_settings()[0]
samples, values = read_specify('model', 'full', 
    product_uniform=False, num_vars=22)
# import index_prodcut which is a array defining the correlations between parameters
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
x_sample = np.loadtxt(f'{output_path}metric_samples.txt')

# Fixing parameters ranked by different PCEs 
# and 1000 samples are used to calculate the uncertainty measures
print(f'--------Calculate uncertainty measures due to FF with PCE-{product_uniform}--------')
from error_fixing import fix_group_ranking
key_use = [f'nsample_{ii}' for ii in [390]]
dist_type = 'full'
filename = f'adaptive-reduce-{dist_type}_552'

num_pce = 100; seed=1000
sa_raw = pd.read_csv(f'{output_path}sa_pce_22.csv')
thresholds = [0.01, 0.1]
for thsd in thresholds[1:]:
    ind_fix = list(sa_raw[sa_raw.ST<thsd].index) 
    partial_order = {f'nsample_{nsample}': {'0': ind_fix}}
    fix_group_ranking(input_path, variable, output_path, samples, values,
        partial_order, index_product, problem, x_fix, x_fix, 
            num_pce, seed, 1000, product_uniform, filename)