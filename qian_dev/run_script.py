import pandas as pd
import numpy as np
from SALib.util import read_param_file

from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from basic.read_data import file_settings, read_specify

from model_sample_process import model_ts_reduced
model_ts_reduced()
# apply_pya to produce the sensitivities of parameters for different PCEs
from apply_pya import run_pya
outpath = file_settings()[0]
# PCE with uniform distributions
product_uniform = False
run_pya(outpath, product_uniform)

# PCE with Beta distributions
product_uniform = True
run_pya(outpath, product_uniform)

# evaluate the uncertainty measures from fixing parameters
# import variables and samples for PCE
input_path = file_settings()[1]
variable, _ = read_specify('parameter', 'reduced', 
    product_uniform=True, num_vars=11)
output_path = file_settings()[0]
samples, values = read_specify('model', 'reduced', 
    product_uniform=False, num_vars=11)
# load partial order results
rankings_all = read_specify('rank', 'reduced', 
    product_uniform=True, num_vars=11)
# import index_prodcut which is a array defining the correlations between parameters
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
sample_range = [1000, 2000, 500]
# if reduce parameters, change samples
if (variable.num_vars()) == 11:
    x_fix_adjust = adjust_sampling(x_fix, index_product, x_fix)

# Fixing parameters ranked by different PCEs 
# and 1000 samples are used to calculate the uncertainty measures
from error_fixing import fix_group_ranking
key_use = [f'nsample_{ii}' for ii in np.arange(104, 131, 13)]
partial_order = dict((key, value) for key, value in rankings_all.items() if key in key_use)
fix_group_ranking(input_path, variable, output_path, samples, values,
    partial_order, index_product, problem, x_fix, x_fix_adjust, sample_range[0])

from error_fixing import fix_increase_sample
# Fixing parameters ranked by a PCE trained with 156 model runs 
# and increasing samples are used to calculate the uncertainty measures
key_use = [f'nsample_{ii}' for ii in [156]]
partial_order = dict((key, value) for key, value in rankings_all.items() if key in key_use)
fix_increase_sample(input_path, variable, output_path, samples, values,
    partial_order, index_product, problem, x_fix, x_fix_adjust, sample_range)

