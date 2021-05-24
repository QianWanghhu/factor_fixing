"""
Script used to run most of the results used in the paper.
Compring different PCEs, uncertainty measures due to factor fixing.
"""
import numpy as np
from SALib.util import read_param_file

from basic.utils import adjust_sampling, dist_return
from basic.read_data import file_settings, read_specify
 
##==============================##============================##
# Combine model results and reform into 11-dimension dataset.
print('Combine and post-process of model results')
from model_sample_process import model_ts_reduced
model_ts_reduced()

##==============================##============================##
# apply_pya to produce the sensitivities of parameters for different PCEs
from apply_pya import run_pya
outpath = file_settings()[0]
num_pce = 10; seed = 727
# fun num folds set to min(num_pce, ntrain_samples)
# so num_pce = np.inf should use leave one out cross validation
# PCE with Exact product distributions
print('--------PCE-E with increasing samples--------')
run_pya(outpath, num_pce, seed, product_uniform='exact')

# PCE with uniform distributions
print('--------PCE-U with increasing samples--------')
run_pya(outpath, num_pce, seed, product_uniform='uniform')

# PCE with 22 parameters
print('----------------PCE-22----------------')
run_pya(outpath, num_pce, seed, product_uniform=False)
##==============================##============================##
# evaluate the uncertainty measures from fixing parameters
# import variables and samples for PCE

# change this to product_uniform='exact' to use new polynomials
product_uniform = 'exact'

input_path = file_settings()[1]
variable, _ = read_specify('parameter', 'reduced', 
    product_uniform=product_uniform, num_vars=11)
output_path = file_settings()[0]
samples, values = read_specify('model', 'reduced', 
    product_uniform=False, num_vars=11)
# load partial order results
rankings_all = read_specify('rank', 'reduced', 
    product_uniform=product_uniform, num_vars=11)
# import index_prodcut which is a array defining the correlations between parameters
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
filename = f'{input_path}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_fix = np.array(problem['bounds']).mean(axis=1).reshape((problem['num_vars'], 1))
sample_range = [1000, 10000, 500]
# if reduce parameters, change samples
if (variable.num_vars()) == 11:
    x_fix_adjust = adjust_sampling(x_fix, index_product, x_fix)

# Fixing parameters ranked by different PCEs 
# and 1000 samples are used to calculate the uncertainty measures
print(f'--------Calculate uncertainty measures due to FF with PCE-{product_uniform}--------')
from error_fixing import fix_group_ranking
key_use = [f'nsample_{ii}' for ii in np.arange(20, 130, 10)]
partial_order = dict((key, value) for key, value in rankings_all.items() if key in key_use)
dist_type = dist_return(product_uniform)
filename = f'adaptive-reduce-{dist_type}_552'

#---------------RUN FACTOR FIXING ---------------#
fix_group_ranking(input_path, variable, output_path, samples, values,
    partial_order, index_product, problem, x_fix, x_fix_adjust, 
        num_pce, seed, sample_range[0], product_uniform, filename)

##==============================##============================##
# Fixing parameters ranked by a PCE trained with 156 model runs 
# and increasing samples are used to calculate the uncertainty measures
print(f'--------Calculate uncertainty measures due to FF with increasing samples and a PCE-{product_uniform}--------')
from error_fixing import fix_increase_sample
key_use = [f'nsample_{ii}' for ii in [70]]
partial_order = dict((key, value) for key, value in rankings_all.items() if key in key_use)
fix_increase_sample(input_path, variable, output_path, samples, values,
    partial_order, index_product, problem, x_fix, x_fix_adjust, 1, 
            seed, sample_range, product_uniform, filename)





