import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform
import json
import os
import time
from pyapprox.random_variable_algebra import product_of_independent_random_variables_pdf

from basic.boots_pya import least_squares, fun
from basic.partial_rank import partial_rank
from basic.utils import variables_prep
from basic.read_data import read_specify, file_settings

start_time = time.time()
# import the original parameter sets
def pya_boot_sensitivity(product_uniform):
    variable = read_specify('parameter', 'reduced', product_uniform, num_vars=11)
    len_params = variable.num_vars()
    samples, values = read_specify('model', 'reduced', product_uniform, num_vars=11)

    # Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking
    # n_strat, n_end, n_setp = [104, 553, 13]
    n_list = [130, 156, 234, 312, 390, 552]
    # loops of fun
    errors_cv_all = {}
    # errors_bt_all = {}
    partial_results = {}
    index_cover_all = {}
    # for i in range(n_strat, n_end+1, n_setp):
    for i in n_list:
        print(i)
        errors_cv, _, total_effects, index_cover = fun(variable, samples, 
                                                    values, degree=2, 
                                                    nboot=500, ntrain_samples=i)
        # partial ranking
        total_effects = np.array(total_effects)
        sa_shape = list(total_effects.shape)[0:2]
        total_effects = total_effects.reshape((sa_shape))
        rankings = partial_rank(total_effects,len_params, conf_level=0.95)
        partial_results[f'nsample_{i}'] = rankings
        errors_cv_all[f'nsample_{i}'] = errors_cv
        index_cover_all[f'nsample_{i}'] = index_cover
        # errors_bt_all[f'nsample_{i}'] = errors_bt
    # End for

    filepath = 'output/paper0915/figure4/test_plot/'
    if product_uniform == True:
        dist_type = 'beta'
    else:
        dist_type = 'uniform'
    filename = f'adaptive-reduce-{dist_type}_552.npz'
    np.savez(f'{filepath}{filename}',errors_cv=errors_cv_all, sensitivity_indices=partial_results, index_cover = index_cover_all)


def run_pya(product_uniform=True):
    filepath = 'output/paper0915/figure4/test_plot/'
    if product_uniform == True:
        dist_type = 'beta'
    else:
        dist_type = 'uniform'
    filename = f'adaptive-reduce-{dist_type}_552.npz'

    if not os.path.exists(f'{filepath}{filename}'):
        pya_boot_sensitivity(product_uniform)

    fileread = np.load(f'{filepath}{filename}', allow_pickle=True)
    errors_cv = fileread[fileread.files[0]][()]
    sensitivity_indices = fileread[fileread.files[1]][()]
    index_cover = fileread[fileread.files[2]][()]
    # Look the error change with the increase of sample size
    errors_cv = pd.DataFrame.from_dict(errors_cv)
    error_stats = pd.DataFrame()
    error_stats['mean'] = np.round(errors_cv.mean(axis=0), 4)
    error_stats['error_lower'] = np.round(np.quantile(errors_cv, 0.025, axis=0), 4)
    error_stats['error_upper'] = np.round(np.quantile(errors_cv, 0.975, axis=0), 4)
    error_stats.index.name = 'index'

    error_stats.to_csv(f'{filepath}error_cv_{dist_type}_552.csv')
    with open(f'{filepath}partial_reduce_{dist_type}_552.json', 'w') as fp:
        json.dump(sensitivity_indices, fp, indent=2)
    with open(f'{filepath}index_cover.json', 'w') as fp:
        json.dump(index_cover, fp, indent=2)
    
if __name__ == "__main__":
    product_uniform = False
    run_pya(product_uniform)

print("--- %s seconds ---" % (time.time() - start_time))