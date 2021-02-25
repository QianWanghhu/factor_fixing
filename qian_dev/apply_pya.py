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
def pya_boot_sensitivity(outpath, product_uniform):
    variable, _ = read_specify('parameter', 'reduced', product_uniform, num_vars=11)
    len_params = variable.num_vars()
    samples, values = read_specify('model', 'reduced', product_uniform, num_vars=11)

    # Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking
    n_strat, n_end, n_setp = [104, 552, 11]
    errors_cv_all = {}
    partial_results = {}
    index_cover_all = {}
    for i in range(n_strat, n_end+1, n_setp):
    # for i in n_list:
        print(i)
        nboot=500; seed=222
        np.random.seed(seed)                                                    
        rand_I = np.random.randint(0, i, size=(nboot, i))
        # import pdb; pdb.set_trace()
        errors_cv, _, total_effects, index_cover = fun(variable, samples, 
                                                    values, degree=2, 
                                                    nboot=nboot, I=rand_I, 
                                                    ntrain_samples=i)

        # partial ranking
        # import pdb; pdb.set_trace()
        total_effects = np.array(total_effects)
        # import pdb; pdb.set_trace()
        sa_shape = list(total_effects.shape)[0:2]
        total_effects = total_effects.reshape((sa_shape))
        rankings = partial_rank(total_effects,len_params, conf_level=0.95)
        partial_results[f'nsample_{i}'] = rankings
        errors_cv_all[f'nsample_{i}'] = errors_cv
        index_cover_all[f'nsample_{i}'] = index_cover
    # End for
    if product_uniform == True:
        dist_type = 'beta'
    else:
        dist_type = 'uniform'
    filename = f'adaptive-reduce-{dist_type}_552.npz'
    np.savez(f'{outpath}{filename}',errors_cv=errors_cv_all, sensitivity_indices=partial_results, index_cover = index_cover_all)


def run_pya(outpath, product_uniform=True):
    if product_uniform == True:
        dist_type = 'beta'
    else:
        dist_type = 'uniform'
    filename = f'adaptive-reduce-{dist_type}_552.npz'

    if not os.path.exists(f'{outpath}{filename}'):
        pya_boot_sensitivity(outpath, product_uniform)

    fileread = np.load(f'{outpath}{filename}', allow_pickle=True)
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

    error_stats.to_csv(f'{outpath}error_cv_{dist_type}_552.csv')
    with open(f'{outpath}partial_reduce_{dist_type}_552.json', 'w') as fp:
        json.dump(sensitivity_indices, fp, indent=2)
    with open(f'{outpath}index_cover.json', 'w') as fp:
        json.dump(index_cover, fp, indent=2)
    
print("--- %s seconds ---" % (time.time() - start_time))