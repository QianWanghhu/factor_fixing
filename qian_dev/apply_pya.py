import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform
import json
import os
import time
from pyapprox.random_variable_algebra import product_of_independent_random_variables_pdf

from basic.boots_pya import least_squares, fun, fun_new
from basic.partial_rank import partial_rank
from basic.utils import variables_prep
from basic.read_data import read_specify, file_settings

def sa_df_format(total_effects, variables, param_names, conf_level=0.95):    
    sa_df = pd.DataFrame(data=np.zeros(shape=(variables.num_vars(), 3)), 
                        columns=['ST', 'ST_conf_lower', 'ST_conf_upper'])
    total_effects = np.array(total_effects)
    sa_df.loc[:, 'ST'] = total_effects.mean(axis=0)
    sa_df.loc[:, 'ST_conf_lower'] = np.quantile(total_effects, [0.025], axis=0)[0]
    sa_df.loc[:, 'ST_conf_upper' ] = np.quantile(total_effects, [0.975], axis=0)[0]
    sa_df.index = param_names
    return sa_df
# End df_format()

start_time = time.time()
# import the original parameter sets
def pya_boot_sensitivity(outpath, nboot, seed, product_uniform):
    variable, _ = read_specify('parameter', 'reduced', product_uniform, num_vars=11)
    len_params = variable.num_vars()
    samples, values = read_specify('model', 'reduced', product_uniform, num_vars=11)

    # Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking
    n_strat, n_end, n_step = [104, 552, 13]
    errors_cv_all = {}
    partial_results = {}
    index_cover_all = {}
    total_effects_all = {}
    for i in range(n_strat, n_end+1, n_step):
    # for i in n_list:
        print(i)
        if (n_end - i)  < n_step:
            i = n_end
        np.random.seed(seed)                                                    
        rand_I = np.random.randint(0, i, size=(nboot, i))
        errors_cv, _, total_effects, index_cover = fun(variable, samples, 
                                                    values, degree=2, 
                                                    nboot=nboot, I=rand_I, 
                                                    ntrain_samples=i)

        # partial ranking
        total_effects = np.array(total_effects)
        sa_shape = list(total_effects.shape)[0:2]
        total_effects = total_effects.reshape((sa_shape))
        rankings = partial_rank(total_effects,len_params, conf_level=0.95)
        partial_results[f'nsample_{i}'] = rankings
        errors_cv_all[f'nsample_{i}'] = errors_cv
        index_cover_all[f'nsample_{i}'] = index_cover
        total_effects_all[f'nsample_{i}'] = total_effects
    # End for
    if product_uniform == True:
        dist_type = 'beta'
    else:
        dist_type = 'uniform'
    filename = f'adaptive-reduce-{dist_type}_552.npz'
    np.savez(f'{outpath}{filename}',errors_cv=errors_cv_all, sensitivity_indices=partial_results, index_cover = index_cover_all, total_effects=total_effects_all)


def pya_boot_sensitivity_new(outpath, nboot, seed, product_uniform, filename):

    variable, _ = read_specify('parameter', 'reduced', product_uniform, num_vars=11)

    len_params = variable.num_vars()
    samples, values = read_specify('model', 'reduced', product_uniform, num_vars=11)

    # Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking
    n_strat, n_end, n_step = [104, 552, 13]
    errors_cv_all = {}
    partial_results = {}
    index_cover_all = {}
    total_effects_all = {}
    for i in range(n_strat, n_end+1, n_step):
    # for i in n_list:
        print(i)
        if (n_end - i)  < n_step:
            i = n_end
        np.random.seed(seed)                                                    
        errors_cv, _, total_effects = fun_new(
            variable, samples[:, :i], values[:i], product_uniform, nboot=nboot)

        # partial ranking
        total_effects = np.array(total_effects)
        sa_shape = list(total_effects.shape)[0:2]
        total_effects = total_effects.reshape((sa_shape))
        rankings = partial_rank(total_effects,len_params, conf_level=0.95)
        partial_results[f'nsample_{i}'] = rankings
        errors_cv_all[f'nsample_{i}'] = errors_cv
        index_cover_all[f'nsample_{i}'] = np.nan # hack because I do not use this anymore
        total_effects_all[f'nsample_{i}'] = total_effects
    # End for
    if product_uniform == True:
        dist_type = 'beta'
    else:
        dist_type = 'uniform'
    np.savez(f'{outpath}{filename}',errors_cv=errors_cv_all, sensitivity_indices=partial_results, index_cover = index_cover_all, total_effects=total_effects_all)


def run_pya(outpath, nboot, seed, product_uniform):
    if product_uniform == 'beta':
        dist_type = 'beta'
    elif product_uniform == 'exact':
        dist_type = 'exact'
    else:
        dist_type = 'uniform'
    filename = f'adaptive-reduce-{dist_type}_552.npz'

    print(f'{outpath}{filename}')
    if not os.path.exists(f'{outpath}{filename}'):
        pya_boot_sensitivity_new(
            outpath, nboot, seed, product_uniform, filename)

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


def pce_22(nboot, seed, ntrain_samples):
    fpath_save = file_settings()[0]
    samples, values = read_specify('model', 'full', product_uniform=False, num_vars=22)
    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    variable, param_all = read_specify('parameter', 'full', product_uniform=False, num_vars=22)
    # Used for PCE fitting
    np.random.seed(seed)
    rand_I = np.random.randint(0, ntrain_samples, size=(nboot, ntrain_samples)) 
    error_cv, _, total_effects, = fun_new(
        variable, samples[:,:ntrain_samples], 
        values[:ntrain_samples], product_uniform=False,
        nboot=nboot)                                          
    sa_df = sa_df_format(total_effects, variable, 
                    param_all, conf_level=0.95)
    sa_df.to_csv(f'{fpath_save}sa_pce_22.csv', index=True)
    error_cv_df = pd.DataFrame(data = error_cv, columns = [f'22_{ntrain_samples}_uniform'],
                    index = np.arange(len(error_cv)))
    error_cv_mean = error_cv_df.apply(np.mean, axis=0)
    error_cv_lower = np.round(error_cv_df.quantile(0.025, axis=0), 4)
    error_cv_upper= np.round(error_cv_df.quantile(0.975, axis=0), 4)
    error_stats_df = pd.DataFrame(data=[error_cv_mean, error_cv_lower, error_cv_upper], 
                    index=['mean', 'lower', 'upper']).T
    error_stats_df.to_csv(f'{fpath_save}error_cv_compare.csv', index=True)
