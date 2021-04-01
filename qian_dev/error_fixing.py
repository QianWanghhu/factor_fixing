import numpy as np
import pandas as pd
import json
import os
import pyapprox as pya
import SALib.sample.latin as latin
from SALib.util import read_param_file

from basic.boots_pya import fun, fun_new
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from basic.read_data import file_settings, read_specify

def fix_group_ranking(input_path, variable, output_path, samples, values,
                      partial_order, index_product, problem, x_fix,
                      x_fix_adjust, num_pce, seed, sample_range,
                      product_uniform):
    # Calculate the corresponding number of bootstrap with use of group_fix
    x_sample = latin.sample(problem, sample_range, seed=88).T
    np.savetxt(f'{output_path}metric_samples.txt', x_sample)
    # if reduce parameters, change samples
    if (variable.num_vars()) == 11:
        x_sample = adjust_sampling(x_sample, index_product, x_fix)

    conf_uncond, error_dict, pool_res, y_uncond = {}, {}, {}, {}
    rand = np.random.randint(
        0, x_sample.shape[1], size=(500, x_sample.shape[1]))
    ci_bounds = [0.025, 0.975]

    for key, value in partial_order.items():
        pce_list = []; cv_temp = np.zeros(num_pce)
        y_temp = np.zeros(shape=(num_pce, x_sample.shape[1]))
        _, sample_size = key.split('_')[0], int(key.split('_')[1])
        print(f'------------------Training samples: {sample_size}------------------------')
        np.random.seed(seed)
        rand_pce = np.random.randint(
            0, sample_size, size=(num_pce, sample_size))
        for i in range(rand_pce.shape[0]):
            poly = fun_new(
                variable, samples[:, rand_pce[i]], values[rand_pce[i]], 
                product_uniform, nboot=1)
        # add the calculation of y_uncond
            pce_list.append(poly)
            y_temp[i, :] = poly(x_sample).flatten()
            cv_temp[i] = np.sqrt(poly.variance())[0] / poly.mean()[0]
        y_uncond[key] = y_temp
        conf_uncond[key] = uncond_cal(y_uncond[key], ci_bounds, rand)
        conf_uncond[key]['cv'] = cv_temp[i].mean()
        conf_uncond[key]['cv_low'], conf_uncond[key]['cv_up'] = \
            np.quantile(cv_temp, ci_bounds)
        error_dict[key], pool_res = group_fix(value, pce_list, x_sample, y_uncond[key], x_fix_adjust, rand, {}, file_exist=True)
    # End for


    # # separate confidence intervals into separate dicts and write results
    save_path = f'{output_path}error_measures/'
    if not os.path.exists(save_path): os.mkdir(save_path)
    # convert the result into dataframe
    key_outer = list(error_dict.keys())
    f_names = list(error_dict[key_outer[0]].keys())
    for ele in f_names:
        dict_measure = {key: error_dict[key][ele] for key in key_outer}
        df = to_df(partial_order, dict_measure)
        df.to_csv(f'{save_path}/{ele}.csv')

    with open(f'{save_path}y_uncond_stats.json', 'w') as fp:
        json.dump(conf_uncond, fp, indent=2)


def fix_increase_sample(input_path, variable, output_path, samples, values,
                        partial_order, index_product, problem, x_fix, x_fix_adjust, num_pce, seed, sample_range, product_uniform):
# if reduce parameters, change samples
    key = list(partial_order.keys())[0]
    _, sample_size = key.split('_')
    value = partial_order[key]
    poly_list = []
    poly = fun_new(variable, samples[:, :int(sample_size)],
                   values[:int(sample_size)], product_uniform,
                   nboot=num_pce)
    poly_list.append(poly)
    conf_uncond, error_dict, pool_res, y_uncond = {}, {}, {}, {}
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
        y_uncond_temp = poly(x_sample).T
        conf_uncond[str(n)] = uncond_cal(y_uncond_temp, ci_bounds, rand)
        conf_uncond[str(n)]['median'] = np.quantile(y_uncond_temp[0][rand], ci_bounds, axis=1).mean(axis=0).mean()
        conf_uncond[str(n)]['mean'] = poly.mean()[0]
        error_dict[str(n)], pool_res = group_fix(value, poly_list, x_sample, y_uncond_temp, 
                                        x_fix_adjust, rand, {}, file_exist=True)
        # End for

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
