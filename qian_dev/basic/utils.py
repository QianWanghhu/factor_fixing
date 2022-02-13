import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform, beta
from basic.read_data import read_specify

def to_df(partial_order, fix_dict):
    """
    Help function to convert difference between 
    conditioned and unconditioned into dataframe.
    Parameters:
    ===========
    partial_order : dict, partial ranking of parameters
    fix_dict : dict, difference between conditioned and unconditional model results.
                (each dict result returned from group_fix / pce_group_fix)

    Returns:
    ========
    fix_df : df, formatted fix_dict
    """
    keys_list = list(partial_order.keys())

    fix_df = {key:None for key in keys_list}

    for key in keys_list:
        len_each_group = []
        len_each_group = [len(value) for k, value in partial_order[key].items()]
        fix_temp = []

        for _, v in fix_dict[key].items():
            if isinstance(v, tuple):
                fix_temp.extend([v[0]])
            else:
                fix_temp.extend([v])
        fix_df[key]  = np.repeat(fix_temp, len_each_group)

    fix_df = pd.DataFrame.from_dict(fix_df)
    fix_df.index = np.arange(fix_df.shape[0], 0, -1)
    return fix_df
    
def cvg_check(partial_order):
    order_temp = []
    num_res_change = []
    for key, value in partial_order.items():
        # import pdb
        # pdb.set_trace()
        if not (value in order_temp):
            order_temp.append(value)
            num_res_change.append(key)
    return order_temp, num_res_change      

def names_match(rankings, param_names):
    """
    This convert the results in partial order of the format index into Veneer_names.
    rankings: dict, partial sort results
    parameters: dataframe, contains names of parameters
    """
    partial_names = {}
    for key, value in rankings.items():
        partial_names[key] = [param_names[v] for v in value]
    # with open(f'{fdir}{fname}', 'w') as fp:
    #     json.dump(partial_names, fp, indent=2)
    return partial_names

def epsi_cal(x, y):
    """
    Calculate the difference in y for every pair of sampling points.
    Parameters:
    ===========
    x : numpy.ndarray, sample points of all input factors, as the shaple of n*D of which
        n is the number of samples and D is the number of parameters.
    y : numpy.ndarray, corresponding y returned from the model

    Rerurns:
    ========
    epsi : numpy.ndarray, difference in y for each pair of x
    """
    n = x.shape[0]
    epsi = np.zeros(shape=(n, n))
    y_ave = np.zeros(shape=(n, n))
    for i in range(1, n):
        for j in range(i):
            epsi[i, j] = y[i] - y[j]
            y_ave[i, j] = np.mean([y[i], y[j]])

    return epsi, y_ave


def adjust_sampling(x_sample, index_product, x_fix):
    samples_adjust = np.copy(x_sample)
    pars_delete = []
    for ii in range(list(index_product.shape)[0]):
        index_temp = index_product[ii]
        samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
        # x_fix[index_temp[0]] = np.prod(x_fix[index_temp], axis=0)
        # samples_adjust[index_temp[1:], :] = 1
        pars_delete.extend(index_temp[1:])
    samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)
    # x_fix = np.delete(x_fix, pars_delete, axis=0)
    x_sample = samples_adjust
    return x_sample

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

# clean the dataframe ordered by the sampling-based sensitivity indices
def read_total_effects(fpath_save, product_uniform):
    dist_type = dist_return(product_uniform)
    filename = f'adaptive-reduce-{dist_type}_552.npz'
    fileread = np.load(f'{fpath_save}{filename}', allow_pickle=True)
    return fileread

def df_read(df, result_type, type_num, product_uniform, num_vars):
    _, parameters = read_specify('parameter', 'full', product_uniform, num_vars)
    df.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
    df['Parameters'] = parameters
    df['Type'] = result_type
    df['Type_num'] = type_num
    return df
# End df_read()

def dist_return(product_uniform):
    if product_uniform == 'beta':
        dist_type = 'beta'
    elif product_uniform == 'exact':
        dist_type = 'exact'
    elif product_uniform== 'uniform':
        dist_type = 'uniform'
    else:
        dist_type = 'full'
    return dist_type