import numpy as np
import pandas as pd

# def to_df(measure_dict, measure_name):
#     """
#     Help funtion to convert the dictionary of results into dataframe.
#     """
#     measure_df = {}
#     k = 0
#     for key, value in measure_dict.items():
#         measure_df[key] =  pd.DataFrame(value, index=measure_name)
#         k += 1
#     return measure_df
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

        for g, v in fix_dict[key].items():
            if isinstance(v, tuple):
                fix_temp.extend([v[0]])
            else:
                fix_temp.extend([v])
        fix_df[key]  = np.repeat(fix_temp, len_each_group)

    fix_df = pd.DataFrame.from_dict(fix_df)
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

def names_match(rankings, parameters, fdir, fname):
    """
    This convert the results in partial order of the format index into Veneer_names.
    rankings: dict, partial sort results
    parameters: dataframe, contains names of parameters
    """
    partial_names = {}
    for key, value in rankings.items():
        partial_names[key] = parameters.loc[value, 'Veneer_name'].values.tolist()
    with open(f'{fdir}{fname}', 'w') as fp:
        json.dump(partial_names, fp, indent=2)
    

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