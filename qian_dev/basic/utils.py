import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform, beta

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

def variables_prep(filename, product_uniform=False, dummy=False):
    """
    Help function for preparing the data training data to fit PCE.
    Parameters:
    ===========
    filename : str
    product_uniform : False do not colapse product into one variable
                      'uniform' uniform distributions are used for product; 
                      'beta', beta distributions are used for variables which 
                      are adapted considering the correlations
                      'exact' the true PDF of the product is used

    """
    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    if (product_uniform is False) or (product_uniform == 'uniform'):    
        ranges = np.loadtxt(
            filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
        univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(0, ranges.shape[0]//2)]
    else:
        param_adjust = pd.read_csv(filename)
        beta_index = param_adjust[param_adjust['distribution']== 'beta'].index.to_list()
        ranges = np.array(param_adjust.loc[:, ['min','max']])
        ranges[:, 1] = ranges[:, 1] - ranges[:, 0]
        # param_names = param_adjust.loc[[0, 2, 8], 'Veneer_name'].values
        univariate_variables = []
        for ii in range(param_adjust.shape[0]):
            if ii in beta_index:
                shape_ab = param_adjust.loc[ii, ['a','b']].values.astype('float')
                univariate_variables.append(beta(shape_ab[0], shape_ab[1], 
                                            loc=ranges[ii][0], scale=ranges[ii][1]))
            else:
                # uniform_args = ranges[ii]
                univariate_variables.append(uniform(ranges[ii][0], ranges[ii][1]))
            # End if
    if dummy == True: univariate_variables.append(uniform(0, 1))
    # import pdb
    # pdb.set_trace()
    variable = pya.IndependentMultivariateRandomVariable(univariate_variables)
    return variable
        # End for()

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
