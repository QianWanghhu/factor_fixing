# Script for preparing training dataset for PCE
# import packages
import pandas as pd
import numpy as np
import os
import pyapprox as pya
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from scipy.stats import beta, uniform, norm as beta, uniform, norm
from scipy import stats
from SALib.util import read_param_file

from basic.read_data import file_settings, read_specify

# read model results
def samples_combine():
    filepath = file_settings()[1]
    # combine TSS results into a file
    filenames = os.listdir(filepath)
    for fn in filenames:
        if 'Tss' in fn:
            print(fn)
            tss_temp = pd.read_csv(f'{filepath}{fn}', index_col='# Date')
            tss_temp.index = pd.to_datetime(tss_temp.index)
            tss_temp = tss_temp.loc['2000-07-01':'2014-06-30',:]
            tss_temp.dropna(axis=1, how='any', inplace=True)
            try:
                f_quantile = np.hstack((f_quantile, tss_temp.sum(axis = 0).values / 14))
            except NameError:
                f_quantile = tss_temp.sum(axis = 0).values / 14                
    file_sample = [fn for fn in filenames if 'sample' in fn]
    f_train = pd.read_csv(f'{filepath}{file_sample[0]}', index_col= 'index')
    f_train.loc[:, 'ave_annual'] = f_quantile
    f_train.to_csv(f'{filepath}2000_2014_ave_annual.csv', index_label='id')

def model_ts_reduced():
    file_names = file_settings()
    fpath_save = file_names[0]
    filename = file_names[2]
    # import samples and values
    if not os.path.exists(filename):
        samples_combine()
    print('run')

    samples, values = read_specify('model', 'full', product_uniform=False, num_vars=22)
    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17], 
                            [6, 5, 7], 
                            [19, 20],
                            ])

    # define variables with Beta distribution
    variable_adjust, param_adjust = read_specify('parameter', 'reduced', product_uniform=True, num_vars=11)
    # beta_index = param_adjust[param_adjust['distribution']== 'beta'].\
    #             index.to_list()

    samples_adjust = np.copy(samples)
    pars_delete = []
    for ii in range(list(index_product.shape)[0]):
        index_temp = index_product[ii]
        samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
        # samples_adjust[index_temp[1:], :] = 1
        pars_delete.extend(index_temp[1:])
    samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)

    samples_adjust = np.append(samples_adjust, [values.flatten()], axis=0)    
    df_adjust = pd.DataFrame(data = samples_adjust.T, 
                            index = np.arange(samples_adjust.shape[1]),
                            columns = [*param_adjust, 'TSS_ave'])
    df_adjust.to_csv(f'{fpath_save}samples_adjust.csv')
