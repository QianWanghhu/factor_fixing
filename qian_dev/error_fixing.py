import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pyapprox as pya
import SALib.sample.latin as latin
from SALib.util import read_param_file

# from utils import cvg_check
from basic.utils import cvg_check
from basic.error_measure import group_fix
from basic.boots_pya import pce_fun 
from scipy.stats import uniform
from basic.boots_pya import least_squares, fun
from basic.utils import variables_prep, to_df

reduce_params = True
if reduce_params == True:
    file_list = ['parameter-adjust', 'samples_adjust', 'reduce_params', 'partial_reduce_params']
else:
    file_list = ['parameter-ranges', 'test_qian', 'full_params', 'partial_pya']

fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/pya_related/'
filename = f'{fpath}{file_list[0]}.csv'
variable = variables_prep(filename, product_uniform=True)

filename = f'{fpath}{file_list[1]}.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
len_params = variable.num_vars()
samples = data[:,:len_params].T
values = data[:,len_params:]
values = values[:,:1]# focus on first qoi
len_params = samples.shape[0]
# generate samples for error metric analysis
filename = f'{fpath}problem.txt'
problem = read_param_file(filename, delimiter=',')
x_sample = latin.sample(problem, 1000, seed=88)
x_sample = x_sample.T

# if reduce parameters, adapt samples
if (variable.num_vars()) == 11:
    index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17], 
                         [6, 5, 7], 
                         [19, 20],
                         ])
    samples_adjust = np.copy(x_sample)
    pars_delete = []
    for ii in range(list(index_product.shape)[0]):
        index_temp = index_product[ii]
        samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
        # samples_adjust[index_temp[1:], :] = 1
        pars_delete.extend(index_temp[1:])
    samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)
    x_sample = samples_adjust

# load partial order results
fpath = f'D:/cloudStor/Research/pce_fixing/output/0709_ave_annual/{file_list[2]}/'
fname = f'{file_list[3]}.json'
with open(f'{fpath}{fname}', 'r') as fp:
    partial_order = json.load(fp)
# Get the unique partial sort results and the corresponding number of bootstrap used for PCE
# order_temp, num_resample = cvg_check(partial_order)

measure1 = {}
measure2 = {}
conf_uncond = {}
option_return='ks'
for key, value in partial_order.items():
    _, sample_size = key.split('_')
    poly, error = pce_fun(variable, samples, values, 
                        ntrain_samples=int(sample_size), degree=2)
    # add the calculation of y_uncond
    y_uncond_temp = poly(x_sample).flatten()
    conf_uncond[key] = np.quantile(y_uncond_temp, [0.025, 0.975])
    measure1[key], measure2[key] = group_fix(value, poly, x_sample, 
                                    y_uncond_temp, 1, option_return='ks', file_exist=True)
# print(kl_dict)

# separate confidence intervals into separate dicts
fpath_save = f'D:/cloudStor/Research/pce_fixing/output/0709_ave_annual/{file_list[2]}/'
if option_return=='conf':
# if the measure is confidence intervals
    low_conf, up_conf = {}, {}
    for k, v in measure2.items():
        low_conf[k] = {ii:conf[0] for ii, conf in v.items()}
        up_conf[k] = {ii:conf[1] for ii, conf in v.items()}
    # convert results in dictionary to 
    co_var = to_df(partial_order, measure1).round(4)
    low_conf_df = to_df(partial_order, low_conf).round(4)
    up_conf_df = to_df(partial_order, up_conf).round(4)

    # save results
    co_var.to_csv(f'{fpath_save}co_var.csv')
    low_conf_df.to_csv(f'{fpath_save}low_95.csv')
    up_conf_df.to_csv(f'{fpath_save}up_95.csv')

    uncond_df = pd.DataFrame.from_dict(conf_uncond)
    width_uncond = (uncond_df.loc[1, :] - uncond_df.loc[0, :]).values
    conf_width_relative = (up_conf_df - low_conf_df).apply(lambda x : x / width_uncond, axis=1)
    conf_width_relative.to_csv(f'{fpath_save}conf_width_relative_95.csv')
elif option_return=='ks':
    # convert results in dictionary to 
    Dvalue = to_df(partial_order, measure1).round(4)
    Pvalue = to_df(partial_order, measure2).round(4)
    # save results
    Dvalue.to_csv(f'{fpath_save}dvalue.csv')
    Pvalue.to_csv(f'{fpath_save}pvalue.csv')