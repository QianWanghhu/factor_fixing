# import packages
import numpy as np
import pandas as pd
import os
import chaospy as cp
import dill
import json
from bisect import bisect
import matplotlib.pyplot as plt

# from utils import cvg_check
import utils
from utils import cvg_check, to_df
from error_measure import group_fix
import source_runner as sr

# import PCE object
# assign distributions to parameters
# parameters = parameters.loc[0:10, :]
param_file = '../../input/upper v1.csv'
parameters = sr.load_parameter_file(param_file) 
parameters = parameters[:-1]
len_params = len(parameters)
param_dist = []
for i in range(len_params):
    low_bd = parameters.loc[i, 'min']
    up_bd = parameters.loc[i, 'max']
    param_dist.append(cp.Uniform(lower=low_bd, upper=up_bd))
dist = cp.J(*param_dist)
# upload PCE
qoi = '0709_ave_annual'
fdir = f'../../output/{qoi}/pce/'
fname = 'pce_default.dill'
with open('{}{}'.format(fdir, fname), 'rb') as fp:
    pce_fit = dill.load(fp)
np.random.seed(seed=121)
samples = dist.sample(1000, rule='L')
y_pce = pce_fit(*samples)
# load partial order results
with open(f'../../output/{qoi}/sensitivity/' + f'{qoi}_total_partial_sort.json', 'r') as fp:
    partial_order = json.load(fp)
# Get the unique partial sort results and the corresponding number of bootstrap used for PCE
order_temp, num_resample = cvg_check(partial_order)

# calculate error_measure
kl_dict = {}
conf_int = {}
for ele in num_resample:
    kl_dict[ele], conf_int[ele] = group_fix(partial_order[ele], pce_fit, samples, y_pce, 1, bins=100, a=None, file_exist=True)
print(kl_dict)
# transform the result into dataframe to enable visulization
index_names = ['ks', 'ks_p', 'r2'] # , 'rmse', 'cond_risk'
measure_df = to_df(kl_dict, index_names)
fig_dir = f'../../output/{qoi}/fig/'

for key, value in measure_df.items():
    x = np.arange(len(value.columns))
    # fig = plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax = fig.add_axes([0, 0, 1, 1])
    value = value.transpose()
    width = 1 / len(index_names)
    ax.bar(x + 0 * width, value[index_names[0]], color='b', width=width)
    ax.bar(x + 1 * width, value[index_names[1]], color='g', width=width)
    ax.bar(x + 2 * width, value[index_names[2]], color='r', width=width)
    ax.set_yticks(np.arange(-0.5, 1.0, 0.1))
    ax.set_ylabel(index_names[0], fontsize=16)
    labels = [f'group{i}' for i in value.index]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=12)
    ax.set_xlabel('Groups of parameters', fontsize=16)
    fig.legend(labels=index_names, fontsize=14, bbox_to_anchor=(0.9, 0.35))
    plt.show()
    # plt.savefig(f'{fig_dir}{key}.png')
    # save values
    # value.to_csv(fig_dir+f'{key}.csv')
# # plot
