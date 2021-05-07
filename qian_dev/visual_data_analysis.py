#!/usr/bin/env ffexplore
import numpy as np
import pandas as pd
import seaborn as sns
import json

from basic.plot import short_name
from basic.read_data import file_settings, read_specify

# clean the dataframe ordered by the sampling-based sensitivity indices
def read_total_effects(fpath_save, product_uniform):
    if product_uniform == 'beta':
        dist_type = 'beta'
    elif product_uniform == 'exact':
        dist_type = 'exact'
    else:
        dist_type = 'uniform'

    filename = f'adaptive-reduce-{dist_type}_552.npz'
    fileread = np.load(f'{fpath_save}{filename}', allow_pickle=True)
    return fileread

def df_read(df, result_type, type_num):
    _, parameters = read_specify('parameter', 'reduced', 'uniform', num_vars=11)
    df.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
    df['Parameters'] = parameters
    df['Type'] = result_type
    df['Type_num'] = type_num
    return df
# End df_read()

fpath_save = '../output/adaptive/'

# read total effects calculated by different PCE settings.
nsample = 90
fileread = read_total_effects(fpath_save, product_uniform='uniform')
df_temp = pd.DataFrame.from_dict(fileread[fileread.files[-1]][()][f'nsample_{nsample}']).T
df_raw = pd.DataFrame(index=['ST', 'ST_lower', 'ST_upper'], columns=list(df_temp.index),
    data = [df_temp.mean(axis=1), df_temp.quantile(q=0.025, axis=1), df_temp.quantile(q=0.975, axis=1)]).T
df_raw = df_read(df_raw, 'PCE-Uniform', 2)

# fileread = read_total_effects(fpath_save, product_uniform='beta')
# df_temp = pd.DataFrame.from_dict(fileread[fileread.files[-1]][()][f'nsample_{nsample}']).T
# df_beta = pd.DataFrame(index=['ST', 'ST_lower', 'ST_upper'], columns=list(df_temp.index),
#     data = [df_temp.mean(axis=1), df_temp.quantile(q=0.025, axis=1), df_temp.quantile(q=0.975, axis=1)]).T
# df_beta = df_read(df_beta, 'PCE-Beta', 2)

fileread = read_total_effects(fpath_save, product_uniform='exact')
df_temp = pd.DataFrame.from_dict(fileread[fileread.files[-1]][()][f'nsample_{nsample}']).T
df_exact = pd.DataFrame(index=['ST', 'ST_lower', 'ST_upper'], columns=list(df_temp.index),
    data = [df_temp.mean(axis=1), df_temp.quantile(q=0.025, axis=1), df_temp.quantile(q=0.975, axis=1)]).T
df_exact = df_read(df_exact, 'PCE-Exact', 2)

# Combine the total effects into one set.
# model_group and the type of calculation to the dataframe
df_exact['Model_group'] = np.arange(df_exact.shape[0])
df_raw['Model_group'] = None

# set parameter groups
for ii in range(df_exact.shape[0]):
    param = df_exact.Parameters[ii]
    df_raw.loc[df_raw[df_raw.Parameters==param].index, 'Model_group'] = df_exact.Model_group[ii]
    # df_exact.loc[df_exact[df_exact.Parameters==param].index, 'Model_group'] = df_exact.Model_group[ii]

df_plot = df_exact.filter(items=['Parameters', 'ST', 'ST_lower', 'ST_upper'])
df_plot.rename(columns={'ST': 'ST_exact'})
# df_plot['ST_Beta'], df_plot['ST_Beta_lower'], df_plot['ST_Beta_upper'] = \
#     df_beta.ST, df_beta.ST_lower, df_beta.ST_upper 
df_plot['ST_Uniform'], df_plot['ST_Uniform_lower'], df_plot['ST_Uniform_upper'] = \
    df_raw.ST, df_raw.ST_lower, df_raw.ST_upper

df_plot = short_name(df_plot)
names_update = ['bankErosionCoeff', 'HillslopeFineSDR', 'Gully_Management_Practice_Factor']
new_short_name = ['BEC-R', 'HFSDR-R', 'GMPF-R']
for ii in range(len(names_update)):
    df_plot.loc[df_plot[df_plot.Parameters==names_update[ii]].index, 'short_name'] = new_short_name[ii]

df_plot = df_plot.sort_values(by=['ST'], 
    ascending=[False]).reset_index(drop=True)
# save df_plot
df_plot.to_csv(f'{fpath_save}/sa_fig.csv')