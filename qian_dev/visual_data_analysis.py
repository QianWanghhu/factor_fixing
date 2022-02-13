#!/usr/bin/env ffexplore
import numpy as np
import pandas as pd
import seaborn as sns
import json

from basic.plot import short_name
from basic.read_data import file_settings, read_specify
from basic.utils import read_total_effects, df_read

fpath_save = file_settings()[0]
# read total effects calculated by different PCE settings.
nsample = 90
fileread = read_total_effects(fpath_save, product_uniform='uniform')
df_temp = pd.DataFrame.from_dict(fileread[fileread.files[-1]][()][f'nsample_{nsample}']).T
df_raw = pd.DataFrame(index=['ST', 'ST_lower', 'ST_upper'], columns=list(df_temp.index),
    data = [df_temp.mean(axis=1), df_temp.quantile(q=0.025, axis=1), df_temp.quantile(q=0.975, axis=1)]).T
df_raw = df_read(df_raw, 'PCE-Uniform', 2, 'uniform', 11)

fileread = read_total_effects(fpath_save, product_uniform='exact')
df_temp = pd.DataFrame.from_dict(fileread[fileread.files[-1]][()][f'nsample_{nsample}']).T
df_exact = pd.DataFrame(index=['ST', 'ST_lower', 'ST_upper'], columns=list(df_temp.index),
    data = [df_temp.mean(axis=1), df_temp.quantile(q=0.025, axis=1), df_temp.quantile(q=0.975, axis=1)]).T
df_exact = df_read(df_exact, 'PCE-Exact', 2, 'exact', 11)

# Combine the total effects into one set.
# model_group and the type of calculation to the dataframe
df_exact['Model_group'] = np.arange(df_exact.shape[0])
df_raw['Model_group'] = None

# set parameter groups
for ii in range(df_exact.shape[0]):
    param = df_exact.Parameters[ii]
    df_raw.loc[df_raw[df_raw.Parameters==param].index, 'Model_group'] = df_exact.Model_group[ii]
    
df_plot = df_exact.filter(items=['Parameters', 'ST', 'ST_lower', 'ST_upper'])
df_plot.rename(columns={'ST': 'ST_exact'})

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