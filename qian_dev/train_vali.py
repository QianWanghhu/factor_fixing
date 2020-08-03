# Script for preparing training dataset for PCE

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# read model results
filepath = 'output/Run0730/'
# combine TSS results into a file
filenames = os.listdir(filepath)
for fn in filenames:
    if 'Tss' in fn:
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