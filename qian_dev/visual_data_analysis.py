import numpy as np
import pandas as pd
import seaborn as sns
import json

# sensitivity plot
def short_name(df):
    fp = '../data/'
    name_df = pd.read_csv(f'{fp}parameter-reimplement.csv')
    df['short_name'] = None
    for ii in range(df.shape[0]):
        df.loc[ii, 'short_name'] = name_df[name_df.Veneer_name == df.Parameters[ii]]['short_name'].values 
    return df
# End short_name()

def df_read(fpath, fname, result_type, type_num):
    df = pd.read_csv(f'{fpath}{fname}')
    df.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
    df['Type'] = result_type
    df['Type_num'] = type_num
    return df
# End df_read()

# clean the dataframe ordered by the sampling-based sensitivity indices
fpath_save = '../output/paper/'
filename = ['sa_samples_product_test', 'sa_pce_uniform_test', 'sa_pce_beta_test']
df_raw = df_read(fpath_save, f'{filename[1]}.csv', 'PCE-Uniform', 2)
df_sampling = df_read(fpath_save, f'{filename[0]}.csv', 'Sampling', 0)
df_sampling = df_sampling.sort_values(by='ST', ascending=False)
df_beta = df_read(fpath_save, f'{filename[2]}.csv', 'PCE-Beta', 2)

index_product = np.load('../data/index_product.npy', allow_pickle=True)

# model_group and the type of calculation to the dataframe
df_sampling['Model_group'] = np.arange(df_sampling.shape[0])
df_raw['Model_group'] = None

# set parameter groups
for ii in range(df_sampling.shape[0]):
    param = df_sampling.Parameters[ii]
    df_raw.loc[df_raw[df_raw.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
    df_beta.loc[df_beta[df_beta.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
# if ((df_raw.shape[0]) == (df_sampling.shape[0])) == True:
#     df_plot = pd.concat([df_raw, df_sampling, df_beta])

# df_plot = df_plot.sort_values(by=['Model_group', 'Type_num', 'ST'], 
#                             ascending=[True, True, False]).reset_index(drop=True)

df_plot = df_sampling.filter(items=['Parameters', 'ST', 'ST_conf_lower', 'ST_conf_upper'])
df_plot.rename(columns={'ST': 'ST_sampling'})
df_plot['ST_Beta'], df_plot['ST_Beta_conf_lower'], df_plot['ST_Beta_conf_upper'] = \
    df_beta.ST, df_beta.ST_conf_lower, df_beta.ST_conf_upper 
df_plot['ST_Uniform'], df_plot['ST_Uniform_conf_lower'], df_plot['ST_Uniform_conf_upper'] = \
    df_raw.ST, df_raw.ST_conf_lower, df_raw.ST_conf_upper

df_plot = short_name(df_plot)
names_update = ['bankErosionCoeff', 'HillslopeFineSDR', 'Gully_Management_Practice_Factor']
new_short_name = ['new_BEC', 'new_HFSDR', 'new_GMPF']
for ii in range(len(names_update)):
    df_plot.loc[df_plot[df_plot.Parameters==names_update[ii]].index, 'short_name'] = new_short_name[ii]

# save df_plot
df_plot.to_csv(f'{fpath_save}/sa_fig2_0812.csv')


# clean the dataframe ordered by the sampling-based sensitivity indices
filename = ['sa_pce_raw', 'sa_samples_product']
df_raw = df_read(fpath_save, f'{filename[0]}.csv', 'PCE', 1)
df_sampling = df_read(fpath_save, f'{filename[1]}.csv', 'Sampling', 0)
df_sampling = df_sampling.sort_values(by='ST', ascending=False)
# model_group and the type of calculation to the dataframe
df_sampling['Model_group'] = np.arange(df_sampling.shape[0])
df_raw['Model_group'] = None

for ii in range(df_sampling.shape[0]):
    param = df_sampling.Parameters[ii]
    df_raw.loc[df_raw[df_raw.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]

for jj in index_product:
    df_raw.loc[jj[1:], 'Model_group'] = df_raw.Model_group[jj[0]]

df_plot = pd.concat([df_raw, df_sampling])
df_plot = df_plot.sort_values(by=['Model_group', 'Type_num', 'ST'], ascending=[True, True, False]).reset_index(drop=True)
df_plot = short_name(df_plot)

names_update = ['bankErosionCoeff', 'HillslopeFineSDR', 'Gully_Management_Practice_Factor']
new_short_name = ['new_BEC', 'new_HFSDR', 'new_GMPF']
for ii in range(len(names_update)):
    df_plot.loc[df_plot[(df_plot.Parameters==names_update[ii]) & 
            (df_plot.Type=='Sampling')].index, 'short_name'] = new_short_name[ii]

df_plot.to_csv(f'{fpath_save}/sa_fig1_0811.csv')