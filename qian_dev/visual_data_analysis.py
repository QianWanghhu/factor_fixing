import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
rc("text", usetex=False)

# prepare the data for plot: confidence intervals and the coefficient of variation
fpath_save = 'D:/cloudStor/Research/pce_fixing/output/0709_ave_annual/full_params/'
fpath_save_average = fpath_save + 'fix_average/'
filename = ['low_95', 'up_95', 'co_var', 'conf_width_relative_95', 'dvalue', 'pvalue']
y_stats = {"full_25": 36581791.287638865,
            "full_75": 44691132.165343955,
            "full_average": 40969683.0438159,
            "reduce_25": 36637156.96412801,
            "reduce_75": 44637839.449896894,
            "reduce_average": 40971895.7445479
            }
# get data in the corresponding column
num_params = 22
col_filter = ['nsample_552']
df = pd.DataFrame(index=np.arange(num_params), columns=filename)
x_range = [0, num_params + 1]
for f in filename:
    df[f] = pd.read_csv(f'{fpath_save}{f}.csv').loc[:, col_filter]
df.index = np.arange(df.shape[0], 0, -1)
df.index.name = 'Num of parameters fixed'
df.reset_index(inplace=True)
# read data from the fix_average directory
df_average = pd.DataFrame(index=np.arange(num_params), columns=filename)
for f in filename:
    df_average[f] = pd.read_csv(f'{fpath_save_average}{f}.csv').loc[:, col_filter]
df_average.index = np.arange(df_average.shape[0], 0, -1)
df_average.index.name = 'Num of parameters fixed'
df_average.reset_index(inplace=True)

# Line plot
fig, axes = plt.subplots(3, 1, sharex=True)
ax = df.plot(x='Num of parameters fixed', y=filename[0:2], 
            figsize=(7, 15), marker='o', ax=axes[0])
df_average.plot(x='Num of parameters fixed', y=filename[0:2], 
            marker='d', ax=axes[0])  
for i in range(3): ax.axhline(c='grey',linestyle='--', y=list(y_stats.values())[i])          
ax.set_ylabel('TSS load(Kg)')
ax.set_ylim(3e7, 6e7)
ax.legend(['low_default', 'up_default', 'low_average', 'up_average'], fontsize=10)
ax1 = df.plot('Num of parameters fixed', filename[2], 
            ax=axes[1], marker='o')
df_average.plot('Num of parameters fixed', filename[2], 
            ax=axes[1], marker='d')
ax1.set_ylim(-0.01, 0.16)        
ax1.set_ylabel('Coefficient of variation')
ax1.legend(['default', 'average'], fontsize=10)
ax2 = df.plot('Num of parameters fixed',filename[4:], marker='o', ax=axes[2])
df_average.plot('Num of parameters fixed',filename[4:], marker='d', ax=axes[2])
ax2.set_xlim(*x_range)
ax2.set_yticks([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_ylabel('KS_statistic')
ax2.legend(['D_default', 'pvalue_default', 'D_average', 'pvalue_average'], fontsize=10)
plt.savefig(f'{fpath_save}measures_combine.png', format='png', dpi=300, bbox_inches='tight')

# heatmap
df = pd.read_csv(f'{fpath_save}normalized_median.csv')
df.drop(columns={'Unnamed: 0'}, inplace=True)
sns.set(font_scale=1.2)
def plot_heatmap(df, save=False, save_name=None):    
    df.index = np.arange(df.shape[0], 0, -1)
    df = df.rename(columns={col: col.split('_')[1] for col in df.columns})
    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df.loc[:, df.columns[0:-4]], annot=True, fmt=".3f", annot_kws={"size": 10})
    ax.set_xlabel('Number of sample size')
    ax.set_ylabel('Number of parameter fixed')
    if save:
        plt.savefig(f'{fpath_save}{save_name}.png', format='png', dpi=300)
plot_heatmap(df, save=True, save_name='normalized_median')


# sensitivity plot
def short_name(df):
    fp = '../data/'
    name_df = pd.read_csv(f'{fp}parameter-reimplement.csv')
    df['short_name'] = None
    for ii in range(df.shape[0]):
        df.loc[ii, 'short_name'] = name_df[name_df.Veneer_name == df.Parameters[ii]]['short_name'].values 
    return df

# clean the dataframe ordered by the sampling-based sensitivity indices
fpath_save = '../output/paper/'
filename = ['sa_samples_product', 'sa_pce_uniform', 'sa_pce_beta']
df_raw = pd.read_csv(f'{fpath_save}{filename[1]}.csv').filter(
                    items=['Unnamed: 0', 'ST', 'ST_conf'])
df_raw.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
# df_raw.set_index('Parameters', drop=True)

df_sampling = pd.read_csv(f'{fpath_save}{filename[0]}.csv').filter(items=
                    ['Unnamed: 0', 'ST', 'ST_conf'])
df_sampling.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
df_sampling = df_sampling.sort_values(by='ST', ascending=False)
index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17], 
                         [6, 5, 7], 
                         [19, 20],
                         ])

# model_group and the type of calculation to the dataframe
df_sampling['Model_group'] = np.arange(df_sampling.shape[0])
df_sampling['Type'] = 'Sampling'
df_sampling['Type_num'] = 0
df_raw['Model_group'] = None
df_raw['Type'] = 'PCE-Uniform'
df_raw['Type_num'] = 2
df_beta = pd.read_csv(f'{fpath_save}{filename[2]}.csv').filter(items=
                ['Unnamed: 0', 'ST', 'ST_conf'])
df_beta.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
df_beta['Type'] = 'PCE-Beta'
df_beta['Type_num'] = 1

for ii in range(df_sampling.shape[0]):
    param = df_sampling.Parameters[ii]
    df_raw.loc[df_raw[df_raw.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
    df_beta.loc[df_beta[df_beta.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
if ((df_raw.shape[0]) == (df_sampling.shape[0])) == True:
    df_plot = pd.concat([df_raw, df_sampling, df_beta])

df_plot = df_plot.sort_values(by=['Model_group', 'Type_num', 'ST'], ascending=[True, True, False]).reset_index(drop=True)


df_plot = short_name(df_plot)

names_update = ['bankErosionCoeff', 'HillslopeFineSDR', 'Gully_Management_Practice_Factor']
new_short_name = ['new_BEC', 'new_HFSDR', 'new_GMPF']
for ii in range(len(names_update)):
    df_plot.loc[df_plot[(df_plot.Parameters==names_update[ii]) & 
            (df_plot.Type=='Sampling')].index, 'short_name'] = new_short_name[ii]


# the new style of plot for sensitivity
df_plot = df_sampling.filter(items=['Parameters', 'ST', 'ST_conf'])
df_plot.rename(columns={'ST': 'ST_sampling'})
df_plot['ST_Beta'], df_plot['ST_conf_Beta'] = df_beta.ST, df_beta.ST_conf
df_plot['ST_Uniform'], df_plot['ST_conf_Uniform'] = df_raw.ST, df_raw.ST_conf

df_plot = short_name(df_plot)
names_update = ['bankErosionCoeff', 'HillslopeFineSDR', 'Gully_Management_Practice_Factor']
new_short_name = ['new_BEC', 'new_HFSDR', 'new_GMPF']
for ii in range(len(names_update)):
    df_plot.loc[df_plot[df_plot.Parameters==names_update[ii]].index, 'short_name'] = new_short_name[ii]

# save df_plot
df_plot.to_csv(f'{fpath_save}/sa_fig2.csv')


# clean the dataframe ordered by the sampling-based sensitivity indices
fpath_save = '../output/paper/'
filename = ['sa_pce_raw', 'sa_samples_product']
df_raw = pd.read_csv(f'{fpath_save}{filename[0]}.csv').filter(
                    items=['Unnamed: 0', 'ST', 'ST_conf'])
df_raw.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)

df_sampling = pd.read_csv(f'{fpath_save}{filename[1]}.csv').filter(items=
                    ['Unnamed: 0', 'ST', 'ST_conf'])
df_sampling.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
df_sampling = df_sampling.sort_values(by='ST', ascending=False)

# model_group and the type of calculation to the dataframe
df_sampling['Model_group'], df_sampling['Type'], df_sampling['Type_num'] = \
    np.arange(df_sampling.shape[0]), 'Sampling', 0
df_raw['Model_group'], df_raw['Type'], df_raw['Type_num'] = None, 'PCE', 1

for ii in range(df_sampling.shape[0]):
    param = df_sampling.Parameters[ii]
    df_raw.loc[df_raw[df_raw.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
    # df_beta.loc[df_beta[df_beta.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
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

df_plot.to_csv(f'{fpath_save}/sa_fig1.csv')

