import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
rc("text", usetex=False)
# prepare the data for plot: confidence intervals and the coefficient of variation
fpath_save = 'D:/cloudStor/Research/pce_fixing/output/0709_ave_annual/reduce_params/'
filename = ['low_95', 'up_95', 'co_var', 'conf_width_relative_95', 'dvalue', 'pvalue']
# get data in the corresponding column
num_params = 11
df = pd.DataFrame(index=np.arange(num_params), columns=filename)
col_filter = ['nsample_178']
for f in filename:
    df[f] = pd.read_csv(f'{fpath_save}{f}.csv').loc[:, col_filter]
df.index = np.arange(df.shape[0], 0, -1)
df.index.name = 'Num of parameters fixed'
df.reset_index(inplace=True)

fig, axes = plt.subplots(1, 2)
ax = df.plot(x='Num of parameters fixed', y=filename[0:2], 
            figsize=(12, 5), marker='o', ax=axes[0])
ax.set_ylabel('TSS load(Kg)')
ax1 = df.plot('Num of parameters fixed', filename[2:4], 
            secondary_y=True, ax=ax, marker='d')
# # ax1.set_ylabel('CV and width_conf', fontsize=8)
ax2 = df.plot('Num of parameters fixed',filename[4:], marker='o', ax=axes[1])
ax2.set_ylabel('KS_statistic')
ax2.legend(filename[4:])
# plt.savefig(f'{fpath_save}measures.png', format='png', dpi=300)

# heatmap
conf_width_relative.index = np.arange(conf_width_relative.shape[0], 0, -1)
conf_width_relative = conf_width_relative.rename(columns={col: col.split('_')[1] for col in conf_width_relative.columns})
fig = plt.figure(figsize=(12, 8))
ax = sns.heatmap(conf_width_relative.loc[:, conf_width_relative.columns[2:-4]], annot=True, fmt=".3f")
ax.set_xlabel('Number of sample_size')
ax.set_ylabel('Number of parameter fixed')
plt.savefig(f'{fpath_save}conf_width_relative_heatmap.png', format='png', dpi=300)

# sensitivity plot
# clean the dataframe ordered by the sampling-based sensitivity indices
fpath_save = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
filename = ['sa_pce_raw', 'sa_samples_product', 'sa_pce_uniform', 'sa_pce_beta']
df_raw = pd.read_csv(f'{fpath_save}{filename[2]}.csv').filter(
                    items=['Unnamed: 0', 'ST', 'ST_conf'])
df_raw.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
df_raw.set_index('Parameters', drop=True)

df_sampling = pd.read_csv(f'{fpath_save}{filename[1]}.csv').filter(items=
                    ['Unnamed: 0', 'ST', 'ST_conf'])
df_sampling.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
df_sampling.sort_values(by='ST', ascending=False)
index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17], 
                         [6, 5, 7], 
                         [19, 20],
                         ])

# model_group and the type of calculation to the dataframe
df_sampling['Model_group'] = np.arange(df_sampling.shape[0])
df_sampling['Type'] = 'Sampling'
df_sampling['Type_num'] = 0
df_raw['Model_group'] = None
if ((df_raw.shape[0]) == (df_sampling.shape[0])) == True:
    df_raw['Type'] = 'PCE-Uniform'
    df_raw['Type_num'] = 2
    df_beta = pd.read_csv(f'{fpath_save}{filename[3]}.csv').filter(items=
                    ['Unnamed: 0', 'ST', 'ST_conf'])
    df_beta.rename(columns={'Unnamed: 0' : 'Parameters'}, inplace=True)
    df_beta['Type'] = 'PCE-Beta'
    df_beta['Type_num'] = 1
else:
    df_raw['Type'] = 'PCE'
    df_raw['Type_num'] = 1

for ii in range(df_sampling.shape[0]):
    param = df_sampling.Parameters[ii]
    df_raw.loc[df_raw[df_raw.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
    df_beta.loc[df_beta[df_beta.Parameters==param].index, 'Model_group'] = df_sampling.Model_group[ii]
if ((df_raw.shape[0]) == (df_sampling.shape[0])) == True:
    df_plot = pd.concat([df_raw, df_sampling, df_beta])
else:
    for jj in index_product:
        df_raw.loc[jj[1:], 'Model_group'] = df_raw.Model_group[jj[0]]
    df_plot = pd.concat([df_raw, df_sampling])
df_plot = df_plot.sort_values(by=['Model_group', 'Type_num', 'ST'], ascending=[True, True, False]).reset_index(drop=True)

def short_name(df):
    fp = 'D:/cloudStor/Research/pce_fixing/pyfile/pya_related/'
    name_df = pd.read_csv(f'{fp}parameter-ranges.csv')
    df['short_name'] = None
    for ii in range(df.shape[0]):
        df.loc[ii, 'short_name'] = name_df[name_df.Veneer_name == df.Parameters[ii]]['short_name'].values
    
    return df
df_plot = short_name(df_plot)

sns.set_style('whitegrid')
fig = plt.figure(figsize=(8, 6))
if (df_raw.shape[0]) == (df_sampling.shape[0]):
    colors=['b', 'g', 'violet']*11
else:
    colors = tuple(np.where(df_plot.Type_num == 0, 'b', 'g'))
ax = df_plot.plot(x='short_name', y='ST', kind='bar', 
                yerr='ST_conf', color=colors, legend=False)
ax.set_ylabel('Total effects', fontsize=10)
ax.set_xlabel('Parameters', fontsize=10)
ax.tick_params(axis='x', which='major', labelsize=8)           
SPL = mpatches.Patch(color='b', label='Sampling')
PCE_Beta = mpatches.Patch(color='g', label='PCE-Beta') # -Beta
PCE_Uniform = mpatches.Patch(color='violet', label='PCE-Uniform')
ax.legend(handles=[SPL, PCE_Beta, PCE_Uniform], fontsize=8, loc=1) 
# plt.show()
plt.savefig(f'{fpath_save}sentivity_fig2.png', format='png', dpi=300, bbox_inches='tight')
