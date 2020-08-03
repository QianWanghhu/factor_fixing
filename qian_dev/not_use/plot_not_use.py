import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from SALib.plotting.bar import plot as barplot
from matplotlib import ticker
from scipy.stats import norm

rc("text", usetex=False)
def plot(sa_df, sav=False, fpath_save, figname=None):
    sns.set_style('whitegrid')
    _ = plt.figure(figsize=(6, 4))
    ax = barplot(sa_df)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Main / Total effects')
    ax.xaxis.set_tick_params(labelsize=12)
    if sav:
        plt.savefig(f'{fpath_save}{figname}', format='png', dpi=300)
# End plot()

def df_format(main_effects, total_effects, variables, param_names, conf_level=0.95):    
    Z = norm.ppf(0.5 + conf_level / 2)
    sa_df = pd.DataFrame(data=np.zeros(shape=(variables.num_vars(), 4)), 
                        columns=['S1', 'S1_conf', 'ST', 'ST_conf'])
    main_effects = np.array(main_effects)
    total_effects = np.array(total_effects)
    sa_df.loc[:, 'S1'] = main_effects.mean(axis=0)
    sa_df.loc[:, 'S1_conf'] = Z * main_effects.std(axis=0, ddof=1)
    sa_df.loc[:, 'ST'] = total_effects.mean(axis=0)
    sa_df.loc[:, 'ST_conf'] = Z * total_effects.std(axis=0, ddof=1)
    sa_df.index = param_names
    return sa_df
# End df_format()

# scatter plot
def scatter_plot(x, y, save_name=None):   
    _ = plt.figure(figsize=(6, 4))
    ax = sns.scatterplot(x, y)
    ax.set_xlabel('Model Outputs', fontsize=12)
    ax.set_ylabel('PCE Outputs (kg)', fontsize=12)
    # ax.set_yticklabels(size=10)
    # ax.set_xticklabels(size=10)
    if save_name is not None:
        plt.savefig(f'{fpath_save}{save_name}', format='png', dpi=300)
# End scatter_plot()
