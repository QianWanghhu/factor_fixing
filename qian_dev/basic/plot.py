# UTF-8 
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_cv(df, save_fig, save_path, fig_name, log_scale):
    """
    Parameters:
    ===========
    df : DataFrame
    save_fig : Bool
    save_path : String

    Resutls:
    ========

    """
    # set ticks for xlabel
    index_temp = list(df.index)
    labels = [i.split('_')[1] for i in index_temp]
    sns.set_style('darkgrid')
    _ = plt.figure(figsize = (8, 6))
    cols = df.columns
    # plt.errorbar(np.arange(df.shape[0]), df['mean'], df['std'], linestyle ='None', marker = 'd');
    plt.errorbar(np.arange(df.shape[0]), df['mean'], yerr = df.loc[:, cols[1:]].T.values, linestyle ='None', marker = 'd');
    if log_scale == True: plt.yscale('log')
    plt.xticks(ticks = np.arange(df.shape[0]), labels = labels, fontsize = 14, rotation=90);
    plt.yticks(fontsize = 14)
    plt.xlabel('Sample size', fontsize = 16)
    plt.ylabel('Error (relative RMSE)', fontsize = 16);
    if save_fig:
        plt.savefig(f'{save_path}{fig_name}.png', dpi = 300, format = 'png')
    else:
        plt.show()


# sensitivity plot
def short_name(df):
    fp = '../data/'
    name_df = pd.read_csv(f'{fp}parameter.csv')
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