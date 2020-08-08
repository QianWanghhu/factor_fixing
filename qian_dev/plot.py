# UTF-8 
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error_cv(df, save_fig, save_path, fig_name):
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
    plt.errorbar(np.arange(df.shape[0]), df['mean'], df['std'], linestyle ='None', marker = 'd');
    plt.xticks(ticks = np.arange(16), labels = labels, fontsize = 14);
    plt.yticks(fontsize = 14)
    plt.xlabel('Sample size', fontsize = 16)
    plt.ylabel('Error (relative RMSE)', fontsize = 16);
    if save_fig:
        plt.savefig(f'{save_path}{fig_name}.png', dpi = 300, format = 'png')
    else:
        plt.show()