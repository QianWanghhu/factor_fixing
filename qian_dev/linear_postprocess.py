#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataframe of ratios
def ratio_compare(analytical_results, numerical_results):
    """
    fnames: list of file names, fnames[0]: analytical ratios, fnames[1]: numerical ratios
    """
    ratio_anly = analytical_results
    ratio_num = numerical_results
    num_params = ratio_anly.shape[0]
    para_names = ratio_anly.columns
    # calculate the difference of numerical and analytical ratios
    ratio_difference = (ratio_num - ratio_anly).abs() / ratio_anly * 100
    para_dict = {ii : [] for ii in para_names}

    for ii in range(num_params):
        ind = para_names[ii]
        for jj in range(ii, num_params):    
            col = para_names[jj] 
            ratio = ratio_difference.loc[ind, col]   
            if (ratio <= 10) & (ratio > 0):
                para_dict[ind].append(col)
                para_dict[col].append(ind)
        # End for 
    # End for
    return ratio_difference, para_dict
# End ratio_compare()

fpath = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
fnames = ['ratios_analytical.csv', 'mian_conf_low.csv', 'mian_conf_up.csv']
analytical_results = pd.read_csv(f'{fpath}{fnames[0]}', index_col='para_name')
main_conf_low = pd.read_csv(f'{fpath}{fnames[1]}', index_col='para_name')
main_conf_up = pd.read_csv(f'{fpath}{fnames[2]}', index_col='para_name')
difference_low, para_dict_low = ratio_compare(analytical_results, main_conf_low)
difference_up, para_dict_up = ratio_compare(analytical_results, main_conf_up)



