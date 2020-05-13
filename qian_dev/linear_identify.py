#import 
import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform
from basic.boots_pya import least_squares, fun, pce_fun
from basic.partial_rank import partial_rank


# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-ranges.csv'
# adjust parameters according to linear correlations
parameters = pd.read_csv(filename)
param_names = parameters.Veneer_name.values
num_paras = parameters.shape[0]

def mian_ratios(I):
     # define functions
    def first_order(range1, range2):
        param1 = np.square(range1[1] - range1[0]) / np.square(range1.sum())
        param2 = np.square(range2[1] - range2[0]) / np.square(range2.sum())
        ratio = param1 / param2
        return ratio
    
    main_effect, _, _, interaction_effects = pya_sens(I, True)
    # define functions
    def interact_order(range1, range2):
        ratio = 3 * np.square(range2.sum()) / np.square(range2[1] - range2[0])
        return ratio
    
    analytical_main = pd.DataFrame(data=np.zeros((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)
    analytical_second = pd.DataFrame(data=np.zeros((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)
    # for second: ratio = D_row / D_rwo_col
    numerical_main = pd.DataFrame(data=np.zeros((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)
    numerical_second = pd.DataFrame(data=np.zeros((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)                    
                                    
    # main_effect, total_effect = pya_sens() # numerical sensitivity indices 
    k = 22                                                         
    for i in range(num_paras):
        for j in range(i, num_paras):
            range1 = parameters.loc[i, 'min':'max'].values
            range2 = parameters.loc[j, 'min':'max'].values
            loc_upper = [param_names[i], param_names[j]]
            analytical_main.loc[loc_upper[0], loc_upper[1]] = first_order(range1, range2)
            numerical_main.loc[loc_upper[0], loc_upper[1]] = (main_effect[i] / main_effect[j])
            
        if i > 0:
            for jj in range(i):
                loc_lower = [param_names[i], param_names[jj]]
                analytical_second.loc[loc_lower[0], loc_lower[1]] = interact_order(range1, range2)
                analytical_second.loc[loc_lower[1], loc_lower[0]] = interact_order(range2, range1)
                numerical_second.loc[loc_lower[0], loc_lower[1]] = main_effect[i] / interaction_effects[k]
                numerical_second.loc[loc_lower[1], loc_lower[0]] = main_effect[jj] / interaction_effects[k]
                k += 1
                
    return analytical_main, analytical_second, numerical_main, numerical_second
# End mian_ratios()

    

def pya_sens(I, interaction=False):
    # import the original parameter setsi
    fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
    filename = f'{fpath}parameter-ranges.csv'
    ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
    univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(0, ranges.shape[0]//2)]
    variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

    filename = f'{fpath}test_qian.csv'
    data = np.loadtxt(filename, delimiter=",", skiprows=1)[:, 1:]
    samples = data[:,:22].T
    values = data[:,22:]
    values = values[:,:1]# focus on first qoi
    ntrain_samples = 552
    poly, _ = pce_fun(variable, samples, values, ntrain_samples, boot_ind=I)
    main_effect, total_effect = pya.get_main_and_total_effect_indices_from_pce(poly.get_coefficients(),poly.get_indices())
    if interaction:
        interaction_terms, interaction_effects = pya.get_sobol_indices(poly.get_coefficients(),poly.get_indices(),max_order=2)
        return main_effect, total_effect, interaction_terms, interaction_effects
    else:
        return main_effect, total_effect
    # End if
# End pya_sens()

# ration from main effects
# confidence intervals of ratios using bootstrap

ntrain_samples = 552
num_boots = 10
analytical_main_boot = np.zeros((num_boots, num_paras, num_paras))
numerical_main_boot, analytical_second_boot, numerical_second_boot = np.zeros_like(analytical_boot), \
                                                                np.zeros_like(analytical_boot), \
                                                                np.zeros_like(analytical_boot)
for bt in range(num_boots):
    I = np.random.randint(0, ntrain_samples, ntrain_samples)
    analytical_main, analytical_second, numerical_main, numerical_second = mian_ratios(I)
    analytical_main_boot[bt, :, :] = analytical_main.values
    analytical_second_boot[bt, :, :] = analytical_second.values
    numerical_main_boot[bt, :, :] = analytical_main.values
    numerical_second_boot[bt, :, :] = analytical_second.values

main_ci = np.quantile(numerical_boot, q=[0.025, 0.975], axis=0)
total_ci = np.quantile(total_boot, q=[0.025, 0.975], axis=0)

fpath = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'

# calculate the difference bewteeen numerical and analytical results
main_difference = np.array([np.abs(numerical_boot[ii, :] - analytical_boot[0, :]) / analytical_boot[0, :] * 100 
                            for ii in range(num_boots)])
# counts the frequency of numerical ratios with 10% difference compared with the analytical results
condition_counts = np.zeros(list(main_difference.shape)[1:])
for ii in range(list(condition_counts.shape)[0]):
    for jj in range(ii, list(condition_counts.shape)[1]):
        condition_counts[ii, jj] = np.count_nonzero(main_difference[:, ii, jj] <= 20)
        condition_counts[jj, ii] = np.count_nonzero(main_difference[:, jj, ii] <= 50)


para_dict = {ii : [] for ii in para_names}
for ii in range(num_paras):
    for jj in range(ii+1, num_paras):    
        counts_main = condition_counts[ii, jj] 
        counts_second = condition_counts[jj, ii] 
        if (counts_main >= 800) & (counts_second >= 500):
            para_dict[param_names[ii]].append(param_names[jj])
            para_dict[param_names[jj]].append(param_names[ii])
