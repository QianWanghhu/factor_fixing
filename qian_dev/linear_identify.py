#import 
import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform
from basic.boots_pya import least_squares, fun, pce_fun
from basic.partial_rank import partial_rank
import matplotlib.pyplot as plt
import seaborn as sns


def mian_ratios(I):
  
    main_effect, _, _, interaction_effects = pya_sens(I, True)
    # for second: ratio = D_row / D_rwo_col
    numerical_main = pd.DataFrame(data=np.ones((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)
    numerical_second = pd.DataFrame(data=np.ones((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)                    
    k = 22                                                         
    for i in range(num_paras):
        for j in range(i, num_paras):
            loc_upper = [param_names[i], param_names[j]]
            numerical_main.loc[loc_upper[0], loc_upper[1]] = (main_effect[i] / main_effect[j])
            
        if i > 0:
            for jj in range(i):
                # row / col
                loc_lower = [param_names[i], param_names[jj]]
                numerical_second.loc[loc_lower[0], loc_lower[1]] = main_effect[i] / interaction_effects[k]
                numerical_second.loc[loc_lower[1], loc_lower[0]] = main_effect[jj] / interaction_effects[k]
                k += 1
                
    return numerical_main, numerical_second
# End mian_ratios()

def analytical_ratios():
    # define functions
    def interact_order(range1, range2):
        ratio = 3 * np.square(range2.sum()) / np.square(range2[1] - range2[0])
        return ratio
    # End interact_order()
    def first_order(range1, range2):
        param1 = np.square(range1[1] - range1[0]) / np.square(range1.sum())
        param2 = np.square(range2[1] - range2[0]) / np.square(range2.sum())
        ratio = param1 / param2
        return ratio
    # End first_order()

    analytical_main = pd.DataFrame(data=np.ones((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)
    analytical_second = pd.DataFrame(data=np.ones((num_paras, num_paras)), 
                                    index=param_names, columns=param_names)
    for i in range(num_paras):
        for j in range(i, num_paras):
            range1 = parameters.loc[i, 'min':'max'].values
            range2 = parameters.loc[j, 'min':'max'].values
            loc_upper = [param_names[i], param_names[j]]
            analytical_main.loc[loc_upper[0], loc_upper[1]] = first_order(range1, range2)
            
        if i > 0:
            for jj in range(i):
                loc_lower = [param_names[i], param_names[jj]]
                range2 = parameters.loc[jj, 'min':'max'].values
                analytical_second.loc[loc_lower[0], loc_lower[1]] = interact_order(range1, range2)
                # analytical_second.loc[loc_lower[1], loc_lower[0]] = interact_order(range2, range1)
                
    return analytical_main, analytical_second


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


# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-ranges.csv'
# adjust parameters according to linear correlations
parameters = pd.read_csv(filename)
param_names = parameters.Veneer_name.values
num_paras = parameters.shape[0]

# confidence intervals of ratios using bootstrap
ntrain_samples = 552
num_boots = 1000
numerical_main_boot = np.ones((num_boots, num_paras, num_paras))
numerical_second_boot = np.ones_like(numerical_main_boot)
for bt in range(num_boots):
    I = np.random.randint(0, ntrain_samples, ntrain_samples)
    numerical_main, numerical_second = mian_ratios(I)
    numerical_main_boot[bt, :, :] = np.round(numerical_main.values, 3)
    numerical_second_boot[bt, :, :] = np.round(numerical_second.values, 3)
# End for
analytical_main, analytical_second = analytical_ratios()
analytical_main = np.round(analytical_main.values, 3)
analytical_second = np.round(analytical_second.values, 3)

fpath = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
# calculate the difference bewteeen numerical and analytical results
main_difference = np.array([np.abs(numerical_main_boot[ii, :] - analytical_main) / analytical_main * 100 
                            for ii in range(num_boots)])
second_difference = np.array([np.abs(numerical_second_boot[ii, :] - analytical_second) / analytical_second * 100 
                            for ii in range(num_boots)])
# save results
param_index = np.arange(0, 22)
index_combs = np.array(list(combinations(param_index, 2)))
param_combs = param_combs_match(index_combs)
df_ratios = numerical_second_boot[:, index_combs[:, 0], index_combs[:, 1]]
df_ratios = pd.DataFrame(data=df_ratios.T, index=param_combs)
df_ratios.index.name = 'param_name'
df_quantiles = df_ratios.quantile(q=[0.05, 0.95, 0.025, 0.975], axis=1)
df_quantiles.index.name = 'quantile'
df_quantiles.loc['mean', :] = df_ratios.mean(axis=1)
df_quantiles.loc['std', :] = df_ratios.std(axis=1)
df_quantiles.loc['analytical', :] = analytical_second[index_combs[:, 0], index_combs[:, 1]]
# df_quantiles.to_csv(f'{fpath}second_quantiles_lower.csv', index_label='parameters', sep=',')

df_ratios = df_ratios.transpose()
df_ratios.to_csv(f'{fpath}numerical_second_upper.csv', sep=',')

df = pd.DataFrame(data=analytical_second, index=param_names, columns=param_names)
df.to_csv(f'{fpath}analytical_second.csv', index_label='parameters', sep=',')


# check the significance of ratios and plot
# give the indices of parameters which are of product relations
from itertools import combinations
from matplotlib import rc
rc("text", usetex=False)
index_product = np.array([[0, 1, 2, 3, 9, 10, 11, 16, 17, 18], 
                         [5, 6, 7], 
                         [19, 20],
                         [0, 5, 19, 4, 8, 12, 13, 14, 15, 21]])
index_combs = np.array(list(combinations(index_product[3], 2)))
# index_combs = np.vstack((index_combs, index_product[-1]))
num_subplots = index_combs.shape[0]

def param_combs_match(index_combs):
    num_subplots = index_combs.shape[0]
    param_combs = []
    for i in range(num_subplots):
        params_temp = [parameters.loc[index_combs[i][0], 'Veneer_name'], 
                        parameters.loc[index_combs[i][1], 'Veneer_name']]
        try:
            param_combs.append(f'{params_temp[0]} & {params_temp[1]}')
        except NameError:
            param_combs = [f'{params_temp[0]} & {params_temp[1]}']
    return param_combs
# End param_combs_match()

data_plot = numerical_main_boot[:, index_combs[:, 0], index_combs[:, 1]]
df_ratios = pd.DataFrame(data=data_plot, columns=param_combs)
fpath = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
sns.set(style='whitegrid')
fig = plt.figure(figsize=(25, 4))
for ii in range(35, 42):
    row = np.floor(ii / 5) + 1
    col = ii % 7 + 1
    print(row, col)
    ax = plt.subplot(1, 7, col)
    ax = sns.violinplot(data=df_ratios.loc[:, param_combs[ii]])
    ax.plot(analytical_main[index_combs[ii][0], index_combs[ii][1]], 
            'ro', markersize=4, label='ratio_analytical')
    ax.set_xticklabels([param_combs[ii]], fontsize=8);
    if ii == 0:
        ax.set_ylabel('ratio_main_effects')
plt.legend()
plt.savefig(f'{fpath}main_nonlinear{row}.png', format='png', dpi=300)

# look at the difference of ratios
fnames = ['numerical_main_boot.csv', 'numerical_second_upper.csv', 'numerical_second_lower.csv']
fname_quantiles = ['main_quantiles.csv', 'second_quantiles_upper.csv', 'second_quantiles_lower.csv']
index = [('main', 'mean'), ('main', 'std'),
        ('second_upper', 'mean'), ('second_upper', 'std'),
        ('second_lower', 'mean'), ('second_lower', 'std')]
cols = pd.read_csv(f'{fpath}{fname_quantiles[ii]}', index_col='parameters').columns
ratio_difference = pd.DataFrame(data = np.zeros(shape=(6, 231)), 
                                index= pd.MultiIndex.from_tuples(index), 
                                columns=cols)
for ii in range(len(fnames)):
    main_quantiles = pd.read_csv(f'{fpath}{fname_quantiles[ii]}', index_col='parameters')
    numerical_main = pd.read_csv(f'{fpath}{fnames[ii]}', index_col='param_name')
    df_difference = numerical_main.apply(lambda x: (x- main_quantiles.loc['analytical', :])/main_quantiles.loc['analytical', :], axis=0).abs()

    ratio_difference.loc[index[ii*2], :] = df_difference.mean(axis=1).values
    ratio_difference.loc[index[ii*2+1], :] = df_difference.std(axis=1).values
# End for
ratio_difference.to_csv(f'{fpath}ratio_difference.csv')