import numpy as np
import pandas as pd
import pyapprox as pya
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from scipy.stats import beta, uniform, norm as beta, uniform, norm
from basic.utils import variables_prep
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from SALib.plotting.bar import plot as barplot
from SALib.util import read_param_file
from basic.boots_pya import least_squares, pce_fun, fun
from basic.partial_rank import partial_rank
from basic.utils import names_match

rc("text", usetex=False)
def plot(sa_df, sav=False, figname=None):
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(6, 4))
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
from matplotlib import ticker
def scatter_plot(x, y, save_name=None):   
    fig = plt.figure(figsize=(6, 4))
    ax = sns.scatterplot(x, y)
    ax.set_xlabel('Model Outputs', fontsize=12)
    ax.set_ylabel('PCE Outputs (kg)', fontsize=12)
    # ax.set_yticklabels(size=10)
    # ax.set_xticklabels(size=10)
    if save_name is not None:
        plt.savefig(f'{fpath_save}{save_name}', format='png', dpi=300)
# End scatter_plot()

fpath_save = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
# import samples and values
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/pya_related/'
filename = f'{fpath}test_qian.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
samples = data[:,:22].T
values = data[:,22:]
values = values[:,:1]# focus on first qoi
# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
filename = f'{fpath}parameter-ranges.csv'
variable = variables_prep(filename, product_uniform=False)
param_all = pd.read_csv(filename).loc[:, 'Veneer_name'].values
index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17], 
                         [6, 5, 7], 
                         [19, 20],
                         ])

# define variables with Beta distribution
filename = f'{fpath}parameter-adjust.csv'
variable_adjust = variables_prep(filename, product_uniform=True)
param_adjust = pd.read_csv(filename)
beta_index = param_adjust[param_adjust['distribution']== 'beta'].\
            index.to_list()

samples_adjust = np.copy(samples)
pars_delete = []
for ii in range(list(index_product.shape)[0]):
    index_temp = index_product[ii]
    samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
    # samples_adjust[index_temp[1:], :] = 1
    pars_delete.extend(index_temp[1:])
samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)

ntrain_samples = 552
# poly_beta, error = pce_fun(variable_adjust, samples_adjust, 
#                     values, ntrain_samples, degree=3)
# approx_values = poly_beta(samples_adjust)                    
# error_vali = np.linalg.norm(approx_values - values) / np.linalg.norm(values)
nboot = 1000
error_cv, error_bt, main_effects, total_effects = fun(variable, samples, 
                                            values, degree=2, nboot=nboot, 
                                            ntrain_samples=ntrain_samples)
print(np.mean(error_cv), np.std(error_cv))

sa_df = df_format(main_effects, total_effects, variable, 
                param_all, conf_level=0.95)
sa_df.to_csv(f'{fpath_save}sa_pce_raw.csv', index=True)

plot(sa_df) # , sav=True, figname='sa_PCE_Beta_degree2_uniform.png'

total_effects = np.array(total_effects)
total_effects = total_effects.reshape(nboot, variable_adjust.num_vars())
rankings = partial_rank(total_effects, variable_adjust.num_vars())
rank_names = names_match(rankings, param_adjust.loc[:, 'Veneer_name'].values)