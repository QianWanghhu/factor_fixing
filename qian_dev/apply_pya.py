import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform
from basic.boots_pya import least_squares, fun
from basic.partial_rank import partial_rank

# define the distribution of the first parameter
from pyapprox.random_variable_algebra import product_of_independent_random_variables_pdf
# import the original parameter sets
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/pya_related/'
filename = f'{fpath}parameter-ranges.csv'
raw_parameters = pd.read_csv(filename)
ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(ranges.shape[0]//2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

filename = f'{fpath}test_qian.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
samples = data[:,:22].T
values = data[:,22:]
values = values[:,:1]# focus on first qoi
len_params = samples.shape[0]

# Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking
n_strat, n_end, n_setp = [276, 828, 46]
# loops of fun
errors_cv_all = {}
errors_bt_all = {}
partial_results = {}
for i in range(n_strat, n_end+1, n_setp):
    errors_cv, errors_bt, sensitivity_indices = fun(variable, samples, values, 
                                                    nboot=1000, ntrain_samples=i)
    # partial ranking
    sensitivity_indices = np.array(sensitivity_indices)
    sa_shape = sensitivity_indices.shape[0:2]
    sensitivity_indices = sensitivity_indices.reshape((sa_shape))
    rankings = partial_rank(sensitivity_indices,len_params, conf_level=0.95)
    partial_results[f'nsample_{i}'] = rankings
    errors_cv_all[f'nsample_{i}'] = errors_cv
    errors_bt_all[f'nsample_{i}'] = errors_bt
# End for

fpath = 'D:/cloudStor/Research/pce_fixing/output/0709_ave_annual/'
filename = f'{fpath}adaptive-cv-data3.npz'
np.savez(filename,errors_cv=errors_cv_all,errors_bt=errors_bt_all,sensitivity_indices=partial_results)

def analytical_sa(save_bool=False):
    names = pd.read_csv(f'{fpath}parameter-ranges.csv', sep=',')['Veneer_name'].values
    _, _, main_sensitivity, sensitivity_indices = fun(variable, samples, values, nboot=1000, ntrain_samples=552)
    sa_df = pd.DataFrame(data=np.zeros(shape=(len_params, 4)), index=names, columns=['S1', 'S1_conf', 'ST', 'ST_conf'])
    from scipy.stats import norm
    Z = norm.ppf(0.5 + 0.95 / 2)
    sa_df['S1'] = np.mean(main_sensitivity, axis=0)
    sa_df['ST'] = np.mean(sensitivity_indices, axis=0)
    sa_df['S1_conf'] = Z * np.std(main_sensitivity, axis=0, ddof=1)
    sa_df['ST_conf'] = Z * np.std(sensitivity_indices, axis=0, ddof=1)

    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from SALib.plotting.bar import plot as barplot
    sns.set(style="whitegrid")
    rc("text", usetex=False)
    fig = plt.figure(figsize=(6, 4))
    ax = barplot(sa_df)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Main / Total effects')
    if save_bool:
        fpath_save = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
        plt.savefig(f'{fpath_save}sa_PCE_analytical.png', format='png', dpi=300)
    else:
        return fig

analytical_sa(save_bool=False)

