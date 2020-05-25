# arbitrary polynomial chaos
import numpy as np
import pandas as pd
import pyapprox as pya
from functools import partial
from pyapprox.arbitrary_polynomial_chaos  import *
from pyapprox.variable_transformations import define_iid_random_variable_transformation
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from pyapprox.indexing import compute_hyperbolic_indices
from scipy.stats import beta as beta, uniform
from basic.boots_pya import least_squares, pce_fun

# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-ranges.csv'
ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(0, ranges.shape[0]//2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17, 18], 
                         [6, 5, 7], 
                         [19, 20],
                         ])
# samples_adjust = np.copy(samples)
# pars_delete = []                
# for ii in range(list(index_product.shape)[0]):
#     index_temp = index_product[ii]
#     samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
#     pars_delete.extend(index_temp[1:])
# samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)

# Check whether the Beta distribution is a proper option
from scipy import stats
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-adjust.csv'
param_adjust = pd.read_csv(filename)
locs = np.array(param_adjust.loc[[0, 2, 8], ['min','max']])
locs[:, 1] = locs[:, 1] - locs[:, 0]
param_names = param_adjust.loc[[0, 2, 8], 'Veneer_name'].values
num_samples = 20000
samples_uniform = generate_independent_random_samples(variable, num_samples)
beta_fit = np.zeros(shape=0)
# for ii in range(list(index_product.shape)[0]):
ii = 1
index_temp = index_product[ii]
samples_uniform[index_temp[0], :] = np.prod(samples_uniform[index_temp, :], axis=0)
# End for()
# fit the Beta distribution
rv_product = samples_uniform[index_temp[0]]
beta_aguments = beta.fit(rv_product,floc=locs[ii][0], fscale=locs[ii][1])
# calculate the KS-statistic
num_boot = 1000
ks_stat= []
for i in range(num_boot):
    I = np.random.randint(0, num_samples, 1000)
    ks_stat.append(stats.kstest(rv_product[I], 
                    'beta', args=(beta_aguments)))
ks_stat = np.array(ks_stat)
ks_mean = ks_stat.mean(axis=0)
ks_stat_std = np.std(ks_stat, axis=0)
critical_value = 1.63 / np.sqrt(1000)

# plot CDFs of the fitted Beta distribution and samples
import matplotlib.pyplot as plt
from matplotlib import rc
rc("text", usetex=False)
import seaborn as sns
fpath_save = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
def plot_dists(beta_aguments, rv_product, ii):
    x = np.sort(rv_product)
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs)+1)/float(len(xs))
        return xs, ys
    fig = plt.figure(figsize=(8,6))
    sns.set_style("darkgrid")
    ax = sns.lineplot(x, stats.beta.cdf(x,*beta_aguments), label='Beta')
    ax.plot(*ecdf(x), label='Product of samples')
    ax.set(xlabel=param_names[ii], ylabel='CDF')
    ax.legend()
    # plt.savefig(f'{fpath_save}{param_names[ii]}.png', format='png', dpi=300)
# End plot_dists()
plot_dists(beta_aguments, rv_product, ii)