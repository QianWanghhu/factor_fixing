import numpy as np
import pandas as pd
import pyapprox as pya
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from scipy.stats import beta as beta, uniform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc("text", usetex=False)

from basic.utils import variables_prep

# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
filepath = 'data/'
filename = f'{filepath}parameter-reimplement.csv'
variable = variables_prep(filename, product_uniform=False)
index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17],
                         [6, 5, 7], 
                         [19, 20],
                         ])

# Check whether the Beta distribution is a proper option
filename = f'{filepath}parameter-adjust.csv'
param_adjust = pd.read_csv(filename)
beta_index = param_adjust[param_adjust['distribution']== 'beta'].\
            index.to_list()
locs = np.array(param_adjust.loc[beta_index, ['min','max']])
locs[:, 1] = locs[:, 1] - locs[:, 0]
param_names = param_adjust.loc[beta_index, 'Veneer_name'].values
num_samples = 20000
samples_uniform = generate_independent_random_samples(variable, num_samples)
beta_fit = np.zeros(shape=(len(param_names), 4))

for ii in range(list(index_product.shape)[0]):
    index_temp = index_product[ii]
    samples_uniform[index_temp[0], :] = np.prod(samples_uniform[index_temp, :], axis=0)
    # fit the Beta distribution
    rv_product = samples_uniform[index_temp[0]]
    beta_aguments = beta.fit(rv_product, floc=locs[ii][0], fscale=locs[ii][1])
    beta_fit[ii, :] = np.round(beta_aguments, 4)
    # calculate the KS-statistic
    num_boot = 1000
    ks_stat= []

    for i in range(num_boot):
        I = np.random.randint(0, num_samples, 1500)
        ks_temp = np.array(stats.kstest(rv_product[I], 
                        'beta', args=(beta_aguments)))
        try:
            ks_stat = np.hstack((ks_stat, ks_temp))
        except NameError:
            ks_stat = ks_temp    
    # End for
# End for
# plot CDFs of the fitted Beta distribution and samples

filepath_save = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
def plot_dists(beta_aguments, rv_product, ii):
    x = np.sort(rv_product)
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs)+1)/float(len(xs))
        return xs, ys
    sns.set_style("darkgrid")
    ax = sns.lineplot(x, stats.beta.cdf(x,*beta_aguments), label='Beta', ax=axes[ii])
    ax.plot(*ecdf(x), label='Product of samples')
    ax.set(xlabel=param_names[ii], ylabel='CDF')
    ax.legend()
# End plot_dists()

_, axes = plt.subplots(1, 3, figsize=(24, 5))
for ii in range(list(index_product.shape)[0]):
    index_temp = index_product[ii]
    rv_product = samples_uniform[index_temp[0], :] 
    beta_aguments = beta_fit[ii]
    parameter = param_names[ii]
    plot_dists(beta_aguments, rv_product, ii)
plt.savefig(f'{filepath_save}beta_compare.png', format='png', dpi=300)