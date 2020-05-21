import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform, norm
from basic.boots_pya import least_squares, fun, pce_fun

fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-ranges.csv'
ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(0, ranges.shape[0]//2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)
filename = f'{fpath}test_qian.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
samples = data[:,:22].T
values = data[:,22:]
values = values[:,:1]# focus on first qoi
ntrain_samples = 552
poly, _ = pce_fun(variable, samples, values, ntrain_samples)

from pyapprox.probability_measure_sampling import generate_independent_random_samples
import SALib
from SALib.util import read_param_file
from SALib.sample import saltelli, sobol_sequence
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

problem = read_param_file(f'{fpath}problem.txt', delimiter=',')
Nsamples = 2000
sample_saltelli = np.round(saltelli.sample(problem, 
                                N=Nsamples, 
                                skip_values=10, 
                                calc_second_order=False, 
                                seed=121), 4)
y_true = np.round(poly(sample_saltelli.T), 4).reshape(sample_saltelli.shape[0])

for i in range(index_product.shape[0]):
    index_temp = index_product[i]
    sample_saltelli[:, index_temp[0]] = np.prod(sample_saltelli[:, index_temp], axis=1)
    sample_saltelli[:, index_temp[1:]] = 1
y_product = np.round(poly(sample_saltelli.T), 4).reshape(sample_saltelli.shape[0])

Y = np.round(poly(sample_saltelli.T), 4).reshape(sample_saltelli.shape[0])
sa = sobol.analyze(problem, Y, calc_second_order=False, 
            num_resamples=100, conf_level=0.95, seed=88)

main_effect_pce, total_effect_pce = \
    pya.get_main_and_total_effect_indices_from_pce(
        poly.get_coefficients(),poly.get_indices()
        )
sa_df['S1_PCE'] = main_effect_pce.reshape(main_effect_pce.shape[0])
sa_df['ST_PCE'] = main_effect_pce.reshape(total_effect_pce.shape[0])
sa_df.index.name = 'param_name'

# with adjustment to parameter ranges
index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17, 18], 
                         [6, 5, 7], 
                         [20, 19],
                         ])

fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}problem_adjust.txt'
problem_adjust = read_param_file(filename, delimiter=',')

def samples_product():
    Nsamples = 2000
    sample_new = np.round(saltelli.sample(problem, 
                                    N=Nsamples, 
                                    calc_second_order=False, 
                                    skip_values=10,
                                    deleted=False,
                                    product_dist=index_product,
                                    problem_adjust=problem_adjust,
                                    seed=121), 4)
    df = sample_new
    return df

# sampling with the only ranges of parameters changed
def ranges_adjust():
    Nsamples = 2000
    sample_range_change = np.round(saltelli.sample(problem_adjust, 
                                    N=Nsamples, 
                                    calc_second_order=False, 
                                    seed=121), 4)
    df = sample_range_change
    return df

# expand the samples with other parameters set to a fixed value
df = samples_product()

samples_expand = np.ones(shape=(df.shape[0], problem['num_vars']))
all_names = problem['names']
adjust_names = problem_adjust['names']
for ii in range(len(adjust_names)):
    col = all_names.index(adjust_names[ii])
    # print(ii, adjust_names[ii], col, all_names[col])
    samples_expand[:, col] = df[:, ii]
y_range_change = np.round(poly(samples_expand.T), 4).reshape(list(samples_expand.shape)[0])

sa = sobol.analyze(problem_adjust, y_range_change, calc_second_order=False, 
            num_resamples=100, conf_level=0.95, seed=88)

# filename = f'{fpath}problem_truncate.txt'
# problem_truncate = read_param_file(filename, delimiter=',')
# plot using plot in SALib
sns.set(style="whitegrid")
rc("text", usetex=False)
sa_df = pd.DataFrame.from_dict(sa)
sa_df.index = problem['names']
fig = plt.figure(figsize=(6, 4))
ax = barplot(sa_df)
ax.set_xlabel('Parameters')
ax.set_ylabel('Main / Total effects')
fpath_save = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
plt.savefig(f'{fpath_save}sa_sampling_raw.png', format='png', dpi=300)



