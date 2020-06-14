import numpy as np
import pandas as pd
from scipy.stats import uniform, norm
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from basic.boots_pya import least_squares, fun, pce_fun
from basic.utils import variables_prep, names_match
from basic.partial_rank import partial_rank

fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/pya_related/'
filename = f'{fpath}parameter-adjust.csv'
variable = variables_prep(filename, product_uniform=True)
filename = f'{fpath}samples_adjust.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
samples = data[:,:11].T
values = data[:,11:]
values = values[:,:1]# focus on first qoi
ntrain_samples = 198
poly, error = pce_fun(variable, samples, values, ntrain_samples, degree=2)

# with adjustment to parameter ranges
index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17], 
                         [6, 5, 7], 
                         [20, 19],
                         ])
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/pya_related/'

problem = read_param_file(f'{fpath}problem.txt', delimiter=',')
filename = f'{fpath}problem_adjust.txt'
problem_adjust = read_param_file(filename, delimiter=',')
problem = read_param_file(f'{fpath}problem.txt', delimiter=',')

def samples_df(sample_method='samples_product', Nsamples = 2000):
    def samples_product(Nsamples=Nsamples):
        sample_new = np.round(saltelli.sample(problem, 
                                        N=Nsamples, 
                                        calc_second_order=False, 
                                        skip_values=100,
                                        deleted=True,
                                        product_dist=index_product,
                                        problem_adjust=problem_adjust,
                                        seed=121), 4)
        return sample_new

    # sampling with the only ranges of parameters changed
    def ranges_adjust(Nsamples=Nsamples):
        sample_range_change = np.round(saltelli.sample(problem_adjust, 
                                        N=Nsamples, 
                                        calc_second_order=False, 
                                        seed=121), 4)
        return sample_range_change

    if sample_method == 'samples_product':
        df = samples_product()
    else:
        df = ranges_adjust()
    return df

# expand the samples with other parameters set to a fixed value
rankings = {}
rank_names = {}
for n in range(200, 2001, 200):
    df = samples_df(sample_method='samples_product', Nsamples = n)
    y_range_change = np.round(poly(df.T), 4).reshape(list(df.shape)[0])
    sa, main_resample, total_resample = sobol.analyze(problem_adjust, 
                                        y_range_change, calc_second_order=False, 
                                        num_resamples=1000, conf_level=0.95, seed=88)

    rankings[f'nsample_{n}'] = partial_rank(total_resample.T, problem_adjust['num_vars'])
    rank_names[f'nsample_{n}'] = names_match(rankings[f'nsample_{n}'], problem_adjust['names'])


sa_df = pd.DataFrame.from_dict(sa)
sa_df.index = problem_adjust['names']

fpath_save = 'D:/cloudStor/Research/pce_fixing/output/linear_dep/'
sa_df.to_csv(f'{fpath_save}sa_samples_product.csv', index=True)
# plot using plot in SALib
sns.set(style="whitegrid")
rc("text", usetex=False)
fig = plt.figure(figsize=(6, 4))
ax = barplot(sa_df)
ax.set_xlabel('Parameters')
ax.set_ylabel('Main / Total effects');
plt.savefig(f'{fpath_save}sampling_product.png', format='png', dpi=300)
