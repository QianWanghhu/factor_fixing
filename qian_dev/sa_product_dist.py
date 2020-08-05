import numpy as np
import pandas as pd
from scipy.stats import uniform, norm
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib import rc
from basic.boots_pya import least_squares, fun, pce_fun
from basic.utils import variables_prep, names_match
from basic.partial_rank import partial_rank

def samples_df(sample_method = 'samples_product', Nsamples = 2000):
    def samples_product(Nsamples=Nsamples):
        sample_new = np.round(saltelli.sample(problem, 
                                        N=Nsamples, 
                                        calc_second_order=False, 
                                        skip_values=1000,
                                        deleted=True,
                                        product_dist=index_product,
                                        problem_adjust=problem_adjust,
                                        seed=121), 4)
        return sample_new

    # sampling for parameters with the only ranges changed and the distributions kept as uniform
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

# expand the samples with other parameters fixed
input_path = '../data/'
filename = f'{input_path}parameter-adjust.csv'
variable = variables_prep(filename, product_uniform=True)

filepath = '../output/paper/'
filename = f'{filepath}samples_adjust.csv'
data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
samples = data[:,:11].T
values = data[:,11:]
values = values[:,:1]# focus on first qoi
ntrain_samples = 156
poly, error = pce_fun(variable, samples, values, ntrain_samples, degree=2)

# with adjustment to parameter ranges
index_product = np.load(f'{input_path}index_prodcut.npy', allow_pickle=True)

# import problem for SALib
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')
filename = f'{input_path}problem_adjust.txt'
problem_adjust = read_param_file(filename, delimiter=',')
rankings, rank_names = {}, {}

for n in range(200, 1001, 100):
    print(n)
    df = samples_df(sample_method='samples_product', Nsamples = n)
    y_range_change = np.round(poly(df.T), 4).reshape(list(df.shape)[0])
    sa, main_resample, total_resample = sobol.analyze(problem_adjust, 
                                        y_range_change, calc_second_order=False, 
                                        num_resamples=500, conf_level=0.95)

    rankings[f'nsample_{n}'] = partial_rank(total_resample.T, problem_adjust['num_vars'])
    rank_names[f'nsample_{n}'] = names_match(rankings[f'nsample_{n}'], problem_adjust['names'])

sa = {key: value for key, value in sa.items() if 'ci' not in key}
sa_df = pd.DataFrame.from_dict(sa)
sa_df.index = problem_adjust['names']
fpath_save = '../output/paper/'

# save results
sa_df.to_csv(f'{fpath_save}sa_samples_product.csv', index=True)
with  open(f'{fpath_save}ranking_sampling.json', 'w') as fp:
    json.dump(rankings, fp, indent=2)    

with  open(f'{fpath_save}ranking_samplin_name.json', 'w') as fp:
    json.dump(rank_names, fp, indent=2)    
