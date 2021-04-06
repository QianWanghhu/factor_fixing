"""
Script used to generate the sampling-based total-order effects.
"""
import numpy as np
import pandas as pd
from scipy.stats import uniform
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
import time
from basic.boots_pya import fun
from basic.utils import names_match
from basic.partial_rank import partial_rank
from basic.read_data import file_settings, read_specify

start_time = time.time()
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
input_path = file_settings()[1]
variable, _ = read_specify('parameter', 'reduced', product_uniform=False, num_vars=11)
output_path = file_settings()[0]
samples, values = read_specify('model', 'reduced', product_uniform=False, num_vars=11)
# with adjustment to parameter ranges
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
# import problem for SALib
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')
filename = f'{input_path}problem_adjust.txt'
problem_adjust = read_param_file(filename, delimiter=',')

num_pce = 500
seed = 222
ntrain_samples = 156
np.random.seed(seed)
rand = np.random.randint(0, ntrain_samples, size=(num_pce, ntrain_samples))
for n in range(600, 601, 100):
    print(n)
    df = samples_df(sample_method='samples_product', Nsamples = n)
    for ii in range(num_pce):
        poly = fun(variable, samples, values, degree=2, 
            nboot=1, I=rand[ii], ntrain_samples=ntrain_samples)
        y_range_change = np.round(poly(df.T), 4).reshape(list(df.shape)[0])
        sa, main_resample, total_resample = sobol.analyze(problem_adjust, 
                                        y_range_change, calc_second_order=False, 
                                        num_resamples=500, conf_level=0.95)
        try:
            rankings = np.append(total_indices, total_resample, axis=1)
        except NameError:
            total_indices = total_resample[:]                                 
    rankings = partial_rank(total_indices.T, problem_adjust['num_vars'])
    rank_names = names_match(rankings, problem_adjust['names'])
    # End For
# End For

sa_df = pd.DataFrame(data = [total_indices.mean(axis=1), *np.quantile(total_indices, [0.025, 0.975], axis=1)],
                index=['ST', 'ST_conf_lower', 'ST_conf_upper'], columns=problem_adjust['names']).T

# save results of partial rankings for parameter index and names
sa_df.to_csv(f'{output_path}sa_sampling.csv', index=True)
with open(f'{output_path}ranking_sampling.json', 'w') as fp:
    json.dump(rankings, fp, indent=2)

with  open(f'{output_path}ranking_sampling_name.json', 'w') as fp:
    json.dump(rank_names, fp, indent=2)    

print("--- %s seconds ---" % (time.time() - start_time))