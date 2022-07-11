#! usr/bin/env/ oed

"""
Script used to generate the sampling-based total-order effects.
"""
import numpy as np
import pandas as pd
import pickle
from scipy.stats import uniform
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
import time
from basic.utils import names_match, dist_return
from basic.partial_rank import partial_rank
from basic.read_data import file_settings, read_specify
from sample_adjust import sample

def samples_df(problem, problem_adjust, product_dist, sample_method = 'samples_product', Nsamples = 2000):
    def samples_product(Nsamples=Nsamples):
        sample_new = np.round(sample(problem, 
                                        N=Nsamples, 
                                        calc_second_order=False, 
                                        skip_values=1000,
                                        deleted=True,
                                        product_dist=product_dist,
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

def run_sa_sampling():

    # expand the samples with other parameters fixed
    input_path = file_settings()[1]
    variable, _ = read_specify('parameter', 'reduced', product_uniform=False, num_vars=11)
    output_path = file_settings()[0]
    # with adjustment to parameter ranges
    index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
    # import problem for SALib
    problem = read_param_file(f'{input_path}problem.txt', delimiter=',')
    problem['dists'] = ['unif'] * 22
    filename = f'{input_path}problem_adjust.txt'
    problem_adjust = read_param_file(filename, delimiter=',')
    seed = 222
    ntrain_samples = 90
    np.random.seed(seed)
    # rand = np.random.randint(0, ntrain_samples, size=(num_pce, ntrain_samples))
    product_uniform = ['exact', 'uniform', False] 
    dist_type = dist_return(product_uniform[1])
    filename = f'adaptive-reduce-{dist_type}_552'
    pce_list = pickle.load(open(f'{output_path}{filename}-approx-list.pkl', "rb"))['nsample_90']

    for n in range(1000, 1001, 100):
        print(n)
        df = samples_df(problem=problem, problem_adjust=problem_adjust, 
                        product_dist=index_product, sample_method='samples_product', 
                        Nsamples = n)
        for poly in pce_list:
            y_range_change = np.round(poly(df.T), 4).reshape(list(df.shape)[0])
            sa = sobol.analyze(problem_adjust, y_range_change, calc_second_order=False, 
                                num_resamples=100, conf_level=0.95, keep_resamples=True)
            try:
                total_indices = np.append(total_indices, sa['ST_conf_all'], axis=0)
            except NameError:
                total_indices = sa['ST_conf_all']                                
        # End For
    # End For

    rankings = partial_rank(total_indices, problem_adjust['num_vars'])
    rank_names = names_match(rankings, problem_adjust['names'])
    sa_df = pd.DataFrame(data = [total_indices.mean(axis=0), *np.quantile(total_indices, [0.025, 0.975], axis=0)],
                    index=['ST', 'ST_conf_lower', 'ST_conf_upper'], columns=problem_adjust['names']).T
    # save results of partial rankings for parameter index and names
    return sa_df

if __name__ == "__main__":
    start_time = time.time()
    sa_df = run_sa_sampling()
    output_path = file_settings()[0]
    sa_df.to_csv(f'{output_path}sa_sampling.csv', index=True)
    print("--- %s seconds ---" % (time.time() - start_time))
