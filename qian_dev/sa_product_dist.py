import numpy as np
import pandas as pd
from scipy.stats import uniform
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
import time
from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, names_match
from basic.partial_rank import partial_rank

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
input_path = '../data/'
filename = f'{input_path}parameter-adjust.csv'
variable = variables_prep(filename, product_uniform=False)

filepath = '../output/paper/'
filename = f'{filepath}samples_adjust.csv'
data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
samples = data[:,:11].T
values = data[:,11:]
values = values[:,:1]
ntrain_samples = 156

# with adjustment to parameter ranges
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)
# import problem for SALib
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')
filename = f'{input_path}problem_adjust.txt'
problem_adjust = read_param_file(filename, delimiter=',')
rankings, rank_names = {}, {}

nboot = 500
rand = np.random.randint(0, ntrain_samples, size=(nboot, ntrain_samples))
for n in range(600, 601, 100):
    print(n)
    df = samples_df(sample_method='samples_product', Nsamples = n)
    for ii in range(nboot):
        poly, error = pce_fun(variable, samples, values, 
                            ntrain_samples, degree=2, boot_ind=rand[ii])
        y_range_change = np.round(poly(df.T), 4).reshape(list(df.shape)[0])
        sa, main_resample, total_resample = sobol.analyze(problem_adjust, 
                                        y_range_change, calc_second_order=False, 
                                        num_resamples=500, conf_level=0.95)
        try:
            total_indices = np.append(total_indices, total_resample, axis=1)
        except NameError:
            total_indices = total_resample[:]                                 
    rankings[f'nsample_{n}'] = partial_rank(total_indices.T, problem_adjust['num_vars'])
    rank_names[f'nsample_{n}'] = names_match(rankings[f'nsample_{n}'], problem_adjust['names'])
    # End For
# End For

sa_df = pd.DataFrame(data = [total_indices.mean(axis=1), *np.quantile(total_indices, [0.025, 0.975], axis=1)],
                index=['ST', 'ST_conf_lower', 'ST_conf_upper'], columns=problem_adjust['names']).T

fpath_save = '../output/paper/'

# save results of partial rankings for parameter index and names
# sa_df.to_csv(f'{fpath_save}sa_samples_product_test.csv', index=True)

# with open(f'{fpath_save}ranking_sampling.json', 'w') as fp:
#     json.dump(rankings, fp, indent=2)

# with  open(f'{fpath_save}ranking_sampling_name.json', 'w') as fp:
#     json.dump(rank_names, fp, indent=2)    

print("--- %s seconds ---" % (time.time() - start_time))