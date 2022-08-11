import numpy as np
from SALib.util import read_param_file
from SALib.analyze import sobol
import pickle

from basic.utils import dist_return
from basic.read_data import variables_prep
import SAFEpython.VBSA as VB
import scipy.stats as st
from sa_sampling import samples_df
import timeit

# Calculate the corresponding number of bootstrap with use of group_fix
ntrain_samples = 552
nboot = 500
samples_files = ['2000_2014_ave_annual.csv', 'samples_adjust.csv']
param_files = ['parameter.csv', 'parameter-adjust.csv']


fpath_save = '../output/adaptive/'
filename = f'{fpath_save}{samples_files[0]}'

fpath_input = '../data/'
filename = f'{fpath_input}{param_files[0]}'
dummy = True; product_uniform = False
variable = variables_prep(filename, dummy=dummy, product_uniform=product_uniform)
# Read problem.txt file and add a dummy parameter to it
input_path = '../data/'
problem_adjust = read_param_file(f'{input_path}problem_adjust.txt', delimiter=',')
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')
problem['dists'] = ['unif'] * 22
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)

filename = f'{input_path}problem_adjust.txt'
problem_adjust = read_param_file(filename, delimiter=',')


product_uniform = ['exact', 'uniform', False] 
dist_type = dist_return(product_uniform[0])
filename = f'adaptive-reduce-{dist_type}_552'
pce_list = pickle.load(open(f'{fpath_save}{filename}-approx-list.pkl', "rb"))['nsample_70']
fun_test = pce_list[0]
samp_strat = 'lhs'

N = 15000
M = problem['num_vars']
distr_fun = st.uniform
distr_par = [[bound[0], bound[1]-bound[0]] for bound in problem['bounds']]
# X = AAT_sampling(samp_strat, M, distr_fun ,distr_par, 2*N)
X = samples_df(problem=problem, problem_adjust=problem_adjust, 
                product_dist=index_product, sample_method='samples_product', Nsamples = N)

[XA, XB, XC ] = VB.vbsa_resampling(X)

YA = fun_test(XA.T)
YB = fun_test(XB.T)
YC = fun_test(XC.T)

Si, STi, Sdummy, STdummy = VB.vbsa_indices(YA, YB, YC, problem_adjust['num_vars'], Nboot=100, dummy=True)

y_range_change = np.round(fun_test(X.T), 4).reshape(list(X.shape)[0])

sa = sobol.analyze(problem_adjust, y_range_change, calc_second_order=False, 
                    num_resamples=100, conf_level=0.95, keep_resamples=True)