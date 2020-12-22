import numpy as np
import pandas as pd
import json
import os
from SALib.util import read_param_file
from SALib.sample import saltelli
from SALib.analyze import sobol
import time

from basic.boots_pya import fun, pce_fun
from basic.utils import variables_prep, to_df, adjust_sampling
from basic.group_fix import group_fix, uncond_cal
from SAFEpython.VBSA import vbsa_indices

# Calculate the corresponding number of bootstrap with use of group_fix
ntrain_samples = 552
nboot = 500
samples_files = ['2000_2014_ave_annual.csv', 'samples_adjust.csv']
param_files = ['parameter-implement.csv', 'parameter-adjust.csv']


fpath_save = '../output/paper0915/'
filename = f'{fpath_save}{samples_files[0]}'
data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
samples = data[:, :-1].T
values = data[:, -1:]

fpath_input = '../data/'
filename = f'{fpath_input}{param_files[0]}'
dummy = False; product_uniform = False
variable = variables_prep(filename, dummy=dummy, product_uniform=product_uniform)
poly, error, total_index, _ = fun(variable, samples, values, 
                    ntrain_samples=ntrain_samples, degree=2, nboot=1)

# Read problem.txt file and add a dummy parameter to it
input_path = '../data/'
problem_adjust = read_param_file(f'{input_path}problem_adjust.txt', delimiter=',')
problem = read_param_file(f'{input_path}problem.txt', delimiter=',')
index_product = np.load(f'{input_path}index_product.npy', allow_pickle=True)

N = 600
problem['names'].append('dummy')
problem['groups'].append('dummy')
problem['num_vars'] += 1
problem['bounds'].append([0, 1])

x_saltelli = saltelli.sample(problem, N)
y_saltelli = poly(x_saltelli[:, :-1].T).flatten()
sa = sobol.analyze(problem, y_saltelli, num_resamples=500)

