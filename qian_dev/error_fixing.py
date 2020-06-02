import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pyapprox as pya
import SALib.sample.latin as latin

# from utils import cvg_check
from basic.utils import cvg_check
from basic.error_measure import group_fix
from basic.boots_pya import pce_fun 
from scipy.stats import uniform
from basic.boots_pya import least_squares, fun

fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/pya_related/'
filename = f'{fpath}parameter-ranges.csv'
ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(ranges.shape[0]//2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

filename = f'{fpath}test_qian.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
samples = data[:,:22].T
values = data[:,22:]
values = values[:,:1]# focus on first qoi
len_params = samples.shape[0]
# generate samples for error metric analysis
bounds = [[ranges[2*ii], ranges[2*ii+1]] for ii in range(ranges.shape[0]//2)]
problem = {
    'num_vars': len_params,
    'names': [f'x{i}' for i in range(len_params)],
    'bounds': bounds
        }
x_sample = latin.sample(problem, 1000, seed=88)
x_sample = x_sample.T

# generate PCE
poly, error = pce_fun(variable, samples, values, ntrain_samples=None)
y_uncond = poly(x_sample).flatten()

# load partial order results
fpath = 'D:/cloudStor/Research/pce_fixing/output/0709_ave_annual/'
fname = 'partial_pya2.json'
with open(f'{fpath}{fname}', 'r') as fp:
    partial_order = json.load(fp)
# Get the unique partial sort results and the corresponding number of bootstrap used for PCE
order_temp, num_resample = cvg_check(partial_order)

error_dict = {}
for ele in num_resample[:-2]:
    error_dict[ele] = group_fix(partial_order[ele], poly, x_sample, y_uncond, 1, file_exist=True)
# print(kl_dict)