import numpy as np
import pyapprox as pya
from scipy.stats import uniform
from basic.boots_pya import least_squares, fun
from basic.partial_rank import partial_rank

fpath = '../../../Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-ranges.csv'
ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(ranges.shape[0]//2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

filename = f'{fpath}test_qian.csv'
data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
samples = data[:,:22].T
values = data[:,22:]
values = values[:,:1]# focus on first qoi

# Using pyapprox for sensitivity analysis
errors, condition_numbers, sensitivity_indices = fun(fpath, variable, samples, values, 
                                                    ntrials=100, ntrain_samples=None)
len_params = samples.shape[0]

# Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking
n_strat, n_end, n_setp = [276, 828, 50]
# loops of fun
partial_results = {}
for i in range(n_strat, n_end+1, n_setp):
    errors, condition_numbers, sensitivity_indices = fun(fpath, variable, samples, values, 
                                                        ntrials=500, ntrain_samples=i)

    # partial ranking
    sensitivity_indices = np.array(sensitivity_indices)
    sa_shape = sensitivity_indices.shape[0:2]
    sensitivity_indices = sensitivity_indices.reshape((sa_shape))
    rankings = partial_rank(sensitivity_indices,len_params, conf_level=0.95)
    partial_results[f'nsample_{i}'] = rankings
# End for


