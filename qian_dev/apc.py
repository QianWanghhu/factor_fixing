# arbitrary polynomial chaos
import numpy as np
import pyapprox as pya
from functools import partial
from pyapprox.arbitrary_polynomial_chaos  import *
from pyapprox.variable_transformations import define_iid_random_variable_transformation
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from pyapprox.indexing import compute_hyperbolic_indices
from scipy.stats import beta as beta, uniform
from basic.boots_pya import least_squares, pce_fun

# import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-adjust.csv'
ranges = np.loadtxt(filename,delimiter=",",usecols=[3,4],skiprows=1).flatten()
univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(0, ranges.shape[0]//2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

filename = f'{fpath}test_qian.csv'
data = np.loadtxt(filename, delimiter=",", skiprows=1)[:, 1:]
samples = data[:,:22].T
values = data[:,22:]
values = values[:,:1]# focus on first qoi

index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17, 18], 
                         [6, 5, 7], 
                         [20, 19],
                         ])
samples_adjust = np.copy(samples)
pars_delete = []                
for ii in range(list(index_product.shape)[0]):
    index_temp = index_product[ii]
    samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
    pars_delete.extend(index_temp[1:])
samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)
 
# define the pce_opts
num_vars = variable.num_vars()
degree=3
ntrain_samples = 858
pce_var_trans = define_iid_random_variable_transformation(uniform(0,1),num_vars)
pce_opts = {'alpha_poly':0,'beta_poly':0,'var_trans':pce_var_trans,
                    'poly_type':'legendre'}

compute_moment_matrix_function = partial(
            compute_moment_matrix_from_samples,samples=samples_adjust)
pce = APC(compute_moment_matrix_function)
pce.configure(pce_opts)
indices = compute_hyperbolic_indices(num_vars,degree,1.0)
pce.set_indices(indices)
train_samples = samples_adjust[:, 0:ntrain_samples]
train_values = values[0:ntrain_samples]
coef = least_squares(pce.basis_matrix,train_samples ,train_values)
pce.set_coefficients(coef)

validation_samples = samples_adjust[:,ntrain_samples:]
validation_values = values[ntrain_samples:]
approx_values = pce(validation_samples)
error = np.linalg.norm(approx_values-validation_values)/np.linalg.norm(values)

# generate poly
fpath = 'D:/cloudStor/Research/pce_fixing/pyfile/John/'
filename = f'{fpath}parameter-ranges.csv'
ranges = np.loadtxt(filename,delimiter=",",usecols=[2,3],skiprows=1).flatten()
univariate_variables = [uniform(ranges[2*ii],ranges[2*ii+1]-ranges[2*ii]) for ii in range(0, ranges.shape[0]//2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

degree=2
ntrain_samples = 552
poly, error = pce_fun(variable, samples, values, ntrain_samples)
samples_vali = generate_independent_random_samples(
            variable,10000)
y_original = poly(samples_vali)
# adjust and calculate new results with the new PCE
pars_delete = []                
for ii in range(list(index_product.shape)[0]):
    index_temp = index_product[ii]
    samples_vali[index_temp[0], :] = np.prod(samples_vali[index_temp, :], axis=0)
    pars_delete.extend(index_temp[1:])
samples_vali = np.delete(samples_vali, pars_delete, axis=0)
y_new = pce(samples_vali)

err = np.linalg.norm(y_new-y_original)/np.linalg.norm(y_original)
