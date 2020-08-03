import numpy as np
import pandas as pd
import pyapprox as pya
from scipy.stats import uniform
from basic.boots_pya import least_squares, fun
from basic.partial_rank import partial_rank
from basic.utils import variables_prep

# define the distribution of the first parameter
from pyapprox.random_variable_algebra import product_of_independent_random_variables_pdf
# import the original parameter sets
def main():
    file_input = 'data/'
    filename = f'{file_input}parameter-adjust.csv'
    # raw_parameters = pd.read_csv(filename)
    variable = variables_prep(filename, product_uniform=True)

    file_sample = 'output/paper/'
    filename = f'{file_sample}samples_adjust.csv'
    data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
    len_params = variable.num_vars()
    samples = data[:,:len_params].T
    values = data[:,len_params:]

    # Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking
    n_strat, n_end, n_setp = [78, 234, 10]
    # loops of fun
    errors_cv_all = {}
    errors_bt_all = {}
    partial_results = {}
    for i in range(n_strat, n_end+1, n_setp):
        errors_cv, errors_bt, main_effects, total_effects = fun(variable, samples, 
                                                        values, degree=2, 
                                                        nboot=100, ntrain_samples=i)
        # partial ranking
        total_effects = np.array(total_effects)
        sa_shape = list(total_effects.shape)[0:2]
        total_effects = total_effects.reshape((sa_shape))
        rankings = partial_rank(total_effects,len_params, conf_level=0.95)
        partial_results[f'nsample_{i}'] = rankings
        errors_cv_all[f'nsample_{i}'] = errors_cv
        errors_bt_all[f'nsample_{i}'] = errors_bt
    # End for

    filepath = 'output/paper/'
    filename = f'{filepath}adaptive-reduce-params.npz'
    np.savez(filename,errors_cv=errors_cv_all,errors_bt=errors_bt_all,sensitivity_indices=partial_results)

if __name__ == "__main__":
    main()
