import numpy as np
import pandas as pd
import pyapprox as pya
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from scipy.stats import beta, uniform, norm as beta, uniform, norm
from basic.utils import variables_prep
from scipy import stats
from SALib.util import read_param_file

from basic.boots_pya import fun

def df_format(total_effects, variables, param_names, conf_level=0.95):    
    sa_df = pd.DataFrame(data=np.zeros(shape=(variables.num_vars(), 3)), 
                        columns=['ST', 'ST_conf_lower', 'ST_conf_upper'])
    total_effects = np.array(total_effects)
    sa_df.loc[:, 'ST'] = total_effects.mean(axis=0)
    # import pdb; pdb.set_trace()
    sa_df.loc[:, 'ST_conf_lower'] = np.quantile(total_effects, [0.025], axis=0)[0]
    sa_df.loc[:, 'ST_conf_upper' ] = np.quantile(total_effects, [0.975], axis=0)[0]
    sa_df.index = param_names
    return sa_df
# End df_format()

def main():
    fpath_save = 'output/paper/'
    filename = f'{fpath_save}2000_2014_ave_annual.csv'
    data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
    samples = data[:, :22].T
    values = data[:, 22:]

    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    fpath_input = 'data/'
    filename = f'{fpath_input}parameter-reimplement.csv'
    variable = variables_prep(filename, product_uniform=False)
    param_all = pd.read_csv(filename).loc[:, 'Veneer_name'].values

    # Used for PCE fitting 
    ntrain_samples = 552
    nboot = 500
    error_cv, _, total_effects = fun(variable, samples, 
                                                values, degree=2, nboot=nboot, 
                                                ntrain_samples=ntrain_samples)
    # import pdb; pdb.set_trace()                                                
    sa_df = df_format(total_effects, variable, 
                    param_all, conf_level=0.95)
    sa_df.to_csv(f'{fpath_save}sa_pce_raw_test.csv', index=True)
    error_cv_df = pd.DataFrame(data = error_cv, columns = [f'22_{ntrain_samples}_uniform'],
                    index = np.arange(len(error_cv)))

    # PCE with 11 parameters of uniform distributions
    filename = f'{fpath_input}parameter-adjust.csv'
    variable_uniform = variables_prep(filename, product_uniform=False)
    variable_beta = variables_prep(filename, product_uniform=True)
    param_reduce = pd.read_csv(filename).loc[:, 'Veneer_name'].values

    # import samples and values
    filename = f'{fpath_save}samples_adjust.csv'
    data = np.loadtxt(filename, delimiter=",", skiprows=1)[:,1:]
    samples_adjust = data[:, :11].T
    
    for ntrain in [156]:
        error_cv_uniform, _, total_effects = fun(variable_uniform, samples_adjust, 
                                                values, degree=2, 
                                                ntrain_samples=ntrain, nboot=nboot)
        error_cv_df[f'11_{ntrain}_uniform'] = error_cv_uniform
        if ntrain == 156:
            sa_df = df_format(total_effects, variable_uniform, 
                    param_reduce, conf_level=0.95)
            sa_df.to_csv(f'{fpath_save}sa_pce_uniform_test.csv', index=True)

    for ntrain in [156]:
        error_cv_beta, main_effects, total_effects = fun(variable_beta, samples_adjust, 
                                                values, degree=2, 
                                                ntrain_samples=ntrain, nboot=nboot)
        error_cv_df[f'11_{ntrain}_beta'] = error_cv_beta
        if ntrain == 156:
            sa_df = df_format(total_effects, variable_beta, 
                    param_reduce, conf_level=0.95)
            sa_df.to_csv(f'{fpath_save}sa_pce_beta_test.csv', index=True)

    error_cv_mean = error_cv_df.apply(np.mean, axis=0)
    error_cv_std = error_cv_df.apply(np.std, axis=0)
    error_stats_df = pd.DataFrame(data=[error_cv_mean, error_cv_std], 
                    index=['mean', 'std']).T
    error_stats_df.index = error_cv_df.columns
    # error_stats_df.to_csv(f'{fpath_save}error_cv_compare.csv', index=True)

if __name__ == "__main__":
    main()
    