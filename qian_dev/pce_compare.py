import numpy as np
import pandas as pd
import pyapprox as pya
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from scipy.stats import beta, uniform, norm as beta, uniform, norm
from basic.utils import variables_prep
from scipy import stats
from SALib.util import read_param_file

from basic.boots_pya import fun
from basic.read_data import read_specify, file_settings

def df_format(total_effects, variables, param_names, conf_level=0.95):    
    sa_df = pd.DataFrame(data=np.zeros(shape=(variables.num_vars(), 3)), 
                        columns=['ST', 'ST_conf_lower', 'ST_conf_upper'])
    total_effects = np.array(total_effects)
    sa_df.loc[:, 'ST'] = total_effects.mean(axis=0)
    sa_df.loc[:, 'ST_conf_lower'] = np.quantile(total_effects, [0.025], axis=0)[0]
    sa_df.loc[:, 'ST_conf_upper' ] = np.quantile(total_effects, [0.975], axis=0)[0]
    sa_df.index = param_names
    return sa_df
# End df_format()

def compare():
    fpath_save = file_settings()[0]
    samples, values = read_specify('model', 'full', product_uniform=False, num_vars=22)
    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    variable, param_all = read_specify('parameter', 'full', product_uniform=False, num_vars=22)
    # Used for PCE fitting 
    ntrain_samples = 552
    nboot = 500
    error_cv, _, total_effects = fun(variable, samples, 
                                        values, degree=2, nboot=nboot, 
                                            ntrain_samples=ntrain_samples)                                          
    sa_df = df_format(total_effects, variable, 
                    param_all, conf_level=0.95)
    # sa_df.to_csv(f'{fpath_save}sa_pce_raw_test.csv', index=True)
    error_cv_df = pd.DataFrame(data = error_cv, columns = [f'22_{ntrain_samples}_uniform'],
                    index = np.arange(len(error_cv)))

    # PCE with 11 parameters of uniform distributions
    variable_uniform, param_reduce = read_specify('parameter', 'reduced', product_uniform=False, num_vars=11)
    variable_beta, _ = read_specify('parameter', 'reduced', product_uniform=True, num_vars=11)
    # import samples and values
    samples_adjust, _ = read_specify('model', 'reduced', product_uniform=False, num_vars=11)
    
    for ntrain in [156]:
        error_cv_uniform, _, total_effects = fun(variable_uniform, samples_adjust, 
                                                values, degree=2, 
                                                ntrain_samples=ntrain, nboot=nboot)
        error_cv_df[f'11_{ntrain}_uniform'] = error_cv_uniform
        if ntrain == 156:
            sa_df = df_format(total_effects, variable_uniform, 
                    param_reduce, conf_level=0.95)
            # sa_df.to_csv(f'{fpath_save}sa_pce_uniform.csv', index=True)

    for ntrain in [156]:
        error_cv_beta, main_effects, total_effects = fun(variable_beta, samples_adjust, 
                                                values, degree=2, 
                                                ntrain_samples=ntrain, nboot=nboot)
        error_cv_df[f'11_{ntrain}_beta'] = error_cv_beta
        if ntrain == 156:
            sa_df = df_format(total_effects, variable_beta, 
                    param_reduce, conf_level=0.95)
            # sa_df.to_csv(f'{fpath_save}sa_pce_beta.csv', index=True)

    error_cv_mean = error_cv_df.apply(np.mean, axis=0)
    # error_cv_std = error_cv_df.apply(np.std, axis=0)
    error_cv_lower = np.round(error_cv_df.quantile(0.025, axis=0), 4)
    error_cv_upper= np.round(error_cv_df.quantile(0.975, axis=0), 4)
    error_stats_df = pd.DataFrame(data=[error_cv_mean, error_cv_lower, error_cv_upper], 
                    index=['mean', 'lower', 'upper']).T
    error_stats_df.index = error_cv_df.columns
    # error_stats_df.to_csv(f'{fpath_save}error_cv_compare.csv', index=True)

if __name__ == "__main__":
    compare()
    