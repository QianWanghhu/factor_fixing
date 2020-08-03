import numpy as np
import pandas as pd
import pyapprox as pya
from pyapprox.probability_measure_sampling import generate_independent_random_samples
from scipy.stats import beta, uniform, norm as beta, uniform, norm
from basic.utils import variables_prep
from scipy import stats
from SALib.util import read_param_file

def main():
    fpath_save = 'output/paper/'
    # import samples and values
    fpath = 'output/Run0730/'
    filename = f'{fpath}2000_2014_ave_annual.csv'
    data = np.loadtxt(filename,delimiter=",",skiprows=1)[:,1:]
    samples = data[:,:22].T
    values = data[:,22:]

    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    fpath_input = 'data/'
    filename = f'{fpath_input}parameter-reimplement.csv'
    # variable = variables_prep(filename, product_uniform=False)
    # param_all = pd.read_csv(filename).loc[:, 'Veneer_name'].values
    index_product = np.array([[1, 0, 2, 3, 9, 10, 11, 16, 17], 
                            [6, 5, 7], 
                            [19, 20],
                            ])

    # define variables with Beta distribution
    filename = f'{fpath_input}parameter-adjust.csv'
    variable_adjust = variables_prep(filename, product_uniform=True)
    param_adjust = pd.read_csv(filename)
    beta_index = param_adjust[param_adjust['distribution']== 'beta'].\
                index.to_list()

    samples_adjust = np.copy(samples)
    pars_delete = []
    for ii in range(list(index_product.shape)[0]):
        index_temp = index_product[ii]
        samples_adjust[index_temp[0], :] = np.prod(samples_adjust[index_temp, :], axis=0)
        # samples_adjust[index_temp[1:], :] = 1
        pars_delete.extend(index_temp[1:])
    samples_adjust = np.delete(samples_adjust, pars_delete, axis=0)
    # import pdb
    # pdb.set_trace()

    samples_adjust = np.append(samples_adjust, [values.flatten()], axis=0)    
    df_adjust = pd.DataFrame(data = samples_adjust.T, 
                            index = np.arange(samples_adjust.shape[1]),
                            columns = [*param_adjust.loc[:, 'Veneer_name'], 'TSS_ave'])
    df_adjust.to_csv(f'{fpath_save}samples_adjust.csv')

    # # Used for PCE fitting
    # ntrain_samples = 552
    # nboot = 1000
    # if ntrain_samples == 552:
    #     error_cv, error_bt, main_effects, total_effects = fun(variable, samples, 
    #                                                 values, degree=2, nboot=nboot, 
    #                                                 ntrain_samples=ntrain_samples)
    #     sa_df = df_format(main_effects, total_effects, variable, 
    #                     param_all, conf_level=0.95)
    #     sa_df.to_csv(f'{fpath_save}sa_pce_raw.csv', index=True)
    #     plot(sa_df, sav=True, figname='sa_PCE_Beta_degree2_uniform.png')
    # else:
    #     poly_beta, error = pce_fun(variable_adjust, samples_adjust, 
    #                     values, ntrain_samples, degree=2)
    #     approx_values = poly_beta(samples_adjust)                    
    #     error_vali = np.linalg.norm(approx_values - values) / np.linalg.norm(values)

    # total_effects = np.array(total_effects)
    # total_effects = total_effects.reshape(nboot, variable_adjust.num_vars())
    # rankings = partial_rank(total_effects, variable_adjust.num_vars())
    # rank_names = names_match(rankings, param_adjust.loc[:, 'Veneer_name'].values)

if __name__ == "__main__":
    main()