import numpy as np
from scipy.stats import entropy
from bisect import bisect
from scipy import stats
from scipy.stats import median_absolute_deviation as mad
from sklearn.metrics import r2_score, mean_squared_error
from pyapprox.multivariate_polynomials import conditional_moments_of_polynomial_chaos_expansion as cond_moments

def group_fix(partial_result, func, x, y_true, x_default, 
            rand, pool_results, file_exist=False):
    """
    Function for compare results between conditioned and unconditioned QoI.
    Fix parameters from the least influential group 
    based on results from partially sorting.

    Four error measure types will be returned.

    Parameters
    ----------
    partial_result : dict,
        dictionary of parameter groups, results of partial sort

    func : function,
        function for analysis (analytical formula or model)

    x : np.array,
        Input with shape of N * D where N is sampling size and 
        D is the number of parameters

    y_true : list,
        Function results with all x varying (the raw sampling matrix of x)

    x_default : int, float, list,
        Default values of x as a scalar or list of scalars

    rand : np.ndarray,
        Resample index in bootstrap, shape of R * N, 
        where R is the number of resamples

    pool_results : dict,
        Index of fixed parameters and the corresponding results

    a : np.array (Default: None),
        Coefficients used in `func`

    file_exist : bool (default: False), 
        If true, reads cached partial-ranking results from a file.
        Otherwise, calculates results.
    
    Returns
    ----------
    Tuple of:

    dict_return:
        mae : dict, 
            Changes in absolute mean error of the func results due to fixing 
            parameters

        var : dict, 
            Changes in variance of the func results due to fixing parameters

        ks : dict, 
            Changes in pearson correlation coefficients 
            of the func results due to fixing parameters

        mae_lower : dict,
            Lowest absolute MAE values

        var_lower :  dict, 
            Lowest variance

        ppmc_lower :  dict,
            Lowest PPMC

        mae_upper :  dict,
            Largest absolute MAE values

        var_upper :  dict,
            Largest variance values

        ppmc_upper :  dict,
            Largest PPMC values

    pool_results:

    """
    num_group = len(partial_result) - 1

    # store results from fixing parameters in dict
    cf_upper = {i: None for i in range(num_group)}
    cf_lower, cv, ks, pvalue = dict(cf_upper), dict(cf_upper), dict(cf_upper), dict(cf_upper)
    cf_upper_upper, cf_upper_lower, ks_upper, pvalue_upper = dict(cf_upper), dict(cf_upper), dict(cf_upper), dict(cf_upper)
    cf_lower_lower, cf_lower_upper, ks_lower, pvalue_lower = dict(cf_upper), dict(cf_upper), dict(cf_upper), dict(cf_upper)
    cf_width, cf_width_lower, cf_width_upper, cond_mean = dict(cf_upper), dict(cf_upper), dict(cf_upper), dict(cf_upper)
    ind_fix = []
    conf_level = [0.025, 0.975]
    measures_all = [cf_upper, cf_lower, ks, pvalue, cv, 
                    cf_upper_upper, cf_upper_lower, cf_lower_upper, 
                    cf_lower_lower, ks_lower, ks_upper, 
                    pvalue_lower,  pvalue_upper,
                    cf_width,  cf_width_lower, 
                    cf_width_upper,  cond_mean]
    
    for i in range(num_group, -1, -1):
        if file_exist:
            try:
                ind_fix.extend(partial_result[str(i)])
            except NameError:
                ind_fix = partial_result[str(i)]
        else:
            try:
                ind_fix.extend(partial_result[i])
            except NameError:
                ind_fix = partial_result[i]
        ind_fix.sort()

        x_temp = x_default[ind_fix]
        # check whether results existing        
        skip_calcul = results_exist(ind_fix, pool_results)
        # print(skip_calcul)

        if skip_calcul == False:
            x_copy = np.copy(x)
            x_copy[ind_fix, :] = x_temp
            # compare results with insignificant parameters fixed
            Nresample = rand.shape[0]
            num_func = len(func)
            total_resample = num_func * Nresample
            pvalue_bt,  ks_bt,  cf_upper_bt, cf_lower_bt, cf_width_bt, y_true_width = \
            np.zeros(total_resample), np.zeros(total_resample), np.zeros(total_resample), np.zeros(total_resample), \
            np.zeros(total_resample), np.zeros(total_resample)
            ## Add the bootstrap of PCE
            for jj in range(num_func):
                fun = func[jj]
                results_fix = fun(x_copy).flatten()
                for ii in range(Nresample):            
                    I = rand[ii]
                    ind_resample = jj * Nresample + ii
                    [cf_lower_bt[ind_resample], cf_upper_bt[ind_resample], ks_bt[ind_resample], pvalue_bt[ind_resample],  y_true_width[ind_resample]] \
                        = error_measure(I, y_true[jj], results_fix, conf_level)
            cf_width_bt = (cf_upper_bt - cf_lower_bt) / y_true_width
            # End for
            
            cf_upper[i], cf_lower[i], ks[i], pvalue[i] = cf_upper_bt.mean(), cf_lower_bt.mean(), ks_bt.mean(), pvalue_bt.mean()
            cf_upper_lower[i], cf_upper_upper[i] = np.quantile(cf_upper_bt, conf_level)
            cf_lower_lower[i], cf_lower_upper[i] = np.quantile(cf_lower_bt, conf_level)
            cf_width[i], cf_width_lower[i], cf_width_upper[i] = cf_width_bt.mean(), *np.quantile(cf_width_bt, conf_level)
            ks_lower[i], ks_upper[i] = np.quantile(ks_bt, conf_level)
            pvalue_lower[i], pvalue_upper[i] = np.quantile(pvalue_bt, conf_level)
            cond_mean[i] = results_fix.mean()

            if len(ind_fix) == x.shape[0]:
                cv[i] = 0
                # cond_mean[i] = func(x_temp)[0][0]
            else:
                mean, variance = cond_moments(fun, x_temp, ind_fix, return_variance=True)
                # cond_mean[i] = mean[0]
                
                cv[i] = (np.sqrt(variance) / mean)[0]
            # End If

            # update pool_results
            measure_list = [measure_ele[i] for measure_ele in measures_all]
            pool_results = pool_update(ind_fix, measure_list, pool_results)
        else:
            # map index to calculated values
            for ele in range(len(measures_all)):
                measures_all[ele][i] = skip_calcul[ele]
        # End if
    # End for()

    names = ['cf_upper', 'cf_lower', 'ks', 'pvalue', 'cv', 
            'cf_upper_upper', 'cf_upper_lower', 'cf_lower_upper', 
            'cf_lower_lower', 'ks_lower', 'ks_upper', 
            'pvalue_lower', 'pvalue_upper',
            'cf_width', 'cf_width_lower', 
            'cf_width_upper', 'cond_mean']
    dict_return = dict(zip(names, measures_all))

    return dict_return, pool_results

def error_measure(I, y_true, results_fix, conf_level):
    """
    Calculate the error measures with a resample dataset.

    Parameters:
    ----------
    I : np.array
        the random index of each  bootstrap
    y_true : list,
            Function results with all x varying (the raw sampling matrix of x)
    result_fix : list,
            Conditional results with all some x fixed

    Returns:
    ----------
    List, values of error measures
    """
    y_true_resample = y_true[I]
    results_fix_resample = results_fix[I]                
    cf_lower_temp, cf_upper_temp = np.quantile(results_fix_resample, conf_level)
    ks_bt_temp, pvalue_bt_temp = stats.ks_2samp(y_true_resample, results_fix_resample)
    y_true_width_temp = np.quantile(y_true_resample, conf_level[1]) - np.quantile(y_true_resample, conf_level[0])
    return [cf_lower_temp, cf_upper_temp, ks_bt_temp, pvalue_bt_temp, y_true_width_temp]

def uncond_cal(y_true, conf_level, rand):
    """
    Calculate the unconditional results
    Parameters:
    ----------
    partial_key: key of the partail_results

    Returns:
    ----------

    """
    # if rand is None:
    #     y_true_bt = y_true
    # elif isinstance(rand, np.ndarray):
    #     y_true_bt = y_true[rand]
    # else:
    #     AssertionError
    y_true_bt = np.zeros(shape=(y_true.shape[0], rand.shape[0], y_true.shape[1]))
    for ii in range(y_true.shape[0]):
        y_true_bt[ii] = y_true[ii][rand]
    uncond_cf_bt = np.quantile(y_true_bt, conf_level, axis=2)
    # import pdb; pdb.set_trace()
    uncond_cf_low, uncond_cf_up = {},  {}
    uncond_cf_low['mean'] = uncond_cf_bt[0].mean()   
    uncond_cf_low['low'], uncond_cf_low['up'] = np.quantile(uncond_cf_bt[0], conf_level)
    uncond_cf_up['mean'] = uncond_cf_bt[1].mean()   
    uncond_cf_up['low'], uncond_cf_up['up'] = np.quantile(uncond_cf_bt[1], conf_level)
    uncond_dict = {
                    'uncond_cf_low' : uncond_cf_low, 
                    'uncond_cf_up' : uncond_cf_up, 
                    'uncond_mean': y_true_bt.mean()
                    }
    return uncond_dict




def results_exist(parms_fixed, pool_results):
    """
    Helper function to determine whether results exist.

    Parameters
    ----------
    parms_fixed : list, 
        Index of parameters to fix

    pool_results : dict, 
        Contains both index of parameters fixed and the corresponding results

    Returns
    -------
    skip_cal : bool
    """ 
    if pool_results == {}:
        skip_cal = False
    elif parms_fixed in pool_results['parms']:
        index_measure = pool_results['parms'].index(parms_fixed) 
        skip_cal = pool_results[f'measures_{index_measure}']
    else:
        skip_cal = False

    return skip_cal


def pool_update(parms_fixed, measure_list, pool_results):
    """Update pool_results with new values.

    Parameters
    ----------
    parms_fixed : list, 
        Index of parameters to fix

    measure_list : list, 
        Measures newly calculated for parameters in parms_fixed

    pool_results : dict, 
        Contains both index of parameters fixed and the corresponding results

    Returns
    ----------
    Updated pool_results
    """
    try:
        pool_results['parms'].append(parms_fixed[:])
    except KeyError:
        pool_results['parms'] = [parms_fixed[:]]      
    index_measure = pool_results['parms'].index(parms_fixed)
    pool_results[f'measures_{index_measure}'] = measure_list

    return pool_results
