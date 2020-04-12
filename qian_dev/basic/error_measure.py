import numpy as np
from scipy.stats import entropy
from bisect import bisect
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

def group_fix(partial_result, func, x, y_true, x_default, a=None, file_exist=False):
    """
    Function for compare results between conditioned and unconditioned QoI.
    Fix parameters from the least influential group 
    based on results from partially sorting.
    Four types of error measures will be returned.
    Parameters:
    ===========
    partial_result: dictionary of parameter groups, results of partial sort
    func: function for analysis (analytical formula or model)
    x: numpy array of input, shape of N * D where N is sampling size and 
        D is the number of parameters
    y_true: list of func results with all x varying (the raw sampling matrix of x)
    x_default: scalar or listdefault values of x
    file_exist : Boolean for checking whether the partial results is from calculation
                or reading from the existing file.
    a: coefficients used in func
    
    Return:
    =======
    compare_mean : dict, changes in mean of the func results due to fixing parameters
    compare_mae :dict, changes in absolute mean error of the func results due to fixing parameters
    compare_var : dict, changes in variance of the func results due to fixing parameters
    pearsons : dict, changes in pearson correlation coefficients 
                of the func results due to fixing parameters
    """
    num_group = len(partial_result) - 1
    # store results from fixing parameters in dict
    ks_stats = {i: None for i in range(num_group)}
    ks_pvalue = {i: None for i in range(num_group)}
    co_var = {i: None for i in range(num_group)}
    conf_int = {i: None for i in range(num_group)}
    mean_abs = {i: None for i in range(num_group)}
    result_cond = {i: None for i in range(num_group)}
    # sample_fix = {i: None for i in range(num_group)}
    ind_fix = []
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
        sample_copy = np.copy(x) # for pce, x is sampled by dist.sample

        if  a is None:
            sample_copy[ind_fix, :] = [x_default]
            results_fix = func(sample_copy).flatten()
        else:
            sample_copy[:, ind_fix] = [x_default]
            results_fix = func(sample_copy, a)
        # calculate measures of error
        co_var[i] = stats.variation(results_fix)
        ks_stats[i], ks_pvalue[i] = stats.ks_2samp(y_true, results_fix)
        conf_int[i] = np.quantile(results_fix, [0.10, 0.90])
        temp = np.abs(y_true - results_fix) / y_true
        mean_abs[i] = np.average(temp)
        result_cond[i] = results_fix
    return  [mean_abs,result_cond] #[ks_stats, ks_pvalue] #[conf_int, co_var]
# End group_fix()

def probability_cal(y_true, y_cond, bins=100, epsi=1e-50):
    """
    Calculate probability density of the continuous variables.
    Parameters:
    ===========
    y_true: array, unconditional values of model outputs.
    y_cond: array, conditional values of model outputs with some parameters fixing.
    bins: int, the number of intervals to generate within the range of y_true.
    espi: float, to avoid the probability to be 0.

    Returns:
    ========
    prob: array, the probability density of y_cond.
    """
    if not isinstance(y_cond, np.ndarray):
        raise TypeError('input must be an array')
    else:
        prob = []
        min_y = y_true.min()
        max_y = y_true.max()
        y_intervals = np.linspace(min_y, max_y, bins)
        # sort y in ascending order
        y_cond.sort()
        y_len = len(y_cond)
        prob_temp = [bisect(y_cond, i)/y_len for i in y_intervals]
        prob.append(0)
        prob.extend([prob_temp[i+1] - prob_temp[i] for i in range(bins - 1)])
        prob.append(0)
        prob = np.array(prob)
        prob = prob + epsi
    return prob
# End probability_cal()

def ecdf(yt, yc):
    """
    Calculate probability density of the continuous variables.
    Parameters:
    ===========
    y_true: array, unconditional values of model outputs.
    y_cond: array, conditional values of model outputs with some parameters fixing.
    bins: int, the number of intervals to generate within the range of y_true.

    Returns:
    ========
    cumu_prob: array, the cumulative probability distribution of y_cond.
    """
    if not isinstance(yc, np.ndarray):
        raise TypeError('input must be an array')

    N = len(yt)
    F = np.linspace(1, N, N) / N

    yt = np.flip(np.sort(yt), axis=0)
    yt, iu = np.unique(yt, return_index=True)
    iu = N - 1 - iu

    F = F[iu]
    N = len(F)
    y_cond = np.unique(yc)
    # interpolate the empirical CDF for conditional y
    Fi = np.ones((len(yc), ))

    for j in range(N-1, -1, -1):
        Fi[yc[:]<=yt[j]] = F[j]
    
    Fi[yc[:] < yt[0]] = 0

    return Fi
    min_y = y_true.min()
    max_y = y_true.max()
    # print(min_y, max_y)
    y_intervals = np.linspace(min_y, max_y, bins)
    # sort y in ascending order
    y_cond.sort()
    y_len = len(y_cond)
    cumu_prob =np.array([bisect(y_cond, i)/y_len for i in y_intervals])
    return cumu_prob
# End cumulative_prob
