import numpy as np
from scipy.stats import entropy
from bisect import bisect
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

def group_fix(partial_result, func, x, y_true, x_default, option_return='conf', file_exist=False):
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
    option_return: determine what results to return, 
                    default is upper and lower bounds of confidence_intervals, denoted as'conf'
    a: coefficients used in func
    file_exist : Boolean for checking whether the partial results is from calculation
                or reading from the existing file.
    
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
    measure1 = {i: None for i in range(num_group)}
    measure2 = {i: None for i in range(num_group)}
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

        sample_copy[ind_fix, :] = [x_default]
        results_fix = func(sample_copy).flatten()

        # calculate measures of error
        if option_return == 'conf':
            measure1[i] = stats.variation(results_fix)
            measure2[i] = np.quantile(results_fix, [0.025, 0.975])
        elif option_return =='ks':
            measure1[i], measure2[i] = stats.ks_2samp(y_true, results_fix)
        elif option_return == 'raw':
            measure1[i] = results_fix

    return [measure1, measure2]#[co_var] #[ks_stats, ks_pvalue]， result_cond
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
