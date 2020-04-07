import numpy as np
from scipy.stats import norm
from toposort import toposort

# Define helper function for partial sorting
def partial_rank(sa, len_params, conf_level=0.95):
    """
    Help function for partial ranking.
    Parameters:
    ===========
    sa: ndarray, matrix of sensitivity indices, 
        of shape (N * P) where N is the number of bootstraps, P is the number of parameters
    len_params: int, the number of parameters
    conf_level: float, the confidence level used for the calculation of confidence intervals of rankings

    Returns:
    ========
    partial_rankings: dict, dictionary of partial rankings of parameters
                    e.g. {0:['2'], 1:['0', '1']} 
                        which means the parameter of index 2 is in group 0 which is more sensitive than the group 1 
    """
    num_boot = sa.shape[0]
    print(num_boot)
    rankings = np.zeros([num_boot, len_params])
    ranking_ci = np.zeros([2, len_params])
    for resample in range(num_boot):
        rankings[resample,:] = np.argsort(sa[resample,:]).argsort()

    ## Generate confidence intervals of rankings based on quantiles (if the num_resample is big enough)
    # ranking_ci = np.quantile(rankings,[(1-conf_level)/2, 0.5 + conf_level/2], axis=0)
    rci = norm.ppf(0.5 + conf_level / 2) * rankings.std(ddof=1, axis=0)
    ranking_ci = [rankings.mean(axis=0) - rci, rankings.mean(axis=0) + rci]
    # ranking_ci = np.rint(ranking_ci)
    # ranking_ci[ranking_ci<0] = 0        
    conf_low = ranking_ci[0]
    conf_up = ranking_ci[1]
    rank_conf = {j:None for j in range(len_params)}
    abs_sort= {j:None for j in range(len_params)}

    for j in range(len_params):
        rank_conf[j] = [conf_low[j], conf_up[j]]  
    # End for

    for m in range(len_params):
        list_temp = np.where(conf_low >= conf_up[m])
        set_temp = set()
        if len(list_temp) > 0:
            for ele in list_temp[0]:
                set_temp.add(ele)
        abs_sort[m] = set_temp
        # End if
    # End for

    order_temp = list(toposort(abs_sort))
    partial_rankings = {j: list(order_temp[j]) for j in range(len(order_temp))}
    return partial_rankings