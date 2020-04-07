import numpy as np
import pandas as pd
import SALib
import os
import json
import chaospy as cp
import sklearn
from scipy.stats import norm
from toposort import toposort

import saffix
from source_runner import load_parameter_file
import utils.cvg_check as cvg_check

# import parameters
param_file = '../../input/upper v2.csv'
parameters = load_parameter_file(param_file) 
# the parameter: dailyFlowPowerFactor is not included in this case
parameters = parameters[:-3]
len_params = len(parameters)

bounds = parameters.loc[:, 'min':'max'].values
problem = {
    'num_vars': len(parameters),
    'names': parameters.loc[:, 'Veneer_name'].values,
    'bounds': bounds
        }

# import sensitivity indices
sa_dir = '../../output/sensitivity/'
sa_name = '0709_ave_annual_total.txt'
sa = np.loadtxt('{}{}'.format(sa_dir, sa_name))
# partial sort factors
num_boots = sa.shape[0]

# Define helper function for partial sorting
def partial_sort(s, ini_size, num_boots, parameters, conf_level=0.95, step=10):
    partial_order = {}
    for num_pce in range(ini_size, num_boots, step):
        sa_trunc = s[0:num_pce, :]
        len_params = len(parameters)
        rankings = np.zeros([num_pce, len_params])
        ranking_ci = np.zeros([2, len_params])
        for resample in range(num_pce):
            rankings[resample,:] = np.argsort(sa_trunc[resample,:]).argsort()
        ## Generate confidence intervals of rankings based on quantiles (if the num_resample is big enough)
        # ranking_ci = np.quantile(rankings,[(1-conf_level)/2, 0.5 + conf_level/2], axis=0)
        # if the number of resamples is not big enough, the way using quantiles is not recommmended
        # # Generate confidence intervals of rankings based on parametic calculation
        rci = norm.ppf(0.5 + conf_level / 2) * rankings.std(ddof=1, axis=0)
        ranking_ci = [rankings.mean(axis=0) - rci, rankings.mean(axis=0) + rci]
        # import pdb
        # pdb.set_trace()
        ranking_ci = np.rint(ranking_ci)
        ranking_ci[ranking_ci<0] = 0        
        conf_low = ranking_ci[0]
        conf_up = ranking_ci[1]
        rank_conf = {j:None for j in range(len_params)}
        for j in range(len_params):
            rank_conf[j] = [conf_low[j], conf_up[j]]  
        abs_sort= {j:None for j in range(len_params)}

        for m in range(len_params):
            list_temp = np.where(conf_low >= conf_up[m])
            set_temp = set()
            if len(list_temp) > 0:
                for ele in list_temp[0]:
                    set_temp.add(ele)
            abs_sort[m] = set_temp
        order_temp = list(toposort(abs_sort))

        partial_order['result_'+str(num_pce)] = {j: list(order_temp[j]) for j in range(len(order_temp))}
    return partial_order

partial_order = partial_sort(sa, 100, num_boots, parameters, conf_level=0.95, step=10)
fname_split, _ = os.path.splitext(sa_name)        
with open('{}{}{}'.format(sa_dir, fname_split, '_partial_sort.json'), 'w') as fp:
    json.dump(partial_order, fp, indent=2) 

# Check the convergence of partial sorting with the increase of the number of bootstrap
order_temp, num_resample = cvg_check(partial_order)
print(order_temp, num_resample)
# End

