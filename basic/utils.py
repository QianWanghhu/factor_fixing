import numpy as np
import pandas as pd

def to_df(measure_dict, measure_name):
    """
    Help funtion to convert the dictionary of results into dataframe.
    """
    measure_df = {}
    k = 0
    for key, value in measure_dict.items():
        measure_df[key] =  pd.DataFrame(value, index=measure_name)
        k += 1
    return measure_df
    
def cvg_check(partial_order):
    order_temp = []
    num_res_change = []
    for key, value in partial_order.items():
        # import pdb
        # pdb.set_trace()
        if not (value in order_temp):
            order_temp.append(value)
            num_res_change.append(key)
    return order_temp, num_res_change      

def names_match(rankings, parameters, fdir, fname):
    """
    This convert the results in partial order of the format index into Veneer_names.
    rankings: dict, partial sort results
    parameters: dataframe, contains names of parameters
    """
    partial_names = {}
    for key, value in rankings.items():
        partial_names[key] = parameters.loc[value, 'Veneer_name'].values.tolist()
    with open(f'{fdir}{fname}', 'w') as fp:
        json.dump(partial_names, fp, indent=2)
    

