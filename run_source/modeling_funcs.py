# RUN SOURCE to generate observation_ensemble.csv
import pandas as pd
import numpy as np
import veneer
from veneer.pest_runtime import *
from veneer.manage import start,kill_all_now
import time
import os
# Import packages
from SALib.sample import latin
from source_runner import *

def modeling_settings():
    """
    Set and return user_defined arguments for model configuration, recording and retrieving results.
    Returns:
    NODEs: list, list of elements to record
    things_to_record, list of dict, to confifure the model
    criteria: dict, criteria used to filter results
    start_date, end_date
    """
    NODEs = ['gauge_124001B_OConnellRvStaffordsCrossing']
    things_to_record = [{'NetworkElement':node,'RecordingVariable':'Constituents@Sediment - Fine@Downstream Flow Mass'} for node in NODEs]
    criteria = {'NetworkElement': NODEs[0],'RecordingVariable':'Constituents@Sediment - Fine@Downstream Flow Mass'}
    start_date = '01/07/1998'; end_date='30/06/2014'
    assert isinstance(start_date, str),"start_date has to be time str."
    assert isinstance(end_date, str),"end_date has to be time str."
    assert isinstance(things_to_record, list),"things_to_record has to be a list of dict."
    return NODEs, things_to_record, criteria, start_date, end_date


def generate_observation_default(v, criteria, start_date, end_date, retrieve_time):
    """
    Run the model to obtain observations without noise.
    Parameters:
    ===========
    vs: veneer object
    criteria: dict, criteria used to configure the model
    start_date, end_date: str of dates when the model simulates, e.g., 
        start_date = '01/07/2014'; end_date='01/07/2016'

    Returns:
    ===========
    load: pd.DataFrame, loads to return
    """
    # store the results into a .csv file
    time_start = time.time()
    load = pd.DataFrame()
    assert isinstance(v, veneer.general.Veneer),"vs has to be an veneer object."
    
    assert isinstance(retrieve_time, list),"retrieve_time has to be a list of timestamp."
    
    print('-----------Run the model to produce synthetic observations-----------')
    v.drop_all_runs()
    v.run_model(params={'NoHardCopyResults':True}, start = start_date, end = end_date) 
    column_names = ['TSS_obs_load']
    # import pdb; pdb.set_trace()
    get_din = v.retrieve_multiple_time_series(criteria=criteria)
    get_din.columns = column_names
    din = get_din.loc[retrieve_time[0]:retrieve_time[1]]
    # store the daily results and the index of sampling
    load = pd.concat([load, din], axis=1)
    v.drop_all_runs()
    time_end = time.time()
    print(f'{time_end - time_start} seconds')
    print('----------Finished generate_observation_noised-----------')
    return load
# End generate_observation_noised()

def generate_observation_ensemble(vs_list, criteria, start_date, end_date, parameters, retrieve_time, 
    initial_vals, param_vename_dic, param_vename, parameter_info, fromList):
    """
    Run the model to obtain observations without noise.
    Parameters:
    ===========
    vs: veneer object
    criteria: dict, criteria used to configure the model
    start_date, end_date: str of dates when the model simulates, e.g., 
        start_date = '01/07/2014'; end_date='01/07/2016'

    Returns:
    ===========
    load: pd.DataFrame, loads to return
    """
    assert isinstance(vs_list, list),"vs has to be a list of veneer objects."
    num_copies = len(vs_list)     
    # install the results into a csv
    load = pd.DataFrame()
    time_start = time.time()
    # parameters = parameters.iloc[0:12, :]
    num_runs = parameters.shape[0]
    group_loops = np.floor_divide(num_runs, num_copies) + 1
    total_runs = 0
    for index_group in range(group_loops):
        group_run_responses = []
        if index_group == (group_loops - 1):
            num_copies_loop = (num_runs - index_group * num_copies)
        else:
            num_copies_loop = num_copies

        for j in range(num_copies_loop):
            total_runs += 1
            if (index_group * num_copies + j) >= num_runs: break

            vs= vs_list[j]
            vs.drop_all_runs()
            parameter_dict = parameters.iloc[total_runs-1]
            # Make sure names of parameters are correct!
            vs = change_param_values(vs, parameter_dict, initial_vals, 
                param_vename_dic, param_vename, parameter_info, fromList=fromList)
            response = vs.run_model(params={'NoHardCopyResults':True}, 
                start=start_date, end=end_date, run_async=True)
            group_run_responses.append(response)

        for j in range(num_copies_loop):
            run_index = index_group * num_copies + j
            if (run_index) >= num_runs: break
                
            vs = vs_list[j]
            r = group_run_responses[j]   
            code = r.getresponse().getcode() # wait until the job finished   
            column_names = [run_index]
            get_din = vs.retrieve_multiple_time_series(criteria=criteria)
            get_din.columns = column_names
            din = get_din.loc[retrieve_time[0]:retrieve_time[1]]
            # store the daily results and the index of sampling
            load = pd.concat([load, din], axis=1)
        
        print(f'Finish {total_runs} runs')

    # kill_all_now()
    time_end = time.time()
    print(f'{time_end - time_start} seconds')
    print('----------Finished generate_observation_ensemble-----------')
    return load
# End generate_observation_ensemble() 

def generate_parameter_ensemble(nsample, param_ensemble, datapath, seed=None):
    """
    The function is used to generate the parameter and data ensemble.
    The parameters are generated with Sobol' sampling or LHS.
    Parameters:
    ===========
    nsample: int, the number of parameter samples to generate (e.g., 512)
    param_ensemble: the name containing the relative path of parameter ensemble
    seed: int, the random seed to generate samples, default is None

    Returns:
    ===========
    None. Save parameter results to the given path.
    """
    fname = param_ensemble
    if not os.path.exists(fname):      
        parameters = pd.read_csv(datapath + 'parameters.csv', index_col = 'Index')

        problem = {
            'num_vars': parameters.shape[0],
            'names': parameters.Veneer_name.values,
            'bounds': parameters.loc[:, ['Min', 'Max']].values
            }
        parameters_ensemble = latin.sample(problem, nsample, seed=88)
        df = pd.DataFrame(data=parameters_ensemble, index = np.arange(nsample), columns=problem['names'])
        df.index.name = 'index'
        df.to_csv(param_ensemble)
    else:
        print(f'The file of parameter ensemble exists under the folder')

def change_param_values(v, pvalue_dict, initial_vals, param_vename_dic, param_vename, parameters, fromList, abs_value=False):
    assert isinstance(v, veneer.general.Veneer),"vs has to be an veneer object."
    for k in range(parameters.shape[0]):
        name = parameters.Veneer_name[k]
        param_new_factor = pvalue_dict[name]
        param_value_ini = initial_vals[name]
        if abs_value:
            param_value_new = param_new_factor
        else:
            param_value_new = [param_new_factor * value for value in param_value_ini]
        #set parameter values
        if name in param_vename_dic[param_vename[0]]:
            assert v.model.catchment.generation.set_param_values(name, param_value_new, fromList=fromList)
        if name in  param_vename_dic[param_vename[1]]:
            assert v.model.link.constituents.set_param_values(name, param_value_new,fromList=fromList)
        if name in  param_vename_dic[param_vename[2]]:
            assert v.model.node.set_param_values(name, param_value_new,fromList=fromList)
        if name in  param_vename_dic[param_vename[3]]:
            assert v.model.link.routing.set_param_values(name, param_value_new,fromList=fromList)

    return v

def vs_settings(ports, things_to_record):
    """
    Set up the vs objects.
    Parameters:
    ===========
    ports: list, list of ports to generate veneer objects
    things_to_record: dict, configurations of vs objects

    Returns:
    ===========
    vs_list: list, list of vs ojeects

    """
    
    vs_list = []
    assert isinstance(ports, list),"vs has to be a list of int."

    for veneer_port in ports:
        # veneer_port = first_port 
        vs_list.append(veneer.Veneer(port=veneer_port))
    
    for vs in vs_list:
        vs.configure_recording(disable=[{}], enable=things_to_record)
    return vs_list
