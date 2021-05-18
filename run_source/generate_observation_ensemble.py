"""
RUN SOURCE to generate observation_ensemble.csv
"""
import pandas as pd
import numpy as np
import veneer
from veneer.pest_runtime import *
from veneer.manage import start,kill_all_now
import os

import source_runner as sr 
from source_runner import parameter_funcs

from modeling_funcs import vs_settings, \
    change_param_values, generate_observation_ensemble, generate_parameter_ensemble,\
        modeling_settings


first_port=15000
num_copies = 1
# define catchment_project and veneer_exe
project_path = 'pest_source/'
# project_path = 'pest_source/'
catchment_project= project_path + 'MW_BASE_RC8_411_4712.rsproj'

# Setup Veneer
# define paths to veneer command and the catchment project
veneer_path = project_path + 'vcmd412/FlowMatters.Source.VeneerCmd.exe'
# veneer_path = project_path + 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
#Now, go ahead and start source
processes, ports = start(catchment_project,
                        n_instances=num_copies,
                        ports=first_port,
                        debug=True,
                        veneer_exe=veneer_path,
                        remote=False,
                        overwrite_plugins=True)

NODEs, things_to_record, criteria, start_date, end_date = modeling_settings()
vs_list = vs_settings(ports, things_to_record)

# generate parameter emsenble
datapath = 'data/'
nsample = 200
param_ensemble = 'samples.csv'
generate_parameter_ensemble(nsample, param_ensemble, datapath, seed=88)
para_info = parameter_funcs.load_parameter_file(datapath + 'parameters.csv')

# obtain the initial values of parameter 
param_names, param_vename_dic, param_vename, param_types = sr.group_parameters(para_info)
initial_values = parameter_funcs.get_initial_param_vals(vs_list[0], param_names, param_vename, param_vename_dic)
# initial_values = obtain_initials(vs_list[0])


# run to generate observation with default parameter values in the model
print('------------------Generate observation with default parameter values-----------------')
retrieve_time = [pd.Timestamp('2000-07-01'), pd.Timestamp('2014-06-30')]

# run to generate observation ensemble with parameter ensemble
print('------------------Generate observation ensemble-----------------')
obs_ensemble_name = 'Tss_124001B'   
parameters = pd.read_csv('samples.csv', index_col='index')

# generate the observation ensemble
def run_obs_ensemble(vs, criteria, start_date, end_date, parameters, 
    obs_ensemble_name, retrieve_time, datapath):
    if not os.path.exists(f'{datapath}{obs_ensemble_name}.csv'):
        load = generate_observation_ensemble(vs_list, criteria, 
            start_date, end_date, parameters, retrieve_time, initial_values, 
                param_vename_dic, param_vename, para_info, fromList=fromList)
        load.to_csv(f'{datapath}{obs_ensemble_name}.csv')
    else:
        print(f'{obs_ensemble_name}.csv exists.')

fromList=True
run_obs_ensemble(vs_list, criteria, start_date, end_date, parameters, 
    obs_ensemble_name, retrieve_time, datapath)

# set parameter to the initial values
for vs in vs_list:
    change_param_values(vs, initial_values, initial_values, param_vename_dic, 
        param_vename, para_info, fromList=fromList, abs_value=True)

kill_all_now(processes)


