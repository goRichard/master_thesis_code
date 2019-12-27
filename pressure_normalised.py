from WaterNetWorkBasics import *
from data.get_file_path import *
import pandas as pd

# get the pressure simulation results
epanet_pressure_results = pd.read_csv(get_result_file_path("epanet"))

# define class
c_town_plot = PlotWaterNetWorks(get_inp_file_path("c-town"), "c_town")
junctions_name_with_demand = c_town_plot.name_list['node_name_list']["junctions_with_demand"]
junction_names = c_town_plot.name_list['node_name_list']['junction_names']
time_series = c_town_plot.get_time_steps

# get the pressure simulation results w.r.t junctions with demand
epanet_pressure_results_with_demand = epanet_pressure_results.loc[:, junctions_name_with_demand].values  # (169, 334)


# note that the shape of data (334,169), 334 junctions with demand and 169 simulation time step
X_epanet = epanet_pressure_results_with_demand.T  # (334, 169)

X_epanet_normalised = normalise_pressure(X_epanet)