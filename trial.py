from Cluster_Algorithm.ClusterAlgorithm import *
from data.GetFilePath import *
import pandas as pd
from WaterNetWorkBasics import *
from sklearn.preprocessing import *
import functools


c_town = WaterNetWorkBasics(get_inp_flie_path("c-town"), "c_town")
junctions_name_with_demand = c_town.name_list['node_name_list']["junctions_with_demand"]

# define preprocessing methods
mms = MinMaxScaler()
ss = StandardScaler()

# read epanet results
epanet_pressure_results = pd.read_csv(get_result_file_path("epanet"))

# get data
X = epanet_pressure_results.loc[:, junctions_name_with_demand].values.T

# get scaled data
X_transformed_mms = mms.fit_transform(X)

find_best_min_samples_scipy_fixed_X = functools.partial(find_best_min_samples_scipy, X_transformed_mms)
eps_list = np.linspace(0.3, 0.4, 11)
within_eps_list = list(map(find_best_min_samples_scipy_fixed_X, eps_list))
print(len(within_eps_list))




