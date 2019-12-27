from Cluster_Algorithm.ClusterAlgorithm import *
from data.GetFilePath import *
import pandas as pd
from WaterNetWorkBasics import *
from sklearn.preprocessing import *


c_town = WaterNetWorkBasics("c-town_true_network.inp", "c_town")
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
X_transformed_ss = ss.fit_transform(X)

def epsilon(data, MinPts):
    '''计算最佳半径
    input:  data(mat):训练数据
            MinPts(int):半径内的数据点的个数
    output: eps(float):半径
    '''
    data = data.T
    m, n = np.shape(data)
    xMax = np.max(data, 0)
    xMin = np.min(data, 0)
    eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps


result = find_best_epsilon_formular(X_transformed_ss, 3)
print(result)


