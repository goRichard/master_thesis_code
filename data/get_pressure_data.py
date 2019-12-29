from data.get_file_path import *
import pandas as pd
from WaterNetWorkBasics import *
from sklearn.preprocessing import *


def data_normalization(data):
    scaling_factor = data.sum(axis=1) / data.shape[1]
    data_normalized = np.zeros(data.shape)
    for i in range(data.shape[0]):
        data_normalized[i, :] = data[i, :] / scaling_factor[i]
    return data_normalized


def get_data_scaled(data):
    # get scaled data
    mms = MinMaxScaler()
    ss = StandardScaler()
    X_transformed_mms = mms.fit_transform(data)
    X_transformed_ss = ss.fit_transform(data)
    return [X_transformed_mms, X_transformed_ss]


class GetPressureData(object):

    # define preprocessing methods

    def __init__(self, town_name, name_of_simulator):
        # get file path
        self.town_name = town_name
        self.name_of_simulator = name_of_simulator
        self.ctown = WaterNetWorkBasics(inp_file_path, self.town_name)
        self.junctions_names = self.ctown.name_list['node_name_list']["junction_names"]
        self.time_steps = self.ctown.get_time_steps

    @property
    def get_data(self):
        # read epanet results
        print(result_file_paths_epanet)
        epanet_pressure_results = pd.read_csv(result_file_paths_epanet[0])
        wntr_pressure_results = pd.read_csv(result_file_paths_wntr[0])
        # get data
        if self.name_of_simulator == "epanet":
            X_epanet = epanet_pressure_results.loc[:, self.junctions_names].values.T
            return X_epanet
        else:
            X_wntr = wntr_pressure_results.loc[:, self.junctions_names].values.T
            return X_wntr


# plot certain pressure
GetPressureData = GetPressureData("c_town", "epanet")

# get time steps
time_step = GetPressureData.time_steps

# get data
X_epanet = GetPressureData.get_data
X_epanet_mms = get_data_scaled(X_epanet)[0]
X_epanet_ss = get_data_scaled(X_epanet)[1]
X_normalized = data_normalization(X_epanet)

if __name__ == "__main__":
    # plot the result
    plt.figure()
    plt.title("pressure comparison")

    plt.subplot(311)
    plt.plot(time_step, X_epanet[1, :], "r-", label="normal pressure")
    plt.xlabel("time step")
    plt.ylabel("pressure")
    plt.legend()
    plt.grid(True)

    plt.subplot(312)
    plt.plot(time_step, X_epanet_mms[1, :], "b-", label="min max scale pressure")
    plt.plot(time_step, X_epanet_ss[1, :], "g-", label="standard scale pressure")
    plt.xlabel("time step")
    plt.ylabel("pressure")
    plt.legend()
    plt.grid(True)

    plt.subplot(313)
    plt.plot(time_step, X_normalized[1, :], "r-", label="normalized pressure")
    plt.xlabel("time step")
    plt.ylabel("pressure")
    plt.legend()
    plt.grid(True)
    plt.show()
