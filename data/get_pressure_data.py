from data.get_file_path import *
import pandas as pd
from WaterNetWorkBasics import *
from sklearn.preprocessing import *


class GetPressureData(object):

    # define preprocessing methods

    def __init__(self, town_name, name_of_simulator):
        # get file path
        file_path = get_inp_file_path(data_path)
        print(file_path)
        self.town_name = town_name
        self.name_of_simulator = name_of_simulator
        self.ctown = WaterNetWorkBasics(file_path, self.town_name)
        self.junctions_names = self.ctown.name_list['node_name_list']["junction_names"]
        # read epanet results
        self.result_file_path = get_result_file_path(self.name_of_simulator, data_path)
        epanet_pressure_results = pd.read_csv(self.result_file_path)
        # get data
        self.X = epanet_pressure_results.loc[:, self.junctions_names].values.T
        # get scaled data
        mms = MinMaxScaler()
        ss = StandardScaler()
        self.X_transformed_mms = mms.fit_transform(self.X)
        self.X_transformed_ss = ss.fit_transform(self.X)

    def data_normalization(self):
        scaling_factor = self.X.sum(axis=1) / self.X.shape[1]
        X_normalized = np.zeros(self.X.shape)
        for i in range(self.X.shape[0]):
            X_normalized[i, :] = self.X[i, :] / scaling_factor[i]
        return X_normalized


# plot certain pressure
if __name__ == "__main__":
    get_data = GetPressureData("c_town", "epanet")
    # get time steps
    time_step = get_data.ctown.get_time_steps
    # plot the result
    plt.figure()
    plt.title("pressure comparison")

    plt.subplot(311)
    plt.plot(time_step, get_data.X[1, :], "r-", label="normal pressure")
    plt.xlabel("time step")
    plt.ylabel("pressure")
    plt.legend()
    plt.grid(True)

    plt.subplot(312)
    plt.plot(time_step, get_data.X_transformed_mms[1, :], "b-", label="min max scale pressure")
    plt.plot(time_step, get_data.X_transformed_ss[1, :], "g-", label="standard scale pressure")
    plt.xlabel("time step")
    plt.ylabel("pressure")
    plt.legend()
    plt.grid(True)

    plt.subplot(313)
    plt.plot(time_step, get_data.data_normalization()[1, :], "r-", label="normalized pressure")
    plt.xlabel("time step")
    plt.ylabel("pressure")
    plt.legend()
    plt.grid(True)
    plt.show()
