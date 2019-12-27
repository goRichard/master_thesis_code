from WaterNetWorkBasics import *
from data.get_file_path import *


class GetConnectivityMatrix(object):

    def __init__(self, name_of_town):
        self.name = name_of_town
        print(data_path)
        self.file_path = get_inp_file_path(data_path)
        print(self.file_path)
        self.ctown = WaterNetWorkBasics(self.file_path, self.name)
        # initialise a connectivity matrix
        self.n_junctions = self.ctown.describe["junctions"]
        # junction name
        self.junction_names = np.array(self.ctown.name_list["node_name_list"]["junction_names"])
        # pipe name
        pipe_names = self.ctown.name_list["link_name_list"]["pipe_names"]
        pipes = self.ctown.get_link(pipe_names)
        self.pipe_start_nodes = [pipe.start_node_name for pipe in pipes]
        self.pipe_end_nodes = [pipe.end_node_name for pipe in pipes]

    def get_matrix(self):
        A = np.zeros([self.n_junctions, self.n_junctions])
        for pipe_start_node, pipe_end_node in zip(self.pipe_start_nodes, self.pipe_end_nodes):
            i = np.where(self.junction_names == pipe_start_node)
            j = np.where(self.junction_names == pipe_end_node)
            A[i, j] = 1
            A[j, i] = 1

        print(A)
        return A


if __name__ == "__main__":
    get_connectivity_matrix = GetConnectivityMatrix("c_town")
    matrix = get_connectivity_matrix.get_matrix()
    print(matrix)
    plt.figure()
    plt.spy(matrix)
    plt.show()
