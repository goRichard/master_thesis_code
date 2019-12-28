from data.get_file_path import *
from WaterNetWorkBasics import *
import os

data_path = os.getcwd()

file_path = [x for x in os.listdir(data_path) if os.path.isfile(x)]
inp_file_path = [x for x in file_path if os.path.splitext(x)[1] == ".inp"]
result_file_paths = [x for x in file_path if os.path.splitext(x)[1] == ".csv"]

for result_file_path in result_file_paths:
    if "epanet" in os.path.splitext(result_file_path)[0]:
        result_file_path_epanet = result_file_path
    else:
        result_file_path_wntr = result_file_path


# get the number of junctions
ctown = WaterNetWorkBasics(inp_file_path, "c_town")
n_junctions = ctown.describe["junctions"]
print(n_junctions)

# junction name
junction_names = np.array(ctown.name_list["node_name_list"]["junction_names"])

# pipe name
pipe_names = ctown.name_list["link_name_list"]["pipe_names"]
pipes = ctown.get_link(pipe_names)
pipe_start_nodes = [pipe.start_node_name for pipe in pipes]
pipe_end_nodes = [pipe.end_node_name for pipe in pipes]
print(pipe_end_nodes)


# initialise a connectivity matrix
A = np.zeros([n_junctions, n_junctions])
for pipe_start_node, pipe_end_node in zip(pipe_start_nodes, pipe_end_nodes):
    i = np.where(junction_names == pipe_start_node)
    j = np.where(junction_names == pipe_end_node)
    A[i, j] = 1
    A[j, i] = 1

print(A)
plt.figure()
plt.spy(A)
plt.show()
