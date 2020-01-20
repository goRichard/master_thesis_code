import os

data_path = os.getcwd()
file_path = [x for x in os.listdir(data_path) if os.path.isfile(x)]
inp_file_path = [x for x in file_path if os.path.splitext(x)[1] == ".inp"][0]
result_file_paths = [x for x in file_path if os.path.splitext(x)[1] == ".csv"]
result_file_paths_epanet = [x for x in result_file_paths if "epanet" in os.path.splitext(x)[0]]
result_file_paths_wntr = [x for x in result_file_paths if "wntr" in os.path.splitext(x)[0]]


if __name__ == "__main__":

    print("the file path is : {}".format(file_path))
    print("the inp file path is : {}".format(inp_file_path))
    print("the result file path is : {}".format(result_file_paths))
    print("the result file path epanet is : {}".format(result_file_paths_epanet))
    print("the result file path wntr is : {}".format(result_file_paths_wntr))


