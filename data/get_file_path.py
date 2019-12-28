import os

data_path = "/home/ruizhiluo/Documents/git/master_thesis/master_thesis_code/data"
file_path = [x for x in os.listdir(data_path) if os.path.isfile(x)]
inp_file_path = [x for x in file_path if os.path.splitext(x)[1] == ".inp"][0]
result_file_paths = [x for x in file_path if os.path.splitext(x)[1] == ".csv"]

for result_file_path in result_file_paths:
    if "epanet" in os.path.splitext(result_file_path)[0]:
        result_file_path_epanet = result_file_path
    else:
        result_file_path_wntr = result_file_path

if __name__ == "__main__":

    print("the file path is : {}".format(file_path))
    print("the inp file path is : {}".format(inp_file_path))
    print("the result file path is : {}".format(result_file_path))



