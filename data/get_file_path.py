import os

# get the inp file path

global data_path
data_path = "/home/ruizhiluo/Documents/git/master_thesis/Code/data"


def get_file_path(path):
    return [x for x in os.listdir(path) if os.path.isfile(x)]


def get_inp_file_path(path):
    file_path = get_file_path(path)
    global inp_file_path
    inp_file_path = [x for x in file_path if os.path.splitext(x)[1] == ".inp"]
    return inp_file_path


def get_result_file_path(name_of_simulator,path):
    file_path = get_file_path(path)
    result_file_paths = [x for x in file_path if os.path.splitext(x)[1] == ".csv"]
    for result_file_path in result_file_paths:
        if name_of_simulator in os.path.splitext(result_file_path)[0]:
            return result_file_path


if __name__ == "__main__":
    file_path = get_file_path(data_path)
    inp_file_path = get_inp_file_path(data_path)
    result_file_path = get_result_file_path("epanet", data_path)
    print("the file path is : {}".format(file_path))
    print("the inp file path is : {}".format(inp_file_path))
    print("the result file path is : {}".format(result_file_path))
