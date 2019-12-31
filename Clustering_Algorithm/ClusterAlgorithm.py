import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score
from sklearn.metrics.cluster import silhouette_score
from tqdm import *
import math
import scipy.spatial
from data.get_pressure_data import *
from Clustering_Algorithm.get_connectivity_matrix import *
from sklearn.metrics import pairwise_distances

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

# junction name
junction_names = np.array(ctown.name_list["node_name_list"]["junction_names"])

# pipe name
pipe_names = ctown.name_list["link_name_list"]["pipe_names"]
pipes = ctown.get_link(pipe_names)
pipe_start_nodes = [pipe.start_node_name for pipe in pipes]
pipe_end_nodes = [pipe.end_node_name for pipe in pipes]


# initialise a connectivity matrix
# A is the adjacency  matrix
# A is a symmetric matrix with diagonal entries equal zero
A = np.zeros([n_junctions, n_junctions])
for pipe_start_node, pipe_end_node in zip(pipe_start_nodes, pipe_end_nodes):
    i = np.where(junction_names == pipe_start_node)
    j = np.where(junction_names == pipe_end_node)
    A[i, j] = 1
    A[j, i] = 1

# calculate the pairwise distance between each pair of points
# X_normalized shape (388, 169) 388 data and each data has 169 feature
# W is the similarity matrix
# create similarity matrix based on k-nearest neighbour
# the diagonal of W_knn is 0

W_knn = pairwise_distances(X_normalized, metric="euclidean")  # shape (388,388)

# get the weighted matrix by considering the distance between two connected points
W = A * W_knn


def get_diagonal(matrix):
    # assert isinstance(matrix, np.ndarray)
    col, row = matrix.shape
    diagonal = []
    for i in range(col):
        for j in range(row):
            if i == j:
                diagonal.append(matrix[i, j])

    return np.array([diagonal])


def matrix_normalize(matrix, D):
    assert isinstance(matrix, np.ndarray)
    diagonal = np.diag(1 / np.sqrt(D))
    return np.dot(np.dot(diagonal, matrix), diagonal)


def running_KMeans(data, n_clusters_list):
    """
    run the KMeans cluster with different n_clusters searval time and print the result out
    """
    KMeans_dict = dict()
    sum_of_squared_distances = []
    for n_clusters in tqdm(n_clusters_list):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        KMeans_dict[n_clusters] = kmeans.labels_
        sum_of_squared_distances.append(kmeans.inertia_)
    return [KMeans_dict, sum_of_squared_distances]


def running_KMeans_random(data, n_clusters_list):
    KMeans_random_dict = dict()
    sum_of_squared_distances_random = []
    for n_clusters in tqdm(n_clusters_list):
        kmeans_random = KMeans(n_clusters=n_clusters, init='random')
        kmeans_random.fit(data)
        KMeans_random_dict[n_clusters] = kmeans_random.labels_
        sum_of_squared_distances_random.append(kmeans_random.inertia_)
    return [KMeans_random_dict, sum_of_squared_distances_random]


def running_agg_with_connectivity_matrix(data, n_clusters_list, which_matrix):
    """
    do not allow to predict the new data, use fit_predict method
    """
    Agg_dict = dict()
    if which_matrix == "adjacency":
        for n_clusters in tqdm(n_clusters_list):
            # linkage = ward will minimize the variance within cluster
            agg = AgglomerativeClustering(n_clusters=n_clusters, connectivity=A, linkage="ward")
            labels = agg.fit_predict(data)
            Agg_dict[n_clusters] = labels
    elif which_matrix == "W_knn":
        for n_clusters in tqdm(n_clusters_list):
            # linkage = ward will minimize the variance within cluster
            agg = AgglomerativeClustering(n_clusters=n_clusters, connectivity=W_knn, linkage="ward")
            labels = agg.fit_predict(data)
            Agg_dict[n_clusters] = labels
    elif which_matrix == "W":
        for n_clusters in tqdm(n_clusters_list):
            # linkage = ward will minimize the variance within cluster
            agg = AgglomerativeClustering(n_clusters=n_clusters, connectivity=W, linkage="ward")
            labels = agg.fit_predict(data)
            Agg_dict[n_clusters] = labels
    return Agg_dict


def running_agg_without_connectivity_matrix(data, n_clusters_list):
    """
    do not allow to predict the new data, use fit_predict method
    """
    Agg_dict = dict()
    for n_clusters in tqdm(n_clusters_list):
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = agg.fit_predict(data)
        Agg_dict[n_clusters] = labels
    return Agg_dict


def running_DBSCAN(data, eps_list, min_samples_list):
    DBSCAN_dict_labels = dict()
    DBSCAN_dict_points = dict()

    for eps, min_samples in tqdm(zip(eps_list, min_samples_list)):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        points, counts = np.unique(labels, return_counts=True)
        DBSCAN_dict_labels[(eps, min_samples)] = labels
        #DBSCAN_dict_points[points] = counts
    return DBSCAN_dict_labels


def running_GMM(data, n_components_list):
    gmm_dict = dict()
    for n_components in tqdm(n_components_list):
        gmm = GaussianMixture(n_components=n_components, covariance_type="full")
        gmm.fit(data)
        labels = gmm.fit_predict(data)
        gmm_dict[n_components] = labels
    return gmm_dict


# return a dictionary with {cluster_indedata: [name of junctions within this cluster]}
def cluster_dictionary(junction_list, labels, n_clusters):
    """
    :param junction_list:   list, the name of junctions with demand
    :param labels:    list, the cluster result from KMeans, Agglomerative or DBSCAN
    :param n_clusters:  number of clusters defined in KMeans or Agglomerative
    :return:  dictionary, {indedata of cluster: [junction name]}
    """

    cluster_junction_dictionary = dict()
    for cluster in range(n_clusters):
        label_cluster = np.where(labels == cluster)[0]
        cluster_junction_dictionary[cluster] = list((np.array(junction_list)[label_cluster]))
    return cluster_junction_dictionary


# evaluate the clustering algorithm
def evaluate_Algorithm(data, algorithm_dict, name_of_algorithm):
    silhouette_score_list = []
    davies_bouldin_score_list = []
    calinski_harabasz_score_list = []
    print("Result of {}: \n".format(name_of_algorithm))
    for n_cluster in tqdm(algorithm_dict.keys()):
        print("cluster: {}".format(n_cluster))
        print("silhouette_score: {}     ".format(silhouette_score(data, algorithm_dict[n_cluster])))
        print("davies_bouldin_score: {}     ".format(davies_bouldin_score(data, algorithm_dict[n_cluster])))
        print("calinski_harabasz_score: {}\n".format(calinski_harabasz_score(data, algorithm_dict[n_cluster])))
        silhouette_score_list.append(silhouette_score(data, algorithm_dict[n_cluster]))
        davies_bouldin_score_list.append(davies_bouldin_score(data, algorithm_dict[n_cluster]))
        calinski_harabasz_score_list.append(calinski_harabasz_score(data, algorithm_dict[n_cluster]))
        print("*" * 50)

    return [silhouette_score_list, davies_bouldin_score_list, calinski_harabasz_score_list]


def silhouette_visualizer(data, n_clusters, Algorithm):
    algorithm = Algorithm(n_clusters=n_clusters)
    visualizer = SilhouetteVisualizer(algorithm, colors='yellowbrick')
    visualizer.fit(data)
    visualizer.show()


def plot_junctons_pressure_within_same_cluster(result_dataframe, labels, junction_name_list_with_demand, n_clusters,
                                               cluster_indedata,
                                               cluster_method_name):
    """

    :param cluster_method_name:  string, name of the clustering method used
    :param labels:  list, the cluster result from KMeans, Agglomerative or DBSCAN
    :param junction_name_list_with_demand:  list, names of junctions with demand
    :param n_clusters:   int, the number of clusters defined in KMeans or Agglomerative
    :param cluster_indedata: int, the indedata of clusters
    :return:  plot of the pressure of junctions within same cluster
    """
    cluster_junction_dict = cluster_dictionary(junction_name_list_with_demand, labels, n_clusters)
    junctions_names_within_same_cluster = cluster_junction_dict[cluster_indedata]
    junction_pressure_within_same_cluster = result_dataframe.loc[:, junctions_names_within_same_cluster].values
    data = np.linspace(0, 169, 170)[1:]
    plt.ion()
    plt.figure(figsize=[20, 4])
    plt.title("Junctions' Pressure within Cluster {} ({}, n_clusters: {}) ".format(cluster_indedata, cluster_method_name,
                                                                                   n_clusters))
    for i in range(len(junctions_names_within_same_cluster)):
        plt.plot(data, junction_pressure_within_same_cluster[:, i])
    plt.datalabel("simulation time (h)")
    plt.ylabel("pressure (m)")
    plt.legend(labels=[junctions_names for junctions_names in junctions_names_within_same_cluster])
    plt.grid(True)
    plt.ioff()
    plt.show()


def find_same_cluster_of_different_method(cluster_dictionary_1, cluster_dictionary_2, indedata):
    set_a = set(cluster_dictionary_1[indedata])
    length = 0
    for key in cluster_dictionary_2.keys():
        set_b = set(cluster_dictionary_2[key])
        intersection = list(set_a.intersection(set_b))
        if length < len(intersection):
            length = len(intersection)
            i = key
    return i


def add_noise_to_pressure(pressure_result_dataframe):
    # pressure_result_dataframe = pressure_result_dataframe.reshape(-1,1)
    noise = np.random.uniform(0, 10, pressure_result_dataframe.shape[0]).reshape(-1, 1)
    return pressure_result_dataframe + noise


# methods for DBSCAN to find the best epsilon
def find_best_epsilon_formular(data, MinPts):
    """
    Calculate the best radius
    input:  data(mat): training data
            MinPts(int): the minimum points inside circle with radius
    output: eps(float): radius
    using the algorithm from internet, the meaning
    this method must use the data preprocessed with standard scaler
    which means data has zero mean and variance 1
    """
    data = data.T
    m, n = np.shape(data)
    print("m,n {} {}".format(m, n))
    dataMadata = np.madata(data, 0)
    dataMin = np.min(data, 0)
    # range flow error when the parameter of gamma function reaches more than 350
    eps = ((np.prod(dataMadata - dataMin) * MinPts * (math.gamma(n*0.5+1))) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps


def find_best_epsilon_neighboring(data, n_neighbors):
    k_neighbours = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    """
    the kneighbors method returns two arrays, 
    one which contains the distance to the closest n_neighbors points and 
    the other which contains the indedata for each of those points.
    """
    distances, indices = k_neighbours.kneighbors()
    ## return the distances of the k nearest neighbor of each point
    ## the values of each row in distances array are in a order of closest to farthest neighbouring points w.r.t
    ## single point, indices are the corresponding serial number

    distances = np.sort(distances, adatais=0)
    # sort distances along adatais 0
    distances = distances[:, n_neighbors - 1]
    # 选择最外面的一列，行已经通过np.sort从小到大排列完成，现在需要取的是在点 a 的 k 个紧邻中最远的那一个， 即将最外面的一列plot出来得到的
    # elbow plot 中的拐点对应的值为eps，同时该eps对应的min points  应该是 紧邻的数量 k
    # 拐点处距离陡增，证明点A大部分的紧邻都落在这个拐点所对应的距离内，所以拐点对应的距离应该就为紧邻数量k所对应的最佳eps
    return distances


def find_best_min_samples(data, eps):
    within_eps = []
    row = data.shape[0]
    for i in range(row):
        distance = np.sqrt(np.sum((data[i, :] - data) ** 2, adatais=1))
        within_eps.append(len(list(filter(lambda n: n <= eps, distance))))

    return within_eps


def find_best_min_samples_scipy(data, eps):
    """

    :param data: data
    :param eps: the eps radius
    :return: list, contains the number of samples inside eps as a single element
    """
    distance_matridata = scipy.spatial.distance.cdist(data, data)
    distance_within_eps_list = []
    for i in range(distance_matridata.shape[0]):
        distance = distance_matridata[i, :]
        distance_within_eps_list.append(len(distance[distance <= eps]))
    return distance_within_eps_list


if __name__ == "__main__":
    GetPressureData = GetPressureData("c_town", "epanet")
    # get data
    data_epanet = GetPressureData.get_data
    data_epanet_mms = get_data_scaled(data_epanet)[0]
    data_epanet_ss = get_data_scaled(data_epanet)[1]
    data_normalized = data_normalization(data_epanet)

    model = AgglomerativeClustering()
    #visualizer_3 = SilhouetteVisualizer(model, colors="yellowbrick")
    #visualizer_3.fit(gd.data_normalized_mms)
    #visualizer_3.show()

    visualizer_3 = KElbowVisualizer(model, k=(5,101), metric="calinski_harabasz")
    visualizer_3.fit(data_normalized)
    visualizer_3.show()

    """

    visualizer_2 = KElbowVisualizer(model, k=(5, 101), metric="calinski_harabasz")
    visualizer_2.fit(data_transformed_mms)
    visualizer_2.show()

    visualizer_3 = KElbowVisualizer(model, k=(5, 101), metric='silhouette')
    visualizer_3.fit(data_transformed_ss)
    visualizer_3.show()

    visualizer_4 = KElbowVisualizer(model, k=(5, 101), metric="calinski_harabasz")
    visualizer_4.fit(data_transformed_ss)
    visualizer_4.show()

    visualizer_5 = KElbowVisualizer(model, k=(5, 101), metric='silhouette')
    visualizer_5.fit(data)
    visualizer_5.show()

    visualizer_6 = KElbowVisualizer(model, k=(5, 101), metric="calinski_harabasz")
    visualizer_6.fit(data)
    visualizer_6.show()
    """





