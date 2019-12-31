from Clustering_Algorithm.ClusterAlgorithm import *
n_clusters_list = list(range(5,30))
agg_dict1 = running_agg_with_connectivity_matrix(X_normalized, n_clusters_list, "W")
