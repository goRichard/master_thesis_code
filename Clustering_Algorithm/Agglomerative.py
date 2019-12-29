from Clustering_Algorithm.ClusterAlgorithm import *
from data.get_pressure_data import *

# define a list with n_clusters
n_clusters_list = list(range(5, 101))    # get 5 to 100

# run Agglomerative Algorithm
"""
Agg_dict_raw = running_agg_with_connectivity_matrix(X, n_clusters_list) # data without preprocessing
Agg_dict_mms = running_agg_with_connectivity_matrix(X_transformed_mms, n_clusters_list) # data preprocessed with mms
Agg_dict_ss = running_agg_with_connectivity_matrix(X_transformed_ss, n_clusters_list) # data preprocessed with ss
"""
Agg_dict_raw = running_agg_with_connectivity_matrix(X_normalized, n_clusters_list) # data without preprocessing
Agg_dict_mms = running_agg_with_connectivity_matrix(X_epanet_mms, n_clusters_list) # data preprocessed with mms
Agg_dict_ss = running_agg_with_connectivity_matrix(X_epanet_ss, n_clusters_list) # data preprocessed with ss

# evaluate the Agg results with 3 methods:
Agg_list_1 = evaluate_Algorithm(X_normalized, Agg_dict_raw, "Agg")
Agg_list_2 = evaluate_Algorithm(X_epanet_mms, Agg_dict_mms, "Agg")
Agg_list_3 = evaluate_Algorithm(X_epanet_ss, Agg_dict_ss, "Agg")

# plot scores
plt.figure(figsize=(10, 8))
plt.subplot(311)
plt.title("Evaluation for Agglomerative (silhouette_score)")
plt.plot(n_clusters_list, Agg_list_1[0], "r-", marker="o")
plt.plot(n_clusters_list, Agg_list_2[0], "g-", marker="o")
plt.plot(n_clusters_list, Agg_list_3[0], "b-", marker="o")
plt.xlabel("k")
plt.ylabel("scores")
plt.legend(["raw data", "data preprocessing with mms", "data preprocessing with ss"], loc="upper left")

plt.subplot(312)
plt.title("Evaluation for Agglomerative (davies_bouldin_score)")
plt.plot(n_clusters_list, Agg_list_1[1], "r-", marker="o")
plt.plot(n_clusters_list, Agg_list_2[1], "g-", marker="o")
plt.plot(n_clusters_list, Agg_list_3[1], "b-", marker="o")
plt.xlabel("k")
plt.ylabel("scores")
plt.legend(["raw data", "data preprocessing with mms", "data preprocessing with ss"], loc="upper left")

plt.subplot(313)
plt.title("Evaluation for Agglomerative (calinski_harabasz_score)")
plt.plot(n_clusters_list, Agg_list_1[2], "r-", marker="o")
plt.plot(n_clusters_list, Agg_list_2[2], "g-", marker="o")
plt.plot(n_clusters_list, Agg_list_3[2], "b-", marker="o")
plt.xlabel("k")
plt.ylabel("scores")
plt.legend(["raw data", "data preprocessing with mms", "data preprocessing with ss"], loc="upper left")
plt.show()