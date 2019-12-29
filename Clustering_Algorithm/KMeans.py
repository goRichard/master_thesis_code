from Clustering_Algorithm.ClusterAlgorithm import *
from data.get_pressure_data import *

# define a list with n_clusters
n_clusters_list = list(range(5, 101))    # get 5 to 100

# run KMeans ++ Algorithm
"""
result_raw = running_KMeans(X, n_clusters_list) # data without preprocessing
result_mms = running_KMeans(X_transformed_mms, n_clusters_list) # data preprocessed with mms
result_ss = running_KMeans(X_transformed_ss, n_clusters_list) # data preprocessed with ss
"""


result_raw = running_KMeans(X_epanet, n_clusters_list) # data without normalization
result_mms = running_KMeans(X_epanet_mms, n_clusters_list) # data preprocessed with mms
result_ss = running_KMeans(X_epanet_ss, n_clusters_list) # data preprocessed with ss
result_normalization = running_KMeans(X_normalized, n_clusters_list) # data normalized

KMeans_dict_raw = result_raw[0]
KMeans_dict_mms = result_mms[0]
KMeans_dict_ss = result_ss[0]
KMeans_dict_normalization = result_normalization[0]

sum_of_squared_distances_raw = result_raw[1]
sum_of_squared_distances_mms = result_mms[1]
sum_of_squared_distances_ss = result_ss[1]
sum_of_squared_distances_normalization = result_normalization[1]

"""
# run KMeans random Algorithm (faster than KMeans ++)
result_raw_random = running_KMeans_random(X, n_clusters_list) # data without preprocessing
result_mms_random = running_KMeans_random(X_transformed_mms, n_clusters_list) # data preprocessed with mms
result_ss_random = running_KMeans_random(X_transformed_ss, n_clusters_list) # data preprocessed with ss

KMeans_dict_random_raw = result_raw_random[0]
KMeans_dict_random_mms = result_mms_random[0]
KMeans_dict_random_ss = result_ss_random[0]

sum_of_squared_distances_random_raw = result_raw_random[1]
sum_of_squared_distances_random_mms = result_mms_random[1]
sum_of_squared_distances_random_ss = result_ss_random[1]
"""


# evaluate the KMeans results with 3 methods:
KMeans_list_1 = evaluate_Algorithm(X_epanet, KMeans_dict_raw, "KMeans")
KMeans_list_2 = evaluate_Algorithm(X_epanet_mms, KMeans_dict_mms, "KMeans")
KMeans_list_3 = evaluate_Algorithm(X_epanet_ss, KMeans_dict_ss, "KMeans")
KMeans_list_4 = evaluate_Algorithm(X_normalized, KMeans_dict_normalization, "KMeans")

# plot optimal k by sum_squared_distances with kmeans ++ by elbow methods
"""
plt.figure(figsize=(10, 8))
plt.subplot(311)
plt.title("Elbow Method for Optimal K by KMeans ++ (raw data)")
plt.plot(n_clusters_list, sum_of_squared_distances_raw, "r-", marker="o")
plt.xlabel("k")
plt.ylabel("sum_of_squared_distances")
plt.legend("raw data")

plt.subplot(312)
plt.plot(n_clusters_list, sum_of_squared_distances_mms, "b-", marker="o")
plt.xlabel("k")
plt.ylabel("sum_of_squared_distances")
plt.legend(["data preprocessing with mms"], loc="upper right")

plt.subplot(313)
plt.plot(n_clusters_list, sum_of_squared_distances_ss, "g-", marker="o")
plt.xlabel("k")
plt.ylabel("sum_of_squared_distances")
plt.legend(["data preprocessing with ss"], loc="upper right")
plt.show()
"""

# plot scores
"""
plt.figure(figsize=(10, 8))
plt.subplot(311)
plt.title("Evaluation for KMeans (silhouette_score)")
plt.plot(n_clusters_list, KMeans_list_1[0], "r-", marker="o")
plt.plot(n_clusters_list, KMeans_list_2[0], "g-", marker="o")
plt.plot(n_clusters_list, KMeans_list_3[0], "b-", marker="o")
plt.xlabel("k")
plt.ylabel("silhouette scores")
plt.legend(["raw data", "data preprocessing with mms", "data preprocessing with ss"], loc="upper left")

plt.subplot(312)
plt.title("Evaluation for KMeans (davies_bouldin_score)")
plt.plot(n_clusters_list, KMeans_list_1[1], "r-", marker="o")
plt.plot(n_clusters_list, KMeans_list_2[1], "g-", marker="o")
plt.plot(n_clusters_list, KMeans_list_3[1], "b-", marker="o")
plt.xlabel("k")
plt.ylabel("davies bouldin scores")
plt.legend(["raw data", "data preprocessing with mms", "data preprocessing with ss"], loc="upper right")

plt.subplot(313)
plt.title("Evaluation for KMeans (calinski_harabasz_score)")
plt.plot(n_clusters_list, KMeans_list_1[2], "r-", marker="o")
plt.plot(n_clusters_list, KMeans_list_2[2], "g-", marker="o")
plt.plot(n_clusters_list, KMeans_list_3[2], "b-", marker="o")
plt.xlabel("k")
plt.ylabel("scores")
plt.legend(["raw data", "data preprocessing with mms", "data preprocessing with ss"], loc="upper left")

plt.show()
"""
