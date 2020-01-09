from Clustering_Algorithm.ClusterAlgorithm import *
from data.get_pressure_data import *
import csv

# define a list with n_clusters
n_clusters_list = list(range(2, 101))    # n_clusters from 2 to 100

# run KMeans ++ Algorithm

#result_raw = running_KMeans(X_epanet, n_clusters_list) # data without normalization
#result_mms = running_KMeans(X_epanet_mms, n_clusters_list) # data preprocessed with mms
#result_ss = running_KMeans(X_epanet_ss, n_clusters_list) # data preprocessed with ss
result_normalization = running_KMeans(X_normalized.T, n_clusters_list) # data normalized
#result_reduced = running_KMeans(X_normalized.T, n_clusters_list)

#KMeans_dict_raw = result_raw[0]
#KMeans_dict_mms = result_mms[0]
#KMeans_dict_ss = result_ss[0]
KMeans_dict_normalization = result_normalization[0]

#sum_of_squared_distances_raw = result_raw[1]
#sum_of_squared_distances_mms = result_mms[1]
#sum_of_squared_distances_ss = result_ss[1]
sum_of_squared_distances_normalization = result_normalization[1]


# evaluate the KMeans results with 3 methods and store the value in .csv file
# create the csv file
f1 = open("cluster_result/K-Means/kmeans_normalization_new.csv", "w")
#f2 = open("kmeans_score_mms.csv", "w")
#f3 = open("kmeans_score_ss.csv", "w")
#f4 = open("kmeans_score_normalization.csv", "w")

writer1 = csv.writer(f1)
#writer2 = csv.writer(f2)
#writer3 = csv.writer(f3)
#writer4 = csv.writer(f4)

writer1.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer2.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer3.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer4.writerow(["n_cluster", "Silhouette", "davies", "calinski"])


KMeans_list_1 = evaluate_Algorithm(X_normalized, KMeans_dict_normalization, "KMeans")
#KMeans_list_2 = evaluate_Algorithm(X_epanet_mms, KMeans_dict_mms, "KMeans")
#KMeans_list_3 = evaluate_Algorithm(X_epanet_ss, KMeans_dict_ss, "KMeans")
#KMeans_list_4 = evaluate_Algorithm(X_normalized, KMeans_dict_normalization, "KMeans")

for i in range(len(n_clusters_list)):
    writer1.writerow([n_clusters_list[i], KMeans_list_1[0][i], KMeans_list_1[1][i], KMeans_list_1[2][i]])
    #writer2.writerow([n_clusters_list[i], KMeans_list_2[0][i], KMeans_list_2[1][i], KMeans_list_2[2][i]])
    #writer3.writerow([n_clusters_list[i], KMeans_list_3[0][i], KMeans_list_3[1][i], KMeans_list_3[2][i]])
    #writer4.writerow([n_clusters_list[i], KMeans_list_4[0][i], KMeans_list_4[1][i], KMeans_list_4[2][i]])

f1.close()
#f2.close()
#f3.close()
#f4.close()