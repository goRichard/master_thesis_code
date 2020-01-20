from Clustering_Algorithm.ClusterAlgorithm import *
from data.get_pressure_data import *
import csv
from multiprocessing import Pool
import time

# define a list with n_clusters
n_clusters_list = list(range(2, 101))  # n_cluster from 2 to 100

# run Agglomerative Algorithm

Agg_dict_raw = running_agg_with_connectivity_matrix(X, n_clusters_list) # data without preprocessing
Agg_dict_mms = running_agg_with_connectivity_matrix(X_transformed_mms, n_clusters_list) # data preprocessed with mms
Agg_dict_ss = running_agg_with_connectivity_matrix(X_transformed_ss, n_clusters_list) # data preprocessed with ss


# agg clustering
# the reason why problems like
# UserWarning: the number of connected components of the connectivity matrix is 5 > 1. Completing it to avoid stopping the tree early.
#   affinity='euclidean')
# appears is because the connectivity matrix should be considered as the reconstruction from a full connected matrix but
#  the adjacency matrix has some zero elements which means it is not fully connected
print("clustering beginns...")
#Agg_dict_adjacency = running_agg_without_connectivity_matrix(X_reduced, n_clusters_list)
#Agg_dict_adjacency = running_agg_with_connectivity_matrix(X_normalized.T, n_clusters_list, "adjacency")
#Agg_dict_adjacency_mms = running_agg_with_connectivity_matrix(X_normalized_mms, n_clusters_list, "adjacency")
Agg_dict_W_knn = running_agg_with_connectivity_matrix(X_normalized.T, n_clusters_list, "W_knn")
#Agg_dict_W_knn_mms = running_agg_with_connectivity_matrix(X_normalized_mms, n_clusters_list, "W_knn")
#Agg_dict_W = running_agg_with_connectivity_matrix(X_normalized, n_clusters_list, "W")
#Agg_dict_W_mms = running_agg_with_connectivity_matrix(X_normalized_mms, n_clusters_list, "W")
print("Clustering ends")
# evaluate the Agg results with 3 methods:
# evaluate the KMeans results with 3 methods and store the value in .csv file
print("Evaluation begins...")
#list1 = evaluate_Algorithm(X_normalized.T, Agg_dict_adjacency, "Agg")
#list2 = evaluate_Algorithm(X_normalized_mms, Agg_dict_adjacency_mms, "Agg")
list3 = evaluate_Algorithm(X_normalized, Agg_dict_W_knn, "Agg")
#list4 = evaluate_Algorithm(X_normalized_mms, Agg_dict_W_knn_mms, "Agg")
#list5 = evaluate_Algorithm(X_normalized, Agg_dict_W, "Agg")
#list6 = evaluate_Algorithm(X_normalized_mms, Agg_dict_W_mms, "Agg")
print("Evaluation ends")

# create the csv file
f1 = open("cluster_result/Agglomerative/agg_score_new.csv", "w")
#f1 = open("cluster_result/Agglomerative/agg_score_adjacency.csv", "w")
#f2 = open("cluster_result/Agglomerative/agg_score_adjacency_mms.csv", "w")
#f3 = open("cluster_result/Agglomerative/agg_score_W_knn.csv", "w")
#f4 = open("cluster_result/Agglomerative/agg_score_W_knn_mms.csv", "w")
#f5 = open("cluster_result/Agglomerative/agg_score_W.csv", "w")
#f6 = open("cluster_result/Agglomerative/agg_score_W_mms.csv", "w")

writer1 = csv.writer(f1)
#writer2 = csv.writer(f2)
#writer3 = csv.writer(f3)
#writer4 = csv.writer(f4)
#writer5 = csv.writer(f5)
#writer6 = csv.writer(f6)

writer1.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer2.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer3.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer4.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer5.writerow(["n_cluster", "Silhouette", "davies", "calinski"])
#writer6.writerow(["n_cluster", "Silhouette", "davies", "calinski"])

for x in range(len(n_clusters_list)):
    writer1.writerow([n_clusters_list[x], list1[0][x], list1[1][x], list1[2][x]])
    #writer2.writerow([n_clusters_list[x], list2[0][x], list2[1][x], list2[2][x]])
    #writer3.writerow([n_clusters_list[x], list3[0][x], list3[1][x], list3[2][x]])
    #writer4.writerow([n_clusters_list[x], list4[0][x], list4[1][x], list4[2][x]])
    #writer5.writerow([n_clusters_list[x], list5[0][x], list5[1][x], list5[2][x]])
    #writer6.writerow([n_clusters_list[x], list6[0][x], list6[1][x], list6[2][x]])

f1.close()
#f2.close()
#f3.close()
#f4.close()
#f5.close()
#f6.close()
