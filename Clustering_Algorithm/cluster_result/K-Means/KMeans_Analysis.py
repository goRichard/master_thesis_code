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