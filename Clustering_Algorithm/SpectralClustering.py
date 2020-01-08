from Clustering_Algorithm.ClusterAlgorithm import *
import csv



# normalize the laplacian matrix by Ng-Jordan-Weiss Algorithm
# L_matrix_norm = matrix_normalize(L_matrix, D_matrix)

# find the eigenvalues and eigenvectors of L_matrix
vals, vecs = np.linalg.eig(L_matrix)
# vals_weighted, vecs_weighted = np.linalg.eig(L_matrix_weighted_norm)

# sort the vals and vecs
# np.linalg.eig will return the eigenvalue and eigen vector with complex value
# using np.real to transform the data into real numbers
"""
vecs = np.real(vecs[:, np.argsort(vals)])
vals = vals[np.argsort(vals)]
vecs_weighted = np.real(vecs_weighted[:, np.argsort(vals_weighted)])
vals_weighted = vals_weighted[np.argsort(vals_weighted)]
"""
vecs = np.real(vecs)
# vecs_weighted = np.real(vecs_weighted)


# running KMeans for the eigenvectors matrix
n_clusters = np.arange(2, 102)
result_sc_vecs = dict()
# result_sc_vecs_weighted = dict()

# create csv file to evaluate score
f1 = open("cluster_result/sc/result_dimension_7.csv", "w")
writer1 = csv.writer(f1)
"""
f2 = open("cluster_result/sc/davies_score_W.csv", "w")
writer2 = csv.writer(f2)
f3 = open("cluster_result/sc/calinski_score_W.csv", "w")
writer3 = csv.writer(f3)
"""


#writer1.writerow(["n_cluster" , "silhouette_score" , "davis_score" , "calinski_score"])
"""
writer2.writerow(["n_cluster"] + list(n_clusters))
writer3.writerow(["n_cluster"] + list(n_clusters))
"""

KMeans_dict = running_KMeans(vecs[:, :7], n_clusters)[0]  # (388,k)


print(KMeans_dict[30])


# evaluate the algorithm
score_list = evaluate_Algorithm(vecs[:, :7], KMeans_dict, "KMeans")
score_list_1 = score_list[0]
score_list_2 = score_list[1]
score_list_3 = score_list[2]


for n, i,j,k in zip(n_clusters, score_list_1, score_list_2, score_list_3):
    writer1.writerow([n, i, j, k])

f1.close()

# KMeans_list_vecs_weighted = evaluate_Algorithm(vecs_weighted, result_sc_vecs_weighted, "KMeans")

# plot the result

"""
plt.figure(figsize=(10, 8))
plt.title("Evaluation for KMeans (silhouette_score)")
plt.plot(n_clusters, KMeans_list_vecs[0], "r-", marker="o")
plt.plot(n_clusters, KMeans_list_vecs[1], "g-", marker="o")
plt.plot(n_clusters, KMeans_list_vecs[2], "b-", marker="o")
plt.xlabel("k")
plt.ylabel("scores")
plt.legend(["silhouette", "davies", "calinski"], loc="upper left")

plt.show()


"""
