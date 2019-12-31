from Clustering_Algorithm.ClusterAlgorithm import *
from Clustering_Algorithm.get_connectivity_matrix import *
import csv

# calculate the Degree Matrix
D_matrix = np.sum(W, axis=1)

# calculate the laplacian matrix
L_matrix = D_matrix - W

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
n_clusters = np.arange(2, 102, 1)
result_sc_vecs = dict()
result_sc_vecs_weighted = dict()
k_list = np.arange(2, 51, 1)

# create csv file to evaluate score
f1 = open("silhouette_score_new.csv", "w")
writer1 = csv.writer(f1)
f2 = open("davies_score_new.csv", "w")
writer2 = csv.writer(f2)
f3 = open("calinski_score_new.csv", "w")
writer3 = csv.writer(f3)

writer1.writerow(["n_cluster"] + list(n_clusters))
writer2.writerow(["n_cluster"] + list(n_clusters))
writer3.writerow(["n_cluster"] + list(n_clusters))

for k in tqdm(k_list):
    KMeans_dict = running_KMeans(vecs[:, :k], n_clusters)  # (388,k)
    # evaluate the algorithm
    score_list = evaluate_Algorithm(vecs[:, :k], KMeans_dict, "KMeans")
    score_list_1 = score_list[0]
    score_list_2 = score_list[1]
    score_list_3 = score_list[2]

    score_list_1.insert(0, k)
    score_list_2.insert(0, k)
    score_list_3.insert(0, k)

    writer1.writerow(score_list_1)
    writer2.writerow(score_list_2)
    writer3.writerow(score_list_3)

f1.close()
f2.close()
f3.close()

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
