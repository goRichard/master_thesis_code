from Clustering_Algorithm.ClusterAlgorithm import *
import networkx as nx
import seaborn as sns
from data.get_pressure_data import *
from sklearn.metrics import pairwise_distances
from Clustering_Algorithm.get_connectivity_matrix import *
import csv


def get_diagonal(matrix):
    #assert isinstance(matrix, np.ndarray)
    col, row = matrix.shape
    diagonal = []
    for i in range(col):
        for j in range(row):
            if i == j:
                diagonal.append(matrix[i, j])

    return np.array([diagonal])


def matrix_normalize(matrix, D):
    assert isinstance(matrix, np.ndarray)
    diagonal = np.diag(1/np.sqrt(D))
    return np.dot(np.dot(diagonal, matrix), diagonal)



# calculate the pairwise distance between each pair of points
# X_normalized shape (388, 169) 388 data and each data has 169 feature
# W is the similarity matrix
# create similarity matrix based on k-nearest neighbour
# the diagonal of W_knn is 0
W_knn = pairwise_distances(X_normalized, metric="euclidean")  # shape (388,388)

# calculate the Degree Matrix
D_matrix = np.sum(W_knn, axis=1)

# calculate the laplacian matrix
L_matrix = D_matrix - W_knn

# normalize the laplacian matrix by Ng-Jordan-Weiss Algorithm
#L_matrix_norm = matrix_normalize(L_matrix, D_matrix)

# find the eigenvalues and eigenvectors of L_matrix
vals, vecs = np.linalg.eig(L_matrix)
#vals_weighted, vecs_weighted = np.linalg.eig(L_matrix_weighted_norm)

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
f1 = open("silhouette_score.csv", "w")
writer1 = csv.writer(f1)
f2 = open("davies_score.csv", "w")
writer2 = csv.writer(f2)
f3 = open("calinski_score.csv", "w")
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
