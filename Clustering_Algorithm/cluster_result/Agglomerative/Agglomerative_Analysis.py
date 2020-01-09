import pandas as pd
import matplotlib.pyplot as plt

result = pd.read_csv("agg_score_adjacency.csv")
n_clusters = result.values[:, 0]
scores = result.values[:, 1:]

plt.figure()
plt.subplot(311)
plt.title("silhouette score")
plt.plot(n_clusters, scores[:, 0], "r-")
plt.xlabel("n_cluster")
plt.ylabel("silhouette scores")
plt.grid()

plt.subplot(312)
plt.title("davies score")
plt.plot(n_clusters, scores[:, 1], "b-")
plt.xlabel("n_cluster")
plt.ylabel("davies scores")
plt.grid()

plt.subplot(313)
plt.title("calinski score")
plt.plot(n_clusters, scores[:, 2], "g-")
plt.xlabel("n_cluster")
plt.ylabel("calinski scores")
plt.grid()
plt.show()
