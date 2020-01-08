from Clustering_Algorithm.ClusterAlgorithm import *


def cal_gradient(n_array):
    n_array = np.array(n_array)
    return np.gradient(n_array)


vals, vecs = np.linalg.eig(L_matrix)
vecs = np.real(vecs)
X_train_1 = vecs[:, :10]
print(X_train_1.shape)
X_train_2 = X_reduced
n_components_list = np.arange(1, 100)

models_1 = [GaussianMixture(n_components=n, covariance_type="full", random_state=0).fit(X_train_1) for n in
            n_components_list]

models_2 = [GaussianMixture(n_components=n, covariance_type="full", random_state=0).fit(X_train_2) for n in
            n_components_list]


bic_1 = [m.bic(X_train_1) for m in models_1]
bic_1_gradient = cal_gradient(bic_1)
aic_1 = [m.aic(X_train_1) for m in models_1]
aic_1_gradient = cal_gradient(aic_1)

bic_2 = [m.bic(X_train_2) for m in models_2]
aic_2 = [m.aic(X_train_2) for m in models_2]
bic_2_gradient = cal_gradient(bic_2)
aic_2_gradient = cal_gradient(aic_2)


plt.figure()
plt.subplot(211)
plt.title("GMM with laplacian matrix")
plt.plot(n_components_list, bic_1, label="BIC")
plt.plot(n_components_list, aic_1, label="AIC")
plt.legend(loc="best")
plt.xlabel("n_components")

plt.subplot(212)
plt.title("gradient of BIC and AIC")
plt.plot(n_components_list, bic_1_gradient, label="BIC")
plt.plot(n_components_list, aic_1_gradient, label="AIC")
plt.legend(loc="best")
plt.xlabel("n_components")
plt.show()

plt.figure()
plt.subplot(211)
plt.title("GMM with PCA")
plt.plot(n_components_list, bic_2, label="BIC")
plt.plot(n_components_list, aic_2, label="AIC")
plt.legend(loc="best")
plt.xlabel("n_components")

plt.subplot(212)
plt.title("gradient of BIC and AIC")
plt.plot(n_components_list, bic_2_gradient, label="BIC")
plt.plot(n_components_list, aic_2_gradient, label="AIC")
plt.legend(loc="best")
plt.xlabel("n_components")
plt.show()
