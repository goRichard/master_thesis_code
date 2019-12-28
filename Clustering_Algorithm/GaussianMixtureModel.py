from data.get_pressure_data import *
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import silhouette_score
import tqdm

n_components_list = list(range(1, 30))
models = [GaussianMixture(n_components=n, covariance_type="full").fit(X_transformed_ss) for n in
          tqdm(n_components_list)]


silhouette_score(X_transformed_ss, models[0].fit_predict(X_transformed_ss))


plt.figure()
plt.plot(n_components_list, [m.bic(X_transformed_ss) for m in models], label="BIC")
plt.plot(n_components_list, [m.aic(X_transformed_ss) for m in models], label="AIC")
plt.xlabel("n_components")
plt.show()

