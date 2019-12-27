from get_data import *
import functools
from ClusterAlgorithm import *


# define list of min samples
min_samples_list = list(range(1, 21))

# find the best eps value by two methods
results_formular = []
results_neighbouring_mms = []
results_neighbouring_ss = []
for min_samples in min_samples_list:
    # formular method only works for preprocessing with standard scaler
    #results_formular.append(find_best_epsilon_formular(X_transformed_ss, min_samples))
    results_neighbouring_mms.append(find_best_epsilon_neighboring(X_transformed_mms, min_samples))
    results_neighbouring_ss.append(find_best_epsilon_neighboring(X_transformed_ss, min_samples))



# define the list of eps
# eps for ss 2.5 - 5, minpts 5
# eps for mss 0.5 - 1, minpts 10
eps_list_ss = np.linspace(2.5, 5, 6)
eps_list_mss = np.linspace(0.5, 1, 6)

# define two partial functions
find_best_min_samples_scipy_fixed_X_ss = functools.partial(find_best_min_samples_scipy, X_transformed_ss)
find_best_min_samples_scipy_fixed_X_mms = functools.partial(find_best_min_samples_scipy, X_transformed_mms)

# define the result list
within_eps_list_ss = list(map(find_best_min_samples_scipy_fixed_X_ss, eps_list_ss))
within_eps_list_mss = list(map(find_best_min_samples_scipy_fixed_X_mms, eps_list_mss))


## the large the number of the min samples, the less the number of cluster and it will generate more noise points
# evaluate
DBSCAN_dict = running_DBSCAN(X_transformed_mms, eps_list_mss, [10,10,10,10,10])
evaluate_Algorithm(X_transformed_mms, DBSCAN_dict, "DBSCAN")



# see OPTICS
