from Clustering_Algorithm.ClusterAlgorithm import *


def weighted_euclidean(X, V, weights):
    dists = X - V
    return np.sqrt(np.sum((dists * weights) ** 2))


def single_delta(X, V, F):
    d = X[F] - V[F]
    return d


def cal_beta(X, d):
    n = X.shape[0]
    for b in np.linspace(0, 1.2, 1000):
        p = 1 / (1 + b * d)
        p = np.triu(p, 1)
        para = (2 / (n * (n - 1))) * np.sum(p)
        if para < 0.5:
            return b


def return_weights(X, beta, d):
    max_iter = 1000
    threshold = 0.00001
    w = np.empty((1, X.shape[1]))  # initialize w matrix with the same shape as features in X data set and elements 1
    w.fill(1)
    rho_1 = 1 / (1 + beta * d)
    n = X.shape[0]
    E_old = 1
    for i in range(0, max_iter):
        d = pairwise_distances(X, X, metric=weighted_euclidean, **{'weights': w})
        grad_w = np.empty((1, X.shape[1]))
        part_rho_d = -beta / (1 + beta * d) ** 2
        rho_w = 1 / (1 + beta * d)
        E = (2 / (n * (n - 1))) * np.sum(np.triu(.5 * (rho_w * (1 - rho_1) + rho_1 * (1 - rho_w)), 1))
        print(f'E : {E}')
        if E_old - E < threshold:
            break
        E_old = E
        part_E_rho = (1 - 2 * rho_1)
        w_valid = np.where(w > 0)[1]
        print(f'w_valid = {w_valid} \n')
        for j in w_valid:
            d_w = pairwise_distances(X, X, metric=single_delta, **{'F': j})
            print(f'dw = {d_w}')
            part_w = w[0, j] * (d_w ** 2) / d
            part_w = np.triu(part_w, 1)
            grad_w_j = 1 / (n * (n - 1)) * part_E_rho * part_rho_d * part_w
            grad_w_j = np.triu(grad_w_j, 1)
            grad_w[0, j] = np.nansum(grad_w_j)

        grad_w = grad_w * 10
        w = w - grad_w
        print(f"w = {w}")
        w = w.clip(min=0)

    w_max = np.max(w)
    w = w / w_max # normalize w, because the minimum of w is zero, so w/(w_max - w_min) = w/(w_max - 0)
    print(w)
    return w


X = X_normalized
n = X.shape[0]
d = pairwise_distances(X, X, metric="euclidean")  # normal d
beta = cal_beta(X, d)
print(f'beta: {beta}')
w = return_weights(X, beta, d)
print(f"w = {w}")
