"""Intrinsic-dimension tools: global TwoNN, ID-vs-scale (decimation), local ID, baseline."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def twoNN_id(X):
    """Closed-form TwoNN MLE of global ID (Facco et al. 2017). mu = r2/r1."""
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    good = (r1 > 0) & (r2 > r1)
    mu = r2[good] / r1[good]
    return good.sum() / np.sum(np.log(mu))


def id_vs_scale_decimation(X, n_min=50, seed=0):
    """ID vs scale, swept by decimation (1, 1/2, 1/4, ...). Curve SHAPE is the signal."""
    r = np.random.default_rng(seed)
    n = len(X); sizes, ids = [], []; size = n
    while size >= n_min:
        sub = X if size == n else X[r.choice(n, size, replace=False)]
        ids.append(twoNN_id(sub)); sizes.append(size); size //= 2
    return np.array(sizes), np.array(ids)


def local_id_mle(X, k):
    """Pointwise local ID via Levina-Bickel MLE on first k neighbours (scale set by k)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    d = dist[:, 1 : k + 1]; rk = d[:, -1][:, None]
    with np.errstate(divide="ignore"):
        logs = np.log(rk / d[:, :-1])
    return (k - 1) / logs.sum(axis=1)


def knn_dist(X, k):
    """Naive anomaly baseline: mean distance to k nearest neighbours."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    return dist[:, 1:].mean(axis=1)
