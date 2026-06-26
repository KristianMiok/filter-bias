"""Synthetic crayfish-like feature data with CONTROLLED contamination."""
import numpy as np


class Manifold:
    """Curved d-dim manifold embedded in D-dim ambient space; fixed embedding so
    on-manifold contamination reuses the exact same map.

    d : true intrinsic dimension. CHECK AGAINST REAL DATA: run twoNN_id on the real
        feature matrix, then set d to match.
    D : ambient dimension (number of stored environmental features, ~20).
    """
    def __init__(self, d=5, D=20, curvature=0.6, seed=0):
        self.d, self.D, self.curv = d, D, curvature
        self.W = np.random.default_rng(seed).normal(size=(d, D))

    def embed(self, z):
        X = z @ self.W
        X[:, : self.d] += self.curv * np.sin(z)
        return X

    def sample(self, n, manifold_noise=0.01, seed=0):
        r = np.random.default_rng(seed)
        z = r.normal(size=(n, self.d))
        X = self.embed(z) + manifold_noise * r.normal(size=(n, self.D))
        return X, z


def contaminate_offmanifold(X, frac, sigma, seed=0):
    """Bad-coordinate model: displace a fraction of points off the manifold."""
    r = np.random.default_rng(seed)
    n = len(X); mask = np.zeros(n, dtype=bool)
    idx = r.choice(n, int(frac * n), replace=False); mask[idx] = True
    Xc = X.copy(); Xc[idx] += sigma * r.normal(size=(len(idx), X.shape[1]))
    return Xc, mask


def contaminate_onmanifold(mani, X, frac, seed=0):
    """MisID model: replace points with other VALID points from the same manifold."""
    r = np.random.default_rng(seed)
    n = len(X); mask = np.zeros(n, dtype=bool)
    idx = r.choice(n, int(frac * n), replace=False); mask[idx] = True
    Xc = X.copy(); Xc[idx] = mani.embed(r.normal(size=(len(idx), mani.d)))
    return Xc, mask
