import numpy as np

class SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.VT = None

    def fit(self, X):
        # Compute SVD
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        # Keep only the first n_components
        self.U = U[:, :self.n_components]
        self.S = sigma[:self.n_components]
        self.VT = VT[:self.n_components, :]
        return self

    def transform(self, X):
        # Project data
        return np.dot(X, self.VT.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)