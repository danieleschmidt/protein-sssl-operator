"""3D structure encoder using RBF expansion of pairwise distances."""

import numpy as np


class StructureEncoder:
    """Encode 3D coordinates as RBF-expanded pairwise distance matrices.

    from_coords: (n, 3) -> (n, n, n_rbf)
    flatten:     (n, n, n_rbf) -> (n, n_rbf)  via mean pooling over axis 1
    """

    def __init__(self, n_rbf: int = 16, cutoff: float = 10.0):
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        # RBF centres uniformly spaced in [0, cutoff]
        self.mu = np.linspace(0.0, cutoff, n_rbf)   # (n_rbf,)
        self.sigma = cutoff / n_rbf                  # scalar bandwidth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_coords(self, coords: np.ndarray) -> np.ndarray:
        """Compute RBF-encoded pairwise distances.

        Args:
            coords: np.ndarray of shape (n, 3)

        Returns:
            np.ndarray of shape (n, n, n_rbf)
        """
        n = coords.shape[0]
        # Pairwise distances: (n, n)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 3)
        dist = np.linalg.norm(diff, axis=-1)                         # (n, n)

        # RBF expansion: (n, n, n_rbf)
        dist_exp = dist[:, :, np.newaxis]           # (n, n, 1)
        mu_exp = self.mu[np.newaxis, np.newaxis, :] # (1, 1, n_rbf)
        rbf = np.exp(-((dist_exp - mu_exp) / self.sigma) ** 2)
        return rbf.astype(np.float32)               # (n, n, n_rbf)

    def flatten(self, encoded: np.ndarray) -> np.ndarray:
        """Reduce (n, n, n_rbf) to (n, n_rbf) via mean pooling over neighbors."""
        return encoded.mean(axis=1)  # (n, n_rbf)
