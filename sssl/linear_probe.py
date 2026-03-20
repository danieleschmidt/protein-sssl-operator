"""Linear probe for secondary structure prediction."""

import numpy as np
from typing import List

SS_CLASSES = ['H', 'E', 'C']  # helix, sheet, coil


class LinearProbe:
    """Multinomial logistic regression trained with SGD.

    Predicts secondary structure class probabilities from encoder features.
    """

    def __init__(self, input_dim: int, n_classes: int = 3):
        self.input_dim = input_dim
        self.n_classes = n_classes
        rng = np.random.default_rng(0)
        self.weights = rng.standard_normal((input_dim, n_classes)).astype(np.float32) * 0.01
        self.bias = np.zeros(n_classes, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _cross_entropy_grad(self, features: np.ndarray, labels: np.ndarray):
        """Compute gradients and loss for softmax cross-entropy.

        Args:
            features: (n, input_dim)
            labels:   (n,) int class indices

        Returns:
            loss (float), grad_W (input_dim, n_classes), grad_b (n_classes,)
        """
        n = features.shape[0]
        logits = features @ self.weights + self.bias  # (n, n_classes)
        probs  = self._softmax(logits)                # (n, n_classes)

        # Cross-entropy loss
        correct_probs = probs[np.arange(n), labels]
        loss = float(-np.log(correct_probs + 1e-9).mean())

        # Gradient of cross-entropy w.r.t. logits
        d_logits = probs.copy()
        d_logits[np.arange(n), labels] -= 1.0
        d_logits /= n  # (n, n_classes)

        grad_W = features.T @ d_logits  # (input_dim, n_classes)
        grad_b = d_logits.sum(axis=0)   # (n_classes,)

        return loss, grad_W, grad_b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return softmax probabilities.

        Args:
            features: (n, input_dim)

        Returns:
            np.ndarray of shape (n, n_classes)
        """
        logits = features @ self.weights + self.bias
        return self._softmax(logits)

    def train(self, features: np.ndarray, labels: np.ndarray,
              n_epochs: int = 50, lr: float = 0.01) -> List[float]:
        """Train with SGD, returning list of per-epoch losses.

        Args:
            features: (n, input_dim)
            labels:   (n,) int indices in [0, n_classes)
            n_epochs: number of epochs
            lr:       learning rate

        Returns:
            List of float losses, one per epoch.
        """
        losses = []
        for _ in range(n_epochs):
            loss, grad_W, grad_b = self._cross_entropy_grad(features, labels)
            self.weights -= lr * grad_W
            self.bias    -= lr * grad_b
            losses.append(loss)
        return losses
