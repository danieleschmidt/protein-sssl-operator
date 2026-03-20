"""NT-Xent (InfoNCE) contrastive loss for sequence-structure alignment."""

import numpy as np


class ContrastiveLoss:
    """NT-Xent loss on (batch, dim) L2-normalised embedding pairs.

    For each sample i, treats (z_seq[i], z_struct[i]) as the positive pair
    and all other j != i as negatives.
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def __call__(self, z_seq: np.ndarray, z_struct: np.ndarray) -> float:
        """Compute symmetric NT-Xent loss.

        Args:
            z_seq:    (batch, dim) – L2-normalised sequence projections
            z_struct: (batch, dim) – L2-normalised structure projections

        Returns:
            Scalar float loss.
        """
        batch = z_seq.shape[0]

        # Cosine similarity matrix: (batch, batch)
        sim = z_seq @ z_struct.T / self.temperature  # (batch, batch)

        # Labels: positive pair for sample i is column i
        labels = np.arange(batch)

        # Cross-entropy for each row (z_seq vs z_struct)
        loss_seq = self._cross_entropy(sim, labels)
        # Cross-entropy for each column (z_struct vs z_seq)
        loss_struct = self._cross_entropy(sim.T, labels)

        return float((loss_seq + loss_struct) / 2.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
        """Numerically stable row-wise cross-entropy averaged over batch."""
        # Subtract row-max for numerical stability
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
        positive_logits = shifted[np.arange(len(labels)), labels]
        return float((-positive_logits + log_sum_exp).mean())
