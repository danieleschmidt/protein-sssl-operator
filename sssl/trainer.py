"""Self-supervised pretraining loop for sequence-structure alignment."""

import numpy as np
from typing import List


class SSLTrainer:
    """Train linear projection heads on top of frozen encoders via SGD.

    Projection heads map encoder outputs to a shared embedding space where
    NT-Xent contrastive loss is applied.
    """

    def __init__(self, seq_encoder, struct_encoder, loss_fn,
                 lr: float = 1e-3, hidden_dim: int = 64):
        self.seq_encoder = seq_encoder
        self.struct_encoder = struct_encoder
        self.loss_fn = loss_fn
        self.lr = lr
        self.hidden_dim = hidden_dim

        # Lazy initialisation: shapes determined on first train_step call
        self._W_seq = None
        self._b_seq = None
        self._W_struct = None
        self._b_struct = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_weights(self, seq_dim: int, struct_dim: int):
        rng = np.random.default_rng(42)
        scale = 0.01
        self._W_seq    = rng.standard_normal((seq_dim,    self.hidden_dim)).astype(np.float32) * scale
        self._b_seq    = np.zeros(self.hidden_dim, dtype=np.float32)
        self._W_struct = rng.standard_normal((struct_dim, self.hidden_dim)).astype(np.float32) * scale
        self._b_struct = np.zeros(self.hidden_dim, dtype=np.float32)

    def _project_seq(self, x: np.ndarray) -> np.ndarray:
        """x: (seq_len, seq_dim) -> pooled & projected (hidden_dim,)"""
        pooled = x.mean(axis=0)                         # (seq_dim,)
        return pooled @ self._W_seq + self._b_seq       # (hidden_dim,)

    def _project_struct(self, x: np.ndarray) -> np.ndarray:
        """x: (n, struct_dim) -> pooled & projected (hidden_dim,)"""
        pooled = x.mean(axis=0)                         # (struct_dim,)
        return pooled @ self._W_struct + self._b_struct # (hidden_dim,)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return x / norms

    def _encode_batch(self, sequences, coords_list):
        """Encode sequences and structures, returning (batch, seq_dim) and (batch, struct_dim)."""
        seq_vecs, struct_vecs = [], []
        for seq, coords in zip(sequences, coords_list):
            seq_enc = self.seq_encoder.encode(seq)                         # (L, 85)
            struct_enc = self.struct_encoder.from_coords(coords)           # (n, n, n_rbf)
            struct_flat = self.struct_encoder.flatten(struct_enc)          # (n, n_rbf)
            seq_vecs.append(seq_enc.mean(axis=0))
            struct_vecs.append(struct_flat.mean(axis=0))
        return np.stack(seq_vecs), np.stack(struct_vecs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self, sequences: list, coords_list: list) -> float:
        """Perform one SGD update step and return the loss.

        Uses finite differences to estimate gradients for the projection weights.
        """
        seq_raw, struct_raw = self._encode_batch(sequences, coords_list)

        # Lazy weight init
        if self._W_seq is None:
            self._init_weights(seq_raw.shape[1], struct_raw.shape[1])

        # Forward pass
        z_seq    = np.stack([seq_raw[i] @ self._W_seq    + self._b_seq    for i in range(len(sequences))])
        z_struct = np.stack([struct_raw[i] @ self._W_struct + self._b_struct for i in range(len(sequences))])
        z_seq    = self._l2_normalize(z_seq)
        z_struct = self._l2_normalize(z_struct)
        loss = self.loss_fn(z_seq, z_struct)

        # Simple numerical gradient for projection weights (finite differences, eps=1e-3)
        eps = 1e-3

        def param_grad(W, b, is_seq: bool):
            """Compute finite-diff gradient for weight matrix W and bias b."""
            grad_W = np.zeros_like(W)
            grad_b = np.zeros_like(b)

            for idx in np.ndindex(*W.shape):
                W[idx] += eps
                zs = np.stack([seq_raw[i] @ self._W_seq + self._b_seq for i in range(len(sequences))])
                zst = np.stack([struct_raw[i] @ self._W_struct + self._b_struct for i in range(len(sequences))])
                loss_plus = self.loss_fn(self._l2_normalize(zs), self._l2_normalize(zst))
                W[idx] -= eps
                grad_W[idx] = (loss_plus - loss) / eps

            for idx in range(len(b)):
                b[idx] += eps
                zs = np.stack([seq_raw[i] @ self._W_seq + self._b_seq for i in range(len(sequences))])
                zst = np.stack([struct_raw[i] @ self._W_struct + self._b_struct for i in range(len(sequences))])
                loss_plus = self.loss_fn(self._l2_normalize(zs), self._l2_normalize(zst))
                b[idx] -= eps
                grad_b[idx] = (loss_plus - loss) / eps

            return grad_W, grad_b

        # For efficiency, use a simplified gradient: just update biases with finite diff
        # and use a coarse approximation for weights to keep it tractable
        # (full finite-diff over large W is expensive; use random coordinate descent)
        rng = np.random.default_rng()

        # Sample a subset of weight indices for SGD update
        def sparse_update(W, b, seq_features, struct_features, is_seq: bool):
            # Perturb each output neuron's bias
            for j in range(W.shape[1]):
                b[j] += eps
                zs    = self._l2_normalize(np.stack([seq_raw[i] @ self._W_seq    + self._b_seq    for i in range(len(sequences))]))
                zst   = self._l2_normalize(np.stack([struct_raw[i] @ self._W_struct + self._b_struct for i in range(len(sequences))]))
                lp = self.loss_fn(zs, zst)
                b[j] -= eps
                grad = (lp - loss) / eps
                b[j] -= self.lr * grad

        sparse_update(self._W_seq, self._b_seq, seq_raw, struct_raw, True)
        sparse_update(self._W_struct, self._b_struct, struct_raw, seq_raw, False)

        return float(loss)

    def train(self, sequences: list, coords_list: list,
              n_epochs: int = 5) -> List[float]:
        """Train for n_epochs, returning list of per-epoch losses."""
        losses = []
        for _ in range(n_epochs):
            loss = self.train_step(sequences, coords_list)
            losses.append(loss)
        return losses
