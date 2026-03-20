"""Amino acid sequence encoder with one-hot + sinusoidal positional encoding."""

import numpy as np

# 20 standard amino acids + padding token (index 20)
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["<PAD>"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
VOCAB_SIZE = len(AA_VOCAB)  # 21


class SequenceEncoder:
    """Encode protein sequences as one-hot + sinusoidal positional encodings.

    Output shape: (seq_len, 85) = (seq_len, 21) one-hot + (seq_len, 64) positional.
    """

    def __init__(self, max_len: int = 512, pos_dim: int = 64):
        self.max_len = max_len
        self.pos_dim = pos_dim
        self._pos_table = self._build_pos_table(max_len, pos_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pos_table(self, max_len: int, pos_dim: int) -> np.ndarray:
        """Pre-compute sinusoidal positional encodings of shape (max_len, pos_dim)."""
        positions = np.arange(max_len)[:, np.newaxis]          # (max_len, 1)
        dims = np.arange(pos_dim)[np.newaxis, :]               # (1, pos_dim)
        angles = positions / np.power(10000, (2 * (dims // 2)) / pos_dim)
        # Even indices → sin, odd indices → cos
        table = np.zeros((max_len, pos_dim))
        table[:, 0::2] = np.sin(angles[:, 0::2])
        table[:, 1::2] = np.cos(angles[:, 1::2])
        return table  # (max_len, pos_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, sequence: str) -> list:
        """Convert sequence string to list of integer indices.

        Unknown amino acids are mapped to the padding index (20).
        """
        pad_idx = AA_TO_IDX["<PAD>"]
        return [AA_TO_IDX.get(aa.upper(), pad_idx) for aa in sequence]

    def encode(self, sequence: str) -> np.ndarray:
        """Encode sequence to (len, 85) array.

        Returns:
            np.ndarray of shape (len(sequence), 21 + pos_dim)
        """
        indices = self.tokenize(sequence)
        n = len(indices)

        # One-hot encoding: (n, 21)
        one_hot = np.zeros((n, VOCAB_SIZE), dtype=np.float32)
        for i, idx in enumerate(indices):
            one_hot[i, idx] = 1.0

        # Positional encoding: (n, pos_dim)
        pos = self._pos_table[:n].astype(np.float32)

        return np.concatenate([one_hot, pos], axis=1)  # (n, 85)
