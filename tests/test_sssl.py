"""Tests for the SSSL protein learning library."""

import numpy as np
import pytest
import sys
import os

# Ensure the repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sssl.sequence_encoder import SequenceEncoder, AA_VOCAB, VOCAB_SIZE
from sssl.structure_encoder import StructureEncoder
from sssl.contrastive_loss import ContrastiveLoss
from sssl.trainer import SSLTrainer
from sssl.linear_probe import LinearProbe, SS_CLASSES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seq_enc():
    return SequenceEncoder()


@pytest.fixture
def struct_enc():
    return StructureEncoder()


@pytest.fixture
def loss_fn():
    return ContrastiveLoss(temperature=0.1)


@pytest.fixture
def sample_sequences():
    return ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ"]


@pytest.fixture
def sample_coords():
    rng = np.random.default_rng(1)
    return [
        rng.random((20, 3)).astype(np.float32) * 10,
        rng.random((22, 3)).astype(np.float32) * 10,
    ]


# ---------------------------------------------------------------------------
# SequenceEncoder tests
# ---------------------------------------------------------------------------

def test_sequence_encoder_encodes_valid_amino_acids(seq_enc):
    """SequenceEncoder should produce non-zero output for valid amino acids."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    encoded = seq_enc.encode(seq)
    assert encoded is not None
    assert encoded.sum() > 0


def test_sequence_encoder_output_shape(seq_enc):
    """SequenceEncoder output shape must be (len, 85)."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    encoded = seq_enc.encode(seq)
    assert encoded.shape == (len(seq), 85), f"Expected ({len(seq)}, 85), got {encoded.shape}"


def test_sequence_encoder_short_sequence(seq_enc):
    """Shape must be (len, 85) for any sequence length."""
    seq = "MKAT"
    encoded = seq_enc.encode(seq)
    assert encoded.shape == (4, 85)


def test_sequence_encoder_unknown_amino_acid(seq_enc):
    """Unknown amino acids should map to padding (index 20) gracefully."""
    seq = "AXZ"  # X and Z are not standard
    indices = seq_enc.tokenize(seq)
    pad_idx = 20
    assert indices[0] == 0    # 'A' → index 0
    assert indices[1] == pad_idx
    assert indices[2] == pad_idx


def test_sequence_encoder_one_hot_valid(seq_enc):
    """One-hot slice should have exactly one 1 per position."""
    seq = "ACDE"
    encoded = seq_enc.encode(seq)
    one_hot_part = encoded[:, :21]
    row_sums = one_hot_part.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(seq)), atol=1e-6)


# ---------------------------------------------------------------------------
# StructureEncoder tests
# ---------------------------------------------------------------------------

def test_structure_encoder_from_coords_shape(struct_enc):
    """from_coords output must be (n, n, n_rbf)."""
    n, n_rbf = 10, 16
    coords = np.random.default_rng(0).random((n, 3)).astype(np.float32)
    encoded = struct_enc.from_coords(coords)
    assert encoded.shape == (n, n, n_rbf), f"Expected ({n}, {n}, {n_rbf}), got {encoded.shape}"


def test_structure_encoder_flatten_shape(struct_enc):
    """flatten must return (n, n_rbf) via mean pooling."""
    n, n_rbf = 8, 16
    coords = np.random.default_rng(2).random((n, 3)).astype(np.float32)
    encoded = struct_enc.from_coords(coords)
    flat = struct_enc.flatten(encoded)
    assert flat.shape == (n, n_rbf), f"Expected ({n}, {n_rbf}), got {flat.shape}"


def test_structure_encoder_rbf_values_in_range(struct_enc):
    """RBF values should be in [0, 1]."""
    coords = np.random.default_rng(3).random((5, 3)).astype(np.float32) * 10
    encoded = struct_enc.from_coords(coords)
    assert encoded.min() >= 0.0
    assert encoded.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# ContrastiveLoss tests
# ---------------------------------------------------------------------------

def test_contrastive_loss_returns_scalar(loss_fn):
    """ContrastiveLoss __call__ must return a float scalar."""
    rng = np.random.default_rng(4)
    batch, dim = 4, 32
    z_seq = rng.standard_normal((batch, dim)).astype(np.float32)
    z_struct = rng.standard_normal((batch, dim)).astype(np.float32)
    # L2 normalize
    z_seq    /= np.linalg.norm(z_seq,    axis=1, keepdims=True)
    z_struct /= np.linalg.norm(z_struct, axis=1, keepdims=True)
    loss = loss_fn(z_seq, z_struct)
    assert isinstance(loss, float)


def test_contrastive_loss_identical_lower_than_random(loss_fn):
    """Loss should be lower when pairs are identical vs completely random."""
    rng = np.random.default_rng(5)
    batch, dim = 4, 32

    # Identical pairs
    z = rng.standard_normal((batch, dim)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    loss_identical = loss_fn(z, z.copy())

    # Random pairs
    z_seq = rng.standard_normal((batch, dim)).astype(np.float32)
    z_struct = rng.standard_normal((batch, dim)).astype(np.float32)
    z_seq    /= np.linalg.norm(z_seq,    axis=1, keepdims=True)
    z_struct /= np.linalg.norm(z_struct, axis=1, keepdims=True)
    loss_random = loss_fn(z_seq, z_struct)

    assert loss_identical < loss_random, (
        f"Expected identical loss {loss_identical:.4f} < random loss {loss_random:.4f}"
    )


# ---------------------------------------------------------------------------
# SSLTrainer tests
# ---------------------------------------------------------------------------

def test_ssl_trainer_train_step_returns_float(seq_enc, struct_enc, loss_fn, sample_sequences, sample_coords):
    """train_step must return a float."""
    trainer = SSLTrainer(seq_enc, struct_enc, loss_fn)
    loss = trainer.train_step(sample_sequences, sample_coords)
    assert isinstance(loss, float), f"Expected float, got {type(loss)}"


def test_ssl_trainer_train_returns_correct_length(seq_enc, struct_enc, loss_fn, sample_sequences, sample_coords):
    """train must return a list of losses with length == n_epochs."""
    n_epochs = 3
    trainer = SSLTrainer(seq_enc, struct_enc, loss_fn)
    losses = trainer.train(sample_sequences, sample_coords, n_epochs=n_epochs)
    assert isinstance(losses, list)
    assert len(losses) == n_epochs, f"Expected {n_epochs} losses, got {len(losses)}"
    assert all(isinstance(l, float) for l in losses)


# ---------------------------------------------------------------------------
# LinearProbe tests
# ---------------------------------------------------------------------------

def test_linear_probe_predict_output_shape():
    """LinearProbe.predict must return (n, 3)."""
    probe = LinearProbe(input_dim=16)
    features = np.random.default_rng(6).random((10, 16)).astype(np.float32)
    probs = probe.predict(features)
    assert probs.shape == (10, 3), f"Expected (10, 3), got {probs.shape}"


def test_linear_probe_predict_probs_sum_to_one():
    """Softmax probabilities must sum to 1 for each sample."""
    probe = LinearProbe(input_dim=16)
    features = np.random.default_rng(7).random((5, 16)).astype(np.float32)
    probs = probe.predict(features)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-5)


def test_linear_probe_train_reduces_loss():
    """Training should reduce the loss over epochs."""
    rng = np.random.default_rng(8)
    n, dim = 20, 16
    features = rng.standard_normal((n, dim)).astype(np.float32)
    labels = rng.integers(0, 3, size=n)
    probe = LinearProbe(input_dim=dim)
    losses = probe.train(features, labels, n_epochs=100, lr=0.05)
    assert losses[0] > losses[-1], (
        f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_linear_probe_train_returns_loss_list():
    """train must return a list of float losses."""
    features = np.random.default_rng(9).random((6, 8)).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2])
    probe = LinearProbe(input_dim=8)
    losses = probe.train(features, labels, n_epochs=10)
    assert isinstance(losses, list)
    assert len(losses) == 10


def test_ss_classes_definition():
    """SS_CLASSES should contain exactly H, E, C."""
    assert SS_CLASSES == ['H', 'E', 'C']
