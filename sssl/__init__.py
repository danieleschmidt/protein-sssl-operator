"""SSSL: Self-supervised structure-sequence learning for proteins."""

from .sequence_encoder import SequenceEncoder
from .structure_encoder import StructureEncoder
from .contrastive_loss import ContrastiveLoss
from .trainer import SSLTrainer
from .linear_probe import LinearProbe

__all__ = [
    "SequenceEncoder",
    "StructureEncoder",
    "ContrastiveLoss",
    "SSLTrainer",
    "LinearProbe",
]
