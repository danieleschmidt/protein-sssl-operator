from .models.ssl_encoder import SequenceStructureSSL
from .models.neural_operator import NeuralOperatorFold
from .models.structure_decoder import StructurePredictor
from .data.sequence_dataset import ProteinDataset
from .data.structure_dataset import StructureDataset
from .training.ssl_trainer import SSLTrainer
from .training.folding_trainer import FoldingTrainer
from .analysis.domain_analysis import MultiScaleAnalyzer, DomainSegmenter

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "SequenceStructureSSL",
    "NeuralOperatorFold", 
    "StructurePredictor",
    "ProteinDataset",
    "StructureDataset",
    "SSLTrainer",
    "FoldingTrainer",
    "MultiScaleAnalyzer",
    "DomainSegmenter",
]