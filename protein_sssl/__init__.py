# Torch-free imports for autonomous execution
try:
    import torch
    TORCH_AVAILABLE = True
    # Full torch-based imports
    from .models.ssl_encoder import SequenceStructureSSL
    from .models.neural_operator import NeuralOperatorFold
    from .models.structure_decoder import StructurePredictor
    from .data.sequence_dataset import ProteinDataset
    from .data.structure_dataset import StructureDataset
    from .training.ssl_trainer import SSLTrainer
    from .training.folding_trainer import FoldingTrainer
    from .analysis.domain_analysis import MultiScaleAnalyzer, DomainSegmenter
except ImportError:
    TORCH_AVAILABLE = False
    # Torch-free fallback implementations
    from .utils.torch_mock import MockModule
    
    # Create mock classes for autonomous execution
    SequenceStructureSSL = MockModule
    NeuralOperatorFold = MockModule
    StructurePredictor = MockModule
    ProteinDataset = object
    StructureDataset = object
    SSLTrainer = MockModule
    FoldingTrainer = MockModule
    MultiScaleAnalyzer = MockModule
    DomainSegmenter = MockModule

# Config system always available
from .config import ConfigManager, load_config, save_config, get_default_ssl_config, get_default_folding_config

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
    "ConfigManager",
    "load_config",
    "save_config",
    "get_default_ssl_config",
    "get_default_folding_config",
    "TORCH_AVAILABLE"
]