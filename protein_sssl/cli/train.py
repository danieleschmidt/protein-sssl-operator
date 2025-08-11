"""
Training command for protein-sssl CLI.
"""
import torch
import yaml
from pathlib import Path
import sys
from omegaconf import OmegaConf

from ..training.ssl_trainer import SSLTrainer
from ..training.folding_trainer import FoldingTrainer
from ..models.ssl_encoder import SequenceStructureSSL
from ..models.neural_operator import NeuralOperatorFold
from ..data.sequence_dataset import ProteinDataset
from ..data.structure_dataset import StructureDataset
from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)


def train_command(args):
    """Execute model training."""
    logger.info("Starting protein folding model training...")
    
    # Load configuration
    try:
        config = OmegaConf.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine training type from config
    train_type = config.get('training_type', 'ssl')  # default to SSL pre-training
    
    if train_type == 'ssl':
        _train_ssl(args, config)
    elif train_type == 'folding':
        _train_folding(args, config)
    else:
        logger.error(f"Unknown training type: {train_type}")
        sys.exit(1)


def _train_ssl(args, config):
    """Train SSL encoder."""
    logger.info("Training SSL encoder...")
    
    # Create dataset
    try:
        dataset = ProteinDataset.from_fasta(
            args.data,
            max_length=config.data.get('max_length', 1024),
            clustering_threshold=config.data.get('clustering_threshold', 0.9)
        )
        logger.info(f"Loaded {len(dataset)} sequences")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        sys.exit(1)
    
    # Create model
    model_config = config.model
    model = SequenceStructureSSL(
        d_model=model_config.get('d_model', 1280),
        n_layers=model_config.get('n_layers', 33),
        n_heads=model_config.get('n_heads', 20),
        ssl_objectives=model_config.get('ssl_objectives', 
                                      ['masked_modeling', 'contrastive', 'distance_prediction'])
    )
    
    # Create trainer
    trainer = SSLTrainer(
        model=model,
        learning_rate=config.training.get('learning_rate', 1e-4),
        warmup_steps=config.training.get('warmup_steps', 10000),
        gradient_checkpointing=config.training.get('gradient_checkpointing', True),
        use_wandb=config.logging.get('use_wandb', False),
        project_name=config.logging.get('project_name', 'protein-ssl')
    )
    
    # Train model
    trainer.pretrain(
        dataset,
        epochs=config.training.get('epochs', 10),
        batch_size=config.training.get('batch_size', 32),
        num_gpus=args.gpus,
        mixed_precision=config.training.get('mixed_precision', True),
        save_dir=args.output,
        resume_from=args.resume
    )
    
    # Save final model
    model.save_pretrained(f"{args.output}/final_model")
    logger.info("SSL training completed!")


def _train_folding(args, config):
    """Train structure prediction model."""
    logger.info("Training folding model...")
    
    # Load pre-trained SSL encoder
    ssl_model_path = config.model.get('ssl_model_path')
    if ssl_model_path:
        base_model = SequenceStructureSSL.from_pretrained(ssl_model_path)
        logger.info(f"Loaded pre-trained SSL model from {ssl_model_path}")
    else:
        logger.warning("No pre-trained SSL model specified, training from scratch")
        model_config = config.model
        base_model = SequenceStructureSSL(
            d_model=model_config.get('d_model', 1280),
            n_layers=model_config.get('n_layers', 33),
            n_heads=model_config.get('n_heads', 20)
        )
    
    # Create folding model
    folding_model = NeuralOperatorFold(
        encoder=base_model,
        operator_layers=config.model.get('operator_layers', 12),
        fourier_features=config.model.get('fourier_features', 256),
        uncertainty_method=config.model.get('uncertainty_method', 'ensemble')
    )
    
    # Create structure dataset
    try:
        dataset = StructureDataset.from_pdb(
            args.data,
            resolution_cutoff=config.data.get('resolution_cutoff', 3.0),
            remove_redundancy=config.data.get('remove_redundancy', True)
        )
        logger.info(f"Loaded {len(dataset)} structures")
    except Exception as e:
        logger.error(f"Failed to create structure dataset: {e}")
        sys.exit(1)
    
    # Create trainer
    trainer = FoldingTrainer(
        model=folding_model,
        loss_weights=config.training.get('loss_weights', {
            'distance_map': 1.0,
            'torsion_angles': 0.5,
            'secondary_structure': 0.3
        }),
        use_wandb=config.logging.get('use_wandb', False),
        project_name=config.logging.get('project_name', 'protein-folding')
    )
    
    # Train model
    trainer.fit(
        dataset,
        epochs=config.training.get('epochs', 50),
        batch_size=config.training.get('batch_size', 16),
        validation_split=config.training.get('validation_split', 0.1),
        save_dir=args.output,
        resume_from=args.resume,
        num_gpus=args.gpus
    )
    
    # Save final model
    torch.save(folding_model.state_dict(), f"{args.output}/folding_model_final.pt")
    logger.info("Folding training completed!")