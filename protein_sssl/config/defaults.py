"""
Default configuration templates for protein-sssl-operator.
"""
from omegaconf import OmegaConf, DictConfig


def get_default_ssl_config() -> DictConfig:
    """Get default SSL pre-training configuration."""
    config = {
        'training_type': 'ssl',
        
        'model': {
            'd_model': 1280,
            'n_layers': 33,
            'n_heads': 20,
            'vocab_size': 21,
            'max_length': 1024,
            'ssl_objectives': ['masked_modeling', 'contrastive', 'distance_prediction'],
            'dropout': 0.1
        },
        
        'data': {
            'max_length': 1024,
            'clustering_threshold': 0.9,
            'min_length': 30,
            'augmentation': {
                'mask_probability': 0.15,
                'random_mask_prob': 0.1,
                'unchanged_prob': 0.1
            }
        },
        
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 10000,
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'max_grad_norm': 1.0,
            
            'ssl_loss_weights': {
                'masked_lm': 1.0,
                'contrastive': 0.5,
                'distance_prediction': 0.3
            }
        },
        
        'optimizer': {
            'name': 'adamw',
            'eps': 1e-8,
            'betas': [0.9, 0.95]
        },
        
        'scheduler': {
            'name': 'cosine_with_warmup',
            'min_lr_ratio': 0.1
        },
        
        'logging': {
            'use_wandb': True,
            'project_name': 'protein-ssl',
            'log_interval': 100,
            'save_interval': 1000,
            'eval_interval': 5000
        },
        
        'hardware': {
            'num_gpus': 1,
            'num_workers': 4,
            'pin_memory': True
        },
        
        'paths': {
            'output_dir': './ssl_checkpoints',
            'logs_dir': './logs',
            'cache_dir': './cache'
        },
        
        'security': {
            'max_sequence_length': 2048,
            'max_batch_size': 128,
            'parameter_limit': 10e9,
            'memory_limit_gb': 32
        },
        
        'seed': 42
    }
    
    return OmegaConf.create(config)


def get_default_folding_config() -> DictConfig:
    """Get default structure folding configuration."""
    config = {
        'training_type': 'folding',
        
        'model': {
            'ssl_model_path': './ssl_checkpoints/final_model',
            'd_model': 1280,
            'n_layers': 33,
            'n_heads': 20,
            'operator_layers': 12,
            'fourier_features': 256,
            'attention_type': 'efficient',
            'uncertainty_method': 'ensemble',
            'num_ensemble_models': 5,
            'dropout': 0.1
        },
        
        'data': {
            'resolution_cutoff': 3.0,
            'remove_redundancy': True,
            'min_length': 30,
            'max_length': 1024,
            'quality_filters': {
                'min_coverage': 0.8,
                'max_missing_residues': 0.1,
                'require_backbone': True
            }
        },
        
        'training': {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_steps': 5000,
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'max_grad_norm': 1.0,
            'validation_split': 0.1,
            
            'loss_weights': {
                'distance_map': 1.0,
                'torsion_angles': 0.5,
                'secondary_structure': 0.3,
                'uncertainty': 0.2
            },
            
            'num_recycles': 3,
            'recycle_early_stop_tolerance': 1e-2
        },
        
        'optimizer': {
            'name': 'adamw',
            'eps': 1e-8,
            'betas': [0.9, 0.95]
        },
        
        'scheduler': {
            'name': 'cosine_with_warmup',
            'min_lr_ratio': 0.1
        },
        
        'logging': {
            'use_wandb': True,
            'project_name': 'protein-folding',
            'log_interval': 50,
            'save_interval': 500,
            'eval_interval': 1000
        },
        
        'evaluation': {
            'metrics': ['tm_score', 'gdt_ts', 'gdt_ha', 'lddt', 'rmsd'],
            'confidence_thresholds': [0.5, 0.7, 0.9],
            'validate_stereochemistry': True
        },
        
        'hardware': {
            'num_gpus': 1,
            'num_workers': 4,
            'pin_memory': True,
            'max_memory_gb': 24
        },
        
        'paths': {
            'output_dir': './folding_checkpoints',
            'logs_dir': './folding_logs',
            'predictions_dir': './predictions',
            'cache_dir': './folding_cache'
        },
        
        'security': {
            'max_sequence_length': 1024,
            'max_structure_size': 2048,
            'memory_limit_gb': 32,
            'time_limit_hours': 48
        },
        
        'seed': 42
    }
    
    return OmegaConf.create(config)


def get_minimal_config(config_type: str = 'ssl') -> DictConfig:
    """Get minimal configuration for quick setup."""
    if config_type == 'ssl':
        config = {
            'training_type': 'ssl',
            'model': {
                'd_model': 512,  # Smaller model for quick testing
                'n_layers': 12,
                'n_heads': 8,
                'ssl_objectives': ['masked_modeling']
            },
            'training': {
                'epochs': 1,
                'batch_size': 8,
                'learning_rate': 1e-4
            },
            'logging': {'use_wandb': False}
        }
    
    elif config_type == 'folding':
        config = {
            'training_type': 'folding',
            'model': {
                'd_model': 512,
                'operator_layers': 6,
                'uncertainty_method': 'none'
            },
            'training': {
                'epochs': 5,
                'batch_size': 4,
                'learning_rate': 1e-4,
                'loss_weights': {'distance_map': 1.0}
            },
            'logging': {'use_wandb': False}
        }
    
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return OmegaConf.create(config)