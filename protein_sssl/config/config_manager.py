"""
Configuration management utilities for protein-sssl-operator.
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
from omegaconf import OmegaConf, DictConfig

from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ConfigManager:
    """Manager for configuration files and validation."""
    
    def __init__(self):
        self.config_cache = {}
        
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """Load configuration from file with caching."""
        config_path = Path(config_path)
        
        if str(config_path) in self.config_cache:
            logger.debug(f"Using cached config from {config_path}")
            return self.config_cache[str(config_path)]
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            config = OmegaConf.load(config_path)
            self._validate_config(config)
            self.config_cache[str(config_path)] = config
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def save_config(self, config: DictConfig, output_path: Union[str, Path]):
        """Save configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                OmegaConf.save(config, f)
            logger.info(f"Saved configuration to {output_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save configuration: {e}")
    
    def merge_configs(self, base_config: DictConfig, override_config: DictConfig) -> DictConfig:
        """Merge two configurations with override taking precedence."""
        try:
            merged = OmegaConf.merge(base_config, override_config)
            self._validate_config(merged)
            return merged
            
        except Exception as e:
            raise ValueError(f"Failed to merge configurations: {e}")
    
    def _validate_config(self, config: DictConfig):
        """Validate configuration parameters."""
        # Check required top-level keys
        if 'model' not in config:
            raise ValueError("Configuration must contain 'model' section")
            
        if 'training' not in config:
            raise ValueError("Configuration must contain 'training' section")
        
        # Validate model parameters
        model_config = config.model
        if 'd_model' in model_config and model_config.d_model <= 0:
            raise ValueError("model.d_model must be positive")
            
        if 'n_layers' in model_config and model_config.n_layers <= 0:
            raise ValueError("model.n_layers must be positive")
            
        if 'n_heads' in model_config and model_config.n_heads <= 0:
            raise ValueError("model.n_heads must be positive")
        
        # Validate training parameters
        training_config = config.training
        if 'batch_size' in training_config and training_config.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
            
        if 'learning_rate' in training_config and training_config.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
            
        if 'epochs' in training_config and training_config.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        
        # Validate specific training types
        if hasattr(config, 'training_type'):
            if config.training_type == 'ssl':
                self._validate_ssl_config(config)
            elif config.training_type == 'folding':
                self._validate_folding_config(config)
        
        logger.debug("Configuration validation passed")
    
    def _validate_ssl_config(self, config: DictConfig):
        """Validate SSL-specific configuration."""
        model_config = config.model
        
        if 'ssl_objectives' in model_config:
            valid_objectives = {'masked_modeling', 'contrastive', 'distance_prediction'}
            for objective in model_config.ssl_objectives:
                if objective not in valid_objectives:
                    raise ValueError(f"Invalid SSL objective: {objective}")
        
        if 'data' in config and 'mask_probability' in config.data.get('augmentation', {}):
            mask_prob = config.data.augmentation.mask_probability
            if not 0 <= mask_prob <= 1:
                raise ValueError("data.augmentation.mask_probability must be between 0 and 1")
    
    def _validate_folding_config(self, config: DictConfig):
        """Validate folding-specific configuration."""
        model_config = config.model
        
        if 'operator_layers' in model_config and model_config.operator_layers <= 0:
            raise ValueError("model.operator_layers must be positive")
            
        if 'fourier_modes' in model_config and model_config.fourier_modes <= 0:
            raise ValueError("model.fourier_modes must be positive")
        
        if 'loss_weights' in config.training:
            weights = config.training.loss_weights
            for weight_name, weight_value in weights.items():
                if weight_value < 0:
                    raise ValueError(f"training.loss_weights.{weight_name} must be non-negative")
    
    def get_default_config(self, config_type: str = 'ssl') -> DictConfig:
        """Get default configuration for specified type."""
        if config_type == 'ssl':
            from .defaults import get_default_ssl_config
            return get_default_ssl_config()
        elif config_type == 'folding':
            from .defaults import get_default_folding_config
            return get_default_folding_config()
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    def update_config_from_args(self, config: DictConfig, args: Dict[str, Any]) -> DictConfig:
        """Update configuration with command line arguments."""
        updated_config = config.copy()
        
        # Map common CLI arguments to config paths
        arg_mappings = {
            'batch_size': 'training.batch_size',
            'learning_rate': 'training.learning_rate', 
            'epochs': 'training.epochs',
            'num_gpus': 'hardware.num_gpus',
            'mixed_precision': 'training.mixed_precision',
            'gradient_checkpointing': 'training.gradient_checkpointing'
        }
        
        for arg_name, config_path in arg_mappings.items():
            if arg_name in args and args[arg_name] is not None:
                # Navigate nested config structure
                keys = config_path.split('.')
                current = updated_config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = args[arg_name]
        
        self._validate_config(updated_config)
        return updated_config
    
    def clear_cache(self):
        """Clear configuration cache."""
        self.config_cache.clear()
        logger.debug("Configuration cache cleared")


# Global config manager instance
_config_manager = ConfigManager()

def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration file."""
    return _config_manager.load_config(config_path)

def save_config(config: DictConfig, output_path: Union[str, Path]):
    """Save configuration file."""
    return _config_manager.save_config(config, output_path)