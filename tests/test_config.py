"""
Configuration management tests.
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from omegaconf import OmegaConf

from protein_sssl.config import (
    ConfigManager,
    load_config,
    save_config,
    get_default_ssl_config,
    get_default_folding_config
)
from protein_sssl.config.defaults import get_minimal_config


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config_manager = ConfigManager()
        self.config_manager.clear_cache()
    
    def test_load_valid_config(self):
        """Test loading valid configuration."""
        config_data = {
            'model': {
                'd_model': 512,
                'n_layers': 12,
                'n_heads': 8
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001,
                'epochs': 10
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = self.config_manager.load_config(config_path)
            assert config.model.d_model == 512
            assert config.training.batch_size == 16
        finally:
            Path(config_path).unlink()
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            self.config_manager.load_config("nonexistent.yaml")
    
    def test_config_caching(self):
        """Test configuration caching."""
        config_data = {'model': {'d_model': 256}, 'training': {'epochs': 5}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load config twice
            config1 = self.config_manager.load_config(config_path)
            config2 = self.config_manager.load_config(config_path)
            
            # Should be the same object (cached)
            assert config1 is config2
        finally:
            Path(config_path).unlink()
    
    def test_save_config(self):
        """Test saving configuration."""
        config = OmegaConf.create({
            'model': {'d_model': 128},
            'training': {'epochs': 1}
        })
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            self.config_manager.save_config(config, output_path)
            
            # Load back and verify
            loaded_config = self.config_manager.load_config(output_path)
            assert loaded_config.model.d_model == 128
            assert loaded_config.training.epochs == 1
        finally:
            Path(output_path).unlink()
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = OmegaConf.create({
            'model': {'d_model': 512, 'n_layers': 12},
            'training': {'epochs': 10, 'batch_size': 32}
        })
        
        override_config = OmegaConf.create({
            'model': {'d_model': 256},  # Override
            'training': {'learning_rate': 0.001}  # Add new
        })
        
        merged = self.config_manager.merge_configs(base_config, override_config)
        
        assert merged.model.d_model == 256  # Overridden
        assert merged.model.n_layers == 12  # Preserved
        assert merged.training.epochs == 10  # Preserved
        assert merged.training.batch_size == 32  # Preserved
        assert merged.training.learning_rate == 0.001  # Added
    
    def test_validate_config_missing_required(self):
        """Test validation with missing required sections."""
        invalid_config = OmegaConf.create({
            'model': {'d_model': 512}
            # Missing 'training' section
        })
        
        with pytest.raises(ValueError, match="must contain 'training'"):
            self.config_manager._validate_config(invalid_config)
    
    def test_validate_config_invalid_values(self):
        """Test validation with invalid parameter values."""
        invalid_config = OmegaConf.create({
            'model': {'d_model': -1},  # Invalid negative value
            'training': {'batch_size': 16}
        })
        
        with pytest.raises(ValueError, match="must be positive"):
            self.config_manager._validate_config(invalid_config)
    
    def test_validate_ssl_config(self):
        """Test SSL-specific configuration validation."""
        ssl_config = OmegaConf.create({
            'training_type': 'ssl',
            'model': {
                'd_model': 512,
                'ssl_objectives': ['invalid_objective']  # Invalid objective
            },
            'training': {'batch_size': 16}
        })
        
        with pytest.raises(ValueError, match="Invalid SSL objective"):
            self.config_manager._validate_config(ssl_config)
    
    def test_validate_folding_config(self):
        """Test folding-specific configuration validation."""
        folding_config = OmegaConf.create({
            'training_type': 'folding',
            'model': {
                'd_model': 512,
                'operator_layers': -1  # Invalid negative value
            },
            'training': {
                'batch_size': 16,
                'loss_weights': {
                    'distance_map': -0.5  # Invalid negative weight
                }
            }
        })
        
        with pytest.raises(ValueError, match="must be positive"):
            self.config_manager._validate_config(folding_config)
    
    def test_get_default_configs(self):
        """Test getting default configurations."""
        ssl_config = self.config_manager.get_default_config('ssl')
        assert ssl_config.training_type == 'ssl'
        assert 'masked_modeling' in ssl_config.model.ssl_objectives
        
        folding_config = self.config_manager.get_default_config('folding')
        assert folding_config.training_type == 'folding'
        assert folding_config.model.operator_layers > 0
    
    def test_update_config_from_args(self):
        """Test updating configuration from command line arguments."""
        base_config = OmegaConf.create({
            'model': {'d_model': 512},
            'training': {'batch_size': 32, 'epochs': 10},
            'hardware': {'num_gpus': 1}
        })
        
        args = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'num_gpus': 2
        }
        
        updated_config = self.config_manager.update_config_from_args(base_config, args)
        
        assert updated_config.training.batch_size == 64  # Updated
        assert updated_config.training.epochs == 10  # Preserved
        assert updated_config.training.learning_rate == 0.001  # Added
        assert updated_config.hardware.num_gpus == 2  # Updated


class TestDefaultConfigs:
    """Test default configuration templates."""
    
    def test_default_ssl_config(self):
        """Test default SSL configuration."""
        config = get_default_ssl_config()
        
        assert config.training_type == 'ssl'
        assert config.model.d_model == 1280
        assert config.model.n_layers == 33
        assert config.model.n_heads == 20
        assert len(config.model.ssl_objectives) == 3
        assert config.training.epochs == 10
        assert config.training.batch_size == 32
        assert config.logging.use_wandb is True
    
    def test_default_folding_config(self):
        """Test default folding configuration."""
        config = get_default_folding_config()
        
        assert config.training_type == 'folding'
        assert config.model.d_model == 1280
        assert config.model.operator_layers == 12
        assert config.model.uncertainty_method == 'ensemble'
        assert config.training.epochs == 50
        assert config.training.batch_size == 16
        assert len(config.evaluation.metrics) >= 3
    
    def test_minimal_ssl_config(self):
        """Test minimal SSL configuration for testing."""
        config = get_minimal_config('ssl')
        
        assert config.training_type == 'ssl'
        assert config.model.d_model == 512  # Smaller than default
        assert config.model.n_layers == 12   # Smaller than default
        assert config.training.epochs == 1   # Quick training
        assert config.logging.use_wandb is False  # No external logging
    
    def test_minimal_folding_config(self):
        """Test minimal folding configuration for testing."""
        config = get_minimal_config('folding')
        
        assert config.training_type == 'folding'
        assert config.model.d_model == 512
        assert config.model.operator_layers == 6  # Smaller than default
        assert config.model.uncertainty_method == 'none'  # No uncertainty
        assert config.training.epochs == 5
        assert config.logging.use_wandb is False
    
    def test_minimal_config_invalid_type(self):
        """Test minimal config with invalid type."""
        with pytest.raises(ValueError, match="Unknown config type"):
            get_minimal_config('invalid_type')


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_load_config_function(self):
        """Test standalone load_config function."""
        config_data = {
            'model': {'d_model': 256},
            'training': {'batch_size': 8}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.model.d_model == 256
            assert config.training.batch_size == 8
        finally:
            Path(config_path).unlink()
    
    def test_save_config_function(self):
        """Test standalone save_config function."""
        config = OmegaConf.create({
            'model': {'d_model': 128},
            'training': {'epochs': 1}
        })
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            save_config(config, output_path)
            
            # Verify file exists and is readable
            assert Path(output_path).exists()
            
            # Load back to verify content
            loaded = load_config(output_path)
            assert loaded.model.d_model == 128
        finally:
            Path(output_path).unlink()


class TestConfigValidation:
    """Test comprehensive configuration validation."""
    
    def test_config_with_all_sections(self):
        """Test configuration with all required sections."""
        complete_config = OmegaConf.create({
            'training_type': 'ssl',
            'model': {
                'd_model': 1280,
                'n_layers': 33,
                'n_heads': 20,
                'ssl_objectives': ['masked_modeling', 'contrastive']
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 1e-4
            },
            'optimizer': {
                'name': 'adamw',
                'eps': 1e-8
            },
            'logging': {
                'use_wandb': True,
                'project_name': 'test-project'
            }
        })
        
        manager = ConfigManager()
        # Should not raise any exception
        manager._validate_config(complete_config)
    
    def test_config_edge_cases(self):
        """Test configuration validation edge cases."""
        manager = ConfigManager()
        
        # Zero values where positive required
        config = OmegaConf.create({
            'model': {'d_model': 0},
            'training': {'batch_size': 16}
        })
        with pytest.raises(ValueError, match="must be positive"):
            manager._validate_config(config)
        
        # Very large values (potential security issue)
        config = OmegaConf.create({
            'model': {'d_model': 512},
            'training': {'batch_size': 999999}  # Extremely large
        })
        # Should pass validation (batch_size limit is high)
        manager._validate_config(config)
    
    def test_nested_config_validation(self):
        """Test validation of nested configuration structures."""
        nested_config = OmegaConf.create({
            'model': {
                'd_model': 512,
                'architecture': {
                    'layers': {
                        'encoder': {'depth': 12},
                        'decoder': {'depth': 6}
                    }
                }
            },
            'training': {
                'optimization': {
                    'optimizer': {
                        'name': 'adamw',
                        'params': {
                            'lr': 1e-4,
                            'weight_decay': 0.01
                        }
                    }
                },
                'batch_size': 32
            }
        })
        
        manager = ConfigManager()
        # Should handle deeply nested structures
        manager._validate_config(nested_config)


if __name__ == "__main__":
    pytest.main([__file__])