# Yaml-free config for autonomous execution
try:
    import yaml
    from .config_manager import ConfigManager, load_config, save_config
except ImportError:
    # Fallback yaml-free config system
    class ConfigManager:
        def __init__(self):
            self.config = {
                "model": {"d_model": 1280, "n_layers": 33, "n_heads": 20},
                "training": {"learning_rate": 1e-4, "batch_size": 128},
                "data": {"max_length": 512}
            }
        def get(self, key, default=None):
            return self.config.get(key, default)
        def set(self, key, value):
            self.config[key] = value
    
    def load_config(path=None):
        return ConfigManager().config
    
    def save_config(config, path):
        pass  # No-op for autonomous execution
# Import torch-free defaults for autonomous execution
try:
    from .defaults import get_default_ssl_config, get_default_folding_config
except ImportError:
    from .torch_free_defaults import get_default_ssl_config, get_default_folding_config

__all__ = [
    "ConfigManager", 
    "load_config", 
    "save_config",
    "get_default_ssl_config",
    "get_default_folding_config"
]