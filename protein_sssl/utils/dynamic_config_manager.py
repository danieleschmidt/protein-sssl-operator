"""
Dynamic Configuration Management for protein-sssl-operator
Provides enterprise-grade configuration management with dynamic updates,
validation schemas, environment-specific configs, and hot reloading.
"""

import os
import time
import json
import yaml
import threading
import logging
import hashlib
import importlib
from typing import (
    Dict, List, Optional, Any, Union, Callable, 
    Type, TypeVar, Generic, Protocol
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import weakref
import uuid

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    import jsonschema
    from jsonschema import validate, ValidationError as JsonSchemaValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    from omegaconf import OmegaConf, DictConfig, ListConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

try:
    import pydantic
    from pydantic import BaseModel, Field, validator, ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"
    PYTHON = "python"

class ConfigSource(Enum):
    """Configuration sources"""
    FILE = "file"
    ENVIRONMENT = "environment"
    REDIS = "redis"
    DATABASE = "database"
    HTTP = "http"
    CONSUL = "consul"
    ETCD = "etcd"

class ValidationLevel(Enum):
    """Configuration validation levels"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"

@dataclass
class ConfigChangeEvent:
    """Configuration change event"""
    timestamp: float
    source: str
    key: str
    old_value: Any
    new_value: Any
    event_type: str  # "added", "modified", "removed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'source': self.source,
            'key': self.key,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'event_type': self.event_type,
            'metadata': self.metadata
        }

@dataclass
class ConfigSchema:
    """Configuration validation schema"""
    name: str
    version: str
    description: str
    schema_format: str  # "jsonschema", "pydantic", "custom"
    schema_definition: Dict[str, Any]
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    validation_rules: List[Callable] = field(default_factory=list)
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against schema"""
        errors = []
        
        if self.schema_format == "jsonschema" and JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(config, self.schema_definition)
            except JsonSchemaValidationError as e:
                errors.append(f"JSON Schema validation failed: {e.message}")
        
        elif self.schema_format == "pydantic" and PYDANTIC_AVAILABLE:
            try:
                # This would require dynamic model creation
                pass
            except PydanticValidationError as e:
                errors.append(f"Pydantic validation failed: {e}")
        
        # Check required fields
        for field in self.required_fields:
            if field not in config:
                errors.append(f"Required field missing: {field}")
        
        # Run custom validation rules
        for rule in self.validation_rules:
            try:
                result = rule(config)
                if isinstance(result, str):
                    errors.append(result)
                elif not result:
                    errors.append(f"Validation rule failed: {rule.__name__}")
            except Exception as e:
                errors.append(f"Validation rule error: {e}")
        
        return len(errors) == 0, errors

class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration files"""
    
    def __init__(self, config_manager: 'DynamicConfigManager'):
        self.config_manager = config_manager
        self.watched_files = set()
    
    def add_file(self, file_path: Path):
        """Add file to watch list"""
        self.watched_files.add(str(file_path.resolve()))
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and str(event.src_path) in self.watched_files:
            logger.info(f"Configuration file modified: {event.src_path}")
            self.config_manager._handle_file_change(event.src_path)
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory and str(event.src_path) in self.watched_files:
            logger.info(f"Configuration file created: {event.src_path}")
            self.config_manager._handle_file_change(event.src_path)

class ConfigValidator:
    """Configuration validation engine"""
    
    def __init__(self):
        self.schemas: Dict[str, ConfigSchema] = {}
        self.global_validators: List[Callable] = []
    
    def register_schema(self, name: str, schema: ConfigSchema):
        """Register configuration schema"""
        self.schemas[name] = schema
        logger.info(f"Registered config schema: {name}")
    
    def add_global_validator(self, validator: Callable[[Dict[str, Any]], Union[bool, str]]):
        """Add global validation function"""
        self.global_validators.append(validator)
    
    def validate_config(self, config: Dict[str, Any], schema_name: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        all_errors = []
        
        # Global validators
        for validator in self.global_validators:
            try:
                result = validator(config)
                if isinstance(result, str):
                    all_errors.append(result)
                elif not result:
                    all_errors.append(f"Global validator failed: {validator.__name__}")
            except Exception as e:
                all_errors.append(f"Global validator error: {e}")
        
        # Schema-specific validation
        if schema_name and schema_name in self.schemas:
            schema = self.schemas[schema_name]
            is_valid, errors = schema.validate(config)
            all_errors.extend(errors)
        
        return len(all_errors) == 0, all_errors
    
    def validate_model_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model configuration"""
        errors = []
        
        # Required fields
        required = ['d_model', 'n_layers', 'n_heads']
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Type and range validation
        if 'd_model' in config:
            if not isinstance(config['d_model'], int) or config['d_model'] <= 0:
                errors.append("d_model must be positive integer")
            elif config['d_model'] % config.get('n_heads', 1) != 0:
                errors.append("d_model must be divisible by n_heads")
        
        if 'n_layers' in config:
            if not isinstance(config['n_layers'], int) or config['n_layers'] <= 0:
                errors.append("n_layers must be positive integer")
            elif config['n_layers'] > 100:
                errors.append("n_layers too large (max 100)")
        
        if 'n_heads' in config:
            if not isinstance(config['n_heads'], int) or config['n_heads'] <= 0:
                errors.append("n_heads must be positive integer")
            elif config['n_heads'] > 64:
                errors.append("n_heads too large (max 64)")
        
        if 'dropout' in config:
            if not isinstance(config['dropout'], (int, float)) or not (0 <= config['dropout'] <= 1):
                errors.append("dropout must be float between 0 and 1")
        
        return len(errors) == 0, errors
    
    def validate_training_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate training configuration"""
        errors = []
        
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("learning_rate must be positive number")
            elif lr > 1.0:
                errors.append("learning_rate very high (>1.0)")
        
        if 'batch_size' in config:
            bs = config['batch_size']
            if not isinstance(bs, int) or bs <= 0:
                errors.append("batch_size must be positive integer")
            elif bs > 1024:
                errors.append("batch_size very large (>1024)")
        
        if 'epochs' in config:
            epochs = config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                errors.append("epochs must be positive integer")
        
        if 'weight_decay' in config:
            wd = config['weight_decay']
            if not isinstance(wd, (int, float)) or wd < 0:
                errors.append("weight_decay must be non-negative number")
        
        return len(errors) == 0, errors

class ConfigProvider(Protocol):
    """Configuration provider interface"""
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from source"""
        ...
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to source"""
        ...
    
    def watch_changes(self, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Watch for configuration changes"""
        ...

class FileConfigProvider:
    """File-based configuration provider"""
    
    def __init__(self, file_path: Union[str, Path], format: ConfigFormat = ConfigFormat.YAML):
        self.file_path = Path(file_path)
        self.format = format
        self.last_modified = 0
        self._lock = threading.RLock()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        with self._lock:
            if not self.file_path.exists():
                return {}
            
            try:
                with open(self.file_path, 'r') as f:
                    if self.format == ConfigFormat.JSON:
                        config = json.load(f)
                    elif self.format == ConfigFormat.YAML:
                        config = yaml.safe_load(f)
                    elif self.format == ConfigFormat.PYTHON:
                        # Execute Python file as module
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("config", self.file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        config = {k: v for k, v in module.__dict__.items() if not k.startswith('_')}
                    else:
                        raise ValueError(f"Unsupported format: {self.format}")
                
                self.last_modified = self.file_path.stat().st_mtime
                return config or {}
                
            except Exception as e:
                logger.error(f"Error loading config from {self.file_path}: {e}")
                return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        with self._lock:
            try:
                # Create backup
                if self.file_path.exists():
                    backup_path = self.file_path.with_suffix(f"{self.file_path.suffix}.backup")
                    import shutil
                    shutil.copy2(self.file_path, backup_path)
                
                # Ensure directory exists
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(self.file_path, 'w') as f:
                    if self.format == ConfigFormat.JSON:
                        json.dump(config, f, indent=2)
                    elif self.format == ConfigFormat.YAML:
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                    else:
                        raise ValueError(f"Save not supported for format: {self.format}")
                
                self.last_modified = self.file_path.stat().st_mtime
                return True
                
            except Exception as e:
                logger.error(f"Error saving config to {self.file_path}: {e}")
                return False
    
    def has_changed(self) -> bool:
        """Check if file has been modified"""
        if not self.file_path.exists():
            return False
        return self.file_path.stat().st_mtime > self.last_modified

class EnvironmentConfigProvider:
    """Environment variable configuration provider"""
    
    def __init__(self, prefix: str = "PROTEIN_SSL_"):
        self.prefix = prefix
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.prefix):].lower()
                
                # Try to parse as JSON first, then as string
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Handle boolean strings
                    if value.lower() in ('true', 'false'):
                        config[config_key] = value.lower() == 'true'
                    # Handle numeric strings
                    elif value.isdigit():
                        config[config_key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        config[config_key] = float(value)
                    else:
                        config[config_key] = value
        
        return config
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to environment (limited support)"""
        # Environment variables are typically read-only for the process
        logger.warning("Saving to environment variables not supported")
        return False

class RedisConfigProvider:
    """Redis-based configuration provider"""
    
    def __init__(self, redis_url: str, key_prefix: str = "config:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis library not available")
        
        self.redis_client = redis.from_url(redis_url)
        self.key_prefix = key_prefix
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from Redis"""
        try:
            config = {}
            
            # Get all keys with prefix
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                config_key = key.decode('utf-8')[len(self.key_prefix):]
                value = self.redis_client.get(key)
                
                if value:
                    try:
                        config[config_key] = json.loads(value.decode('utf-8'))
                    except json.JSONDecodeError:
                        config[config_key] = value.decode('utf-8')
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from Redis: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to Redis"""
        try:
            pipe = self.redis_client.pipeline()
            
            for key, value in config.items():
                redis_key = f"{self.key_prefix}{key}"
                if isinstance(value, (dict, list)):
                    pipe.set(redis_key, json.dumps(value))
                else:
                    pipe.set(redis_key, str(value))
            
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to Redis: {e}")
            return False
    
    def watch_changes(self, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """Watch for configuration changes in Redis"""
        try:
            pubsub = self.redis_client.pubsub()
            channel = f"{self.key_prefix}changes"
            pubsub.subscribe(channel)
            
            def listen_loop():
                for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'].decode('utf-8'))
                            callback(data.get('key', ''), data.get('config', {}))
                        except Exception as e:
                            logger.error(f"Error processing Redis config change: {e}")
            
            threading.Thread(target=listen_loop, daemon=True).start()
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Redis config watcher: {e}")
            return False

class DynamicConfigManager:
    """Dynamic configuration management system"""
    
    def __init__(self,
                 primary_provider: ConfigProvider,
                 fallback_providers: Optional[List[ConfigProvider]] = None,
                 validation_level: ValidationLevel = ValidationLevel.BASIC,
                 auto_reload: bool = True,
                 reload_interval: float = 10.0):
        
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self.validation_level = validation_level
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        
        # Configuration state
        self._config: Dict[str, Any] = {}
        self._config_lock = threading.RLock()
        self._config_hash = ""
        
        # Validation
        self.validator = ConfigValidator()
        self._setup_default_validators()
        
        # Change tracking
        self._change_listeners: List[Callable[[ConfigChangeEvent], None]] = []
        self._change_history: deque = deque(maxlen=1000)
        
        # File watching
        self._file_observer = None
        self._file_watcher = None
        if WATCHDOG_AVAILABLE and auto_reload:
            self._setup_file_watcher()
        
        # Background reload thread
        self._reload_thread = None
        self._stop_reload = threading.Event()
        
        # Environment-specific configs
        self._environment = os.getenv('ENVIRONMENT', 'development')
        self._environment_configs: Dict[str, Dict[str, Any]] = {}
        
        # Load initial configuration
        self.reload_config()
        
        # Start background reload if enabled
        if auto_reload:
            self._start_background_reload()
        
        logger.info("Dynamic configuration manager initialized")
    
    def _setup_default_validators(self):
        """Setup default validation rules"""
        # Model config schema
        model_schema = ConfigSchema(
            name="model_config",
            version="1.0",
            description="Model configuration schema",
            schema_format="custom",
            schema_definition={},
            required_fields=['d_model', 'n_layers', 'n_heads'],
            validation_rules=[self.validator.validate_model_config]
        )
        self.validator.register_schema("model", model_schema)
        
        # Training config schema
        training_schema = ConfigSchema(
            name="training_config",
            version="1.0",
            description="Training configuration schema",
            schema_format="custom",
            schema_definition={},
            required_fields=['learning_rate'],
            validation_rules=[self.validator.validate_training_config]
        )
        self.validator.register_schema("training", training_schema)
    
    def _setup_file_watcher(self):
        """Setup file system watcher"""
        if hasattr(self.primary_provider, 'file_path'):
            self._file_watcher = ConfigFileWatcher(self)
            self._file_observer = Observer()
            
            file_path = self.primary_provider.file_path
            watch_dir = file_path.parent
            
            self._file_watcher.add_file(file_path)
            self._file_observer.schedule(self._file_watcher, str(watch_dir), recursive=False)
            self._file_observer.start()
            
            logger.info(f"File watcher started for {file_path}")
    
    def _start_background_reload(self):
        """Start background configuration reload thread"""
        def reload_loop():
            while not self._stop_reload.wait(self.reload_interval):
                try:
                    if hasattr(self.primary_provider, 'has_changed') and self.primary_provider.has_changed():
                        self.reload_config()
                except Exception as e:
                    logger.error(f"Error in background reload: {e}")
        
        self._reload_thread = threading.Thread(target=reload_loop, daemon=True)
        self._reload_thread.start()
    
    def reload_config(self) -> bool:
        """Reload configuration from all providers"""
        try:
            new_config = {}
            
            # Load from primary provider
            primary_config = self.primary_provider.load_config()
            new_config.update(primary_config)
            
            # Load from fallback providers
            for provider in self.fallback_providers:
                try:
                    fallback_config = provider.load_config()
                    # Fallback providers don't override primary
                    for key, value in fallback_config.items():
                        if key not in new_config:
                            new_config[key] = value
                except Exception as e:
                    logger.warning(f"Error loading from fallback provider: {e}")
            
            # Apply environment-specific overrides
            if self._environment in self._environment_configs:
                env_config = self._environment_configs[self._environment]
                new_config.update(env_config)
            
            # Validate configuration
            if self.validation_level != ValidationLevel.NONE:
                is_valid, errors = self._validate_full_config(new_config)
                
                if not is_valid:
                    if self.validation_level == ValidationLevel.STRICT:
                        logger.error(f"Configuration validation failed: {errors}")
                        return False
                    else:
                        logger.warning(f"Configuration validation warnings: {errors}")
            
            # Calculate config hash
            config_str = json.dumps(new_config, sort_keys=True)
            new_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            # Check if config actually changed
            if new_hash == self._config_hash:
                return True
            
            # Update configuration
            with self._config_lock:
                old_config = self._config.copy()
                self._config = new_config
                self._config_hash = new_hash
            
            # Track changes and notify listeners
            self._track_changes(old_config, new_config)
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False
    
    def _validate_full_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate entire configuration"""
        all_errors = []
        
        # Validate model section
        if 'model' in config:
            is_valid, errors = self.validator.validate_config(config['model'], 'model')
            all_errors.extend([f"model.{error}" for error in errors])
        
        # Validate training section
        if 'training' in config:
            is_valid, errors = self.validator.validate_config(config['training'], 'training')
            all_errors.extend([f"training.{error}" for error in errors])
        
        # Global validation
        is_valid, errors = self.validator.validate_config(config)
        all_errors.extend(errors)
        
        return len(all_errors) == 0, all_errors
    
    def _track_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Track configuration changes and notify listeners"""
        changes = self._calculate_changes(old_config, new_config)
        
        for change in changes:
            # Add to history
            self._change_history.append(change)
            
            # Notify listeners
            for listener in self._change_listeners:
                try:
                    listener(change)
                except Exception as e:
                    logger.error(f"Error in change listener: {e}")
    
    def _calculate_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], prefix: str = "") -> List[ConfigChangeEvent]:
        """Calculate configuration changes recursively"""
        changes = []
        timestamp = time.time()
        
        # Find removed keys
        for key in old_config:
            full_key = f"{prefix}.{key}" if prefix else key
            if key not in new_config:
                changes.append(ConfigChangeEvent(
                    timestamp=timestamp,
                    source="config_manager",
                    key=full_key,
                    old_value=old_config[key],
                    new_value=None,
                    event_type="removed"
                ))
        
        # Find added and modified keys
        for key in new_config:
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in old_config:
                # Added
                changes.append(ConfigChangeEvent(
                    timestamp=timestamp,
                    source="config_manager",
                    key=full_key,
                    old_value=None,
                    new_value=new_config[key],
                    event_type="added"
                ))
            elif old_config[key] != new_config[key]:
                # Modified
                if isinstance(old_config[key], dict) and isinstance(new_config[key], dict):
                    # Recurse into nested dictionaries
                    nested_changes = self._calculate_changes(
                        old_config[key], new_config[key], full_key
                    )
                    changes.extend(nested_changes)
                else:
                    changes.append(ConfigChangeEvent(
                        timestamp=timestamp,
                        source="config_manager",
                        key=full_key,
                        old_value=old_config[key],
                        new_value=new_config[key],
                        event_type="modified"
                    ))
        
        return changes
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        with self._config_lock:
            return self._get_nested_value(self._config, key, default)
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """Get nested value using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """Set configuration value by key (supports dot notation)"""
        with self._config_lock:
            old_config = self._config.copy()
            
            # Set the value
            self._set_nested_value(self._config, key, value)
            
            # Validate if required
            if self.validation_level != ValidationLevel.NONE:
                is_valid, errors = self._validate_full_config(self._config)
                
                if not is_valid and self.validation_level == ValidationLevel.STRICT:
                    # Revert change
                    self._config = old_config
                    logger.error(f"Configuration validation failed for {key}: {errors}")
                    return False
            
            # Persist if requested
            if persist:
                success = self.primary_provider.save_config(self._config)
                if not success:
                    logger.warning(f"Failed to persist configuration change for {key}")
            
            # Track change
            old_value = self._get_nested_value(old_config, key, None)
            change = ConfigChangeEvent(
                timestamp=time.time(),
                source="manual",
                key=key,
                old_value=old_value,
                new_value=value,
                event_type="modified" if old_value is not None else "added"
            )
            
            self._change_history.append(change)
            
            # Notify listeners
            for listener in self._change_listeners:
                try:
                    listener(change)
                except Exception as e:
                    logger.error(f"Error in change listener: {e}")
            
            return True
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set nested value using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def delete(self, key: str, persist: bool = True) -> bool:
        """Delete configuration value by key"""
        with self._config_lock:
            old_value = self._get_nested_value(self._config, key, None)
            
            if old_value is None:
                return False
            
            # Delete the value
            self._delete_nested_value(self._config, key)
            
            # Persist if requested
            if persist:
                success = self.primary_provider.save_config(self._config)
                if not success:
                    logger.warning(f"Failed to persist configuration deletion for {key}")
            
            # Track change
            change = ConfigChangeEvent(
                timestamp=time.time(),
                source="manual",
                key=key,
                old_value=old_value,
                new_value=None,
                event_type="removed"
            )
            
            self._change_history.append(change)
            
            # Notify listeners
            for listener in self._change_listeners:
                try:
                    listener(change)
                except Exception as e:
                    logger.error(f"Error in change listener: {e}")
            
            return True
    
    def _delete_nested_value(self, config: Dict[str, Any], key: str):
        """Delete nested value using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                return
            current = current[k]
        
        if keys[-1] in current:
            del current[keys[-1]]
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        with self._config_lock:
            return self._config.copy()
    
    def get_environment_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        env = environment or self._environment
        return self._environment_configs.get(env, {})
    
    def set_environment_config(self, environment: str, config: Dict[str, Any]):
        """Set environment-specific configuration"""
        self._environment_configs[environment] = config
        
        # Reload if this is the current environment
        if environment == self._environment:
            self.reload_config()
    
    def add_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """Add configuration change listener"""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """Remove configuration change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def get_change_history(self, hours: float = 24.0) -> List[ConfigChangeEvent]:
        """Get configuration change history"""
        cutoff_time = time.time() - (hours * 3600)
        return [change for change in self._change_history if change.timestamp >= cutoff_time]
    
    def export_config(self, file_path: str, format: ConfigFormat = ConfigFormat.YAML, 
                     include_metadata: bool = False) -> bool:
        """Export configuration to file"""
        try:
            export_data = self.get_all()
            
            if include_metadata:
                export_data['_metadata'] = {
                    'export_timestamp': time.time(),
                    'environment': self._environment,
                    'config_hash': self._config_hash,
                    'validation_level': self.validation_level.value
                }
            
            with open(file_path, 'w') as f:
                if format == ConfigFormat.JSON:
                    json.dump(export_data, f, indent=2)
                elif format == ConfigFormat.YAML:
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, file_path: str, format: ConfigFormat = ConfigFormat.YAML,
                     merge: bool = True, validate: bool = True) -> bool:
        """Import configuration from file"""
        try:
            with open(file_path, 'r') as f:
                if format == ConfigFormat.JSON:
                    import_data = json.load(f)
                elif format == ConfigFormat.YAML:
                    import_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported import format: {format}")
            
            # Remove metadata if present
            if '_metadata' in import_data:
                del import_data['_metadata']
            
            # Validate imported config
            if validate:
                is_valid, errors = self._validate_full_config(import_data)
                if not is_valid:
                    logger.error(f"Imported configuration validation failed: {errors}")
                    return False
            
            with self._config_lock:
                if merge:
                    # Merge with existing config
                    old_config = self._config.copy()
                    self._config.update(import_data)
                else:
                    # Replace entire config
                    old_config = self._config.copy()
                    self._config = import_data
                
                # Update hash
                config_str = json.dumps(self._config, sort_keys=True)
                self._config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            # Track changes
            self._track_changes(old_config, self._config)
            
            # Persist changes
            self.primary_provider.save_config(self._config)
            
            logger.info(f"Configuration imported from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
    
    def _handle_file_change(self, file_path: str):
        """Handle file change event"""
        logger.info(f"Configuration file changed: {file_path}")
        # Add small delay to ensure file write is complete
        time.sleep(0.1)
        self.reload_config()
    
    def stop(self):
        """Stop configuration manager"""
        # Stop background reload
        self._stop_reload.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=5.0)
        
        # Stop file watcher
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
        
        logger.info("Dynamic configuration manager stopped")

# Context manager
@contextmanager
def config_context(config_manager: DynamicConfigManager):
    """Context manager for configuration"""
    try:
        yield config_manager
    finally:
        config_manager.stop()

# Decorators
def with_config(key: str, default: Any = None):
    """Decorator to inject configuration value"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This would need a global config manager instance
            # For now, just pass through
            return func(*args, **kwargs)
        return wrapper
    return decorator

def watch_config(key: str, callback: Callable[[Any, Any], None]):
    """Decorator to watch configuration changes"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This would need a global config manager instance
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Global configuration manager instance
_global_config_manager: Optional[DynamicConfigManager] = None

def get_global_config_manager() -> Optional[DynamicConfigManager]:
    """Get global configuration manager instance"""
    return _global_config_manager

def initialize_global_config_manager(config_file: str, **kwargs) -> DynamicConfigManager:
    """Initialize global configuration manager"""
    global _global_config_manager
    
    primary_provider = FileConfigProvider(config_file)
    env_provider = EnvironmentConfigProvider()
    
    _global_config_manager = DynamicConfigManager(
        primary_provider=primary_provider,
        fallback_providers=[env_provider],
        **kwargs
    )
    
    return _global_config_manager

# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from global manager"""
    if _global_config_manager:
        return _global_config_manager.get(key, default)
    return default

def set_config(key: str, value: Any, persist: bool = True) -> bool:
    """Set configuration value in global manager"""
    if _global_config_manager:
        return _global_config_manager.set(key, value, persist)
    return False

def reload_global_config() -> bool:
    """Reload global configuration"""
    if _global_config_manager:
        return _global_config_manager.reload_config()
    return False
