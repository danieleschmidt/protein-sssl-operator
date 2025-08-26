#!/usr/bin/env python3
"""
🛡️ GENERATION 2: ROBUST ERROR HANDLING AND VALIDATION SYSTEM
============================================================
Comprehensive error handling, input validation, and recovery mechanisms
"""

import functools
import logging
import traceback
import time
from typing import Any, Callable, Optional, Dict, List, Union, Type
from pathlib import Path
import sys
import os

class ProteinSSLError(Exception):
    """Base exception for protein-sssl-operator."""
    pass

class ConfigurationError(ProteinSSLError):
    """Configuration-related errors."""
    pass

class ModelError(ProteinSSLError):
    """Model loading/inference errors."""
    pass

class DataValidationError(ProteinSSLError):
    """Data validation errors."""
    pass

class ResourceError(ProteinSSLError):
    """Resource allocation/availability errors."""
    pass

class RobustErrorHandler:
    """Comprehensive error handling system with recovery mechanisms."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_recovery: bool = True):
        """Initialize error handler.
        
        Args:
            log_level: Logging level
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            enable_recovery: Enable automatic recovery mechanisms
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_recovery = enable_recovery
        self.error_counts = {}
        
    def with_retry(self, 
                   exceptions: Union[Exception, tuple] = Exception,
                   max_retries: Optional[int] = None,
                   delay: Optional[float] = None):
        """Decorator for automatic retry on failures."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                retry_delay = delay or self.retry_delay
                
                last_exception = None
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < retries:
                            self.logger.warning(
                                f"Attempt {attempt + 1}/{retries + 1} failed: {e}. "
                                f"Retrying in {retry_delay}s..."
                            )
                            time.sleep(retry_delay)
                        else:
                            self.logger.error(f"All {retries + 1} attempts failed for {func.__name__}")
                
                raise last_exception
            return wrapper
        return decorator
    
    def safe_execute(self, 
                     func: Callable,
                     *args,
                     fallback_value: Any = None,
                     reraise: bool = False,
                     **kwargs) -> Any:
        """Execute function with comprehensive error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_key = f"{func.__name__}:{type(e).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            self.logger.error(
                f"Error in {func.__name__}: {e}\n"
                f"Error count for {error_key}: {self.error_counts[error_key]}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            
            if reraise:
                raise
            return fallback_value
    
    def validate_and_sanitize(self, data: Any, validators: List[Callable]) -> Any:
        """Apply multiple validators and sanitizers to data."""
        current_data = data
        
        for i, validator in enumerate(validators):
            try:
                current_data = validator(current_data)
            except Exception as e:
                raise DataValidationError(
                    f"Validation step {i + 1} failed: {e}"
                ) from e
        
        return current_data

class InputValidator:
    """Comprehensive input validation for protein sequences and configurations."""
    
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AMINO_ACIDS = VALID_AMINO_ACIDS | {"X", "B", "Z", "J", "U", "O"}
    
    @staticmethod
    def validate_protein_sequence(sequence: str, 
                                allow_extended: bool = True,
                                min_length: int = 1,
                                max_length: int = 10000) -> str:
        """Validate and sanitize protein sequence."""
        if not isinstance(sequence, str):
            raise DataValidationError(f"Sequence must be string, got {type(sequence)}")
        
        # Remove whitespace and convert to uppercase
        sequence = ''.join(sequence.split()).upper()
        
        if not sequence:
            raise DataValidationError("Empty sequence provided")
        
        if len(sequence) < min_length:
            raise DataValidationError(f"Sequence too short: {len(sequence)} < {min_length}")
        
        if len(sequence) > max_length:
            raise DataValidationError(f"Sequence too long: {len(sequence)} > {max_length}")
        
        # Validate amino acids
        valid_set = InputValidator.EXTENDED_AMINO_ACIDS if allow_extended else InputValidator.VALID_AMINO_ACIDS
        invalid_chars = set(sequence) - valid_set
        
        if invalid_chars:
            raise DataValidationError(f"Invalid amino acids found: {invalid_chars}")
        
        # Replace ambiguous amino acids with X
        if not allow_extended:
            sequence = sequence.replace('B', 'X').replace('Z', 'X').replace('J', 'X')
        
        return sequence
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], 
                          must_exist: bool = True,
                          allowed_extensions: Optional[List[str]] = None) -> Path:
        """Validate file path."""
        if isinstance(path, str):
            path = Path(path)
        
        if not isinstance(path, Path):
            raise DataValidationError(f"Invalid path type: {type(path)}")
        
        if must_exist and not path.exists():
            raise DataValidationError(f"File does not exist: {path}")
        
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise DataValidationError(
                    f"Invalid file extension: {path.suffix}. "
                    f"Allowed: {allowed_extensions}"
                )
        
        return path
    
    @staticmethod
    def validate_config_dict(config: Dict[str, Any], 
                           required_keys: Optional[List[str]] = None,
                           allowed_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ConfigurationError(f"Config must be dict, got {type(config)}")
        
        if required_keys:
            missing_keys = set(required_keys) - set(config.keys())
            if missing_keys:
                raise ConfigurationError(f"Missing required config keys: {missing_keys}")
        
        if allowed_keys:
            invalid_keys = set(config.keys()) - set(allowed_keys)
            if invalid_keys:
                raise ConfigurationError(f"Invalid config keys: {invalid_keys}")
        
        return config
    
    @staticmethod
    def validate_model_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model parameters."""
        validations = {
            'd_model': lambda x: InputValidator._validate_positive_int(x, 'd_model', min_val=1),
            'n_layers': lambda x: InputValidator._validate_positive_int(x, 'n_layers', min_val=1, max_val=100),
            'n_heads': lambda x: InputValidator._validate_positive_int(x, 'n_heads', min_val=1, max_val=64),
            'dropout': lambda x: InputValidator._validate_float_range(x, 'dropout', 0.0, 1.0),
            'learning_rate': lambda x: InputValidator._validate_float_range(x, 'learning_rate', 1e-8, 1.0),
            'batch_size': lambda x: InputValidator._validate_positive_int(x, 'batch_size', min_val=1),
        }
        
        validated_params = {}
        for key, value in params.items():
            if key in validations:
                validated_params[key] = validations[key](value)
            else:
                validated_params[key] = value
        
        return validated_params
    
    @staticmethod
    def _validate_positive_int(value: Any, name: str, min_val: int = 1, max_val: Optional[int] = None) -> int:
        """Validate positive integer."""
        if not isinstance(value, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise DataValidationError(f"{name} must be integer, got {type(value)}")
        
        if value < min_val:
            raise DataValidationError(f"{name} must be >= {min_val}, got {value}")
        
        if max_val and value > max_val:
            raise DataValidationError(f"{name} must be <= {max_val}, got {value}")
        
        return value
    
    @staticmethod
    def _validate_float_range(value: Any, name: str, min_val: float, max_val: float) -> float:
        """Validate float within range."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise DataValidationError(f"{name} must be numeric, got {type(value)}")
        
        if value < min_val or value > max_val:
            raise DataValidationError(f"{name} must be in range [{min_val}, {max_val}], got {value}")
        
        return float(value)

class ResourceMonitor:
    """Monitor system resources and prevent overallocation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_memory_usage(self, required_gb: float = 1.0) -> bool:
        """Check if sufficient memory is available."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_gb < required_gb:
                self.logger.warning(
                    f"Low memory: {available_gb:.1f}GB available, {required_gb:.1f}GB required"
                )
                return False
            
            return True
            
        except ImportError:
            self.logger.warning("psutil not available, skipping memory check")
            return True
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
            return True  # Conservative: assume memory is available
    
    def check_disk_space(self, path: Union[str, Path], required_gb: float = 1.0) -> bool:
        """Check if sufficient disk space is available."""
        try:
            if isinstance(path, str):
                path = Path(path)
            
            # Get parent directory if file doesn't exist
            check_path = path if path.is_dir() else path.parent
            
            import shutil
            _, _, free_bytes = shutil.disk_usage(check_path)
            available_gb = free_bytes / (1024**3)
            
            if available_gb < required_gb:
                self.logger.warning(
                    f"Low disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Disk space check failed: {e}")
            return True  # Conservative: assume space is available

class SecurityValidator:
    """Security-focused validation and sanitization."""
    
    DANGEROUS_PATTERNS = [
        r'\.\./',           # Path traversal
        r'[<>"|*?]',       # Dangerous filename characters
        r'(?i)script',      # Script injections
        r'(?i)eval',        # Code evaluation
        r'(?i)exec',        # Code execution
        r'(?i)import\s+os', # OS imports
        r'(?i)subprocess',  # Subprocess calls
    ]
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe usage."""
        import re
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        # Ensure not empty
        if not filename:
            filename = "sanitized_file"
        
        return filename
    
    @staticmethod
    def validate_safe_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
        """Validate path is safe and within allowed directory."""
        if isinstance(path, str):
            path = Path(path)
        
        # Resolve to absolute path
        abs_path = path.resolve()
        
        # Check if within base directory
        if base_dir:
            base_abs = base_dir.resolve()
            try:
                abs_path.relative_to(base_abs)
            except ValueError:
                raise SecurityValidator.SecurityError(
                    f"Path outside allowed directory: {abs_path}"
                )
        
        # Check for dangerous patterns
        path_str = str(abs_path)
        import re
        for pattern in SecurityValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, path_str):
                raise SecurityValidator.SecurityError(
                    f"Dangerous pattern detected in path: {path_str}"
                )
        
        return abs_path
    
    class SecurityError(ProteinSSLError):
        """Security validation error."""
        pass

# Global instances for convenience
error_handler = RobustErrorHandler()
validator = InputValidator()
resource_monitor = ResourceMonitor()
security_validator = SecurityValidator()

def robust_operation(func=None, *, retries=3, fallback=None, reraise=False):
    """Decorator for robust operations with comprehensive error handling."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return error_handler.safe_execute(
                f, *args, 
                fallback_value=fallback,
                reraise=reraise,
                **kwargs
            )
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

# Example usage and testing functions
def test_error_handling():
    """Test the error handling system."""
    print("🛡️ Testing Robust Error Handling System")
    print("=" * 45)
    
    # Test input validation
    try:
        seq = validator.validate_protein_sequence("MKFLKFSLLT")
        print(f"✅ Valid sequence: {seq}")
    except Exception as e:
        print(f"❌ Sequence validation failed: {e}")
    
    try:
        validator.validate_protein_sequence("MKFLKF123")
        print("❌ Should have failed on invalid characters")
    except DataValidationError:
        print("✅ Correctly caught invalid sequence")
    
    # Test retry mechanism
    @error_handler.with_retry(exceptions=ValueError, max_retries=2, delay=0.1)
    def flaky_function(attempt_count=[0]):
        attempt_count[0] += 1
        if attempt_count[0] <= 2:
            raise ValueError(f"Attempt {attempt_count[0]} failed")
        return f"Success on attempt {attempt_count[0]}"
    
    try:
        result = flaky_function()
        print(f"✅ Retry mechanism worked: {result}")
    except Exception as e:
        print(f"❌ Retry failed: {e}")
    
    # Test resource monitoring
    if resource_monitor.check_memory_usage(0.1):  # Check for 100MB
        print("✅ Memory check passed")
    else:
        print("⚠️  Low memory detected")
    
    print("🛡️ Error handling system tests completed")
    return True

if __name__ == "__main__":
    test_error_handling()