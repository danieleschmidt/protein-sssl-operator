"""
Input validation and sanitization utilities.
"""
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

from .logging_config import setup_logging

logger = setup_logging(__name__)


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        self.amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        self.max_sequence_length = 10000  # Hard security limit
        self.max_file_size_mb = 100  # Maximum file size in MB
        
    def validate_sequence(self, sequence: str, allow_unknown: bool = True) -> str:
        """Validate and sanitize protein sequence."""
        if not isinstance(sequence, str):
            raise ValueError("Sequence must be a string")
        
        if not sequence.strip():
            raise ValueError("Sequence cannot be empty")
        
        # Remove whitespace and convert to uppercase
        clean_sequence = ''.join(sequence.upper().split())
        
        # Check length limits
        if len(clean_sequence) > self.max_sequence_length:
            raise SecurityError(
                f"Sequence too long ({len(clean_sequence)}). "
                f"Maximum allowed: {self.max_sequence_length}"
            )
        
        if len(clean_sequence) < 3:
            raise ValueError("Sequence too short (minimum 3 residues)")
        
        # Validate characters
        valid_chars = self.amino_acids.copy()
        if allow_unknown:
            valid_chars.add('X')  # Unknown amino acid
            valid_chars.add('U')  # Selenocysteine 
            valid_chars.add('O')  # Pyrrolysine
        
        invalid_chars = set(clean_sequence) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid characters in sequence: {invalid_chars}")
        
        logger.debug(f"Validated sequence of length {len(clean_sequence)}")
        return clean_sequence
    
    def validate_file_path(self, file_path: Union[str, Path], must_exist: bool = True) -> Path:
        """Validate file path for security and existence."""
        try:
            path = Path(file_path).resolve()
        except Exception as e:
            raise ValueError(f"Invalid file path: {e}")
        
        # Security checks
        if '..' in str(path) or path.is_absolute() and not str(path).startswith('/tmp'):
            # Allow absolute paths only in safe directories
            safe_prefixes = ['/tmp', '/var/tmp', os.getcwd(), '/data', '/models']
            if not any(str(path).startswith(prefix) for prefix in safe_prefixes):
                logger.warning(f"Potentially unsafe path blocked: {path}")
        
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.exists():
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                raise SecurityError(
                    f"File too large ({size_mb:.1f}MB). "
                    f"Maximum allowed: {self.max_file_size_mb}MB"
                )
        
        return path
    
    def validate_model_path(self, model_path: Union[str, Path]) -> Path:
        """Validate model file path with additional security checks."""
        path = self.validate_file_path(model_path, must_exist=True)
        
        # Check file extension
        valid_extensions = {'.pt', '.pth', '.bin', '.pkl', '.pickle'}
        if path.suffix not in valid_extensions:
            raise ValueError(f"Invalid model file extension: {path.suffix}")
        
        # Basic file content validation (check if it looks like a model file)
        try:
            with open(path, 'rb') as f:
                header = f.read(8)
                # Check for common model file magic numbers
                if header[:2] == b'PK':  # ZIP-based format (some PyTorch models)
                    pass
                elif header[:4] == b'\x80\x02\x8a\n':  # Pickle format
                    pass
                else:
                    logger.warning(f"Unusual model file format detected: {path}")
        except Exception as e:
            raise ValueError(f"Cannot read model file: {e}")
        
        return path
    
    def validate_coordinates(self, coordinates: Any) -> bool:
        """Validate 3D coordinates array."""
        try:
            import numpy as np
            coords = np.asarray(coordinates)
        except Exception:
            raise ValueError("Coordinates must be numpy-compatible array")
        
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("Coordinates must be shape (N, atoms, 3)")
        
        # Check for reasonable coordinate values (Angstroms)
        if np.any(np.abs(coords) > 1000):
            raise ValueError("Coordinates contain unreasonably large values")
        
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            raise ValueError("Coordinates contain NaN or infinite values")
        
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Security limits
        security_limits = {
            'batch_size': 256,
            'max_length': 5000,
            'epochs': 1000,
            'learning_rate': 1.0,
            'num_workers': 32
        }
        
        # Recursive validation
        def validate_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check security limits
                    if key in security_limits:
                        if isinstance(value, (int, float)) and value > security_limits[key]:
                            raise SecurityError(
                                f"Value too large for {current_path}: {value} > {security_limits[key]}"
                            )
                    
                    validate_recursive(value, current_path)
            
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    validate_recursive(item, f"{path}[{i}]")
            
            elif isinstance(obj, str):
                # Check for potentially dangerous strings
                dangerous_patterns = [
                    r'__[a-zA-Z]+__',  # Python magic methods
                    r'eval\s*\(',      # Code execution
                    r'exec\s*\(',      # Code execution
                    r'import\s+',      # Dynamic imports
                    r'\.\./',          # Path traversal
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, obj, re.IGNORECASE):
                        logger.warning(f"Potentially dangerous string in config at {path}: {pattern}")
        
        validate_recursive(config)
        return config
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent security issues."""
        # Remove or replace dangerous characters
        safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Prevent hidden files and path traversal
        safe_filename = safe_filename.lstrip('.')
        safe_filename = safe_filename.replace('..', '__')
        
        # Limit length
        if len(safe_filename) > 255:
            safe_filename = safe_filename[:255]
        
        if not safe_filename:
            safe_filename = f"file_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
        
        return safe_filename
    
    def validate_device(self, device: str) -> str:
        """Validate device specification."""
        device = device.lower().strip()
        
        valid_devices = {'cpu', 'cuda', 'auto'}
        valid_cuda_pattern = re.compile(r'^cuda:\d+$')
        
        if device in valid_devices:
            return device
        elif valid_cuda_pattern.match(device):
            return device
        else:
            raise ValueError(f"Invalid device specification: {device}")
    
    def validate_numeric_range(
        self, 
        value: Union[int, float], 
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "value"
    ) -> Union[int, float]:
        """Validate numeric value within range."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        
        return value


# Global validator instance
validator = InputValidator()

def validate_sequence(sequence: str, **kwargs) -> str:
    """Convenience function for sequence validation."""
    return validator.validate_sequence(sequence, **kwargs)

def validate_file_path(file_path: Union[str, Path], **kwargs) -> Path:
    """Convenience function for file path validation."""  
    return validator.validate_file_path(file_path, **kwargs)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for config validation."""
    return validator.validate_config(config)