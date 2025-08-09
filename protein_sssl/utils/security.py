"""
Security utilities for protein-sssl-operator
Provides input validation, sanitization, and secure model handling
"""

import re
import hashlib
import os
import tempfile
import pickle
import torch
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from functools import wraps
import warnings

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

class InputValidator:
    """Comprehensive input validation for protein sequences and model parameters"""
    
    # Standard amino acid codes
    VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Extended amino acid codes (including ambiguous)
    EXTENDED_AMINO_ACIDS = VALID_AMINO_ACIDS.union(set('BJOUXZ'))
    
    # Maximum reasonable sequence length (configurable)
    MAX_SEQUENCE_LENGTH = 10000
    
    # Maximum reasonable batch size
    MAX_BATCH_SIZE = 1000
    
    @classmethod
    def validate_protein_sequence(cls, 
                                sequence: str, 
                                allow_extended: bool = False,
                                max_length: Optional[int] = None) -> str:
        """
        Validate and sanitize protein sequence
        
        Args:
            sequence: Input protein sequence
            allow_extended: Allow extended amino acid codes
            max_length: Maximum allowed sequence length
            
        Returns:
            Cleaned and validated sequence
            
        Raises:
            SecurityError: If sequence is invalid or potentially malicious
        """
        
        if not isinstance(sequence, str):
            raise SecurityError(f"Sequence must be string, got {type(sequence)}")
        
        # Remove whitespace and convert to uppercase
        sequence = sequence.strip().upper()
        
        if not sequence:
            raise SecurityError("Empty sequence provided")
        
        # Check length
        max_len = max_length or cls.MAX_SEQUENCE_LENGTH
        if len(sequence) > max_len:
            raise SecurityError(f"Sequence too long: {len(sequence)} > {max_len}")
        
        # Check for valid amino acids
        valid_chars = cls.EXTENDED_AMINO_ACIDS if allow_extended else cls.VALID_AMINO_ACIDS
        invalid_chars = set(sequence) - valid_chars
        
        if invalid_chars:
            raise SecurityError(f"Invalid amino acid codes: {invalid_chars}")
        
        # Check for suspicious patterns
        cls._check_suspicious_patterns(sequence)
        
        return sequence
    
    @classmethod
    def _check_suspicious_patterns(cls, sequence: str):
        """Check for suspicious patterns that might indicate attacks"""
        
        # Check for repetitive patterns that might cause DoS
        if len(set(sequence)) == 1 and len(sequence) > 1000:
            raise SecurityError("Suspicious repetitive pattern detected")
        
        # Check for extremely long runs of single amino acid
        for aa in cls.VALID_AMINO_ACIDS:
            if aa * 500 in sequence:  # 500 consecutive same amino acids
                raise SecurityError(f"Suspicious long run of {aa} detected")
        
        # Check for potential SQL injection patterns (paranoid check)
        sql_patterns = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'SELECT', '--', ';']
        for pattern in sql_patterns:
            if pattern in sequence:
                warnings.warn(f"Potential SQL pattern '{pattern}' in sequence", UserWarning)
    
    @classmethod
    def validate_model_parameters(cls, 
                                d_model: int,
                                n_layers: int,
                                n_heads: int,
                                max_length: int,
                                batch_size: Optional[int] = None) -> Dict[str, int]:
        """
        Validate model parameters to prevent resource exhaustion
        
        Args:
            d_model: Model dimension
            n_layers: Number of layers
            n_heads: Number of attention heads
            max_length: Maximum sequence length
            batch_size: Batch size (optional)
            
        Returns:
            Validated parameters dictionary
            
        Raises:
            SecurityError: If parameters are invalid or potentially dangerous
        """
        
        # Validate types
        params = {
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'max_length': max_length
        }
        
        for name, value in params.items():
            if not isinstance(value, int) or value <= 0:
                raise SecurityError(f"Parameter {name} must be positive integer, got {value}")
        
        # Check reasonable bounds to prevent resource exhaustion
        if d_model > 8192:
            raise SecurityError(f"d_model too large: {d_model} > 8192")
        
        if n_layers > 100:
            raise SecurityError(f"n_layers too large: {n_layers} > 100")
        
        if n_heads > 64:
            raise SecurityError(f"n_heads too large: {n_heads} > 64")
        
        if max_length > cls.MAX_SEQUENCE_LENGTH:
            raise SecurityError(f"max_length too large: {max_length} > {cls.MAX_SEQUENCE_LENGTH}")
        
        # Check that d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise SecurityError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Estimate memory usage (rough approximation)
        estimated_memory_mb = cls._estimate_memory_usage(d_model, n_layers, max_length, batch_size)
        if estimated_memory_mb > 32000:  # 32GB limit
            raise SecurityError(f"Estimated memory usage too high: {estimated_memory_mb:.1f}MB")
        
        if batch_size is not None:
            params['batch_size'] = batch_size
            if batch_size > cls.MAX_BATCH_SIZE:
                raise SecurityError(f"Batch size too large: {batch_size} > {cls.MAX_BATCH_SIZE}")
        
        return params
    
    @classmethod
    def _estimate_memory_usage(cls, 
                             d_model: int,
                             n_layers: int,
                             max_length: int,
                             batch_size: Optional[int] = None) -> float:
        """Estimate memory usage in MB"""
        
        batch_size = batch_size or 1
        
        # Rough estimation of transformer memory usage
        # This is a simplified calculation
        
        # Parameter memory (weights)
        param_memory = (
            d_model * d_model * 4 * n_layers +  # Self-attention weights
            d_model * d_model * 4 * n_layers +  # FFN weights
            d_model * 21 +  # Embedding
            d_model * 1000   # Other parameters
        ) * 4 / (1024 * 1024)  # Convert bytes to MB
        
        # Activation memory (forward pass)
        activation_memory = (
            batch_size * max_length * d_model * n_layers * 2  # Activations
        ) * 4 / (1024 * 1024)  # Convert bytes to MB
        
        # Attention memory (scales quadratically with sequence length)
        attention_memory = (
            batch_size * n_layers * max_length * max_length
        ) * 4 / (1024 * 1024)  # Convert bytes to MB
        
        return param_memory + activation_memory + attention_memory
    
    @classmethod
    def validate_file_path(cls, 
                          file_path: Union[str, Path],
                          allowed_extensions: Optional[Set[str]] = None,
                          max_size_mb: float = 1000.0) -> Path:
        """
        Validate file path for security
        
        Args:
            file_path: Path to validate
            allowed_extensions: Set of allowed file extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is invalid or potentially dangerous
        """
        
        if not isinstance(file_path, (str, Path)):
            raise SecurityError(f"File path must be string or Path, got {type(file_path)}")
        
        path = Path(file_path)
        
        # Check for path traversal attacks
        if '..' in str(path) or str(path).startswith('/'):
            # Allow absolute paths but check them carefully
            resolved_path = path.resolve()
            if '..' in str(resolved_path):
                raise SecurityError(f"Path traversal detected: {path}")
        
        # Check file extension
        if allowed_extensions:
            if path.suffix.lower() not in allowed_extensions:
                raise SecurityError(f"File extension {path.suffix} not allowed. Allowed: {allowed_extensions}")
        
        # Check file size if file exists
        if path.exists() and path.is_file():
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                raise SecurityError(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
        
        return path

class SecureModelHandler:
    """Secure model loading and saving with integrity checks"""
    
    @staticmethod
    def save_model_securely(model: torch.nn.Module,
                          file_path: Union[str, Path],
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Save model with integrity check
        
        Args:
            model: PyTorch model to save
            file_path: Path to save model
            metadata: Optional metadata to include
        """
        
        file_path = InputValidator.validate_file_path(
            file_path, 
            allowed_extensions={'.pt', '.pth'}, 
            max_size_mb=5000.0
        )
        
        # Create secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Prepare model data
            model_data = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'metadata': metadata or {}
            }
            
            # Add timestamp and integrity hash
            import time
            model_data['save_timestamp'] = time.time()
            
            # Save to temporary file first
            torch.save(model_data, temp_path)
            
            # Calculate integrity hash
            with open(temp_path, 'rb') as f:
                content = f.read()
                integrity_hash = hashlib.sha256(content).hexdigest()
            
            # Add hash to metadata and re-save
            model_data['integrity_hash'] = integrity_hash
            torch.save(model_data, temp_path)
            
            # Move to final location atomically
            temp_path.replace(file_path)
            
            logger.info(f"Model saved securely to {file_path} with integrity hash {integrity_hash[:16]}...")
            
        finally:
            # Cleanup temporary file if it still exists
            if temp_path.exists():
                temp_path.unlink()
    
    @staticmethod
    def load_model_securely(file_path: Union[str, Path],
                          expected_model_class: Optional[str] = None,
                          verify_integrity: bool = True) -> Dict[str, Any]:
        """
        Load model with integrity verification
        
        Args:
            file_path: Path to model file
            expected_model_class: Expected model class name for validation
            verify_integrity: Whether to verify file integrity
            
        Returns:
            Model data dictionary
            
        Raises:
            SecurityError: If model fails security checks
        """
        
        file_path = InputValidator.validate_file_path(
            file_path,
            allowed_extensions={'.pt', '.pth'},
            max_size_mb=5000.0
        )
        
        if not file_path.exists():
            raise SecurityError(f"Model file not found: {file_path}")
        
        try:
            # Load model data
            model_data = torch.load(file_path, map_location='cpu')
            
            if not isinstance(model_data, dict):
                raise SecurityError("Invalid model file format")
            
            # Verify integrity if hash is present
            if verify_integrity and 'integrity_hash' in model_data:
                stored_hash = model_data.pop('integrity_hash')
                
                # Re-save without hash to verify
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
                    temp_path = Path(temp_file.name)
                
                try:
                    torch.save(model_data, temp_path)
                    
                    with open(temp_path, 'rb') as f:
                        content = f.read()
                        calculated_hash = hashlib.sha256(content).hexdigest()
                    
                    if stored_hash != calculated_hash:
                        raise SecurityError("Model integrity check failed - file may be corrupted")
                    
                    logger.info(f"Model integrity verified: {calculated_hash[:16]}...")
                    
                finally:
                    temp_path.unlink()
            
            # Verify model class if specified
            if expected_model_class and 'model_class' in model_data:
                if model_data['model_class'] != expected_model_class:
                    raise SecurityError(
                        f"Model class mismatch: expected {expected_model_class}, "
                        f"got {model_data['model_class']}"
                    )
            
            return model_data
            
        except (pickle.UnpicklingError, torch.serialization.StorageException) as e:
            raise SecurityError(f"Failed to load model file: {e}")
    
    @staticmethod
    def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize configuration dictionary
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Sanitized configuration
        """
        
        sanitized = {}
        
        # List of allowed config keys
        allowed_keys = {
            'd_model', 'n_layers', 'n_heads', 'vocab_size', 'max_length',
            'dropout', 'learning_rate', 'batch_size', 'epochs', 'warmup_steps',
            'weight_decay', 'gradient_checkpointing', 'mixed_precision',
            'ssl_objectives', 'loss_weights', 'optimizer_type', 'scheduler_type'
        }
        
        for key, value in config.items():
            if key not in allowed_keys:
                logger.warning(f"Ignoring unknown config key: {key}")
                continue
            
            # Type validation
            if key in ['d_model', 'n_layers', 'n_heads', 'vocab_size', 'max_length', 'batch_size', 'epochs', 'warmup_steps']:
                if not isinstance(value, int) or value <= 0:
                    raise SecurityError(f"Config key {key} must be positive integer")
            
            elif key in ['dropout', 'learning_rate', 'weight_decay']:
                if not isinstance(value, (int, float)) or value < 0:
                    raise SecurityError(f"Config key {key} must be non-negative number")
            
            elif key in ['gradient_checkpointing', 'mixed_precision']:
                if not isinstance(value, bool):
                    raise SecurityError(f"Config key {key} must be boolean")
            
            elif key in ['ssl_objectives']:
                if not isinstance(value, list):
                    raise SecurityError(f"Config key {key} must be list")
            
            sanitized[key] = value
        
        return sanitized

def secure_function(allowed_extensions: Optional[Set[str]] = None,
                   max_file_size_mb: float = 100.0,
                   validate_sequences: bool = True):
    """
    Decorator to add security validation to functions
    
    Args:
        allowed_extensions: Allowed file extensions for file arguments
        max_file_size_mb: Maximum file size in MB
        validate_sequences: Whether to validate protein sequences
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            # Validate file path arguments
            if allowed_extensions:
                for arg_name, arg_value in kwargs.items():
                    if isinstance(arg_value, (str, Path)) and Path(arg_value).suffix:
                        try:
                            InputValidator.validate_file_path(
                                arg_value, 
                                allowed_extensions, 
                                max_file_size_mb
                            )
                        except SecurityError as e:
                            raise SecurityError(f"Security validation failed for {arg_name}: {e}")
            
            # Validate sequence arguments
            if validate_sequences:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in ['sequence', 'sequences'] and isinstance(arg_value, str):
                        try:
                            InputValidator.validate_protein_sequence(arg_value)
                        except SecurityError as e:
                            raise SecurityError(f"Sequence validation failed for {arg_name}: {e}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def sanitize_output(data: Any, max_output_size: int = 10**6) -> Any:
    """
    Sanitize output data to prevent information leakage
    
    Args:
        data: Data to sanitize
        max_output_size: Maximum output size in characters
        
    Returns:
        Sanitized data
    """
    
    if isinstance(data, str):
        if len(data) > max_output_size:
            return data[:max_output_size] + "... [TRUNCATED]"
        return data
    
    elif isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Skip potentially sensitive keys
            if any(sensitive in str(key).lower() for sensitive in ['password', 'token', 'key', 'secret']):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_output(value, max_output_size)
        return sanitized
    
    elif isinstance(data, (list, tuple)):
        return type(data)(sanitize_output(item, max_output_size) for item in data)
    
    else:
        return data

# Security audit utilities
def audit_model_security(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Perform security audit on a model
    
    Args:
        model: Model to audit
        
    Returns:
        Audit results
    """
    
    audit_results = {
        'model_class': model.__class__.__name__,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'has_custom_forward': hasattr(model, 'forward') and 'forward' in model.__class__.__dict__,
        'suspicious_methods': [],
        'large_tensors': [],
        'security_score': 100  # Start with perfect score
    }
    
    # Check for suspicious methods
    suspicious_methods = ['exec', 'eval', 'open', '__import__', 'subprocess']
    for method in suspicious_methods:
        if hasattr(model, method):
            audit_results['suspicious_methods'].append(method)
            audit_results['security_score'] -= 20
    
    # Check for unusually large tensors (potential for resource exhaustion)
    for name, param in model.named_parameters():
        if param.numel() > 10**8:  # 100M parameters
            audit_results['large_tensors'].append(name)
            audit_results['security_score'] -= 10
    
    # Check parameter count reasonableness
    if audit_results['parameter_count'] > 10**10:  # 10B parameters
        audit_results['security_score'] -= 30
    
    audit_results['security_level'] = (
        'HIGH' if audit_results['security_score'] >= 80 else
        'MEDIUM' if audit_results['security_score'] >= 60 else
        'LOW'
    )
    
    return audit_results