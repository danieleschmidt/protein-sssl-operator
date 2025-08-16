"""
Enhanced error handling and recovery for protein-sssl-operator
Generation 2: MAKE IT ROBUST - Comprehensive error handling
"""

import logging
import traceback
import functools
import torch
import warnings
from typing import Dict, Any, Optional, Callable, Union, List
from contextlib import contextmanager
import numpy as np
import time

class ProteinSSLError(Exception):
    """Base exception for protein-sssl-operator"""
    pass

class ModelError(ProteinSSLError):
    """Model-related errors"""
    pass

class DataError(ProteinSSLError):
    """Data processing errors"""
    pass

class ValidationError(ProteinSSLError):
    """Input validation errors"""
    pass

class ComputationError(ProteinSSLError):
    """Computation and numerical errors"""
    pass

class DimensionMismatchError(ModelError):
    """Dimension mismatch in neural networks"""
    pass

def robust_error_handler(
    fallback_value: Any = None,
    exception_types: tuple = (Exception,),
    log_level: str = "ERROR",
    reraise: bool = False
):
    """
    Decorator for robust error handling with fallback values
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                logger = logging.getLogger(func.__module__)
                logger.log(
                    getattr(logging, log_level.upper()),
                    f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                
                if reraise:
                    raise
                
                return fallback_value
        return wrapper
    return decorator

def validate_tensor_dimensions(
    tensor: torch.Tensor,
    expected_dims: Union[int, List[int]],
    name: str = "tensor"
) -> torch.Tensor:
    """
    Validate tensor dimensions with automatic fixing when possible
    """
    if isinstance(expected_dims, int):
        expected_dims = [expected_dims]
    
    actual_dims = list(tensor.shape)
    
    # Try to fix common dimension issues
    if len(actual_dims) != len(expected_dims):
        if len(actual_dims) == len(expected_dims) - 1:
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            actual_dims = list(tensor.shape)
        elif len(actual_dims) == len(expected_dims) + 1 and actual_dims[0] == 1:
            # Remove batch dimension
            tensor = tensor.squeeze(0)
            actual_dims = list(tensor.shape)
    
    # Check if dimensions are compatible
    for i, (actual, expected) in enumerate(zip(actual_dims, expected_dims)):
        if expected is not None and expected != -1 and actual != expected:
            raise DimensionMismatchError(
                f"{name} dimension {i}: expected {expected}, got {actual}. "
                f"Full shape: {actual_dims} vs expected: {expected_dims}"
            )
    
    return tensor

def validate_protein_sequence(sequence: str) -> str:
    """
    Validate and clean protein sequence
    """
    if not isinstance(sequence, str):
        raise ValidationError(f"Sequence must be string, got {type(sequence)}")
    
    # Clean sequence
    sequence = sequence.upper().strip()
    
    # Valid amino acid codes
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    # Remove invalid characters and warn
    cleaned_chars = []
    for char in sequence:
        if char in valid_aa:
            cleaned_chars.append(char)
        elif char in "XBZ":
            # Common ambiguous amino acids, replace with closest
            replacements = {"X": "A", "B": "N", "Z": "Q"}
            cleaned_chars.append(replacements[char])
            warnings.warn(f"Replaced ambiguous amino acid {char} with {replacements[char]}")
    
    cleaned_sequence = "".join(cleaned_chars)
    
    if len(cleaned_sequence) == 0:
        raise ValidationError("No valid amino acids found in sequence")
    
    if len(cleaned_sequence) < 3:
        raise ValidationError(f"Sequence too short: {len(cleaned_sequence)} residues (minimum 3)")
    
    if len(cleaned_sequence) > 2048:
        warnings.warn(f"Very long sequence: {len(cleaned_sequence)} residues. Consider splitting.")
    
    return cleaned_sequence

@contextmanager
def error_recovery_context(recovery_strategies: Dict[type, Callable] = None):
    """
    Context manager for automatic error recovery
    """
    if recovery_strategies is None:
        recovery_strategies = {}
    
    try:
        yield
    except Exception as e:
        error_type = type(e)
        
        # Try recovery strategies
        for exception_type, recovery_func in recovery_strategies.items():
            if isinstance(e, exception_type):
                logging.warning(f"Attempting recovery for {error_type.__name__}: {str(e)}")
                try:
                    result = recovery_func(e)
                    logging.info(f"Recovery successful for {error_type.__name__}")
                    return result
                except Exception as recovery_error:
                    logging.error(f"Recovery failed: {str(recovery_error)}")
        
        # If no recovery strategy worked, re-raise
        raise

def safe_tensor_operation(
    operation: Callable,
    *tensors: torch.Tensor,
    fallback_strategy: str = "zeros",
    **kwargs
) -> torch.Tensor:
    """
    Safely perform tensor operations with fallback strategies
    """
    try:
        return operation(*tensors, **kwargs)
    except RuntimeError as e:
        if "size mismatch" in str(e) or "dimension" in str(e):
            # Try to fix dimension mismatches
            logging.warning(f"Dimension mismatch in tensor operation: {str(e)}")
            
            if fallback_strategy == "zeros":
                # Return zeros with appropriate shape
                if tensors:
                    reference_tensor = tensors[0]
                    return torch.zeros_like(reference_tensor)
                else:
                    return torch.zeros(1)
            
            elif fallback_strategy == "interpolate":
                # Try to interpolate to match dimensions
                if len(tensors) >= 2:
                    target_shape = tensors[0].shape
                    fixed_tensors = []
                    for tensor in tensors:
                        if tensor.shape != target_shape:
                            tensor = torch.nn.functional.interpolate(
                                tensor.unsqueeze(0).unsqueeze(0),
                                size=target_shape[-1],
                                mode='linear',
                                align_corners=False
                            ).squeeze()
                        fixed_tensors.append(tensor)
                    return operation(*fixed_tensors, **kwargs)
        
        raise ComputationError(f"Tensor operation failed: {str(e)}")

class ModelValidator:
    """
    Validates model inputs and outputs for consistency
    """
    
    @staticmethod
    def validate_model_input(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
        """Validate and fix model inputs"""
        
        # Validate input_ids
        if not isinstance(input_ids, torch.Tensor):
            raise ValidationError(f"input_ids must be torch.Tensor, got {type(input_ids)}")
        
        if input_ids.dtype not in [torch.long, torch.int]:
            input_ids = input_ids.long()
        
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        elif len(input_ids.shape) != 2:
            raise ValidationError(f"input_ids must be 1D or 2D, got shape {input_ids.shape}")
        
        batch_size, seq_len = input_ids.shape
        
        if seq_len > max_length:
            warnings.warn(f"Sequence length {seq_len} exceeds maximum {max_length}, truncating")
            input_ids = input_ids[:, :max_length]
            seq_len = max_length
        
        # Validate attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            if attention_mask.shape != input_ids.shape:
                if len(attention_mask.shape) == 1 and attention_mask.shape[0] == seq_len:
                    attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1)
                else:
                    raise ValidationError(
                        f"attention_mask shape {attention_mask.shape} incompatible with "
                        f"input_ids shape {input_ids.shape}"
                    )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    @staticmethod
    def validate_model_output(
        outputs: Dict[str, torch.Tensor],
        expected_keys: List[str],
        batch_size: int,
        seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Validate model outputs"""
        
        for key in expected_keys:
            if key not in outputs:
                raise ValidationError(f"Missing expected output key: {key}")
        
        for key, tensor in outputs.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValidationError(f"Output {key} must be torch.Tensor, got {type(tensor)}")
            
            # Check for NaN or Inf values
            if torch.isnan(tensor).any():
                warnings.warn(f"NaN values detected in output {key}")
                outputs[key] = torch.nan_to_num(tensor, nan=0.0)
            
            if torch.isinf(tensor).any():
                warnings.warn(f"Inf values detected in output {key}")
                outputs[key] = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
        
        return outputs

def setup_error_handling():
    """Setup global error handling for the package"""
    
    # Configure warnings
    warnings.filterwarnings("default", category=UserWarning)
    
    # Set up logging for errors
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # PyTorch specific settings for better error messages
    torch.autograd.set_detect_anomaly(True)
    
    return True

# Initialize error handling
setup_error_handling()