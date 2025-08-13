"""
Advanced input validation and sanitization for protein-sssl-operator
Provides comprehensive validation for sequences, models, and configurations
"""

import re
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .logging_config import get_logger
from .security import SecurityValidator

logger = get_logger().get_logger("validation")

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"  
    PERMISSIVE = "permissive"

@dataclass
class ValidationResult:
    """Result of validation with details"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Any = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

class SequenceValidator:
    """Validates and sanitizes protein sequences"""
    
    # Standard amino acid codes
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
    EXTENDED_AA = STANDARD_AA | {"U", "O", "B", "Z", "J", "X"}  # Include ambiguous/modified
    
    # Common substitutions for non-standard characters
    SUBSTITUTION_MAP = {
        'U': 'C',  # Selenocysteine -> Cysteine
        'O': 'K',  # Pyrrolysine -> Lysine
        'B': 'N',  # Asparagine or Aspartic acid -> Asparagine
        'Z': 'Q',  # Glutamine or Glutamic acid -> Glutamine
        'J': 'L',  # Leucine or Isoleucine -> Leucine
    }
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.security_validator = SecurityValidator()
        
    def validate_sequence(self, sequence: str, 
                         sequence_name: str = "unknown",
                         allow_lowercase: bool = True,
                         max_length: int = 10000,
                         min_length: int = 1) -> ValidationResult:
        """
        Comprehensive sequence validation
        
        Args:
            sequence: Protein sequence to validate
            sequence_name: Name/ID for logging
            allow_lowercase: Whether to allow lowercase letters
            max_length: Maximum allowed sequence length
            min_length: Minimum allowed sequence length
            
        Returns:
            ValidationResult with validation status and sanitized sequence
        """
        
        logger.debug(f"Validating sequence '{sequence_name}' (length: {len(sequence) if sequence else 0})")
        
        errors = []
        warnings = []
        sanitized_sequence = sequence
        metadata = {
            'original_length': len(sequence) if sequence else 0,
            'sequence_name': sequence_name,
            'validation_level': self.validation_level.value
        }
        
        # Basic checks
        if not sequence:
            errors.append("Sequence is empty or None")
            return ValidationResult(False, errors, warnings, None, metadata)
            
        if not isinstance(sequence, str):
            errors.append(f"Sequence must be string, got {type(sequence)}")
            return ValidationResult(False, errors, warnings, None, metadata)
        
        # Security check
        security_result = self.security_validator.validate_input(sequence)
        if not security_result.is_safe:
            errors.extend([f"Security: {err}" for err in security_result.threats])
            
        # Length validation
        if len(sequence) > max_length:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Sequence too long: {len(sequence)} > {max_length}")
            else:
                warnings.append(f"Sequence very long: {len(sequence)} residues")
                sanitized_sequence = sequence[:max_length]
                warnings.append(f"Truncated to {max_length} residues")
                
        if len(sequence) < min_length:
            errors.append(f"Sequence too short: {len(sequence)} < {min_length}")
        
        # Character validation and sanitization
        sanitized_sequence, char_errors, char_warnings = self._validate_characters(
            sanitized_sequence, allow_lowercase
        )
        errors.extend(char_errors)
        warnings.extend(char_warnings)
        
        # Composition analysis
        composition = self._analyze_composition(sanitized_sequence)
        metadata.update(composition)
        
        # Quality checks
        quality_warnings = self._check_sequence_quality(sanitized_sequence)
        warnings.extend(quality_warnings)
        
        # Update metadata
        metadata.update({
            'final_length': len(sanitized_sequence),
            'characters_modified': len(sequence) != len(sanitized_sequence) or sequence != sanitized_sequence,
            'amino_acid_types': len(set(sanitized_sequence.upper()) & self.STANDARD_AA)
        })
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.debug(f"Sequence '{sequence_name}' validated successfully")
        else:
            logger.warning(f"Sequence '{sequence_name}' validation failed: {errors}")
            
        return ValidationResult(is_valid, errors, warnings, sanitized_sequence, metadata)
    
    def _validate_characters(self, sequence: str, allow_lowercase: bool) -> Tuple[str, List[str], List[str]]:
        """Validate and sanitize sequence characters"""
        errors = []
        warnings = []
        sanitized = ""
        
        # Convert to uppercase if needed
        if allow_lowercase:
            sequence = sequence.upper()
        
        invalid_chars = set()
        substitution_count = 0
        
        for i, char in enumerate(sequence):
            if char in self.STANDARD_AA:
                sanitized += char
            elif char in self.SUBSTITUTION_MAP:
                # Handle substitutions
                replacement = self.SUBSTITUTION_MAP[char]
                sanitized += replacement
                substitution_count += 1
                if substitution_count <= 5:  # Limit warning spam
                    warnings.append(f"Position {i+1}: '{char}' substituted with '{replacement}'")
            elif char in self.EXTENDED_AA:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"Position {i+1}: Non-standard amino acid '{char}'")
                else:
                    sanitized += 'X'  # Unknown residue
                    warnings.append(f"Position {i+1}: '{char}' replaced with 'X'")
            elif char.isspace():
                # Skip whitespace
                continue
            elif char.isdigit():
                # Skip numbers (common in sequence formats)
                continue
            else:
                invalid_chars.add(char)
                if self.validation_level != ValidationLevel.PERMISSIVE:
                    sanitized += 'X'  # Replace with unknown
                    
        if substitution_count > 5:
            warnings.append(f"Total {substitution_count} character substitutions made")
            
        if invalid_chars:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Invalid characters found: {sorted(invalid_chars)}")
            else:
                warnings.append(f"Invalid characters replaced: {sorted(invalid_chars)}")
                
        return sanitized, errors, warnings
    
    def _analyze_composition(self, sequence: str) -> Dict[str, Any]:
        """Analyze amino acid composition"""
        if not sequence:
            return {}
            
        composition = {}
        for aa in self.STANDARD_AA:
            count = sequence.count(aa)
            if count > 0:
                composition[f"aa_{aa}"] = count
                composition[f"freq_{aa}"] = count / len(sequence)
        
        # Basic properties
        hydrophobic = "AILMFWVYC"
        polar = "NQST"
        charged = "RKDE"
        aromatic = "FYW"
        
        return {
            **composition,
            'hydrophobic_fraction': sum(1 for aa in sequence if aa in hydrophobic) / len(sequence),
            'polar_fraction': sum(1 for aa in sequence if aa in polar) / len(sequence),
            'charged_fraction': sum(1 for aa in sequence if aa in charged) / len(sequence),
            'aromatic_fraction': sum(1 for aa in sequence if aa in aromatic) / len(sequence),
            'unique_residues': len(set(sequence)),
            'complexity': self._calculate_complexity(sequence)
        }
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity (Shannon entropy)"""
        if not sequence:
            return 0.0
            
        from collections import Counter
        counts = Counter(sequence)
        length = len(sequence)
        
        entropy = 0.0
        for count in counts.values():
            p = count / length
            if p > 0:
                entropy -= p * np.log2(p)
                
        # Normalize by max possible entropy
        max_entropy = np.log2(min(len(self.STANDARD_AA), len(set(sequence))))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _check_sequence_quality(self, sequence: str) -> List[str]:
        """Check for common sequence quality issues"""
        warnings = []
        
        if not sequence:
            return warnings
            
        # Check for excessive repeats
        for aa in self.STANDARD_AA:
            max_repeat = self._find_max_repeat(sequence, aa)
            if max_repeat > 10:
                warnings.append(f"Long {aa} repeat found ({max_repeat} residues)")
        
        # Check for low complexity regions
        if len(sequence) >= 20:
            window_size = min(20, len(sequence) // 4)
            for i in range(0, len(sequence) - window_size + 1, window_size):
                window = sequence[i:i+window_size]
                complexity = self._calculate_complexity(window)
                if complexity < 0.5:
                    warnings.append(f"Low complexity region at positions {i+1}-{i+window_size}")
        
        # Check amino acid frequency outliers
        freq_threshold = 0.4  # 40% of sequence
        for aa in self.STANDARD_AA:
            freq = sequence.count(aa) / len(sequence)
            if freq > freq_threshold:
                warnings.append(f"High {aa} frequency: {freq:.1%}")
                
        return warnings
    
    def _find_max_repeat(self, sequence: str, aa: str) -> int:
        """Find maximum consecutive repeat of an amino acid"""
        max_repeat = 0
        current_repeat = 0
        
        for char in sequence:
            if char == aa:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 0
                
        return max_repeat

class ModelValidator:
    """Validates model architectures and parameters"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        
    def validate_model_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate model configuration parameters"""
        errors = []
        warnings = []
        sanitized_config = config.copy()
        
        # Required fields
        required_fields = ['d_model', 'n_layers', 'n_heads']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate dimensions
        if 'd_model' in config:
            d_model = config['d_model']
            if not isinstance(d_model, int) or d_model <= 0:
                errors.append(f"d_model must be positive integer, got {d_model}")
            elif d_model % config.get('n_heads', 1) != 0:
                errors.append(f"d_model ({d_model}) must be divisible by n_heads ({config.get('n_heads')})")
            elif d_model > 8192:
                warnings.append(f"Very large d_model: {d_model}")
                
        # Validate layer count
        if 'n_layers' in config:
            n_layers = config['n_layers']
            if not isinstance(n_layers, int) or n_layers <= 0:
                errors.append(f"n_layers must be positive integer, got {n_layers}")
            elif n_layers > 100:
                warnings.append(f"Very deep model: {n_layers} layers")
                
        # Validate attention heads
        if 'n_heads' in config:
            n_heads = config['n_heads']
            if not isinstance(n_heads, int) or n_heads <= 0:
                errors.append(f"n_heads must be positive integer, got {n_heads}")
                
        return ValidationResult(
            len(errors) == 0, errors, warnings, sanitized_config
        )

class ConfigValidator:
    """Validates configuration files and parameters"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        
    def validate_training_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration"""
        errors = []
        warnings = []
        sanitized_config = config.copy()
        
        # Learning rate validation
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append(f"learning_rate must be positive number, got {lr}")
            elif lr > 1.0:
                warnings.append(f"Very high learning rate: {lr}")
            elif lr < 1e-8:
                warnings.append(f"Very low learning rate: {lr}")
                
        # Batch size validation
        if 'batch_size' in config:
            bs = config['batch_size']
            if not isinstance(bs, int) or bs <= 0:
                errors.append(f"batch_size must be positive integer, got {bs}")
            elif bs > 1024:
                warnings.append(f"Very large batch size: {bs}")
                
        # Epochs validation
        if 'epochs' in config:
            epochs = config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                errors.append(f"epochs must be positive integer, got {epochs}")
            elif epochs > 1000:
                warnings.append(f"Very many epochs: {epochs}")
                
        return ValidationResult(
            len(errors) == 0, errors, warnings, sanitized_config
        )

class DataValidator:
    """Validates datasets and data paths"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.sequence_validator = SequenceValidator(validation_level)
        
    def validate_dataset_path(self, path: Union[str, Path]) -> ValidationResult:
        """Validate dataset file path"""
        errors = []
        warnings = []
        
        path = Path(path)
        
        if not path.exists():
            errors.append(f"Dataset path does not exist: {path}")
            return ValidationResult(False, errors, warnings, None)
            
        if not path.is_file():
            errors.append(f"Dataset path is not a file: {path}")
            return ValidationResult(False, errors, warnings, None)
            
        # Check file extension
        valid_extensions = {'.fasta', '.fa', '.fas', '.txt', '.tsv', '.csv', '.json'}
        if path.suffix.lower() not in valid_extensions:
            warnings.append(f"Unusual file extension: {path.suffix}")
            
        # Check file size
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 1000:  # 1GB
                warnings.append(f"Very large dataset file: {size_mb:.1f} MB")
            elif size_mb == 0:
                errors.append("Dataset file is empty")
        except Exception as e:
            warnings.append(f"Could not check file size: {e}")
            
        return ValidationResult(len(errors) == 0, errors, warnings, str(path))
    
    def validate_batch(self, batch: Dict[str, Any]) -> ValidationResult:
        """Validate a training batch"""
        errors = []
        warnings = []
        
        required_keys = ['input_ids', 'attention_mask']
        for key in required_keys:
            if key not in batch:
                errors.append(f"Missing required batch key: {key}")
        
        # Validate tensor shapes if available
        if 'input_ids' in batch and hasattr(batch['input_ids'], 'shape'):
            shape = batch['input_ids'].shape
            if len(shape) != 2:
                errors.append(f"input_ids should be 2D tensor, got shape {shape}")
            elif shape[1] > 8192:
                warnings.append(f"Very long sequences in batch: max_len={shape[1]}")
                
        return ValidationResult(len(errors) == 0, errors, warnings, batch)

# Convenience functions
def validate_protein_sequence(sequence: str, 
                            sequence_name: str = "unknown",
                            validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """Quick sequence validation"""
    validator = SequenceValidator(validation_level)
    return validator.validate_sequence(sequence, sequence_name)

def validate_config(config: Dict[str, Any], 
                   config_type: str = "training",
                   validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """Quick config validation"""
    if config_type == "training":
        validator = ConfigValidator(validation_level)
        return validator.validate_training_config(config)
    elif config_type == "model":
        validator = ModelValidator(validation_level)
        return validator.validate_model_config(config)
    else:
        return ValidationResult(False, [f"Unknown config type: {config_type}"], [], None)

# Global validators for convenience
_sequence_validator = SequenceValidator()
_model_validator = ModelValidator()
_config_validator = ConfigValidator()
_data_validator = DataValidator()

def get_sequence_validator() -> SequenceValidator:
    return _sequence_validator

def get_model_validator() -> ModelValidator:
    return _model_validator

def get_config_validator() -> ConfigValidator:
    return _config_validator

def get_data_validator() -> DataValidator:
    return _data_validator