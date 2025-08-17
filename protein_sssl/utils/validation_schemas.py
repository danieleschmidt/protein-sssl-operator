"""
Comprehensive Input Validation Schemas for protein-sssl-operator
Provides enterprise-grade validation with protocol buffer support,
schema-based validation, and comprehensive sanitization.
"""

import re
import json
import hashlib
import inspect
from typing import (
    Dict, List, Optional, Any, Union, Type, Callable, 
    get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field, fields
from enum import Enum
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

try:
    import pydantic
    from pydantic import BaseModel, Field, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    from google.protobuf import message as pb_message
    from google.protobuf import descriptor
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    DEVELOPMENT = "development"

class DataType(Enum):
    """Supported data types for validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    UUID = "uuid"
    BASE64 = "base64"
    JSON = "json"
    PROTEIN_SEQUENCE = "protein_sequence"
    DNA_SEQUENCE = "dna_sequence"
    RNA_SEQUENCE = "rna_sequence"
    COORDINATES_3D = "coordinates_3d"
    FILE_PATH = "file_path"
    MODEL_CONFIG = "model_config"
    TRAINING_CONFIG = "training_config"

@dataclass
class ValidationRule:
    """Individual validation rule"""
    name: str
    data_type: DataType
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    sanitizer: Optional[Callable[[Any], Any]] = None
    description: str = ""
    example: Optional[Any] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if other.errors:
            self.is_valid = False
        self.metadata.update(other.metadata)

class BaseValidator(ABC):
    """Base class for all validators"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
    
    @abstractmethod
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate value against rule"""
        pass
    
    @abstractmethod
    def sanitize(self, value: Any, rule: ValidationRule) -> Any:
        """Sanitize value according to rule"""
        pass

class StringValidator(BaseValidator):
    """Validator for string types"""
    
    # Common regex patterns
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'url': r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$',
        'ipv4': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        'ipv6': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
        'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        'base64': r'^[A-Za-z0-9+/]*={0,2}$',
        'protein_sequence': r'^[ACDEFGHIKLMNPQRSTVWYXUOBZ]+$',
        'dna_sequence': r'^[ATCGN]+$',
        'rna_sequence': r'^[AUCGN]+$'
    }
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        # Type check
        if not isinstance(value, str):
            result.add_error(f"Expected string, got {type(value).__name__}")
            return result
        
        # Length checks
        if rule.min_length is not None and len(value) < rule.min_length:
            result.add_error(f"String too short: {len(value)} < {rule.min_length}")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            if self.validation_level == ValidationLevel.STRICT:
                result.add_error(f"String too long: {len(value)} > {rule.max_length}")
            else:
                result.add_warning(f"String very long: {len(value)} characters")
        
        # Pattern validation
        pattern = rule.pattern or self.PATTERNS.get(rule.data_type.value)
        if pattern:
            if not re.match(pattern, value, re.IGNORECASE):
                result.add_error(f"String does not match pattern for {rule.data_type.value}")
        
        # Allowed values check
        if rule.allowed_values and value not in rule.allowed_values:
            result.add_error(f"Value '{value}' not in allowed values: {rule.allowed_values}")
        
        # Custom validator
        if rule.custom_validator:
            try:
                if not rule.custom_validator(value):
                    result.add_error(f"Custom validation failed for {rule.name}")
            except Exception as e:
                result.add_error(f"Custom validator error: {e}")
        
        # Sanitize if valid
        if result.is_valid:
            result.sanitized_data = self.sanitize(value, rule)
        
        return result
    
    def sanitize(self, value: str, rule: ValidationRule) -> str:
        """Sanitize string value"""
        if rule.sanitizer:
            try:
                return rule.sanitizer(value)
            except Exception as e:
                logger.warning(f"Sanitizer failed for {rule.name}: {e}")
        
        # Default sanitization
        sanitized = value.strip()
        
        # Truncate if too long (in non-strict mode)
        if (rule.max_length and 
            len(sanitized) > rule.max_length and 
            self.validation_level != ValidationLevel.STRICT):
            sanitized = sanitized[:rule.max_length]
        
        # Specific sanitization by data type
        if rule.data_type == DataType.PROTEIN_SEQUENCE:
            sanitized = sanitized.upper().replace(' ', '').replace('\n', '')
            # Replace ambiguous amino acids
            replacements = {'B': 'N', 'Z': 'Q', 'J': 'L', 'U': 'C', 'O': 'K'}
            for old, new in replacements.items():
                sanitized = sanitized.replace(old, new)
        
        elif rule.data_type == DataType.DNA_SEQUENCE:
            sanitized = sanitized.upper().replace(' ', '').replace('\n', '')
        
        elif rule.data_type == DataType.RNA_SEQUENCE:
            sanitized = sanitized.upper().replace(' ', '').replace('\n', '')
            sanitized = sanitized.replace('T', 'U')  # Convert DNA to RNA
        
        elif rule.data_type == DataType.EMAIL:
            sanitized = sanitized.lower()
        
        return sanitized

class NumericValidator(BaseValidator):
    """Validator for numeric types"""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        # Type conversion and check
        try:
            if rule.data_type == DataType.INTEGER:
                if not isinstance(value, int):
                    if isinstance(value, str) and value.isdigit():
                        value = int(value)
                    elif isinstance(value, float) and value.is_integer():
                        value = int(value)
                    else:
                        result.add_error(f"Expected integer, got {type(value).__name__}")
                        return result
                
            elif rule.data_type == DataType.FLOAT:
                if not isinstance(value, (int, float)):
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            result.add_error(f"Cannot convert '{value}' to float")
                            return result
                    else:
                        result.add_error(f"Expected number, got {type(value).__name__}")
                        return result
                
        except (ValueError, TypeError) as e:
            result.add_error(f"Type conversion error: {e}")
            return result
        
        # Range checks
        if rule.min_value is not None and value < rule.min_value:
            result.add_error(f"Value too small: {value} < {rule.min_value}")
        
        if rule.max_value is not None and value > rule.max_value:
            if self.validation_level == ValidationLevel.STRICT:
                result.add_error(f"Value too large: {value} > {rule.max_value}")
            else:
                result.add_warning(f"Value very large: {value}")
        
        # Allowed values check
        if rule.allowed_values and value not in rule.allowed_values:
            result.add_error(f"Value {value} not in allowed values: {rule.allowed_values}")
        
        # Custom validator
        if rule.custom_validator:
            try:
                if not rule.custom_validator(value):
                    result.add_error(f"Custom validation failed for {rule.name}")
            except Exception as e:
                result.add_error(f"Custom validator error: {e}")
        
        # Sanitize if valid
        if result.is_valid:
            result.sanitized_data = self.sanitize(value, rule)
        
        return result
    
    def sanitize(self, value: Union[int, float], rule: ValidationRule) -> Union[int, float]:
        """Sanitize numeric value"""
        if rule.sanitizer:
            try:
                return rule.sanitizer(value)
            except Exception as e:
                logger.warning(f"Sanitizer failed for {rule.name}: {e}")
        
        # Clamp to valid range (in non-strict mode)
        if self.validation_level != ValidationLevel.STRICT:
            if rule.min_value is not None:
                value = max(value, rule.min_value)
            if rule.max_value is not None:
                value = min(value, rule.max_value)
        
        return value

class ListValidator(BaseValidator):
    """Validator for list types"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE,
                 item_validator: Optional[BaseValidator] = None,
                 item_rule: Optional[ValidationRule] = None):
        super().__init__(validation_level)
        self.item_validator = item_validator
        self.item_rule = item_rule
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        # Type check
        if not isinstance(value, (list, tuple)):
            result.add_error(f"Expected list/tuple, got {type(value).__name__}")
            return result
        
        # Length checks
        if rule.min_length is not None and len(value) < rule.min_length:
            result.add_error(f"List too short: {len(value)} < {rule.min_length}")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            if self.validation_level == ValidationLevel.STRICT:
                result.add_error(f"List too long: {len(value)} > {rule.max_length}")
            else:
                result.add_warning(f"List very long: {len(value)} items")
        
        # Validate individual items if validator is provided
        sanitized_items = []
        if self.item_validator and self.item_rule:
            for i, item in enumerate(value):
                item_result = self.item_validator.validate(item, self.item_rule)
                if not item_result.is_valid:
                    for error in item_result.errors:
                        result.add_error(f"Item {i}: {error}")
                else:
                    sanitized_items.append(item_result.sanitized_data)
                
                for warning in item_result.warnings:
                    result.add_warning(f"Item {i}: {warning}")
        else:
            sanitized_items = list(value)
        
        # Custom validator
        if rule.custom_validator:
            try:
                if not rule.custom_validator(value):
                    result.add_error(f"Custom validation failed for {rule.name}")
            except Exception as e:
                result.add_error(f"Custom validator error: {e}")
        
        # Sanitize if valid
        if result.is_valid:
            result.sanitized_data = self.sanitize(sanitized_items, rule)
        
        return result
    
    def sanitize(self, value: List[Any], rule: ValidationRule) -> List[Any]:
        """Sanitize list value"""
        if rule.sanitizer:
            try:
                return rule.sanitizer(value)
            except Exception as e:
                logger.warning(f"Sanitizer failed for {rule.name}: {e}")
        
        # Truncate if too long (in non-strict mode)
        if (rule.max_length and 
            len(value) > rule.max_length and 
            self.validation_level != ValidationLevel.STRICT):
            value = value[:rule.max_length]
        
        return value

class ProteinSequenceValidator(StringValidator):
    """Specialized validator for protein sequences"""
    
    STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')
    EXTENDED_AA = STANDARD_AA | {'U', 'O', 'B', 'Z', 'J', 'X'}
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        result = super().validate(value, rule)
        
        if not result.is_valid or not isinstance(value, str):
            return result
        
        # Protein-specific validation
        sequence = value.upper().strip()
        
        # Check for valid amino acids
        allowed_chars = self.EXTENDED_AA if self.validation_level == ValidationLevel.LENIENT else self.STANDARD_AA
        invalid_chars = set(sequence) - allowed_chars
        
        if invalid_chars:
            if self.validation_level == ValidationLevel.STRICT:
                result.add_error(f"Invalid amino acid codes: {sorted(invalid_chars)}")
            else:
                result.add_warning(f"Non-standard amino acids found: {sorted(invalid_chars)}")
        
        # Check for very repetitive sequences
        if len(sequence) > 10:
            complexity = self._calculate_complexity(sequence)
            if complexity < 0.3:
                result.add_warning("Low complexity sequence detected")
        
        # Check for extremely long runs
        for aa in self.STANDARD_AA:
            max_run = self._find_max_run(sequence, aa)
            if max_run > 20:
                result.add_warning(f"Long {aa} run detected: {max_run} residues")
        
        # Update metadata
        result.metadata.update({
            'length': len(sequence),
            'unique_residues': len(set(sequence)),
            'complexity': self._calculate_complexity(sequence),
            'amino_acid_composition': self._analyze_composition(sequence)
        })
        
        return result
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity (Shannon entropy)"""
        from collections import Counter
        import math
        
        if not sequence:
            return 0.0
        
        counts = Counter(sequence)
        length = len(sequence)
        
        entropy = 0.0
        for count in counts.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(min(20, len(set(sequence))))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _find_max_run(self, sequence: str, aa: str) -> int:
        """Find maximum consecutive run of amino acid"""
        max_run = 0
        current_run = 0
        
        for char in sequence:
            if char == aa:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def _analyze_composition(self, sequence: str) -> Dict[str, float]:
        """Analyze amino acid composition"""
        composition = {}
        length = len(sequence)
        
        for aa in self.STANDARD_AA:
            count = sequence.count(aa)
            composition[aa] = count / length if length > 0 else 0.0
        
        return composition

class CoordinatesValidator(BaseValidator):
    """Validator for 3D coordinates"""
    
    def validate(self, value: Any, rule: ValidationRule) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        try:
            import numpy as np
            
            # Convert to numpy array
            coords = np.asarray(value)
            
            # Check dimensions
            if coords.ndim != 3 or coords.shape[-1] != 3:
                result.add_error(f"Coordinates must be shape (N, atoms, 3), got {coords.shape}")
                return result
            
            # Check for valid values
            if np.any(np.isnan(coords)):
                result.add_error("Coordinates contain NaN values")
            
            if np.any(np.isinf(coords)):
                result.add_error("Coordinates contain infinite values")
            
            # Check for reasonable coordinate values (Angstroms)
            if np.any(np.abs(coords) > 1000):
                result.add_warning("Coordinates contain very large values (>1000 Å)")
            
            # Custom validator
            if rule.custom_validator:
                try:
                    if not rule.custom_validator(coords):
                        result.add_error(f"Custom validation failed for {rule.name}")
                except Exception as e:
                    result.add_error(f"Custom validator error: {e}")
            
            if result.is_valid:
                result.sanitized_data = self.sanitize(coords, rule)
            
        except ImportError:
            result.add_error("NumPy required for coordinate validation")
        except Exception as e:
            result.add_error(f"Coordinate validation error: {e}")
        
        return result
    
    def sanitize(self, value: Any, rule: ValidationRule) -> Any:
        """Sanitize coordinates"""
        if rule.sanitizer:
            try:
                return rule.sanitizer(value)
            except Exception as e:
                logger.warning(f"Sanitizer failed for {rule.name}: {e}")
        
        try:
            import numpy as np
            
            coords = np.asarray(value)
            
            # Replace NaN/inf with zeros
            coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
            
            return coords
            
        except ImportError:
            return value
        except Exception as e:
            logger.warning(f"Coordinate sanitization failed: {e}")
            return value

class ValidationSchema:
    """Schema-based validation system"""
    
    def __init__(self, name: str, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.name = name
        self.validation_level = validation_level
        self.rules: Dict[str, ValidationRule] = {}
        self.validators: Dict[DataType, BaseValidator] = self._create_validators()
        self.dependencies: Dict[str, List[str]] = {}  # field -> [dependent_fields]
        self.conditional_rules: Dict[str, Dict[str, ValidationRule]] = {}  # condition -> {field: rule}
    
    def _create_validators(self) -> Dict[DataType, BaseValidator]:
        """Create validator instances for each data type"""
        return {
            DataType.STRING: StringValidator(self.validation_level),
            DataType.INTEGER: NumericValidator(self.validation_level),
            DataType.FLOAT: NumericValidator(self.validation_level),
            DataType.BOOLEAN: BaseValidator(self.validation_level),
            DataType.LIST: ListValidator(self.validation_level),
            DataType.DICT: BaseValidator(self.validation_level),
            DataType.EMAIL: StringValidator(self.validation_level),
            DataType.URL: StringValidator(self.validation_level),
            DataType.IPV4: StringValidator(self.validation_level),
            DataType.IPV6: StringValidator(self.validation_level),
            DataType.UUID: StringValidator(self.validation_level),
            DataType.BASE64: StringValidator(self.validation_level),
            DataType.PROTEIN_SEQUENCE: ProteinSequenceValidator(self.validation_level),
            DataType.DNA_SEQUENCE: StringValidator(self.validation_level),
            DataType.RNA_SEQUENCE: StringValidator(self.validation_level),
            DataType.COORDINATES_3D: CoordinatesValidator(self.validation_level),
            DataType.FILE_PATH: StringValidator(self.validation_level),
        }
    
    def add_rule(self, field_name: str, rule: ValidationRule) -> 'ValidationSchema':
        """Add validation rule for field"""
        self.rules[field_name] = rule
        return self
    
    def add_dependency(self, field: str, dependent_fields: List[str]) -> 'ValidationSchema':
        """Add field dependency"""
        self.dependencies[field] = dependent_fields
        return self
    
    def add_conditional_rule(self, condition: str, field: str, rule: ValidationRule) -> 'ValidationSchema':
        """Add conditional validation rule"""
        if condition not in self.conditional_rules:
            self.conditional_rules[condition] = {}
        self.conditional_rules[condition][field] = rule
        return self
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema"""
        result = ValidationResult(is_valid=True)
        sanitized_data = {}
        
        # Check required fields
        for field_name, rule in self.rules.items():
            if rule.required and field_name not in data:
                result.add_error(f"Required field '{field_name}' is missing")
        
        # Validate each field
        for field_name, value in data.items():
            if field_name in self.rules:
                rule = self.rules[field_name]
                validator = self.validators.get(rule.data_type)
                
                if validator:
                    field_result = validator.validate(value, rule)
                    
                    if not field_result.is_valid:
                        for error in field_result.errors:
                            result.add_error(f"{field_name}: {error}")
                    else:
                        sanitized_data[field_name] = field_result.sanitized_data
                    
                    for warning in field_result.warnings:
                        result.add_warning(f"{field_name}: {warning}")
                    
                    # Update metadata
                    if field_result.metadata:
                        result.metadata[field_name] = field_result.metadata
                else:
                    result.add_warning(f"No validator for data type {rule.data_type}")
                    sanitized_data[field_name] = value
            else:
                # Unknown field
                if self.validation_level == ValidationLevel.STRICT:
                    result.add_error(f"Unknown field '{field_name}'")
                else:
                    result.add_warning(f"Unknown field '{field_name}' ignored")
        
        # Check dependencies
        for field, dependents in self.dependencies.items():
            if field in sanitized_data:
                for dependent in dependents:
                    if dependent not in data:
                        result.add_error(f"Field '{dependent}' is required when '{field}' is present")
        
        # Check conditional rules
        for condition, conditional_rules in self.conditional_rules.items():
            if self._evaluate_condition(condition, sanitized_data):
                for field_name, rule in conditional_rules.items():
                    if field_name in data:
                        validator = self.validators.get(rule.data_type)
                        if validator:
                            field_result = validator.validate(data[field_name], rule)
                            if not field_result.is_valid:
                                for error in field_result.errors:
                                    result.add_error(f"{field_name} (conditional): {error}")
        
        if result.is_valid:
            result.sanitized_data = sanitized_data
        
        return result
    
    def _evaluate_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Evaluate conditional rule (simplified implementation)"""
        # This is a simplified condition evaluator
        # In production, you might want to use a proper expression parser
        try:
            # Replace field names with actual values
            for field_name, value in data.items():
                condition = condition.replace(f"{{{field_name}}}", str(value))
            
            # Simple evaluation (be careful with eval in production!)
            return eval(condition)
        except Exception:
            logger.warning(f"Failed to evaluate condition: {condition}")
            return False
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format"""
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError("jsonschema library required for JSON Schema export")
        
        properties = {}
        required = []
        
        for field_name, rule in self.rules.items():
            if rule.required:
                required.append(field_name)
            
            prop = self._rule_to_json_schema_property(rule)
            properties[field_name] = prop
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": self.validation_level != ValidationLevel.STRICT
        }
        
        return schema
    
    def _rule_to_json_schema_property(self, rule: ValidationRule) -> Dict[str, Any]:
        """Convert validation rule to JSON Schema property"""
        prop = {
            "description": rule.description or f"Field of type {rule.data_type.value}"
        }
        
        if rule.example is not None:
            prop["examples"] = [rule.example]
        
        # Type mapping
        type_mapping = {
            DataType.STRING: "string",
            DataType.INTEGER: "integer",
            DataType.FLOAT: "number",
            DataType.BOOLEAN: "boolean",
            DataType.LIST: "array",
            DataType.DICT: "object",
        }
        
        prop["type"] = type_mapping.get(rule.data_type, "string")
        
        # Add constraints
        if rule.min_value is not None:
            prop["minimum"] = rule.min_value
        if rule.max_value is not None:
            prop["maximum"] = rule.max_value
        if rule.min_length is not None:
            prop["minLength"] = rule.min_length
        if rule.max_length is not None:
            prop["maxLength"] = rule.max_length
        if rule.pattern:
            prop["pattern"] = rule.pattern
        if rule.allowed_values:
            prop["enum"] = rule.allowed_values
        
        return prop

# Pre-defined schemas for common use cases

def create_protein_sequence_schema(validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationSchema:
    """Create schema for protein sequence validation"""
    schema = ValidationSchema("ProteinSequence", validation_level)
    
    schema.add_rule("sequence", ValidationRule(
        name="sequence",
        data_type=DataType.PROTEIN_SEQUENCE,
        required=True,
        min_length=1,
        max_length=10000,
        description="Protein amino acid sequence",
        example="MKILVLFGD"
    ))
    
    schema.add_rule("name", ValidationRule(
        name="name",
        data_type=DataType.STRING,
        required=False,
        max_length=255,
        description="Sequence name or identifier",
        example="Protein_1"
    ))
    
    schema.add_rule("organism", ValidationRule(
        name="organism",
        data_type=DataType.STRING,
        required=False,
        max_length=255,
        description="Source organism",
        example="Homo sapiens"
    ))
    
    return schema

def create_model_config_schema(validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationSchema:
    """Create schema for model configuration validation"""
    schema = ValidationSchema("ModelConfig", validation_level)
    
    schema.add_rule("d_model", ValidationRule(
        name="d_model",
        data_type=DataType.INTEGER,
        required=True,
        min_value=64,
        max_value=8192,
        description="Model dimension",
        example=512
    ))
    
    schema.add_rule("n_layers", ValidationRule(
        name="n_layers",
        data_type=DataType.INTEGER,
        required=True,
        min_value=1,
        max_value=100,
        description="Number of transformer layers",
        example=12
    ))
    
    schema.add_rule("n_heads", ValidationRule(
        name="n_heads",
        data_type=DataType.INTEGER,
        required=True,
        min_value=1,
        max_value=64,
        description="Number of attention heads",
        example=8
    ))
    
    schema.add_rule("max_length", ValidationRule(
        name="max_length",
        data_type=DataType.INTEGER,
        required=True,
        min_value=1,
        max_value=8192,
        description="Maximum sequence length",
        example=1024
    ))
    
    schema.add_rule("dropout", ValidationRule(
        name="dropout",
        data_type=DataType.FLOAT,
        required=False,
        min_value=0.0,
        max_value=1.0,
        description="Dropout probability",
        example=0.1
    ))
    
    # Add dependency: d_model must be divisible by n_heads
    schema.add_conditional_rule(
        "{d_model} % {n_heads} != 0",
        "d_model",
        ValidationRule(
            name="d_model_divisible",
            data_type=DataType.INTEGER,
            custom_validator=lambda x: False,  # This will trigger error
            description="d_model must be divisible by n_heads"
        )
    )
    
    return schema

def create_training_config_schema(validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationSchema:
    """Create schema for training configuration validation"""
    schema = ValidationSchema("TrainingConfig", validation_level)
    
    schema.add_rule("learning_rate", ValidationRule(
        name="learning_rate",
        data_type=DataType.FLOAT,
        required=True,
        min_value=1e-8,
        max_value=1.0,
        description="Learning rate",
        example=1e-4
    ))
    
    schema.add_rule("batch_size", ValidationRule(
        name="batch_size",
        data_type=DataType.INTEGER,
        required=True,
        min_value=1,
        max_value=1024,
        description="Batch size",
        example=32
    ))
    
    schema.add_rule("epochs", ValidationRule(
        name="epochs",
        data_type=DataType.INTEGER,
        required=True,
        min_value=1,
        max_value=1000,
        description="Number of training epochs",
        example=100
    ))
    
    schema.add_rule("optimizer", ValidationRule(
        name="optimizer",
        data_type=DataType.STRING,
        required=False,
        allowed_values=["adam", "adamw", "sgd", "rmsprop"],
        description="Optimizer type",
        example="adamw"
    ))
    
    schema.add_rule("weight_decay", ValidationRule(
        name="weight_decay",
        data_type=DataType.FLOAT,
        required=False,
        min_value=0.0,
        max_value=1.0,
        description="Weight decay",
        example=0.01
    ))
    
    schema.add_rule("warmup_steps", ValidationRule(
        name="warmup_steps",
        data_type=DataType.INTEGER,
        required=False,
        min_value=0,
        max_value=10000,
        description="Number of warmup steps",
        example=1000
    ))
    
    return schema

# Pydantic integration (if available)
if PYDANTIC_AVAILABLE:
    class PydanticSchemaConverter:
        """Convert ValidationSchema to Pydantic models"""
        
        @staticmethod
        def schema_to_pydantic(schema: ValidationSchema) -> Type[BaseModel]:
            """Convert ValidationSchema to Pydantic BaseModel"""
            fields = {}
            validators = {}
            
            for field_name, rule in schema.rules.items():
                field_type = PydanticSchemaConverter._rule_to_pydantic_type(rule)
                field_info = PydanticSchemaConverter._rule_to_pydantic_field(rule)
                fields[field_name] = (field_type, field_info)
                
                if rule.custom_validator:
                    validators[f'validate_{field_name}'] = validator(field_name, allow_reuse=True)(rule.custom_validator)
            
            # Create dynamic Pydantic model
            model_class = type(
                schema.name,
                (BaseModel,),
                {
                    '__annotations__': {name: field_type for name, (field_type, _) in fields.items()},
                    **{name: field_info for name, (_, field_info) in fields.items()},
                    **validators
                }
            )
            
            return model_class
        
        @staticmethod
        def _rule_to_pydantic_type(rule: ValidationRule) -> Type:
            """Convert validation rule to Pydantic field type"""
            type_mapping = {
                DataType.STRING: str,
                DataType.INTEGER: int,
                DataType.FLOAT: float,
                DataType.BOOLEAN: bool,
                DataType.LIST: List[Any],
                DataType.DICT: Dict[str, Any],
            }
            
            base_type = type_mapping.get(rule.data_type, str)
            
            if not rule.required:
                return Optional[base_type]
            
            return base_type
        
        @staticmethod
        def _rule_to_pydantic_field(rule: ValidationRule) -> Field:
            """Convert validation rule to Pydantic Field"""
            kwargs = {}
            
            if rule.description:
                kwargs['description'] = rule.description
            if rule.example is not None:
                kwargs['example'] = rule.example
            if rule.min_value is not None:
                kwargs['ge'] = rule.min_value
            if rule.max_value is not None:
                kwargs['le'] = rule.max_value
            if rule.min_length is not None:
                kwargs['min_length'] = rule.min_length
            if rule.max_length is not None:
                kwargs['max_length'] = rule.max_length
            if rule.pattern:
                kwargs['regex'] = rule.pattern
            
            if not rule.required:
                kwargs['default'] = None
            
            return Field(**kwargs)

# Global registry for validation schemas
_schema_registry: Dict[str, ValidationSchema] = {}

def register_schema(name: str, schema: ValidationSchema):
    """Register a validation schema globally"""
    _schema_registry[name] = schema
    logger.info(f"Registered validation schema: {name}")

def get_schema(name: str) -> Optional[ValidationSchema]:
    """Get registered validation schema"""
    return _schema_registry.get(name)

def list_schemas() -> List[str]:
    """List all registered schema names"""
    return list(_schema_registry.keys())

# Register default schemas
register_schema("protein_sequence", create_protein_sequence_schema())
register_schema("model_config", create_model_config_schema())
register_schema("training_config", create_training_config_schema())

# Convenience validation functions
def validate_protein_sequence(data: Dict[str, Any], 
                            validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """Validate protein sequence data"""
    schema = create_protein_sequence_schema(validation_level)
    return schema.validate(data)

def validate_model_config(data: Dict[str, Any],
                        validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """Validate model configuration data"""
    schema = create_model_config_schema(validation_level)
    return schema.validate(data)

def validate_training_config(data: Dict[str, Any],
                           validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """Validate training configuration data"""
    schema = create_training_config_schema(validation_level)
    return schema.validate(data)
