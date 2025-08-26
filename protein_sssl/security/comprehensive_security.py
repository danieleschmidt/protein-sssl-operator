#!/usr/bin/env python3
"""
🔒 GENERATION 2: COMPREHENSIVE SECURITY FRAMEWORK
=================================================
Multi-layered security system for protein structure prediction pipeline
"""

import hashlib
import secrets
import hmac
import logging
import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import functools

class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    permissions: List[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    audit_enabled: bool = True

class SecureLogger:
    """Security-aware logging system."""
    
    def __init__(self, name: str, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.logger = logging.getLogger(f"secure.{name}")
        self.security_level = security_level
        self.sensitive_patterns = [
            r'password', r'token', r'key', r'secret', r'auth',
            r'credential', r'private', r'confidential'
        ]
        
    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        import re
        sanitized = message
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(
                rf'({pattern}["\s]*[=:]["\s]*)[^"\s\n]+',
                r'\1[REDACTED]',
                sanitized,
                flags=re.IGNORECASE
            )
        return sanitized
    
    def audit(self, message: str, context: Optional[SecurityContext] = None):
        """Log audit message."""
        audit_msg = f"AUDIT: {self._sanitize_message(message)}"
        if context:
            audit_msg += f" | User: {context.user_id} | Session: {context.session_id}"
        self.logger.info(audit_msg)
    
    def security_event(self, event_type: str, message: str, severity: str = "INFO"):
        """Log security event."""
        security_msg = f"SECURITY[{severity}]: {event_type} - {self._sanitize_message(message)}"
        getattr(self.logger, severity.lower())(security_msg)

class InputSanitizer:
    """Comprehensive input sanitization system."""
    
    @staticmethod
    def sanitize_sequence(sequence: str) -> str:
        """Sanitize protein sequence input."""
        if not isinstance(sequence, str):
            raise ValueError(f"Sequence must be string, got {type(sequence)}")
        
        # Remove all non-alphabetic characters
        import re
        sanitized = re.sub(r'[^A-Za-z]', '', sequence)
        
        # Convert to uppercase
        sanitized = sanitized.upper()
        
        # Limit length to prevent DoS
        if len(sanitized) > 50000:  # 50K amino acids max
            raise ValueError("Sequence too long - potential DoS attack")
        
        return sanitized
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe filesystem operations."""
        import re
        
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Remove leading/trailing dots and whitespace
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:250] + ext
        
        # Ensure not empty or dangerous names
        dangerous_names = ['con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
                          'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2',
                          'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9']
        
        if sanitized.lower() in dangerous_names or not sanitized:
            sanitized = 'safe_filename'
        
        return sanitized
    
    @staticmethod
    def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration dictionary."""
        sanitized = {}
        
        for key, value in config.items():
            # Sanitize key
            if not isinstance(key, str):
                continue
            
            # Remove potentially dangerous keys
            if any(danger in key.lower() for danger in ['exec', 'eval', 'import', 'subprocess']):
                continue
            
            # Sanitize value based on type
            if isinstance(value, str):
                # Limit string length
                if len(value) > 10000:
                    value = value[:10000]
                
                # Remove potential code injection
                import re
                if re.search(r'(exec|eval|import|subprocess|os\.|sys\.)', value, re.IGNORECASE):
                    continue
                    
            elif isinstance(value, dict):
                value = InputSanitizer.sanitize_config(value)
            elif isinstance(value, list):
                value = [InputSanitizer.sanitize_config(item) if isinstance(item, dict) else item 
                        for item in value[:1000]]  # Limit list size
            
            sanitized[key] = value
        
        return sanitized

class AccessController:
    """Role-based access control system."""
    
    def __init__(self):
        self.permissions = {
            'guest': ['read:basic'],
            'user': ['read:basic', 'predict:basic', 'save:user_data'],
            'researcher': ['read:all', 'predict:all', 'train:basic', 'save:all'],
            'admin': ['*']  # All permissions
        }
        self.logger = SecureLogger('access_control')
    
    def check_permission(self, user_role: str, required_permission: str) -> bool:
        """Check if user has required permission."""
        if user_role not in self.permissions:
            self.logger.security_event(
                "ACCESS_DENIED", 
                f"Unknown role: {user_role}", 
                "WARNING"
            )
            return False
        
        user_perms = self.permissions[user_role]
        
        # Check for wildcard permission
        if '*' in user_perms:
            return True
        
        # Check exact match
        if required_permission in user_perms:
            return True
        
        # Check pattern match (e.g., 'read:*' matches 'read:basic')
        for perm in user_perms:
            if perm.endswith(':*'):
                prefix = perm[:-1]
                if required_permission.startswith(prefix):
                    return True
        
        self.logger.security_event(
            "ACCESS_DENIED",
            f"Role {user_role} lacks permission {required_permission}",
            "INFO"
        )
        return False
    
    def require_permission(self, permission: str, user_role: str = 'user'):
        """Decorator to require specific permission."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.check_permission(user_role, permission):
                    raise PermissionError(f"Access denied: requires {permission}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

class DataEncryption:
    """Data encryption/decryption utilities."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize with encryption key."""
        if key is None:
            key = self._generate_key()
        self.key = key
        self.logger = SecureLogger('encryption')
    
    def _generate_key(self) -> bytes:
        """Generate a secure encryption key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def encrypt_data(self, data: str) -> Dict[str, str]:
        """Encrypt data using AES-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            # Generate random nonce
            nonce = secrets.token_bytes(12)
            
            # Encrypt data
            aesgcm = AESGCM(self.key)
            ciphertext = aesgcm.encrypt(nonce, data.encode('utf-8'), None)
            
            return {
                'ciphertext': ciphertext.hex(),
                'nonce': nonce.hex(),
                'algorithm': 'AES-GCM'
            }
            
        except ImportError:
            # Fallback to simple base64 encoding (NOT secure!)
            import base64
            self.logger.security_event(
                "CRYPTO_FALLBACK",
                "Using insecure fallback encryption - install cryptography package",
                "WARNING"
            )
            return {
                'ciphertext': base64.b64encode(data.encode('utf-8')).decode('ascii'),
                'nonce': '',
                'algorithm': 'BASE64_FALLBACK'
            }
    
    def decrypt_data(self, encrypted: Dict[str, str]) -> str:
        """Decrypt data."""
        try:
            if encrypted['algorithm'] == 'AES-GCM':
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                
                aesgcm = AESGCM(self.key)
                ciphertext = bytes.fromhex(encrypted['ciphertext'])
                nonce = bytes.fromhex(encrypted['nonce'])
                
                plaintext = aesgcm.decrypt(nonce, ciphertext, None)
                return plaintext.decode('utf-8')
            
            elif encrypted['algorithm'] == 'BASE64_FALLBACK':
                import base64
                return base64.b64decode(encrypted['ciphertext']).decode('utf-8')
            
            else:
                raise ValueError(f"Unknown encryption algorithm: {encrypted['algorithm']}")
                
        except Exception as e:
            self.logger.security_event(
                "DECRYPT_FAILED",
                f"Decryption failed: {e}",
                "ERROR"
            )
            raise

class SecureFileHandler:
    """Secure file operations with validation and sandboxing."""
    
    def __init__(self, base_path: Path, max_file_size: int = 100 * 1024 * 1024):  # 100MB
        """Initialize with base path and limits."""
        self.base_path = Path(base_path).resolve()
        self.max_file_size = max_file_size
        self.logger = SecureLogger('file_handler')
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def validate_path(self, path: Union[str, Path]) -> Path:
        """Validate file path is within allowed directory."""
        if isinstance(path, str):
            path = Path(path)
        
        # Resolve to absolute path
        abs_path = path.resolve()
        
        # Check if within base directory
        try:
            abs_path.relative_to(self.base_path)
        except ValueError:
            raise SecurityError(f"Path outside allowed directory: {abs_path}")
        
        return abs_path
    
    def secure_read(self, path: Union[str, Path], max_size: Optional[int] = None) -> bytes:
        """Securely read file with size limits."""
        validated_path = self.validate_path(path)
        
        if not validated_path.exists():
            raise FileNotFoundError(f"File not found: {validated_path}")
        
        file_size = validated_path.stat().st_size
        size_limit = max_size or self.max_file_size
        
        if file_size > size_limit:
            raise SecurityError(f"File too large: {file_size} > {size_limit}")
        
        self.logger.audit(f"Reading file: {validated_path}")
        
        try:
            with validated_path.open('rb') as f:
                return f.read()
        except Exception as e:
            self.logger.security_event(
                "FILE_READ_ERROR",
                f"Failed to read {validated_path}: {e}",
                "ERROR"
            )
            raise
    
    def secure_write(self, path: Union[str, Path], data: bytes, overwrite: bool = False) -> None:
        """Securely write file with validation."""
        validated_path = self.validate_path(path)
        
        if validated_path.exists() and not overwrite:
            raise SecurityError(f"File exists and overwrite=False: {validated_path}")
        
        if len(data) > self.max_file_size:
            raise SecurityError(f"Data too large: {len(data)} > {self.max_file_size}")
        
        # Create parent directories
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.audit(f"Writing file: {validated_path}")
        
        try:
            with validated_path.open('wb') as f:
                f.write(data)
        except Exception as e:
            self.logger.security_event(
                "FILE_WRITE_ERROR",
                f"Failed to write {validated_path}: {e}",
                "ERROR"
            )
            raise

class SecurityError(Exception):
    """Security-related error."""
    pass

class SecurityAuditor:
    """Security audit and compliance monitoring."""
    
    def __init__(self, audit_log_path: Optional[Path] = None):
        """Initialize auditor with log path."""
        if audit_log_path is None:
            audit_log_path = Path("security_audit.log")
        
        self.audit_log_path = audit_log_path
        self.logger = SecureLogger('auditor')
        self.events = []
        
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        timestamp = time.time()
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'details': details
        }
        
        self.events.append(event)
        
        # Write to audit log
        try:
            with self.audit_log_path.open('a') as f:
                f.write(f"{json.dumps(event)}\n")
        except Exception as e:
            self.logger.security_event(
                "AUDIT_LOG_ERROR",
                f"Failed to write audit log: {e}",
                "ERROR"
            )
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report = {
            'total_events': len(self.events),
            'event_types': {},
            'recent_events': self.events[-10:] if self.events else [],
            'security_recommendations': []
        }
        
        # Count event types
        for event in self.events:
            event_type = event['event_type']
            report['event_types'][event_type] = report['event_types'].get(event_type, 0) + 1
        
        # Generate recommendations
        if report['event_types'].get('ACCESS_DENIED', 0) > 10:
            report['security_recommendations'].append(
                "High number of access denied events - review user permissions"
            )
        
        if report['event_types'].get('CRYPTO_FALLBACK', 0) > 0:
            report['security_recommendations'].append(
                "Install cryptography package for secure encryption"
            )
        
        return report

# Global security instances
secure_logger = SecureLogger('protein_sssl')
input_sanitizer = InputSanitizer()
access_controller = AccessController()
security_auditor = SecurityAuditor()

def secure_operation(permission: str = None, user_role: str = 'user'):
    """Decorator for secure operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check permissions if specified
            if permission:
                if not access_controller.check_permission(user_role, permission):
                    raise PermissionError(f"Access denied: requires {permission}")
            
            # Log operation
            secure_logger.audit(f"Executing {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                security_auditor.log_event('OPERATION_SUCCESS', {
                    'function': func.__name__,
                    'user_role': user_role
                })
                return result
            except Exception as e:
                security_auditor.log_event('OPERATION_FAILED', {
                    'function': func.__name__,
                    'error': str(e),
                    'user_role': user_role
                })
                raise
        
        return wrapper
    return decorator

def test_security_framework():
    """Test the comprehensive security framework."""
    print("🔒 Testing Comprehensive Security Framework")
    print("=" * 50)
    
    # Test input sanitization
    try:
        sequence = input_sanitizer.sanitize_sequence("MKFL-KF_SL##LT")
        print(f"✅ Sanitized sequence: {sequence}")
    except Exception as e:
        print(f"❌ Sanitization failed: {e}")
    
    # Test access control
    try:
        if access_controller.check_permission('user', 'read:basic'):
            print("✅ Access control working")
        else:
            print("❌ Access control denied valid permission")
    except Exception as e:
        print(f"❌ Access control error: {e}")
    
    # Test encryption (basic)
    try:
        encryptor = DataEncryption()
        test_data = "sensitive protein data"
        encrypted = encryptor.encrypt_data(test_data)
        decrypted = encryptor.decrypt_data(encrypted)
        
        if decrypted == test_data:
            print("✅ Encryption/decryption working")
        else:
            print("❌ Encryption/decryption failed")
    except Exception as e:
        print(f"⚠️  Encryption test failed (may need cryptography package): {e}")
    
    # Test secure file operations
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            file_handler = SecureFileHandler(Path(temp_dir))
            test_path = Path("test_file.txt")
            test_content = b"test content"
            
            file_handler.secure_write(test_path, test_content)
            read_content = file_handler.secure_read(test_path)
            
            if read_content == test_content:
                print("✅ Secure file operations working")
            else:
                print("❌ Secure file operations failed")
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
    
    # Generate security report
    try:
        report = security_auditor.generate_security_report()
        print(f"✅ Security report generated: {len(report['event_types'])} event types")
    except Exception as e:
        print(f"❌ Security report failed: {e}")
    
    print("🔒 Security framework tests completed")
    return True

if __name__ == "__main__":
    test_security_framework()