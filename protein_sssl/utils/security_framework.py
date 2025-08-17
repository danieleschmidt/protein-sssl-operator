"""
Comprehensive Security Framework for protein-sssl-operator
Provides enterprise-grade security with audit logging, access control,
data protection, and compliance features.
"""

import os
import time
import hashlib
import hmac
import secrets
import logging
import json
import threading
import ipaddress
import base64
import uuid
from typing import (
    Dict, List, Optional, Any, Union, Callable, 
    Set, Tuple, NamedTuple, Protocol
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import inspect
import weakref

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security enforcement levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class AccessLevel(Enum):
    """User access levels"""
    GUEST = ("guest", 1)
    USER = ("user", 2)
    MODERATOR = ("moderator", 3)
    ADMIN = ("admin", 4)
    SUPER_ADMIN = ("super_admin", 5)
    
    def __init__(self, name: str, level: int):
        self.level_name = name
        self.level = level
    
    def __ge__(self, other):
        return self.level >= other.level
    
    def __le__(self, other):
        return self.level <= other.level

class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    
    # Authorization
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    
    # Data operations
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # System operations
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # Model operations
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    MODEL_EXPORT = "model_export"
    MODEL_IMPORT = "model_import"

class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = ("info", 1)
    LOW = ("low", 2)
    MEDIUM = ("medium", 3)
    HIGH = ("high", 4)
    CRITICAL = ("critical", 5)
    
    def __init__(self, name: str, level: int):
        self.level_name = name
        self.level = level
    
    def __ge__(self, other):
        return self.level >= other.level

@dataclass
class User:
    """User account information"""
    id: str
    username: str
    email: str
    access_level: AccessLevel
    created_at: float
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    locked_until: Optional[float] = None
    password_hash: Optional[str] = None
    api_key_hash: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        return self.locked_until is not None and time.time() < self.locked_until
    
    def can_access(self, required_level: AccessLevel) -> bool:
        """Check if user has required access level"""
        return not self.is_locked() and self.access_level >= required_level
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return not self.is_locked() and permission in self.permissions

@dataclass
class AuditEvent:
    """Security audit event"""
    id: str
    timestamp: float
    event_type: AuditEventType
    user_id: Optional[str]
    username: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    result: str  # "success", "failure", "denied"
    threat_level: ThreatLevel
    details: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type.value,
            'user_id': self.user_id,
            'username': self.username,
            'source_ip': self.source_ip,
            'user_agent': self.user_agent,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'threat_level': self.threat_level.level_name,
            'details': self.details,
            'session_id': self.session_id,
            'request_id': self.request_id
        }

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    description: str
    enabled: bool = True
    
    # Password policy
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_expiry_days: int = 90
    password_history_count: int = 5
    
    # Account lockout policy
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    lockout_threshold_minutes: int = 15
    
    # Session policy
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 3
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_window_minutes: int = 1
    
    # Data protection
    encrypt_data_at_rest: bool = True
    encrypt_data_in_transit: bool = True
    data_retention_days: int = 365
    
    # Audit settings
    audit_all_operations: bool = True
    audit_retention_days: int = 2555  # 7 years
    
    # Network security
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    require_https: bool = True

class PasswordValidator:
    """Password strength validation"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
    
    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against policy"""
        errors = []
        
        if len(password) < self.policy.min_password_length:
            errors.append(f"Password must be at least {self.policy.min_password_length} characters")
        
        if self.policy.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.policy.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.policy.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.policy.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        # Check for common weak patterns
        weak_patterns = [
            ('123456', 'Sequential numbers'),
            ('password', 'Common word'),
            ('qwerty', 'Keyboard pattern'),
            ('admin', 'Common word'),
            ('user', 'Common word')
        ]
        
        for pattern, description in weak_patterns:
            if pattern.lower() in password.lower():
                errors.append(f"Password contains weak pattern: {description}")
        
        return len(errors) == 0, errors
    
    def generate_strong_password(self, length: int = None) -> str:
        """Generate a strong password"""
        import string
        import random
        
        length = length or max(self.policy.min_password_length, 16)
        
        # Character sets
        chars = string.ascii_lowercase
        if self.policy.require_uppercase:
            chars += string.ascii_uppercase
        if self.policy.require_numbers:
            chars += string.digits
        if self.policy.require_special_chars:
            chars += "!@#$%^&*()_+-="
        
        # Generate password
        password = ''.join(secrets.choice(chars) for _ in range(length))
        
        # Ensure requirements are met
        if self.policy.require_uppercase and not any(c.isupper() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_uppercase)
        
        if self.policy.require_lowercase and not any(c.islower() for c in password):
            password = password[:-1] + secrets.choice(string.ascii_lowercase)
        
        if self.policy.require_numbers and not any(c.isdigit() for c in password):
            password = password[:-1] + secrets.choice(string.digits)
        
        if self.policy.require_special_chars and not any(c in "!@#$%^&*()_+-=" for c in password):
            password = password[:-1] + secrets.choice("!@#$%^&*()_+-=")
        
        return password

class PasswordManager:
    """Password hashing and verification"""
    
    def __init__(self):
        self.bcrypt_available = BCRYPT_AVAILABLE
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        if self.bcrypt_available:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        else:
            # Fallback to PBKDF2 (less preferred)
            salt = secrets.token_bytes(32)
            pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return base64.b64encode(salt + pwdhash).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            if self.bcrypt_available and password_hash.startswith('$2'):
                return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
            else:
                # PBKDF2 verification
                data = base64.b64decode(password_hash.encode('utf-8'))
                salt = data[:32]
                stored_hash = data[32:]
                pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
                return hmac.compare_digest(stored_hash, pwdhash)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

class DataEncryption:
    """Data encryption/decryption utilities"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.cryptography_available = CRYPTOGRAPHY_AVAILABLE
        
        if self.cryptography_available:
            if encryption_key:
                self.fernet = Fernet(encryption_key)
            else:
                # Generate new key
                key = Fernet.generate_key()
                self.fernet = Fernet(key)
                logger.info("Generated new encryption key")
        else:
            logger.warning("Cryptography library not available, using base64 encoding")
            self.fernet = None
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if self.fernet:
            return self.fernet.encrypt(data)
        else:
            # Fallback to base64 (not secure!)
            logger.warning("Using insecure base64 encoding as fallback")
            return base64.b64encode(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        if self.fernet:
            return self.fernet.decrypt(encrypted_data)
        else:
            # Fallback to base64
            return base64.b64decode(encrypted_data)
    
    def encrypt_dict(self, data: Dict[str, Any]) -> bytes:
        """Encrypt dictionary data"""
        json_str = json.dumps(data, separators=(',', ':'))
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt dictionary data"""
        decrypted_bytes = self.decrypt(encrypted_data)
        return json.loads(decrypted_bytes.decode('utf-8'))
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate new encryption key"""
        if CRYPTOGRAPHY_AVAILABLE:
            return Fernet.generate_key()
        else:
            return secrets.token_bytes(32)

class IPAddressFilter:
    """IP address filtering and validation"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.allowed_networks = []
        self.blocked_networks = []
        
        # Parse IP ranges
        for ip_range in policy.allowed_ip_ranges:
            try:
                self.allowed_networks.append(ipaddress.ip_network(ip_range, strict=False))
            except ValueError as e:
                logger.error(f"Invalid allowed IP range '{ip_range}': {e}")
        
        for ip_range in policy.blocked_ip_ranges:
            try:
                self.blocked_networks.append(ipaddress.ip_network(ip_range, strict=False))
            except ValueError as e:
                logger.error(f"Invalid blocked IP range '{ip_range}': {e}")
    
    def is_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed"""
        try:
            ip = ipaddress.ip_address(ip_address)
        except ValueError:
            logger.warning(f"Invalid IP address: {ip_address}")
            return False
        
        # Check blocked list first
        for network in self.blocked_networks:
            if ip in network:
                return False
        
        # If no allowed networks specified, allow all (except blocked)
        if not self.allowed_networks:
            return True
        
        # Check allowed list
        for network in self.allowed_networks:
            if ip in network:
                return True
        
        return False

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.RLock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        window_start = current_time - (self.policy.rate_limit_window_minutes * 60)
        
        with self.lock:
            # Clean old requests
            requests = self.requests[identifier]
            while requests and requests[0] < window_start:
                requests.popleft()
            
            # Check limit
            if len(requests) >= self.policy.rate_limit_requests_per_minute:
                return False
            
            # Add current request
            requests.append(current_time)
            return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests in current window"""
        current_time = time.time()
        window_start = current_time - (self.policy.rate_limit_window_minutes * 60)
        
        with self.lock:
            requests = self.requests[identifier]
            # Clean old requests
            while requests and requests[0] < window_start:
                requests.popleft()
            
            return max(0, self.policy.rate_limit_requests_per_minute - len(requests))

class AuditLogger:
    """Security audit logging system"""
    
    def __init__(self, policy: SecurityPolicy, log_file: Optional[str] = None):
        self.policy = policy
        self.log_file = log_file
        self.events: deque = deque(maxlen=100000)  # In-memory buffer
        self.lock = threading.RLock()
        
        # Setup file logging if specified
        if log_file:
            self.file_logger = logging.getLogger('security_audit')
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.file_logger.addHandler(handler)
            self.file_logger.setLevel(logging.INFO)
        else:
            self.file_logger = None
    
    def log_event(self,
                  event_type: AuditEventType,
                  action: str,
                  result: str,
                  user_id: Optional[str] = None,
                  username: Optional[str] = None,
                  source_ip: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  resource: Optional[str] = None,
                  threat_level: ThreatLevel = ThreatLevel.INFO,
                  details: Optional[Dict[str, Any]] = None,
                  session_id: Optional[str] = None,
                  request_id: Optional[str] = None) -> AuditEvent:
        """Log security event"""
        
        event = AuditEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            username=username,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            threat_level=threat_level,
            details=details or {},
            session_id=session_id,
            request_id=request_id
        )
        
        with self.lock:
            self.events.append(event)
        
        # Log to file if configured
        if self.file_logger:
            self.file_logger.info(json.dumps(event.to_dict()))
        
        # Log to console based on threat level
        if threat_level >= ThreatLevel.MEDIUM:
            log_level = {
                ThreatLevel.MEDIUM: logging.WARNING,
                ThreatLevel.HIGH: logging.ERROR,
                ThreatLevel.CRITICAL: logging.CRITICAL
            }.get(threat_level, logging.INFO)
            
            logger.log(log_level, 
                f"Security Event [{threat_level.level_name.upper()}]: {action} - {result} "
                f"(User: {username or 'N/A'}, IP: {source_ip or 'N/A'})"
            )
        
        return event
    
    def get_events(self, 
                   event_types: Optional[List[AuditEventType]] = None,
                   user_id: Optional[str] = None,
                   threat_level: Optional[ThreatLevel] = None,
                   hours: float = 24.0) -> List[AuditEvent]:
        """Get audit events matching criteria"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # Apply filters
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if threat_level:
            events = [e for e in events if e.threat_level >= threat_level]
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    def get_security_summary(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get security events summary"""
        events = self.get_events(hours=hours)
        
        if not events:
            return {
                'total_events': 0,
                'by_type': {},
                'by_threat_level': {},
                'unique_users': 0,
                'unique_ips': 0
            }
        
        by_type = defaultdict(int)
        by_threat_level = defaultdict(int)
        by_result = defaultdict(int)
        unique_users = set()
        unique_ips = set()
        
        for event in events:
            by_type[event.event_type.value] += 1
            by_threat_level[event.threat_level.level_name] += 1
            by_result[event.result] += 1
            
            if event.user_id:
                unique_users.add(event.user_id)
            if event.source_ip:
                unique_ips.add(event.source_ip)
        
        return {
            'total_events': len(events),
            'by_type': dict(by_type),
            'by_threat_level': dict(by_threat_level),
            'by_result': dict(by_result),
            'unique_users': len(unique_users),
            'unique_ips': len(unique_ips),
            'time_range_hours': hours
        }
    
    def export_events(self, file_path: str, 
                     format: str = "json",
                     hours: float = 24.0) -> str:
        """Export audit events to file"""
        events = self.get_events(hours=hours)
        
        if format.lower() == "json":
            data = {
                'export_timestamp': time.time(),
                'events_count': len(events),
                'time_range_hours': hours,
                'events': [event.to_dict() for event in events]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == "csv":
            import csv
            
            with open(file_path, 'w', newline='') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].to_dict().keys())
                    writer.writeheader()
                    for event in events:
                        writer.writerow(event.to_dict())
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(events)} audit events to {file_path}")
        return file_path

class SessionManager:
    """User session management"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.RLock()
    
    def create_session(self, user_id: str, source_ip: str, user_agent: str) -> str:
        """Create new user session"""
        session_id = str(uuid.uuid4())
        current_time = time.time()
        
        with self.lock:
            # Check max concurrent sessions
            user_session_ids = self.user_sessions[user_id]
            if len(user_session_ids) >= self.policy.max_concurrent_sessions:
                # Remove oldest session
                oldest_session_id = min(user_session_ids, 
                                       key=lambda sid: self.sessions[sid]['created_at'])
                self.remove_session(oldest_session_id)
            
            # Create new session
            session_data = {
                'user_id': user_id,
                'created_at': current_time,
                'last_activity': current_time,
                'source_ip': source_ip,
                'user_agent': user_agent,
                'active': True
            }
            
            self.sessions[session_id] = session_data
            self.user_sessions[user_id].add(session_id)
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and update session"""
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            current_time = time.time()
            
            # Check if session is active
            if not session['active']:
                return None
            
            # Check timeout
            timeout_seconds = self.policy.session_timeout_minutes * 60
            if current_time - session['last_activity'] > timeout_seconds:
                session['active'] = False
                return None
            
            # Update last activity
            session['last_activity'] = current_time
            return session
    
    def remove_session(self, session_id: str) -> bool:
        """Remove session"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                user_id = session['user_id']
                
                del self.sessions[session_id]
                self.user_sessions[user_id].discard(session_id)
                
                logger.info(f"Removed session {session_id}")
                return True
            return False
    
    def remove_user_sessions(self, user_id: str) -> int:
        """Remove all sessions for user"""
        with self.lock:
            session_ids = list(self.user_sessions[user_id])
            count = 0
            
            for session_id in session_ids:
                if self.remove_session(session_id):
                    count += 1
            
            return count
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        current_time = time.time()
        timeout_seconds = self.policy.session_timeout_minutes * 60
        expired_sessions = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                if (not session['active'] or 
                    current_time - session['last_activity'] > timeout_seconds):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.remove_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active sessions"""
        with self.lock:
            sessions = []
            for session_id, session in self.sessions.items():
                if session['active'] and (user_id is None or session['user_id'] == user_id):
                    session_copy = session.copy()
                    session_copy['session_id'] = session_id
                    sessions.append(session_copy)
            
            return sessions

class SecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self, 
                 policy: Optional[SecurityPolicy] = None,
                 audit_log_file: Optional[str] = None,
                 encryption_key: Optional[bytes] = None):
        
        self.policy = policy or SecurityPolicy(name="default", description="Default security policy")
        
        # Initialize components
        self.password_validator = PasswordValidator(self.policy)
        self.password_manager = PasswordManager()
        self.data_encryption = DataEncryption(encryption_key)
        self.ip_filter = IPAddressFilter(self.policy)
        self.rate_limiter = RateLimiter(self.policy)
        self.audit_logger = AuditLogger(self.policy, audit_log_file)
        self.session_manager = SessionManager(self.policy)
        
        # User management
        self.users: Dict[str, User] = {}
        self.user_lock = threading.RLock()
        
        # Security state
        self.security_level = SecurityLevel.MEDIUM
        
        logger.info("Security framework initialized")
    
    def create_user(self, 
                   username: str,
                   email: str,
                   password: str,
                   access_level: AccessLevel = AccessLevel.USER,
                   permissions: Optional[Set[str]] = None) -> User:
        """Create new user account"""
        
        # Validate password
        is_valid, errors = self.password_validator.validate(password)
        if not is_valid:
            raise ValueError(f"Password validation failed: {'; '.join(errors)}")
        
        # Hash password
        password_hash = self.password_manager.hash_password(password)
        
        # Create user
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            access_level=access_level,
            created_at=time.time(),
            password_hash=password_hash,
            permissions=permissions or set()
        )
        
        with self.user_lock:
            self.users[user.id] = user
        
        # Log event
        self.audit_logger.log_event(
            AuditEventType.LOGIN_SUCCESS,
            "user_created",
            "success",
            user_id=user.id,
            username=username,
            threat_level=ThreatLevel.INFO,
            details={'access_level': access_level.level_name}
        )
        
        logger.info(f"Created user: {username} ({access_level.level_name})")
        return user
    
    def authenticate_user(self, 
                         username: str, 
                         password: str,
                         source_ip: str,
                         user_agent: str = "") -> Optional[Tuple[User, str]]:
        """Authenticate user and create session"""
        
        # Check IP address
        if not self.ip_filter.is_allowed(source_ip):
            self.audit_logger.log_event(
                AuditEventType.LOGIN_FAILURE,
                "authentication",
                "failure",
                username=username,
                source_ip=source_ip,
                user_agent=user_agent,
                threat_level=ThreatLevel.HIGH,
                details={'reason': 'IP address not allowed'}
            )
            return None
        
        # Check rate limit
        if not self.rate_limiter.is_allowed(source_ip):
            self.audit_logger.log_event(
                AuditEventType.RATE_LIMIT_EXCEEDED,
                "authentication",
                "failure",
                username=username,
                source_ip=source_ip,
                user_agent=user_agent,
                threat_level=ThreatLevel.MEDIUM,
                details={'reason': 'Rate limit exceeded'}
            )
            return None
        
        # Find user
        user = None
        with self.user_lock:
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
        
        if not user:
            self.audit_logger.log_event(
                AuditEventType.LOGIN_FAILURE,
                "authentication",
                "failure",
                username=username,
                source_ip=source_ip,
                user_agent=user_agent,
                threat_level=ThreatLevel.MEDIUM,
                details={'reason': 'User not found'}
            )
            return None
        
        # Check if user is locked
        if user.is_locked():
            self.audit_logger.log_event(
                AuditEventType.LOGIN_FAILURE,
                "authentication",
                "failure",
                user_id=user.id,
                username=username,
                source_ip=source_ip,
                user_agent=user_agent,
                threat_level=ThreatLevel.MEDIUM,
                details={'reason': 'Account locked'}
            )
            return None
        
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            # Update failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if threshold reached
            if user.failed_login_attempts >= self.policy.max_failed_attempts:
                user.locked_until = time.time() + (self.policy.lockout_duration_minutes * 60)
                
                self.audit_logger.log_event(
                    AuditEventType.LOGIN_FAILURE,
                    "authentication",
                    "failure",
                    user_id=user.id,
                    username=username,
                    source_ip=source_ip,
                    user_agent=user_agent,
                    threat_level=ThreatLevel.HIGH,
                    details={'reason': 'Account locked due to failed attempts'}
                )
            else:
                self.audit_logger.log_event(
                    AuditEventType.LOGIN_FAILURE,
                    "authentication",
                    "failure",
                    user_id=user.id,
                    username=username,
                    source_ip=source_ip,
                    user_agent=user_agent,
                    threat_level=ThreatLevel.MEDIUM,
                    details={'reason': 'Invalid password', 'failed_attempts': user.failed_login_attempts}
                )
            
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = time.time()
        
        # Create session
        session_id = self.session_manager.create_session(user.id, source_ip, user_agent)
        
        # Log successful login
        self.audit_logger.log_event(
            AuditEventType.LOGIN_SUCCESS,
            "authentication",
            "success",
            user_id=user.id,
            username=username,
            source_ip=source_ip,
            user_agent=user_agent,
            session_id=session_id,
            threat_level=ThreatLevel.INFO
        )
        
        logger.info(f"User authenticated: {username} from {source_ip}")
        return user, session_id
    
    def authorize_access(self, 
                        session_id: str,
                        resource: str,
                        required_permission: Optional[str] = None,
                        required_access_level: Optional[AccessLevel] = None) -> bool:
        """Authorize user access to resource"""
        
        # Validate session
        session = self.session_manager.validate_session(session_id)
        if not session:
            self.audit_logger.log_event(
                AuditEventType.ACCESS_DENIED,
                "authorization",
                "failure",
                session_id=session_id,
                resource=resource,
                threat_level=ThreatLevel.MEDIUM,
                details={'reason': 'Invalid session'}
            )
            return False
        
        # Get user
        user_id = session['user_id']
        with self.user_lock:
            user = self.users.get(user_id)
        
        if not user:
            self.audit_logger.log_event(
                AuditEventType.ACCESS_DENIED,
                "authorization",
                "failure",
                user_id=user_id,
                session_id=session_id,
                resource=resource,
                threat_level=ThreatLevel.HIGH,
                details={'reason': 'User not found'}
            )
            return False
        
        # Check access level
        if required_access_level and not user.can_access(required_access_level):
            self.audit_logger.log_event(
                AuditEventType.ACCESS_DENIED,
                "authorization",
                "failure",
                user_id=user.id,
                username=user.username,
                session_id=session_id,
                resource=resource,
                threat_level=ThreatLevel.MEDIUM,
                details={'reason': f'Insufficient access level: {user.access_level.level_name} < {required_access_level.level_name}'}
            )
            return False
        
        # Check permission
        if required_permission and not user.has_permission(required_permission):
            self.audit_logger.log_event(
                AuditEventType.ACCESS_DENIED,
                "authorization",
                "failure",
                user_id=user.id,
                username=user.username,
                session_id=session_id,
                resource=resource,
                threat_level=ThreatLevel.MEDIUM,
                details={'reason': f'Missing permission: {required_permission}'}
            )
            return False
        
        # Access granted
        self.audit_logger.log_event(
            AuditEventType.ACCESS_GRANTED,
            "authorization",
            "success",
            user_id=user.id,
            username=user.username,
            session_id=session_id,
            resource=resource,
            threat_level=ThreatLevel.INFO
        )
        
        return True
    
    def logout_user(self, session_id: str) -> bool:
        """Log out user and remove session"""
        session = self.session_manager.validate_session(session_id)
        if not session:
            return False
        
        user_id = session['user_id']
        with self.user_lock:
            user = self.users.get(user_id)
        
        # Remove session
        self.session_manager.remove_session(session_id)
        
        # Log event
        self.audit_logger.log_event(
            AuditEventType.LOGOUT,
            "logout",
            "success",
            user_id=user_id,
            username=user.username if user else None,
            session_id=session_id,
            threat_level=ThreatLevel.INFO
        )
        
        return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        audit_summary = self.audit_logger.get_security_summary()
        active_sessions = self.session_manager.get_active_sessions()
        
        return {
            'security_level': self.security_level.value,
            'policy_name': self.policy.name,
            'user_count': len(self.users),
            'active_sessions': len(active_sessions),
            'audit_summary': audit_summary,
            'policy_settings': {
                'password_expiry_days': self.policy.password_expiry_days,
                'max_failed_attempts': self.policy.max_failed_attempts,
                'session_timeout_minutes': self.policy.session_timeout_minutes,
                'rate_limit_per_minute': self.policy.rate_limit_requests_per_minute
            }
        }

# Decorators for security

def require_authentication(session_manager: SessionManager, audit_logger: AuditLogger):
    """Decorator to require authentication"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session_id from kwargs or first argument
            session_id = kwargs.get('session_id') or (args[0] if args else None)
            
            if not session_id or not session_manager.validate_session(session_id):
                audit_logger.log_event(
                    AuditEventType.ACCESS_DENIED,
                    func.__name__,
                    "failure",
                    threat_level=ThreatLevel.MEDIUM,
                    details={'reason': 'Authentication required'}
                )
                raise PermissionError("Authentication required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_permission(permission: str, framework: SecurityFramework):
    """Decorator to require specific permission"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session_id = kwargs.get('session_id') or (args[0] if args else None)
            resource = f"{func.__module__}.{func.__name__}"
            
            if not framework.authorize_access(session_id, resource, required_permission=permission):
                raise PermissionError(f"Permission '{permission}' required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def audit_action(action: str, framework: SecurityFramework):
    """Decorator to audit function calls"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            session_id = kwargs.get('session_id')
            resource = f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                framework.audit_logger.log_event(
                    AuditEventType.DATA_ACCESS,
                    action,
                    "success",
                    session_id=session_id,
                    resource=resource,
                    threat_level=ThreatLevel.INFO,
                    details={'duration': time.time() - start_time}
                )
                
                return result
                
            except Exception as e:
                framework.audit_logger.log_event(
                    AuditEventType.DATA_ACCESS,
                    action,
                    "failure",
                    session_id=session_id,
                    resource=resource,
                    threat_level=ThreatLevel.MEDIUM,
                    details={'error': str(e), 'duration': time.time() - start_time}
                )
                raise
        
        return wrapper
    return decorator

# Global security framework instance
_global_security_framework: Optional[SecurityFramework] = None

def get_global_security_framework(**kwargs) -> SecurityFramework:
    """Get or create global security framework instance"""
    global _global_security_framework
    if _global_security_framework is None:
        _global_security_framework = SecurityFramework(**kwargs)
    return _global_security_framework

def configure_security(**kwargs) -> SecurityFramework:
    """Configure global security framework"""
    global _global_security_framework
    _global_security_framework = SecurityFramework(**kwargs)
    return _global_security_framework

# Context manager for security
@contextmanager
def security_context(**kwargs):
    """Context manager for security framework"""
    framework = configure_security(**kwargs)
    try:
        yield framework
    finally:
        # Cleanup if needed
        pass
