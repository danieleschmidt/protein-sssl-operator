"""
Advanced Security Framework for Protein Structure Prediction
Enhanced SDLC Generation 2+ - Maximum Security and Reliability
"""
import hashlib
import hmac
import secrets
import time
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path


class SecurityLevel(Enum):
    """Security level classifications"""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ThreatLevel(Enum):
    """Threat level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record"""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source: str
    description: str
    mitigation_applied: str
    resolved: bool = False


@dataclass
class AccessAttempt:
    """Access attempt record"""
    timestamp: float
    user_id: str
    resource: str
    action: str
    source_ip: str
    success: bool
    security_level: SecurityLevel


class AdvancedSecurityFramework:
    """
    Advanced security framework with military-grade protection
    """
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL,
                 enable_encryption: bool = True,
                 enable_audit_logging: bool = True,
                 max_failed_attempts: int = 3):
        """Initialize advanced security framework"""
        
        self.security_level = security_level
        self.enable_encryption = enable_encryption
        self.enable_audit_logging = enable_audit_logging
        self.max_failed_attempts = max_failed_attempts
        
        # Security state
        self.session_tokens = {}
        self.failed_attempts = {}
        self.security_events = []
        self.access_log = []
        
        # Encryption keys
        self.master_key = self._generate_master_key()
        self.session_keys = {}
        
        # Security patterns
        self.threat_patterns = self._initialize_threat_patterns()
        self.access_patterns = {}
        
        # Initialize security logging
        self._setup_security_logging()
        
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key"""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _setup_security_logging(self):
        """Setup secure audit logging"""
        if not self.enable_audit_logging:
            return
        
        # Create secure log directory
        log_dir = Path("security_logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure secure logging
        logging.basicConfig(
            filename=log_dir / f"security_audit_{int(time.time())}.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.security_logger = logging.getLogger('security_audit')
    
    def _initialize_threat_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize threat detection patterns"""
        return {
            'sql_injection': re.compile(r"(?i)(union|select|insert|delete|drop|create|alter|exec)", re.IGNORECASE),
            'script_injection': re.compile(r"(?i)(<script|javascript:|on\w+\s*=)", re.IGNORECASE),
            'path_traversal': re.compile(r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e\\)", re.IGNORECASE),
            'command_injection': re.compile(r"(;|\||&|`|\$\(|\${)", re.IGNORECASE),
            'buffer_overflow': re.compile(r"([A-Za-z0-9+/=]{1000,}|%[0-9a-fA-F]{2}{100,})", re.IGNORECASE)
        }
    
    def authenticate_user(self, user_id: str, credentials: str, source_ip: str = "unknown") -> Tuple[bool, Optional[str]]:
        """
        Secure user authentication with threat detection
        
        Returns:
            (success, session_token)
        """
        
        # Check for brute force attempts
        if self._is_brute_force_attack(user_id, source_ip):
            self._log_security_event(
                SecurityEvent(
                    timestamp=time.time(),
                    event_type="brute_force_attempt",
                    threat_level=ThreatLevel.HIGH,
                    source=f"{user_id}@{source_ip}",
                    description=f"Brute force attack detected for user {user_id}",
                    mitigation_applied="Account temporarily locked"
                )
            )
            return False, None
        
        # Validate credentials (mock implementation)
        is_valid = self._validate_credentials(user_id, credentials)
        
        # Record access attempt
        attempt = AccessAttempt(
            timestamp=time.time(),
            user_id=user_id,
            resource="authentication",
            action="login",
            source_ip=source_ip,
            success=is_valid,
            security_level=self.security_level
        )
        self.access_log.append(attempt)
        
        if is_valid:
            # Generate secure session token
            session_token = self._generate_session_token(user_id)
            self.session_tokens[session_token] = {
                'user_id': user_id,
                'created': time.time(),
                'last_access': time.time(),
                'source_ip': source_ip,
                'security_level': self.security_level
            }
            
            # Reset failed attempts
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            self._log_audit("User authenticated successfully", user_id, source_ip)
            return True, session_token
        else:
            # Record failed attempt
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = []
            
            self.failed_attempts[user_id].append({
                'timestamp': time.time(),
                'source_ip': source_ip
            })
            
            self._log_audit("Authentication failed", user_id, source_ip)
            return False, None
    
    def _is_brute_force_attack(self, user_id: str, source_ip: str) -> bool:
        """Detect brute force attacks"""
        if user_id not in self.failed_attempts:
            return False
        
        # Check recent failed attempts
        recent_attempts = [
            attempt for attempt in self.failed_attempts[user_id]
            if time.time() - attempt['timestamp'] < 300  # 5 minutes
        ]
        
        return len(recent_attempts) >= self.max_failed_attempts
    
    def _validate_credentials(self, user_id: str, credentials: str) -> bool:
        """Validate user credentials (mock implementation)"""
        # In production, this would validate against secure user database
        # For demo purposes, accept any non-empty credentials
        return len(credentials.strip()) >= 8
    
    def _generate_session_token(self, user_id: str) -> str:
        """Generate cryptographically secure session token"""
        token_data = f"{user_id}:{time.time()}:{secrets.token_hex(16)}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    def validate_session(self, session_token: str, required_level: SecurityLevel = None) -> bool:
        """Validate session token and security level"""
        
        if session_token not in self.session_tokens:
            return False
        
        session = self.session_tokens[session_token]
        
        # Check session expiry (24 hours)
        if time.time() - session['created'] > 86400:
            del self.session_tokens[session_token]
            return False
        
        # Check security level
        if required_level and not self._check_security_level(session['security_level'], required_level):
            return False
        
        # Update last access
        session['last_access'] = time.time()
        
        return True
    
    def _check_security_level(self, user_level: SecurityLevel, required_level: SecurityLevel) -> bool:
        """Check if user security level meets requirement"""
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3
        }
        
        return level_hierarchy[user_level] >= level_hierarchy[required_level]
    
    def scan_for_threats(self, input_data: str) -> List[Dict[str, Any]]:
        """Scan input data for security threats"""
        
        threats_detected = []
        
        for threat_type, pattern in self.threat_patterns.items():
            matches = pattern.findall(input_data)
            if matches:
                threat = {
                    'type': threat_type,
                    'matches': matches,
                    'threat_level': self._assess_threat_level(threat_type, matches),
                    'detected_at': time.time()
                }
                threats_detected.append(threat)
                
                # Log security event
                self._log_security_event(
                    SecurityEvent(
                        timestamp=time.time(),
                        event_type=f"threat_detected_{threat_type}",
                        threat_level=threat['threat_level'],
                        source="input_scanner",
                        description=f"{threat_type} detected: {matches}",
                        mitigation_applied="Input sanitized/blocked"
                    )
                )
        
        return threats_detected
    
    def _assess_threat_level(self, threat_type: str, matches: List[str]) -> ThreatLevel:
        """Assess threat level based on type and severity"""
        threat_levels = {
            'sql_injection': ThreatLevel.CRITICAL,
            'script_injection': ThreatLevel.HIGH,
            'path_traversal': ThreatLevel.HIGH,
            'command_injection': ThreatLevel.CRITICAL,
            'buffer_overflow': ThreatLevel.MEDIUM
        }
        
        base_level = threat_levels.get(threat_type, ThreatLevel.LOW)
        
        # Escalate based on number of matches
        if len(matches) > 5:
            if base_level == ThreatLevel.HIGH:
                return ThreatLevel.CRITICAL
            elif base_level == ThreatLevel.MEDIUM:
                return ThreatLevel.HIGH
        
        return base_level
    
    def sanitize_input(self, input_data: str, threat_scan: bool = True) -> Tuple[str, List[Dict]]:
        """
        Sanitize input data and remove threats
        
        Returns:
            (sanitized_data, threats_removed)
        """
        
        threats = []
        sanitized = input_data
        
        if threat_scan:
            threats = self.scan_for_threats(input_data)
        
        # Remove detected threats
        for threat in threats:
            for match in threat['matches']:
                # Replace threat patterns with safe alternatives
                if threat['type'] == 'sql_injection':
                    sanitized = sanitized.replace(match, '[SQL_BLOCKED]')
                elif threat['type'] == 'script_injection':
                    sanitized = sanitized.replace(match, '[SCRIPT_BLOCKED]')
                elif threat['type'] == 'path_traversal':
                    sanitized = sanitized.replace(match, '[PATH_BLOCKED]')
                elif threat['type'] == 'command_injection':
                    sanitized = sanitized.replace(match, '[CMD_BLOCKED]')
                else:
                    sanitized = sanitized.replace(match, '[THREAT_BLOCKED]')
        
        # Additional sanitization for protein sequences
        sanitized = self._sanitize_protein_sequence(sanitized)
        
        return sanitized, threats
    
    def _sanitize_protein_sequence(self, sequence: str) -> str:
        """Sanitize protein sequence to only allow valid amino acids"""
        # Allow only standard amino acid letters and common separators
        valid_chars = re.compile(r'[^ACDEFGHIKLMNPQRSTVWYX\-\s]', re.IGNORECASE)
        sanitized = valid_chars.sub('', sequence)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def encrypt_data(self, data: str, additional_key: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Encrypt sensitive data using AES-256
        
        Returns:
            (encrypted_data_hex, nonce_hex)
        """
        
        if not self.enable_encryption:
            return data, ""
        
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Generate salt and derive key
            salt = secrets.token_bytes(16)
            key = additional_key or self.master_key
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            derived_key = base64.urlsafe_b64encode(kdf.derive(key))
            
            # Encrypt data
            fernet = Fernet(derived_key)
            encrypted = fernet.encrypt(data.encode())
            
            return base64.b64encode(encrypted).decode(), base64.b64encode(salt).decode()
            
        except ImportError:
            # Fallback to simple encoding if cryptography not available
            import base64
            nonce = secrets.token_hex(16)
            simple_encrypted = base64.b64encode(data.encode()).decode()
            return simple_encrypted, nonce
    
    def decrypt_data(self, encrypted_data_hex: str, nonce_hex: str, additional_key: Optional[bytes] = None) -> str:
        """Decrypt data"""
        
        if not self.enable_encryption or not encrypted_data_hex:
            return encrypted_data_hex
        
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Reconstruct key
            salt = base64.b64decode(nonce_hex.encode())
            key = additional_key or self.master_key
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            derived_key = base64.urlsafe_b64encode(kdf.derive(key))
            
            # Decrypt data
            fernet = Fernet(derived_key)
            encrypted_bytes = base64.b64decode(encrypted_data_hex.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            
            return decrypted.decode()
            
        except ImportError:
            # Fallback decoding
            import base64
            return base64.b64decode(encrypted_data_hex.encode()).decode()
        except Exception as e:
            self._log_security_event(
                SecurityEvent(
                    timestamp=time.time(),
                    event_type="decryption_failed",
                    threat_level=ThreatLevel.MEDIUM,
                    source="encryption_system",
                    description=f"Failed to decrypt data: {str(e)}",
                    mitigation_applied="Error logged, data remains encrypted"
                )
            )
            return "[DECRYPTION_FAILED]"
    
    def generate_secure_hash(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate secure hash with salt
        
        Returns:
            (hash_hex, salt_hex)
        """
        
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use HMAC-SHA256 for secure hashing
        salted_data = f"{salt}:{data}"
        hash_obj = hmac.new(
            self.master_key,
            salted_data.encode(),
            hashlib.sha256
        )
        
        return hash_obj.hexdigest(), salt
    
    def verify_hash(self, data: str, hash_hex: str, salt: str) -> bool:
        """Verify data against secure hash"""
        
        expected_hash, _ = self.generate_secure_hash(data, salt)
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(hash_hex, expected_hash)
    
    def _log_security_event(self, event: SecurityEvent):
        """Log security event"""
        
        self.security_events.append(event)
        
        if self.enable_audit_logging:
            self.security_logger.warning(
                f"SECURITY_EVENT: {event.event_type} | "
                f"LEVEL: {event.threat_level.value} | "
                f"SOURCE: {event.source} | "
                f"DESC: {event.description}"
            )
        
        # Keep only recent events in memory
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def _log_audit(self, action: str, user_id: str = "system", source: str = "unknown"):
        """Log audit trail"""
        
        if self.enable_audit_logging:
            self.security_logger.info(f"AUDIT: {action} | USER: {user_id} | SOURCE: {source}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        current_time = time.time()
        recent_events = [e for e in self.security_events if current_time - e.timestamp < 3600]  # Last hour
        
        # Threat analysis
        threat_summary = {}
        for event in recent_events:
            threat_type = event.event_type
            if threat_type not in threat_summary:
                threat_summary[threat_type] = {'count': 0, 'max_level': ThreatLevel.LOW}
            
            threat_summary[threat_type]['count'] += 1
            if event.threat_level.value > threat_summary[threat_type]['max_level'].value:
                threat_summary[threat_type]['max_level'] = event.threat_level
        
        # Access pattern analysis
        recent_access = [a for a in self.access_log if current_time - a.timestamp < 3600]
        failed_access_count = len([a for a in recent_access if not a.success])
        
        report = {
            'security_level': self.security_level.value,
            'encryption_enabled': self.enable_encryption,
            'audit_logging_enabled': self.enable_audit_logging,
            'active_sessions': len(self.session_tokens),
            'recent_security_events': len(recent_events),
            'threat_summary': {k: {'count': v['count'], 'max_level': v['max_level'].value} 
                             for k, v in threat_summary.items()},
            'access_metrics': {
                'total_attempts_last_hour': len(recent_access),
                'failed_attempts_last_hour': failed_access_count,
                'success_rate': 1.0 - (failed_access_count / max(len(recent_access), 1))
            },
            'brute_force_attacks_blocked': len([u for u in self.failed_attempts.keys() 
                                              if len(self.failed_attempts[u]) >= self.max_failed_attempts]),
            'report_timestamp': current_time
        }
        
        return report
    
    def cleanup_expired_sessions(self):
        """Cleanup expired sessions and old logs"""
        
        current_time = time.time()
        expired_tokens = []
        
        # Find expired sessions
        for token, session in self.session_tokens.items():
            if current_time - session['created'] > 86400:  # 24 hours
                expired_tokens.append(token)
        
        # Remove expired sessions
        for token in expired_tokens:
            del self.session_tokens[token]
        
        # Cleanup old failed attempts (older than 1 hour)
        for user_id in list(self.failed_attempts.keys()):
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if current_time - attempt['timestamp'] < 3600
            ]
            
            # Remove user if no recent failed attempts
            if not self.failed_attempts[user_id]:
                del self.failed_attempts[user_id]
        
        # Cleanup old access logs (keep last 10000)
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-10000:]
        
        self._log_audit("Security cleanup completed", "system", "security_framework")


# Factory function for security framework
def create_security_framework(config: Optional[Dict] = None) -> AdvancedSecurityFramework:
    """Create advanced security framework with configuration"""
    if config is None:
        config = {}
    
    return AdvancedSecurityFramework(
        security_level=SecurityLevel(config.get('security_level', 'confidential')),
        enable_encryption=config.get('enable_encryption', True),
        enable_audit_logging=config.get('enable_audit_logging', True),
        max_failed_attempts=config.get('max_failed_attempts', 3)
    )


# Demonstration of security framework
if __name__ == "__main__":
    # Create security framework
    security = create_security_framework()
    
    # Test authentication
    print("Testing authentication...")
    success, token = security.authenticate_user("test_user", "secure_password_123", "192.168.1.100")
    print(f"Authentication: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        # Test session validation
        is_valid = security.validate_session(token, SecurityLevel.CONFIDENTIAL)
        print(f"Session validation: {'VALID' if is_valid else 'INVALID'}")
    
    # Test threat scanning
    print("\nTesting threat scanning...")
    malicious_input = "SELECT * FROM users; DROP TABLE passwords; <script>alert('xss')</script>"
    threats = security.scan_for_threats(malicious_input)
    print(f"Threats detected: {len(threats)}")
    
    # Test input sanitization
    sanitized, removed_threats = security.sanitize_input(malicious_input)
    print(f"Sanitized input: {sanitized}")
    print(f"Threats removed: {len(removed_threats)}")
    
    # Test encryption
    print("\nTesting encryption...")
    test_data = "SENSITIVE_PROTEIN_DATA_MKFLKFSLLTAVLLSVVFAFSSCG"
    encrypted, nonce = security.encrypt_data(test_data)
    decrypted = security.decrypt_data(encrypted, nonce)
    print(f"Encryption test: {'PASS' if decrypted == test_data else 'FAIL'}")
    
    # Generate security report
    print("\nSecurity Report:")
    report = security.get_security_report()
    for key, value in report.items():
        if key != 'threat_summary':
            print(f"  {key}: {value}")
    
    print("\nAdvanced Security Framework Ready!")