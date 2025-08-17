"""
Unified Robustness Framework for protein-sssl-operator
Integrates all enterprise-grade robustness features into a cohesive system.
"""

import time
import logging
import threading
import uuid
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import functools
import json

# Import all robustness components
try:
    from .advanced_error_handling import (
        AdvancedErrorHandler, get_global_error_handler,
        with_circuit_breaker, with_rate_limit, with_error_recovery
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False

try:
    from .validation_schemas import (
        ValidationSchema, ValidationLevel,
        validate_protein_sequence, validate_model_config, validate_training_config
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

try:
    from .enterprise_monitoring import (
        EnterpriseMonitor, get_global_monitor, start_monitoring,
        HealthStatus, AlertSeverity
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from .security_framework import (
        SecurityFramework, get_global_security_framework,
        SecurityLevel, AccessLevel, AuditEventType
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

try:
    from .enterprise_testing import (
        TestRunner, get_global_test_runner,
        PropertyTestGenerator, StressTestGenerator, FuzzTestGenerator
    )
    TESTING_AVAILABLE = True
except ImportError:
    TESTING_AVAILABLE = False

try:
    from .dynamic_config_manager import (
        DynamicConfigManager, initialize_global_config_manager,
        get_config, set_config
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from .compliance_framework import (
        ComplianceManager, ComplianceRegulation,
        create_gdpr_policy, create_ccpa_policy, create_hipaa_policy
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class RobustnessLevel(Enum):
    """Overall robustness enforcement levels"""
    BASIC = "basic"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    MAXIMUM = "maximum"

class FrameworkStatus(Enum):
    """Framework component status"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    NOT_AVAILABLE = "not_available"

@dataclass
class RobustnessConfig:
    """Configuration for the unified robustness framework"""
    level: RobustnessLevel = RobustnessLevel.STANDARD
    
    # Component enablement
    enable_error_handling: bool = True
    enable_input_validation: bool = True
    enable_monitoring: bool = True
    enable_security: bool = True
    enable_testing: bool = True
    enable_config_management: bool = True
    enable_compliance: bool = True
    
    # Error handling config
    circuit_breaker_failure_threshold: int = 5
    rate_limit_requests_per_minute: int = 100
    error_recovery_enabled: bool = True
    
    # Validation config
    validation_level: str = "moderate"
    validate_all_inputs: bool = True
    sanitize_inputs: bool = True
    
    # Monitoring config
    metrics_collection_interval: float = 30.0
    health_check_interval: float = 60.0
    enable_prometheus: bool = True
    enable_opentelemetry: bool = False
    
    # Security config
    security_level: str = "medium"
    audit_all_operations: bool = True
    require_authentication: bool = True
    
    # Testing config
    enable_property_testing: bool = True
    enable_stress_testing: bool = True
    enable_fuzz_testing: bool = True
    
    # Configuration management
    config_file: str = "config.yaml"
    auto_reload_config: bool = True
    validate_config_changes: bool = True
    
    # Compliance config
    applicable_regulations: List[str] = field(default_factory=lambda: ["gdpr"])
    data_retention_days: int = 365
    automated_deletion_enabled: bool = True

class UnifiedRobustnessFramework:
    """Main unified robustness framework orchestrator"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.initialized = False
        self.component_status: Dict[str, FrameworkStatus] = {}
        
        # Component instances
        self.error_handler: Optional[AdvancedErrorHandler] = None
        self.monitor: Optional[EnterpriseMonitor] = None
        self.security_framework: Optional[SecurityFramework] = None
        self.test_runner: Optional[TestRunner] = None
        self.config_manager: Optional[DynamicConfigManager] = None
        self.compliance_manager: Optional[ComplianceManager] = None
        
        # Initialization lock
        self._init_lock = threading.Lock()
        
        logger.info(f"Unified robustness framework created with level: {config.level.value}")
    
    def initialize(self) -> bool:
        """Initialize all framework components"""
        with self._init_lock:
            if self.initialized:
                logger.warning("Framework already initialized")
                return True
            
            logger.info("Initializing unified robustness framework...")
            
            success = True
            
            # Initialize error handling
            if self.config.enable_error_handling and ERROR_HANDLING_AVAILABLE:
                success &= self._init_error_handling()
            else:
                self.component_status['error_handling'] = (
                    FrameworkStatus.NOT_AVAILABLE if not ERROR_HANDLING_AVAILABLE 
                    else FrameworkStatus.DISABLED
                )
            
            # Initialize monitoring
            if self.config.enable_monitoring and MONITORING_AVAILABLE:
                success &= self._init_monitoring()
            else:
                self.component_status['monitoring'] = (
                    FrameworkStatus.NOT_AVAILABLE if not MONITORING_AVAILABLE 
                    else FrameworkStatus.DISABLED
                )
            
            # Initialize security
            if self.config.enable_security and SECURITY_AVAILABLE:
                success &= self._init_security()
            else:
                self.component_status['security'] = (
                    FrameworkStatus.NOT_AVAILABLE if not SECURITY_AVAILABLE 
                    else FrameworkStatus.DISABLED
                )
            
            # Initialize testing
            if self.config.enable_testing and TESTING_AVAILABLE:
                success &= self._init_testing()
            else:
                self.component_status['testing'] = (
                    FrameworkStatus.NOT_AVAILABLE if not TESTING_AVAILABLE 
                    else FrameworkStatus.DISABLED
                )
            
            # Initialize configuration management
            if self.config.enable_config_management and CONFIG_AVAILABLE:
                success &= self._init_config_management()
            else:
                self.component_status['config_management'] = (
                    FrameworkStatus.NOT_AVAILABLE if not CONFIG_AVAILABLE 
                    else FrameworkStatus.DISABLED
                )
            
            # Initialize compliance
            if self.config.enable_compliance and COMPLIANCE_AVAILABLE:
                success &= self._init_compliance()
            else:
                self.component_status['compliance'] = (
                    FrameworkStatus.NOT_AVAILABLE if not COMPLIANCE_AVAILABLE 
                    else FrameworkStatus.DISABLED
                )
            
            # Initialize input validation (always available)
            if self.config.enable_input_validation:
                success &= self._init_validation()
            else:
                self.component_status['validation'] = FrameworkStatus.DISABLED
            
            self.initialized = success
            
            if success:
                logger.info("Unified robustness framework initialized successfully")
                self._log_component_status()
            else:
                logger.error("Failed to initialize unified robustness framework")
            
            return success
    
    def _init_error_handling(self) -> bool:
        """Initialize error handling components"""
        try:
            self.error_handler = get_global_error_handler()
            
            # Configure circuit breakers and rate limiters based on config
            if self.config.level in [RobustnessLevel.ENTERPRISE, RobustnessLevel.MAXIMUM]:
                # Create default circuit breakers for critical operations
                self.error_handler.create_circuit_breaker(
                    "model_inference",
                    failure_threshold=self.config.circuit_breaker_failure_threshold,
                    recovery_timeout=60.0
                )
                
                self.error_handler.create_circuit_breaker(
                    "data_processing",
                    failure_threshold=self.config.circuit_breaker_failure_threshold,
                    recovery_timeout=30.0
                )
                
                # Create rate limiters
                self.error_handler.create_rate_limiter(
                    "api_requests",
                    rate=self.config.rate_limit_requests_per_minute / 60.0,
                    burst_capacity=10
                )
            
            self.component_status['error_handling'] = FrameworkStatus.ENABLED
            logger.info("Error handling initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing error handling: {e}")
            self.component_status['error_handling'] = FrameworkStatus.ERROR
            return False
    
    def _init_monitoring(self) -> bool:
        """Initialize monitoring components"""
        try:
            self.monitor = get_global_monitor(
                metrics_collection_interval=self.config.metrics_collection_interval,
                health_check_interval=self.config.health_check_interval,
                enable_prometheus=self.config.enable_prometheus,
                enable_opentelemetry=self.config.enable_opentelemetry
            )
            
            self.monitor.start()
            
            self.component_status['monitoring'] = FrameworkStatus.ENABLED
            logger.info("Monitoring initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing monitoring: {e}")
            self.component_status['monitoring'] = FrameworkStatus.ERROR
            return False
    
    def _init_security(self) -> bool:
        """Initialize security components"""
        try:
            self.security_framework = get_global_security_framework()
            
            # Configure security level
            if self.config.security_level == "high":
                self.security_framework.security_level = SecurityLevel.HIGH
            elif self.config.security_level == "medium":
                self.security_framework.security_level = SecurityLevel.MEDIUM
            else:
                self.security_framework.security_level = SecurityLevel.LOW
            
            self.component_status['security'] = FrameworkStatus.ENABLED
            logger.info("Security framework initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing security: {e}")
            self.component_status['security'] = FrameworkStatus.ERROR
            return False
    
    def _init_testing(self) -> bool:
        """Initialize testing components"""
        try:
            self.test_runner = get_global_test_runner(
                enable_coverage=self.config.level == RobustnessLevel.MAXIMUM
            )
            
            self.component_status['testing'] = FrameworkStatus.ENABLED
            logger.info("Testing framework initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing testing: {e}")
            self.component_status['testing'] = FrameworkStatus.ERROR
            return False
    
    def _init_config_management(self) -> bool:
        """Initialize configuration management"""
        try:
            self.config_manager = initialize_global_config_manager(
                self.config.config_file,
                auto_reload=self.config.auto_reload_config
            )
            
            self.component_status['config_management'] = FrameworkStatus.ENABLED
            logger.info("Configuration management initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing config management: {e}")
            self.component_status['config_management'] = FrameworkStatus.ERROR
            return False
    
    def _init_compliance(self) -> bool:
        """Initialize compliance framework"""
        try:
            # Create appropriate compliance policy based on regulations
            if "gdpr" in self.config.applicable_regulations:
                policy = create_gdpr_policy()
            elif "ccpa" in self.config.applicable_regulations:
                policy = create_ccpa_policy()
            elif "hipaa" in self.config.applicable_regulations:
                policy = create_hipaa_policy()
            else:
                # Create a basic policy
                from .compliance_framework import CompliancePolicy, ComplianceRegulation
                policy = CompliancePolicy(
                    name="Basic Policy",
                    regulations=[],
                    data_retention_days=self.config.data_retention_days,
                    automated_deletion_enabled=self.config.automated_deletion_enabled
                )
            
            self.compliance_manager = ComplianceManager(policy)
            
            self.component_status['compliance'] = FrameworkStatus.ENABLED
            logger.info("Compliance framework initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing compliance: {e}")
            self.component_status['compliance'] = FrameworkStatus.ERROR
            return False
    
    def _init_validation(self) -> bool:
        """Initialize input validation"""
        try:
            # Validation is always available as it's part of the core framework
            self.component_status['validation'] = FrameworkStatus.ENABLED
            logger.info("Input validation initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing validation: {e}")
            self.component_status['validation'] = FrameworkStatus.ERROR
            return False
    
    def _log_component_status(self):
        """Log the status of all components"""
        logger.info("Framework component status:")
        for component, status in self.component_status.items():
            logger.info(f"  {component}: {status.value}")
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get comprehensive framework status"""
        status = {
            'initialized': self.initialized,
            'robustness_level': self.config.level.value,
            'components': {name: status.value for name, status in self.component_status.items()},
            'timestamp': time.time()
        }
        
        # Add component-specific status if available
        if self.monitor:
            status['monitoring'] = self.monitor.get_system_status()
        
        if self.security_framework:
            status['security'] = self.security_framework.get_security_status()
        
        if self.error_handler:
            status['error_handling'] = {
                'circuit_breakers': self.error_handler.get_circuit_breaker_status(),
                'rate_limiters': self.error_handler.get_rate_limiter_status(),
                'error_statistics': self.error_handler.get_error_statistics()
            }
        
        if self.compliance_manager:
            status['compliance'] = self.compliance_manager.get_compliance_report()
        
        return status
    
    def validate_input(self, data: Any, schema_name: Optional[str] = None) -> bool:
        """Validate input data using configured validation"""
        if not self.config.enable_input_validation:
            return True
        
        try:
            if schema_name == "protein_sequence" and VALIDATION_AVAILABLE:
                if isinstance(data, dict):
                    result = validate_protein_sequence(data)
                    return result.is_valid
            elif schema_name == "model_config" and VALIDATION_AVAILABLE:
                if isinstance(data, dict):
                    result = validate_model_config(data)
                    return result.is_valid
            elif schema_name == "training_config" and VALIDATION_AVAILABLE:
                if isinstance(data, dict):
                    result = validate_training_config(data)
                    return result.is_valid
            
            # Basic validation for other data types
            if data is None:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle error using configured error handling"""
        if not self.config.enable_error_handling or not self.error_handler:
            return False
        
        try:
            error_event = self.error_handler.handle_error(
                error, 
                context=context,
                recovery_enabled=self.config.error_recovery_enabled
            )
            return error_event.recovery_successful
            
        except Exception as e:
            logger.error(f"Error in error handling: {e}")
            return False
    
    def record_metric(self, name: str, value: Union[float, int], 
                     labels: Optional[Dict[str, str]] = None):
        """Record custom metric"""
        if self.monitor:
            self.monitor.metrics_collector.record_custom_metric(name, value, labels)
    
    def audit_operation(self, operation: str, user_id: Optional[str] = None, 
                       result: str = "success", details: Optional[Dict[str, Any]] = None):
        """Audit operation for compliance"""
        if self.security_framework:
            self.security_framework.audit_logger.log_event(
                AuditEventType.DATA_ACCESS,
                operation,
                result,
                user_id=user_id,
                details=details or {}
            )
    
    def shutdown(self):
        """Shutdown all framework components"""
        logger.info("Shutting down unified robustness framework...")
        
        if self.monitor:
            self.monitor.stop()
        
        if self.config_manager:
            self.config_manager.stop()
        
        self.initialized = False
        logger.info("Framework shutdown complete")

# Decorators for easy framework integration

def with_robustness(schema: Optional[str] = None, 
                   require_auth: bool = False,
                   audit: bool = True):
    """Decorator to add robustness features to functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            framework = get_global_framework()
            if not framework or not framework.initialized:
                logger.warning("Robustness framework not initialized")
                return func(*args, **kwargs)
            
            # Input validation
            if schema and len(args) > 0:
                if not framework.validate_input(args[0], schema):
                    raise ValueError(f"Input validation failed for schema: {schema}")
            
            # Authentication check
            if require_auth and framework.security_framework:
                # This would need session context
                pass
            
            # Execute function with error handling
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Record success metric
                duration = time.time() - start_time
                framework.record_metric(
                    f"function_duration_{func.__name__}",
                    duration,
                    {'status': 'success'}
                )
                
                # Audit operation
                if audit:
                    framework.audit_operation(
                        func.__name__,
                        result="success",
                        details={'duration': duration}
                    )
                
                return result
                
            except Exception as e:
                # Handle error
                recovered = framework.handle_error(e, {'function': func.__name__})
                
                # Record error metric
                framework.record_metric(
                    f"function_errors_{func.__name__}",
                    1,
                    {'error_type': type(e).__name__}
                )
                
                # Audit failure
                if audit:
                    framework.audit_operation(
                        func.__name__,
                        result="failure",
                        details={'error': str(e)}
                    )
                
                if not recovered:
                    raise
                
                return None
        
        return wrapper
    return decorator

def robust_api_endpoint(schema: Optional[str] = None):
    """Decorator for robust API endpoints"""
    return with_robustness(schema=schema, require_auth=True, audit=True)

def robust_data_processing(schema: Optional[str] = None):
    """Decorator for robust data processing functions"""
    return with_robustness(schema=schema, require_auth=False, audit=True)

# Context managers

@contextmanager
def robustness_context(config: Optional[RobustnessConfig] = None):
    """Context manager for robustness framework"""
    config = config or RobustnessConfig()
    framework = UnifiedRobustnessFramework(config)
    
    try:
        framework.initialize()
        set_global_framework(framework)
        yield framework
    finally:
        framework.shutdown()
        set_global_framework(None)

@contextmanager
def error_boundary(recovery_enabled: bool = True):
    """Context manager for error boundary with recovery"""
    framework = get_global_framework()
    
    try:
        yield
    except Exception as e:
        if framework and recovery_enabled:
            recovered = framework.handle_error(e)
            if not recovered:
                raise
        else:
            raise

# Global framework instance management

_global_framework: Optional[UnifiedRobustnessFramework] = None
_framework_lock = threading.Lock()

def get_global_framework() -> Optional[UnifiedRobustnessFramework]:
    """Get global framework instance"""
    return _global_framework

def set_global_framework(framework: Optional[UnifiedRobustnessFramework]):
    """Set global framework instance"""
    global _global_framework
    with _framework_lock:
        _global_framework = framework

def initialize_framework(config: Optional[RobustnessConfig] = None) -> UnifiedRobustnessFramework:
    """Initialize global framework"""
    global _global_framework
    
    with _framework_lock:
        if _global_framework:
            logger.warning("Framework already initialized")
            return _global_framework
        
        config = config or RobustnessConfig()
        _global_framework = UnifiedRobustnessFramework(config)
        _global_framework.initialize()
        
        return _global_framework

def shutdown_framework():
    """Shutdown global framework"""
    global _global_framework
    
    with _framework_lock:
        if _global_framework:
            _global_framework.shutdown()
            _global_framework = None

# Convenience functions

def create_basic_config() -> RobustnessConfig:
    """Create basic robustness configuration"""
    return RobustnessConfig(
        level=RobustnessLevel.BASIC,
        enable_monitoring=False,
        enable_security=False,
        enable_testing=False,
        enable_compliance=False
    )

def create_standard_config() -> RobustnessConfig:
    """Create standard robustness configuration"""
    return RobustnessConfig(
        level=RobustnessLevel.STANDARD,
        enable_testing=False,
        enable_compliance=False
    )

def create_enterprise_config() -> RobustnessConfig:
    """Create enterprise robustness configuration"""
    return RobustnessConfig(
        level=RobustnessLevel.ENTERPRISE,
        security_level="high",
        validation_level="strict",
        audit_all_operations=True,
        applicable_regulations=["gdpr", "ccpa"]
    )

def create_maximum_config() -> RobustnessConfig:
    """Create maximum robustness configuration"""
    return RobustnessConfig(
        level=RobustnessLevel.MAXIMUM,
        security_level="high",
        validation_level="strict",
        audit_all_operations=True,
        enable_opentelemetry=True,
        applicable_regulations=["gdpr", "ccpa", "hipaa"]
    )

# Quick setup functions

def setup_basic_robustness() -> UnifiedRobustnessFramework:
    """Setup basic robustness framework"""
    config = create_basic_config()
    return initialize_framework(config)

def setup_enterprise_robustness() -> UnifiedRobustnessFramework:
    """Setup enterprise robustness framework"""
    config = create_enterprise_config()
    return initialize_framework(config)

def setup_maximum_robustness() -> UnifiedRobustnessFramework:
    """Setup maximum robustness framework"""
    config = create_maximum_config()
    return initialize_framework(config)

# Health check function

def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check"""
    framework = get_global_framework()
    
    if not framework:
        return {
            'status': 'error',
            'message': 'Robustness framework not initialized',
            'timestamp': time.time()
        }
    
    if not framework.initialized:
        return {
            'status': 'error',
            'message': 'Robustness framework not properly initialized',
            'timestamp': time.time()
        }
    
    try:
        status = framework.get_framework_status()
        
        # Determine overall health
        component_statuses = status.get('components', {})
        error_count = sum(1 for s in component_statuses.values() if s == 'error')
        
        if error_count == 0:
            overall_status = 'healthy'
        elif error_count <= 2:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'framework_status': status,
            'timestamp': time.time()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Health check failed: {e}',
            'timestamp': time.time()
        }

if __name__ == "__main__":
    # Example usage
    print("Unified Robustness Framework")
    print("=============================")
    
    # Setup enterprise robustness
    framework = setup_enterprise_robustness()
    
    # Print status
    status = framework.get_framework_status()
    print(json.dumps(status, indent=2, default=str))
    
    # Cleanup
    shutdown_framework()
