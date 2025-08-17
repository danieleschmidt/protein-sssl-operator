"""
Advanced Error Handling & Recovery System for protein-sssl-operator
Provides enterprise-grade error handling with circuit breakers, rate limiting,
and comprehensive recovery mechanisms.
"""

import time
import asyncio
import threading
import functools
import traceback
import logging
import json
import hashlib
import statistics
from typing import (
    Dict, List, Optional, Any, Callable, Type, Union, 
    Awaitable, TypeVar, Generic, Protocol
)
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
import weakref

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class ErrorSeverity(Enum):
    """Enhanced error severity levels with priority scores"""
    DEBUG = ("debug", 10)
    INFO = ("info", 20)
    WARNING = ("warning", 30)
    ERROR = ("error", 40)
    CRITICAL = ("critical", 50)
    FATAL = ("fatal", 60)
    
    def __init__(self, name: str, score: int):
        self.level_name = name
        self.score = score
    
    def __lt__(self, other):
        return self.score < other.score
    
    def __le__(self, other):
        return self.score <= other.score

class ErrorCategory(Enum):
    """Comprehensive error categorization"""
    # Infrastructure errors
    NETWORK = auto()
    STORAGE = auto()
    MEMORY = auto()
    CPU = auto()
    GPU = auto()
    
    # Application errors
    VALIDATION = auto()
    AUTHENTICATION = auto()
    AUTHORIZATION = auto()
    BUSINESS_LOGIC = auto()
    
    # Data errors
    DATA_CORRUPTION = auto()
    DATA_FORMAT = auto()
    DATA_MISSING = auto()
    DATA_INCONSISTENT = auto()
    
    # Model errors
    MODEL_LOADING = auto()
    MODEL_INFERENCE = auto()
    MODEL_TRAINING = auto()
    MODEL_VALIDATION = auto()
    
    # Security errors
    SECURITY_BREACH = auto()
    RATE_LIMIT = auto()
    SUSPICIOUS_ACTIVITY = auto()
    
    # External dependencies
    THIRD_PARTY_API = auto()
    DATABASE = auto()
    MESSAGE_QUEUE = auto()
    
    # Unknown/Other
    UNKNOWN = auto()
    SYSTEM = auto()

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class ErrorEvent:
    """Comprehensive error event data structure"""
    timestamp: float
    error_id: str
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'error_id': self.error_id,
            'error_type': self.error_type,
            'message': self.message,
            'category': self.category.name,
            'severity': self.severity.level_name,
            'stack_trace': self.stack_trace,
            'context': self.context,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_strategy': self.recovery_strategy,
            'tags': self.tags,
            'metrics': self.metrics
        }

@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""
    total_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0
    timeout_requests: int = 0
    rejected_requests: int = 0
    average_response_time: float = 0.0
    failure_rate: float = 0.0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_change_count: int = 0
    
    def update_success(self, response_time: float):
        """Update metrics for successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success_time = time.time()
        self._update_average_response_time(response_time)
        self._update_failure_rate()
    
    def update_failure(self, response_time: float = 0.0):
        """Update metrics for failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = time.time()
        if response_time > 0:
            self._update_average_response_time(response_time)
        self._update_failure_rate()
    
    def update_timeout(self):
        """Update metrics for timeout"""
        self.total_requests += 1
        self.timeout_requests += 1
        self.last_failure_time = time.time()
        self._update_failure_rate()
    
    def update_rejection(self):
        """Update metrics for rejected request"""
        self.rejected_requests += 1
    
    def state_changed(self):
        """Record state change"""
        self.state_change_count += 1
    
    def _update_average_response_time(self, new_time: float):
        """Update rolling average response time"""
        if self.total_requests == 1:
            self.average_response_time = new_time
        else:
            # Exponentially weighted moving average
            alpha = 0.1
            self.average_response_time = (
                alpha * new_time + (1 - alpha) * self.average_response_time
            )
    
    def _update_failure_rate(self):
        """Update failure rate"""
        if self.total_requests > 0:
            self.failure_rate = (self.failed_requests + self.timeout_requests) / self.total_requests

class RateLimiter:
    """Token bucket rate limiter with burst capacity"""
    
    def __init__(self, 
                 rate: float,  # requests per second
                 burst_capacity: int = 10,
                 redis_client: Optional[Any] = None,
                 key_prefix: str = "rate_limit"):
        self.rate = rate
        self.burst_capacity = burst_capacity
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        
        # Local state (for non-Redis mode)
        self._tokens = burst_capacity
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    def is_allowed(self, key: str = "default", tokens: int = 1) -> bool:
        """Check if request is allowed under rate limit"""
        if self.redis_client:
            return self._is_allowed_redis(key, tokens)
        else:
            return self._is_allowed_local(tokens)
    
    def _is_allowed_local(self, tokens: int) -> bool:
        """Local token bucket implementation"""
        with self._lock:
            now = time.time()
            time_passed = now - self._last_refill
            
            # Add tokens based on elapsed time
            tokens_to_add = time_passed * self.rate
            self._tokens = min(self.burst_capacity, self._tokens + tokens_to_add)
            self._last_refill = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            
            return False
    
    def _is_allowed_redis(self, key: str, tokens: int) -> bool:
        """Redis-based distributed rate limiting"""
        if not self.redis_client:
            return self._is_allowed_local(tokens)
        
        redis_key = f"{self.key_prefix}:{key}"
        
        try:
            # Lua script for atomic token bucket operation
            lua_script = """
            local key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local rate = tonumber(ARGV[2])
            local requested = tonumber(ARGV[3])
            local now = tonumber(ARGV[4])
            
            local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket[1]) or capacity
            local last_refill = tonumber(bucket[2]) or now
            
            local time_passed = now - last_refill
            tokens = math.min(capacity, tokens + time_passed * rate)
            
            if tokens >= requested then
                tokens = tokens - requested
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
                return 1
            else
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('EXPIRE', key, 3600)
                return 0
            end
            """
            
            result = self.redis_client.eval(
                lua_script, 1, redis_key, 
                self.burst_capacity, self.rate, tokens, time.time()
            )
            
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Redis rate limiting failed, falling back to local: {e}")
            return self._is_allowed_local(tokens)
    
    def get_remaining_tokens(self, key: str = "default") -> int:
        """Get remaining tokens in bucket"""
        if self.redis_client:
            return self._get_remaining_redis(key)
        else:
            return self._get_remaining_local()
    
    def _get_remaining_local(self) -> int:
        """Get remaining tokens locally"""
        with self._lock:
            now = time.time()
            time_passed = now - self._last_refill
            tokens_to_add = time_passed * self.rate
            current_tokens = min(self.burst_capacity, self._tokens + tokens_to_add)
            return int(current_tokens)
    
    def _get_remaining_redis(self, key: str) -> int:
        """Get remaining tokens from Redis"""
        if not self.redis_client:
            return self._get_remaining_local()
        
        redis_key = f"{self.key_prefix}:{key}"
        
        try:
            bucket = self.redis_client.hmget(redis_key, 'tokens', 'last_refill')
            tokens = float(bucket[0] or self.burst_capacity)
            last_refill = float(bucket[1] or time.time())
            
            now = time.time()
            time_passed = now - last_refill
            current_tokens = min(self.burst_capacity, tokens + time_passed * self.rate)
            
            return int(current_tokens)
            
        except Exception as e:
            logger.warning(f"Redis token check failed: {e}")
            return self._get_remaining_local()

class AdvancedCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and recovery strategies"""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception,
                 timeout: float = 30.0,
                 success_threshold: int = 3,  # Half-open -> Closed
                 failure_rate_threshold: float = 0.5,
                 min_requests: int = 10,
                 sliding_window_size: int = 100,
                 name: str = "default"):
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.min_requests = min_requests
        self.sliding_window_size = sliding_window_size
        self.name = name
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_state_change = time.time()
        self._lock = threading.RLock()
        
        # Metrics and monitoring
        self.metrics = CircuitBreakerMetrics()
        self._request_times = deque(maxlen=sliding_window_size)
        self._request_results = deque(maxlen=sliding_window_size)  # True for success, False for failure
        
        # Recovery strategies
        self._recovery_strategies: List[Callable[[], bool]] = []
        self._health_check_func: Optional[Callable[[], bool]] = None
        
        # Callbacks
        self._state_change_callbacks: List[Callable[[CircuitBreakerState, CircuitBreakerState], None]] = []
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def add_recovery_strategy(self, strategy: Callable[[], bool]):
        """Add a recovery strategy function"""
        self._recovery_strategies.append(strategy)
    
    def set_health_check(self, health_check: Callable[[], bool]):
        """Set health check function for recovery validation"""
        self._health_check_func = health_check
    
    def add_state_change_callback(self, callback: Callable[[CircuitBreakerState, CircuitBreakerState], None]):
        """Add callback for state changes"""
        self._state_change_callbacks.append(callback)
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)"""
        return self.state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open (failing)"""
        return self.state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing)"""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator interface"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self._should_reject_request():
                self.metrics.update_rejection()
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is {self._state.value}"
                )
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                # In half-open state, we're testing recovery
                self._test_recovery()
        
        start_time = time.time()
        
        try:
            # Set timeout for the function call
            if asyncio.iscoroutinefunction(func):
                result = asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            else:
                result = self._call_with_timeout(func, *args, **kwargs)
            
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            return result
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self._record_timeout()
            raise CircuitBreakerTimeoutError(
                f"Function call timed out after {self.timeout}s"
            )
            
        except self.expected_exception as e:
            response_time = time.time() - start_time
            self._record_failure(response_time)
            raise
    
    def _call_with_timeout(self, func: Callable, *args, **kwargs):
        """Call function with timeout (for non-async functions)"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except TimeoutError:
                raise CircuitBreakerTimeoutError(
                    f"Function call timed out after {self.timeout}s"
                )
    
    def _should_reject_request(self) -> bool:
        """Determine if request should be rejected"""
        if self._state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if (self._last_failure_time and 
                time.time() - self._last_failure_time >= self.recovery_timeout):
                self._change_state(CircuitBreakerState.HALF_OPEN)
                return False
            return True
        
        return False
    
    def _test_recovery(self):
        """Test if service has recovered (called in half-open state)"""
        if self._health_check_func:
            try:
                if self._health_check_func():
                    logger.info(f"Circuit breaker '{self.name}' health check passed")
                else:
                    logger.warning(f"Circuit breaker '{self.name}' health check failed")
            except Exception as e:
                logger.error(f"Circuit breaker '{self.name}' health check error: {e}")
    
    def _record_success(self, response_time: float):
        """Record successful request"""
        with self._lock:
            self.metrics.update_success(response_time)
            self._request_times.append(response_time)
            self._request_results.append(True)
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._change_state(CircuitBreakerState.CLOSED)
                    self._reset_counts()
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def _record_failure(self, response_time: float):
        """Record failed request"""
        with self._lock:
            self.metrics.update_failure(response_time)
            self._request_times.append(response_time)
            self._request_results.append(False)
            self._last_failure_time = time.time()
            
            if self._state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
                self._failure_count += 1
                
                # Check if we should open the circuit
                if self._should_open_circuit():
                    self._change_state(CircuitBreakerState.OPEN)
                    self._attempt_recovery()
    
    def _record_timeout(self):
        """Record timeout"""
        with self._lock:
            self.metrics.update_timeout()
            self._request_results.append(False)
            self._last_failure_time = time.time()
            
            if self._state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN]:
                self._failure_count += 1
                
                if self._should_open_circuit():
                    self._change_state(CircuitBreakerState.OPEN)
                    self._attempt_recovery()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        # Simple failure count threshold
        if self._failure_count >= self.failure_threshold:
            return True
        
        # Failure rate threshold (if we have enough requests)
        if len(self._request_results) >= self.min_requests:
            recent_failures = sum(1 for result in self._request_results if not result)
            failure_rate = recent_failures / len(self._request_results)
            
            if failure_rate >= self.failure_rate_threshold:
                return True
        
        return False
    
    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state"""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            self._last_state_change = time.time()
            self.metrics.state_changed()
            
            logger.info(
                f"Circuit breaker '{self.name}' state changed: "
                f"{old_state.value} -> {new_state.value}"
            )
            
            # Notify callbacks
            for callback in self._state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
    
    def _reset_counts(self):
        """Reset failure and success counts"""
        self._failure_count = 0
        self._success_count = 0
    
    def _attempt_recovery(self):
        """Attempt recovery using registered strategies"""
        if not self._recovery_strategies:
            return
        
        logger.info(f"Circuit breaker '{self.name}' attempting recovery")
        
        for i, strategy in enumerate(self._recovery_strategies):
            try:
                if strategy():
                    logger.info(
                        f"Circuit breaker '{self.name}' recovery strategy {i} successful"
                    )
                    return
            except Exception as e:
                logger.error(
                    f"Circuit breaker '{self.name}' recovery strategy {i} failed: {e}"
                )
    
    def force_open(self):
        """Manually open the circuit breaker"""
        with self._lock:
            self._change_state(CircuitBreakerState.OPEN)
            self._last_failure_time = time.time()
    
    def force_close(self):
        """Manually close the circuit breaker"""
        with self._lock:
            self._change_state(CircuitBreakerState.CLOSED)
            self._reset_counts()
    
    def force_half_open(self):
        """Manually set circuit breaker to half-open"""
        with self._lock:
            self._change_state(CircuitBreakerState.HALF_OPEN)
            self._reset_counts()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        with self._lock:
            recent_response_times = list(self._request_times)
            recent_results = list(self._request_results)
            
            avg_response_time = (
                statistics.mean(recent_response_times) 
                if recent_response_times else 0.0
            )
            
            p95_response_time = (
                statistics.quantiles(recent_response_times, n=20)[18] 
                if len(recent_response_times) >= 20 else 0.0
            )
            
            success_rate = (
                sum(recent_results) / len(recent_results) 
                if recent_results else 0.0
            )
            
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'last_failure_time': self._last_failure_time,
                'last_state_change': self._last_state_change,
                'total_requests': self.metrics.total_requests,
                'failed_requests': self.metrics.failed_requests,
                'successful_requests': self.metrics.successful_requests,
                'timeout_requests': self.metrics.timeout_requests,
                'rejected_requests': self.metrics.rejected_requests,
                'failure_rate': self.metrics.failure_rate,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'success_rate': success_rate,
                'state_change_count': self.metrics.state_change_count
            }

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreakerTimeoutError(Exception):
    """Raised when circuit breaker times out"""
    pass

class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded"""
    pass

class AdvancedErrorHandler:
    """Enterprise-grade error handling system"""
    
    def __init__(self,
                 enable_circuit_breakers: bool = True,
                 enable_rate_limiting: bool = True,
                 redis_client: Optional[Any] = None,
                 error_storage_backend: str = "memory"):
        
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_rate_limiting = enable_rate_limiting
        self.redis_client = redis_client
        self.error_storage_backend = error_storage_backend
        
        # Error tracking
        self._error_events: deque = deque(maxlen=10000)
        self._error_patterns: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        
        # Circuit breakers registry
        self._circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        
        # Rate limiters registry
        self._rate_limiters: Dict[str, RateLimiter] = {}
        
        # Recovery strategies
        self._recovery_strategies: Dict[str, List[Callable]] = defaultdict(list)
        
        # Error callbacks
        self._error_callbacks: List[Callable[[ErrorEvent], None]] = []
        
        # Correlation tracking
        self._correlation_map: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("Advanced error handler initialized")
    
    def create_circuit_breaker(self, 
                             name: str,
                             **kwargs) -> AdvancedCircuitBreaker:
        """Create and register a circuit breaker"""
        if name in self._circuit_breakers:
            logger.warning(f"Circuit breaker '{name}' already exists")
            return self._circuit_breakers[name]
        
        cb = AdvancedCircuitBreaker(name=name, **kwargs)
        self._circuit_breakers[name] = cb
        
        logger.info(f"Created circuit breaker: {name}")
        return cb
    
    def get_circuit_breaker(self, name: str) -> Optional[AdvancedCircuitBreaker]:
        """Get circuit breaker by name"""
        return self._circuit_breakers.get(name)
    
    def create_rate_limiter(self,
                          name: str,
                          rate: float,
                          burst_capacity: int = 10) -> RateLimiter:
        """Create and register a rate limiter"""
        if name in self._rate_limiters:
            logger.warning(f"Rate limiter '{name}' already exists")
            return self._rate_limiters[name]
        
        rl = RateLimiter(
            rate=rate,
            burst_capacity=burst_capacity,
            redis_client=self.redis_client,
            key_prefix=f"rate_limit:{name}"
        )
        self._rate_limiters[name] = rl
        
        logger.info(f"Created rate limiter: {name} ({rate} req/s)")
        return rl
    
    def get_rate_limiter(self, name: str) -> Optional[RateLimiter]:
        """Get rate limiter by name"""
        return self._rate_limiters.get(name)
    
    def handle_error(self,
                    error: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    correlation_id: Optional[str] = None,
                    recovery_enabled: bool = True) -> ErrorEvent:
        """Handle error with comprehensive processing"""
        
        # Generate error ID
        error_id = self._generate_error_id(error, context)
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_id=error_id,
            error_type=type(error).__name__,
            message=str(error),
            category=self._categorize_error(error),
            severity=self._determine_severity(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
            correlation_id=correlation_id
        )
        
        # Store error event
        self._store_error_event(error_event)
        
        # Update error patterns
        pattern_key = f"{error_event.category.name}:{error_event.error_type}"
        self._error_patterns[pattern_key] += 1
        
        # Track correlation
        if correlation_id:
            self._correlation_map[correlation_id].append(error_id)
        
        # Attempt recovery if enabled
        if recovery_enabled:
            recovery_result = self._attempt_error_recovery(error_event)
            error_event.recovery_attempted = True
            error_event.recovery_successful = recovery_result
        
        # Notify callbacks
        self._notify_error_callbacks(error_event)
        
        logger.error(
            f"Error handled: {error_event.error_type} - {error_event.message}",
            extra={
                'error_id': error_id,
                'category': error_event.category.name,
                'severity': error_event.severity.level_name
            }
        )
        
        return error_event
    
    def _generate_error_id(self, error: Exception, context: Optional[Dict]) -> str:
        """Generate unique error ID"""
        content = f"{type(error).__name__}:{str(error)}:{time.time()}"
        if context:
            content += f":{json.dumps(context, sort_keys=True)}"
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Network errors
        if any(keyword in error_msg for keyword in 
               ['connection', 'network', 'timeout', 'unreachable']):
            return ErrorCategory.NETWORK
        
        # Memory errors
        if any(keyword in error_msg for keyword in 
               ['memory', 'out of memory', 'oom', 'allocation']):
            return ErrorCategory.MEMORY
        
        # GPU errors
        if any(keyword in error_msg for keyword in 
               ['cuda', 'gpu', 'device', 'cublas', 'cudnn']):
            return ErrorCategory.GPU
        
        # Storage errors
        if any(keyword in error_msg for keyword in 
               ['disk', 'storage', 'file not found', 'permission denied']):
            return ErrorCategory.STORAGE
        
        # Model errors
        if any(keyword in error_msg for keyword in 
               ['model', 'checkpoint', 'weights', 'parameters']):
            return ErrorCategory.MODEL_LOADING
        
        # Data format errors
        if any(keyword in error_msg for keyword in 
               ['format', 'parsing', 'decode', 'json', 'yaml']):
            return ErrorCategory.DATA_FORMAT
        
        # Security errors
        if any(keyword in error_msg for keyword in 
               ['permission', 'unauthorized', 'forbidden', 'security']):
            return ErrorCategory.SECURITY_BREACH
        
        # Rate limiting
        if 'rate limit' in error_msg or 'too many requests' in error_msg:
            return ErrorCategory.RATE_LIMIT
        
        # Default categorization by exception type
        type_mapping = {
            'ValueError': ErrorCategory.VALIDATION,
            'TypeError': ErrorCategory.VALIDATION,
            'KeyError': ErrorCategory.DATA_MISSING,
            'FileNotFoundError': ErrorCategory.STORAGE,
            'PermissionError': ErrorCategory.AUTHORIZATION,
            'ConnectionError': ErrorCategory.NETWORK,
            'TimeoutError': ErrorCategory.NETWORK,
            'MemoryError': ErrorCategory.MEMORY,
            'SystemError': ErrorCategory.SYSTEM
        }
        
        return type_mapping.get(error_type, ErrorCategory.UNKNOWN)
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Critical system errors
        if error_type in ['SystemError', 'MemoryError', 'SystemExit']:
            return ErrorSeverity.FATAL
        
        # Security issues
        if any(keyword in error_msg for keyword in 
               ['security', 'breach', 'unauthorized', 'forbidden']):
            return ErrorSeverity.CRITICAL
        
        # Data corruption
        if any(keyword in error_msg for keyword in 
               ['corrupt', 'invalid', 'malformed']):
            return ErrorSeverity.ERROR
        
        # Resource issues
        if any(keyword in error_msg for keyword in 
               ['memory', 'disk', 'connection', 'timeout']):
            return ErrorSeverity.ERROR
        
        # Validation errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.WARNING
        
        return ErrorSeverity.ERROR
    
    def _store_error_event(self, error_event: ErrorEvent):
        """Store error event based on configured backend"""
        with self._lock:
            self._error_events.append(error_event)
        
        # Additional storage backends can be implemented here
        if self.error_storage_backend == "redis" and self.redis_client:
            self._store_error_redis(error_event)
        elif self.error_storage_backend == "database":
            self._store_error_database(error_event)
    
    def _store_error_redis(self, error_event: ErrorEvent):
        """Store error event in Redis"""
        try:
            key = f"error_events:{error_event.error_id}"
            self.redis_client.setex(
                key, 
                86400,  # 24 hours TTL
                json.dumps(error_event.to_dict())
            )
        except Exception as e:
            logger.warning(f"Failed to store error in Redis: {e}")
    
    def _store_error_database(self, error_event: ErrorEvent):
        """Store error event in database (placeholder)"""
        # Implement database storage as needed
        pass
    
    def _attempt_error_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from error"""
        category = error_event.category.name
        strategies = self._recovery_strategies.get(category, [])
        
        for strategy in strategies:
            try:
                if strategy(error_event):
                    error_event.recovery_strategy = strategy.__name__
                    logger.info(
                        f"Recovery successful using {strategy.__name__} "
                        f"for error {error_event.error_id}"
                    )
                    return True
            except Exception as e:
                logger.error(f"Recovery strategy {strategy.__name__} failed: {e}")
        
        return False
    
    def _notify_error_callbacks(self, error_event: ErrorEvent):
        """Notify registered error callbacks"""
        for callback in self._error_callbacks:
            try:
                callback(error_event)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def add_recovery_strategy(self, 
                            category: ErrorCategory,
                            strategy: Callable[[ErrorEvent], bool]):
        """Add recovery strategy for specific error category"""
        self._recovery_strategies[category.name].append(strategy)
        logger.info(f"Added recovery strategy for {category.name}")
    
    def add_error_callback(self, callback: Callable[[ErrorEvent], None]):
        """Add error callback"""
        self._error_callbacks.append(callback)
    
    def get_error_statistics(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_errors = [
                event for event in self._error_events 
                if event.timestamp >= cutoff_time
            ]
        
        if not recent_errors:
            return {'total_errors': 0, 'error_rate': 0.0}
        
        # Calculate statistics
        total_errors = len(recent_errors)
        error_rate = total_errors / (hours * 3600)  # errors per second
        
        # Group by category
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        
        for event in recent_errors:
            by_category[event.category.name] += 1
            by_severity[event.severity.level_name] += 1
            by_type[event.error_type] += 1
        
        # Recovery statistics
        recovery_attempted = sum(1 for e in recent_errors if e.recovery_attempted)
        recovery_successful = sum(1 for e in recent_errors if e.recovery_successful)
        recovery_rate = (
            recovery_successful / recovery_attempted 
            if recovery_attempted > 0 else 0.0
        )
        
        return {
            'period_hours': hours,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'by_category': dict(by_category),
            'by_severity': dict(by_severity),
            'by_type': dict(by_type),
            'recovery_attempted': recovery_attempted,
            'recovery_successful': recovery_successful,
            'recovery_rate': recovery_rate,
            'most_common_pattern': max(
                self._error_patterns.items(), 
                key=lambda x: x[1], 
                default=('None', 0)
            )
        }
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            name: cb.get_metrics() 
            for name, cb in self._circuit_breakers.items()
        }
    
    def get_rate_limiter_status(self) -> Dict[str, Any]:
        """Get status of all rate limiters"""
        return {
            name: {
                'rate': rl.rate,
                'burst_capacity': rl.burst_capacity,
                'remaining_tokens': rl.get_remaining_tokens()
            }
            for name, rl in self._rate_limiters.items()
        }

# Decorators for easy usage

def with_circuit_breaker(name: str, **kwargs):
    """Decorator to add circuit breaker protection"""
    def decorator(func: F) -> F:
        handler = get_global_error_handler()
        cb = handler.create_circuit_breaker(name, **kwargs)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        return wrapper
    return decorator

def with_rate_limit(name: str, rate: float, burst_capacity: int = 10):
    """Decorator to add rate limiting"""
    def decorator(func: F) -> F:
        handler = get_global_error_handler()
        rl = handler.create_rate_limiter(name, rate, burst_capacity)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not rl.is_allowed():
                raise RateLimitExceededError(
                    f"Rate limit exceeded for {name}: {rate} req/s"
                )
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def with_error_recovery(category: ErrorCategory = None):
    """Decorator to add automatic error recovery"""
    def decorator(func: F) -> F:
        handler = get_global_error_handler()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_event = handler.handle_error(
                    e, 
                    context={'function': func.__name__},
                    recovery_enabled=True
                )
                
                if error_event.recovery_successful:
                    # Retry the function after successful recovery
                    return func(*args, **kwargs)
                else:
                    raise
        
        return wrapper
    return decorator

# Global error handler instance
_global_error_handler: Optional[AdvancedErrorHandler] = None

def get_global_error_handler() -> AdvancedErrorHandler:
    """Get or create global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = AdvancedErrorHandler()
    return _global_error_handler

def configure_global_error_handler(**kwargs) -> AdvancedErrorHandler:
    """Configure global error handler"""
    global _global_error_handler
    _global_error_handler = AdvancedErrorHandler(**kwargs)
    return _global_error_handler

# Context managers

@contextmanager
def error_boundary(category: ErrorCategory = None, 
                  recovery_enabled: bool = True,
                  reraise: bool = True):
    """Context manager for error boundary"""
    handler = get_global_error_handler()
    
    try:
        yield
    except Exception as e:
        error_event = handler.handle_error(
            e,
            context={'boundary': True},
            recovery_enabled=recovery_enabled
        )
        
        if not error_event.recovery_successful and reraise:
            raise

@asynccontextmanager
async def async_error_boundary(category: ErrorCategory = None,
                             recovery_enabled: bool = True,
                             reraise: bool = True):
    """Async context manager for error boundary"""
    handler = get_global_error_handler()
    
    try:
        yield
    except Exception as e:
        error_event = handler.handle_error(
            e,
            context={'async_boundary': True},
            recovery_enabled=recovery_enabled
        )
        
        if not error_event.recovery_successful and reraise:
            raise
