"""
Security and input validation tests.
"""
import pytest
import tempfile
import os
from pathlib import Path

from protein_sssl.utils.input_validation import (
    InputValidator,
    SecurityError,
    validate_sequence,
    validate_file_path,
    validate_config
)
from protein_sssl.utils.error_recovery import (
    retry,
    CircuitBreaker,
    GracefulError,
    BatchProcessor,
    error_tracker
)


class TestInputValidator:
    """Test input validation functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.validator = InputValidator()
    
    def test_validate_sequence_valid(self):
        """Test validation of valid protein sequences."""
        # Standard amino acids
        seq = "MKFLKFSLLTAV"
        result = self.validator.validate_sequence(seq)
        assert result == "MKFLKFSLLTAV"
        
        # With unknown amino acids
        seq = "MKFLXFSLLTAV"
        result = self.validator.validate_sequence(seq, allow_unknown=True)
        assert result == "MKFLXFSLLTAV"
        
        # With whitespace
        seq = "MKF LKF SLL TAV"
        result = self.validator.validate_sequence(seq)
        assert result == "MKFLKFSLLTAV"
    
    def test_validate_sequence_invalid(self):
        """Test validation of invalid sequences."""
        # Empty sequence
        with pytest.raises(ValueError, match="empty"):
            self.validator.validate_sequence("")
        
        # Too short
        with pytest.raises(ValueError, match="too short"):
            self.validator.validate_sequence("MK")
        
        # Invalid characters
        with pytest.raises(ValueError, match="Invalid characters"):
            self.validator.validate_sequence("MKFL123LTAV")
        
        # Unknown amino acids not allowed
        with pytest.raises(ValueError, match="Invalid characters"):
            self.validator.validate_sequence("MKFLXFSLLTAV", allow_unknown=False)
    
    def test_validate_sequence_security_limits(self):
        """Test sequence length security limits."""
        # Very long sequence
        long_seq = "A" * (self.validator.max_sequence_length + 1)
        with pytest.raises(SecurityError, match="too long"):
            self.validator.validate_sequence(long_seq)
    
    def test_validate_file_path_valid(self):
        """Test validation of valid file paths."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            result = self.validator.validate_file_path(temp_path)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_path_nonexistent(self):
        """Test validation of non-existent files."""
        nonexistent = "/tmp/nonexistent_file.txt"
        
        # Should raise error when file must exist
        with pytest.raises(FileNotFoundError):
            self.validator.validate_file_path(nonexistent, must_exist=True)
        
        # Should work when file doesn't need to exist
        result = self.validator.validate_file_path(nonexistent, must_exist=False)
        assert isinstance(result, Path)
    
    def test_validate_model_path(self):
        """Test model path validation."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Write some dummy data that looks like a model
            f.write(b'PK\x03\x04\x14\x00\x00\x00')  # ZIP header
            model_path = f.name
        
        try:
            result = self.validator.validate_model_path(model_path)
            assert isinstance(result, Path)
        finally:
            os.unlink(model_path)
    
    def test_validate_model_path_invalid_extension(self):
        """Test model path with invalid extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            invalid_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid model file extension"):
                self.validator.validate_model_path(invalid_path)
        finally:
            os.unlink(invalid_path)
    
    def test_validate_coordinates(self):
        """Test coordinate validation."""
        import numpy as np
        
        # Valid coordinates
        coords = np.random.randn(10, 4, 3)  # N residues, 4 atoms, 3D
        assert self.validator.validate_coordinates(coords) is True
        
        # Invalid shape
        with pytest.raises(ValueError, match="shape"):
            invalid_coords = np.random.randn(10, 3)  # Missing atom dimension
            self.validator.validate_coordinates(invalid_coords)
        
        # NaN values
        with pytest.raises(ValueError, match="NaN"):
            coords_with_nan = coords.copy()
            coords_with_nan[0, 0, 0] = np.nan
            self.validator.validate_coordinates(coords_with_nan)
        
        # Unreasonably large values
        with pytest.raises(ValueError, match="unreasonably large"):
            large_coords = np.ones((10, 4, 3)) * 2000
            self.validator.validate_coordinates(large_coords)
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'nested': {
                'epochs': 10
            }
        }
        result = self.validator.validate_config(valid_config)
        assert result == valid_config
        
        # Config with security limit violation
        with pytest.raises(SecurityError, match="too large"):
            invalid_config = {
                'batch_size': 1000  # Exceeds security limit
            }
            self.validator.validate_config(invalid_config)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Dangerous filename
        dangerous = "../../../etc/passwd"
        safe = self.validator.sanitize_filename(dangerous)
        assert not safe.startswith('.')
        assert '..' not in safe
        assert '/' not in safe
        
        # Normal filename
        normal = "model_v1.pt"
        safe_normal = self.validator.sanitize_filename(normal)
        assert safe_normal == normal
        
        # Empty filename
        empty = ""
        safe_empty = self.validator.sanitize_filename(empty)
        assert len(safe_empty) > 0
        assert safe_empty.startswith("file_")
    
    def test_validate_device(self):
        """Test device validation."""
        # Valid devices
        assert self.validator.validate_device("cpu") == "cpu"
        assert self.validator.validate_device("cuda") == "cuda"
        assert self.validator.validate_device("auto") == "auto"
        assert self.validator.validate_device("cuda:0") == "cuda:0"
        
        # Invalid device
        with pytest.raises(ValueError, match="Invalid device"):
            self.validator.validate_device("invalid_device")
    
    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        # Valid values
        result = self.validator.validate_numeric_range(5, min_val=0, max_val=10)
        assert result == 5
        
        # Below minimum
        with pytest.raises(ValueError, match="must be >="):
            self.validator.validate_numeric_range(-1, min_val=0)
        
        # Above maximum
        with pytest.raises(ValueError, match="must be <="):
            self.validator.validate_numeric_range(15, max_val=10)
        
        # Non-numeric
        with pytest.raises(ValueError, match="must be numeric"):
            self.validator.validate_numeric_range("not_a_number")


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_retry_decorator_success(self):
        """Test retry decorator with successful function."""
        call_count = 0
        
        @retry(max_attempts=3, log_attempts=False)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = always_succeeds()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_decorator_eventual_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.1, log_attempts=False)
        def succeeds_on_third():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = succeeds_on_third()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_decorator_complete_failure(self):
        """Test retry decorator with complete failure."""
        call_count = 0
        
        @retry(max_attempts=2, delay=0.1, log_attempts=False)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
        
        assert call_count == 2
    
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        circuit_breaker = CircuitBreaker(failure_threshold=3)
        
        @circuit_breaker
        def normal_function():
            return "success"
        
        # Should work normally
        result = normal_function()
        assert result == "success"
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after repeated failures."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        call_count = 0
        
        @circuit_breaker
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Function failure")
        
        # First two calls should fail normally
        with pytest.raises(ValueError):
            failing_function()
        with pytest.raises(ValueError):
            failing_function()
        
        # Third call should trigger circuit breaker
        from protein_sssl.utils.error_recovery import CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            failing_function()
        
        assert call_count == 2  # Function only called twice
    
    def test_graceful_error_context(self):
        """Test graceful error handling context."""
        # Success case
        with GracefulError(fallback_value="fallback", log_error=False) as handler:
            result = "success"
        
        assert not handler.error_occurred
        assert handler.exception is None
        
        # Error case
        with GracefulError(fallback_value="fallback", log_error=False, reraise=False) as handler:
            raise ValueError("Test error")
        
        assert handler.error_occurred
        assert isinstance(handler.exception, ValueError)
    
    def test_batch_processor(self):
        """Test batch processing with error recovery."""
        processor = BatchProcessor(max_workers=2, error_threshold=0.5, retry_failed=False)
        
        def process_item(item):
            if item == "fail":
                raise ValueError("Processing failed")
            return f"processed_{item}"
        
        items = ["item1", "item2", "fail", "item3"]
        result = processor.process_batch(items, process_item)
        
        assert len(result['results']) == 3
        assert len(result['failed']) == 1
        assert result['error_rate'] == 0.25
        assert result['failed'][0][0] == "fail"
    
    def test_error_tracker(self):
        """Test error tracking functionality."""
        # Clear any existing errors
        error_tracker.errors.clear()
        
        # Record some errors
        error_tracker.record_error(ValueError("Test error 1"), "test_context")
        error_tracker.record_error(TypeError("Test error 2"), "test_context")
        
        # Get summary
        summary = error_tracker.get_error_summary(hours=1.0)
        
        assert summary['total'] == 2
        assert 'ValueError' in summary['by_type']
        assert 'TypeError' in summary['by_type']
        assert 'test_context' in summary['by_context']
    
    def test_track_errors_decorator(self):
        """Test error tracking decorator."""
        from protein_sssl.utils.error_recovery import track_errors
        
        # Clear any existing errors
        error_tracker.errors.clear()
        
        @track_errors(context="test_function")
        def failing_function():
            raise RuntimeError("Decorator test error")
        
        with pytest.raises(RuntimeError):
            failing_function()
        
        # Check error was tracked
        summary = error_tracker.get_error_summary(hours=1.0)
        assert summary['total'] >= 1
        assert 'test_function' in summary['by_context']


if __name__ == "__main__":
    pytest.main([__file__])