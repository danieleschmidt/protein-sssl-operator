"""
Health monitoring and system metrics tests.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from protein_sssl.utils.health_monitoring import (
    HealthMonitor,
    PerformanceProfiler,
    SystemMetrics,
    ProcessMetrics
)


class TestSystemMetrics:
    """Test system metrics data structure."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation and serialization."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=75.5,
            memory_percent=60.0,
            memory_used_gb=8.5,
            memory_available_gb=6.0,
            disk_used_percent=45.0,
            disk_free_gb=120.0,
            gpu_memory_used_mb=2048.0,
            gpu_memory_total_mb=8192.0,
            gpu_utilization_percent=85.0
        )
        
        assert metrics.cpu_percent == 75.5
        assert metrics.gpu_memory_used_mb == 2048.0
        assert isinstance(metrics.timestamp, float)


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.monitor = HealthMonitor(
            collection_interval=0.1,  # Fast interval for testing
            history_size=10,
            enable_gpu_monitoring=False  # Disable for testing
        )
    
    def teardown_method(self):
        """Cleanup after each test."""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock psutil returns
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(
            percent=70.0,
            used=8 * 1024**3,  # 8GB in bytes
            available=4 * 1024**3  # 4GB in bytes
        )
        mock_disk.return_value = Mock(
            percent=80.0,
            free=100 * 1024**3  # 100GB in bytes
        )
        
        metrics = self.monitor._collect_system_metrics()
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 70.0
        assert metrics.memory_used_gb == 8.0
        assert metrics.memory_available_gb == 4.0
        assert metrics.disk_used_percent == 80.0
        assert metrics.disk_free_gb == 100.0
    
    @patch('psutil.Process')
    def test_collect_process_metrics(self, mock_process_class):
        """Test process metrics collection."""
        # Mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.cpu_percent.return_value = 25.0
        mock_process.memory_percent.return_value = 15.0
        mock_process.memory_info.return_value = Mock(
            rss=512 * 1024**2,  # 512MB RSS
            vms=1024 * 1024**2  # 1GB VMS
        )
        mock_process.num_threads.return_value = 8
        mock_process.io_counters.return_value = Mock(
            read_bytes=1000000,
            write_bytes=500000
        )
        
        mock_process_class.return_value = mock_process
        
        metrics = self.monitor._collect_process_metrics()
        
        assert metrics.pid == 12345
        assert metrics.cpu_percent == 25.0
        assert metrics.memory_percent == 15.0
        assert metrics.memory_rss_mb == 512.0
        assert metrics.memory_vms_mb == 1024.0
        assert metrics.num_threads == 8
        assert metrics.io_read_bytes == 1000000
        assert metrics.io_write_bytes == 500000
    
    @patch.object(HealthMonitor, '_collect_system_metrics')
    @patch.object(HealthMonitor, '_collect_process_metrics')
    def test_monitoring_loop(self, mock_process_metrics, mock_system_metrics):
        """Test monitoring loop functionality."""
        # Mock metrics returns
        mock_system_metrics.return_value = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_used_percent=70.0,
            disk_free_gb=100.0
        )
        
        mock_process_metrics.return_value = ProcessMetrics(
            timestamp=time.time(),
            pid=12345,
            cpu_percent=25.0,
            memory_percent=15.0,
            memory_rss_mb=512.0,
            memory_vms_mb=1024.0,
            num_threads=8,
            io_read_bytes=1000000,
            io_write_bytes=500000
        )
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Let it run for a short time
        time.sleep(0.3)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Should have collected some metrics
        assert len(self.monitor.system_metrics) > 0
        assert len(self.monitor.process_metrics) > 0
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert not self.monitor.monitoring
        assert self.monitor.monitor_thread is None
        
        self.monitor.start_monitoring()
        assert self.monitor.monitoring
        assert self.monitor.monitor_thread is not None
        assert self.monitor.monitor_thread.is_alive()
        
        self.monitor.stop_monitoring()
        assert not self.monitor.monitoring
    
    @patch.object(HealthMonitor, '_collect_system_metrics')
    @patch.object(HealthMonitor, '_collect_process_metrics')
    def test_get_current_metrics(self, mock_process_metrics, mock_system_metrics):
        """Test getting current metrics."""
        mock_system_metrics.return_value = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_used_percent=70.0,
            disk_free_gb=100.0
        )
        
        mock_process_metrics.return_value = ProcessMetrics(
            timestamp=time.time(),
            pid=12345,
            cpu_percent=25.0,
            memory_percent=15.0,
            memory_rss_mb=512.0,
            memory_vms_mb=1024.0,
            num_threads=8,
            io_read_bytes=1000000,
            io_write_bytes=500000
        )
        
        current_metrics = self.monitor.get_current_metrics()
        
        assert 'system' in current_metrics
        assert 'process' in current_metrics
        assert current_metrics['system']['cpu_percent'] == 50.0
        assert current_metrics['process']['pid'] == 12345
    
    def test_health_check(self):
        """Test comprehensive health check."""
        with patch.object(self.monitor, 'get_current_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = {
                'system': {
                    'memory_percent': 75.0,
                    'cpu_percent': 60.0,
                    'disk_used_percent': 50.0,
                    'gpu_memory_used_mb': 4096.0,
                    'gpu_memory_total_mb': 8192.0
                },
                'process': {
                    'memory_percent': 20.0
                }
            }
            
            health_status = self.monitor.check_health()
            
            assert 'timestamp' in health_status
            assert 'overall_status' in health_status
            assert 'issues' in health_status
            assert 'metrics' in health_status
            
            # Should be healthy with these metrics
            assert health_status['overall_status'] == 'healthy'
    
    def test_health_check_with_issues(self):
        """Test health check with system issues."""
        with patch.object(self.monitor, 'get_current_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = {
                'system': {
                    'memory_percent': 90.0,  # High memory
                    'cpu_percent': 95.0,     # High CPU
                    'disk_used_percent': 90.0,  # Low disk space
                    'gpu_memory_used_mb': 7680.0,  # High GPU memory
                    'gpu_memory_total_mb': 8192.0
                },
                'process': {
                    'memory_percent': 20.0
                }
            }
            
            health_status = self.monitor.check_health()
            
            # Should have issues
            assert len(health_status['issues']) > 0
            assert health_status['overall_status'] in ['warning', 'critical']
            
            # Check specific issues
            issues = health_status['issues']
            assert any('memory' in issue.lower() for issue in issues)
            assert any('cpu' in issue.lower() for issue in issues)
            assert any('disk' in issue.lower() for issue in issues)
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        import tempfile
        import json
        
        # Add some fake metrics
        fake_system_metric = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_used_percent=70.0,
            disk_free_gb=100.0
        )
        
        self.monitor.system_metrics.append(fake_system_metric)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            export_path = f.name
        
        try:
            self.monitor.export_metrics(export_path, hours=1.0)
            
            # Verify file was created and contains data
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'export_timestamp' in exported_data
            assert 'period_hours' in exported_data
            assert 'system_metrics' in exported_data
            assert 'process_metrics' in exported_data
            
            assert exported_data['period_hours'] == 1.0
            assert len(exported_data['system_metrics']) >= 1
            
        finally:
            import os
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.profiler = PerformanceProfiler()
    
    def test_function_profiling(self):
        """Test function execution profiling."""
        @self.profiler.profile("test_function")
        def test_function(delay):
            time.sleep(delay)
            return "result"
        
        # Call function multiple times
        result1 = test_function(0.01)
        result2 = test_function(0.02)
        
        assert result1 == "result"
        assert result2 == "result"
        
        # Check profiling data
        profiles = self.profiler.get_profiles()
        assert "test_function" in profiles
        
        profile_data = profiles["test_function"]
        assert profile_data['call_count'] == 2
        assert profile_data['success_count'] == 2
        assert profile_data['failure_count'] == 0
        assert profile_data['avg_time'] > 0
        assert profile_data['success_rate'] == 1.0
    
    def test_function_profiling_with_failures(self):
        """Test profiling functions that sometimes fail."""
        call_count = 0
        
        @self.profiler.profile("failing_function")
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Test error")
            return "success"
        
        # First call succeeds
        result1 = failing_function()
        assert result1 == "success"
        
        # Second call fails
        with pytest.raises(ValueError):
            failing_function()
        
        # Third call succeeds
        result3 = failing_function()
        assert result3 == "success"
        
        # Check profiling data
        profiles = self.profiler.get_profiles()
        profile_data = profiles["failing_function"]
        
        assert profile_data['call_count'] == 3
        assert profile_data['success_count'] == 2
        assert profile_data['failure_count'] == 1
        assert profile_data['success_rate'] == 2/3
    
    def test_reset_profiles(self):
        """Test resetting profiling data."""
        @self.profiler.profile("test_reset")
        def test_function():
            return "test"
        
        # Call function to generate data
        test_function()
        
        # Should have profile data
        profiles = self.profiler.get_profiles()
        assert "test_reset" in profiles
        
        # Reset profiles
        self.profiler.reset_profiles()
        
        # Should be empty now
        profiles = self.profiler.get_profiles()
        assert len(profiles) == 0
    
    def test_concurrent_profiling(self):
        """Test profiling with concurrent execution."""
        @self.profiler.profile("concurrent_function")
        def concurrent_function(worker_id):
            time.sleep(0.01)  # Small delay
            return f"worker_{worker_id}"
        
        # Run multiple threads
        threads = []
        results = {}
        
        def worker(worker_id):
            results[worker_id] = concurrent_function(worker_id)
        
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 5
        for i in range(5):
            assert results[i] == f"worker_{i}"
        
        # Check profiling data
        profiles = self.profiler.get_profiles()
        profile_data = profiles["concurrent_function"]
        
        assert profile_data['call_count'] == 5
        assert profile_data['success_count'] == 5
        assert profile_data['failure_count'] == 0


class TestIntegrationMonitoring:
    """Integration tests for monitoring functionality."""
    
    def test_monitor_real_system_briefly(self):
        """Test monitoring real system for a brief period."""
        monitor = HealthMonitor(
            collection_interval=0.1,
            history_size=5,
            enable_gpu_monitoring=False
        )
        
        try:
            # Start monitoring
            monitor.start_monitoring()
            
            # Let it collect a few samples
            time.sleep(0.5)
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Should have collected metrics
            assert len(monitor.system_metrics) > 0
            assert len(monitor.process_metrics) > 0
            
            # Get summary
            summary = monitor.get_metrics_summary(hours=0.1)
            assert 'system' in summary
            assert 'process' in summary
            
        finally:
            if monitor.monitoring:
                monitor.stop_monitoring()
    
    def test_health_check_integration(self):
        """Test integrated health check."""
        monitor = HealthMonitor(enable_gpu_monitoring=False)
        
        try:
            health_status = monitor.check_health()
            
            # Should get valid health status
            assert 'timestamp' in health_status
            assert 'overall_status' in health_status
            assert health_status['overall_status'] in ['healthy', 'warning', 'critical']
            assert 'metrics' in health_status
            assert 'issues' in health_status
            
            # Metrics should be real
            metrics = health_status['metrics']
            assert 'system' in metrics
            assert 'process' in metrics
            assert metrics['system']['cpu_percent'] >= 0
            assert metrics['system']['memory_percent'] >= 0
            
        finally:
            pass  # No cleanup needed for this test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])