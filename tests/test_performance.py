"""
Performance and optimization tests.
"""
import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch
import threading
from concurrent.futures import ThreadPoolExecutor

from protein_sssl.utils.performance_optimization import (
    AdaptiveCache,
    ParallelProcessor,
    MemoryOptimizer,
    BatchOptimizer,
    adaptive_cache,
    profile_performance
)
from protein_sssl.utils.gpu_optimization import (
    GPUMemoryManager,
    EfficientAttention,
    ModelOptimizer
)


class TestAdaptiveCache:
    """Test adaptive caching functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cache = AdaptiveCache(max_size=10, ttl=1.0)
    
    def test_basic_cache_operations(self):
        """Test basic cache get/put operations."""
        # Test put and get
        self.cache.put("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        # Test default value for missing key
        assert self.cache.get("missing", "default") == "default"
    
    def test_cache_size_limit(self):
        """Test cache size limiting and eviction."""
        # Fill cache beyond capacity
        for i in range(15):
            self.cache.put(f"key{i}", f"value{i}")
        
        # Should not exceed max size
        assert len(self.cache._cache) <= 10
        
        # Recent items should be preserved
        assert self.cache.get("key14") == "value14"
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        self.cache.put("expire_key", "expire_value")
        
        # Should be available immediately
        assert self.cache.get("expire_key") == "expire_value"
        
        # Wait for TTL to expire
        time.sleep(1.5)
        
        # Should be expired
        assert self.cache.get("expire_key", "default") == "default"
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Perform some operations
        self.cache.put("key1", "value1")
        self.cache.get("key1")
        self.cache.get("missing")
        
        stats = self.cache.get_stats()
        
        assert stats['size'] == 1
        assert stats['total_requests'] == 2
        assert stats['cache_hits'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_adaptive_cache_decorator(self):
        """Test adaptive cache decorator."""
        call_count = 0
        
        @adaptive_cache(max_size=5)
        def expensive_function(x, y=1):
            nonlocal call_count
            call_count += 1
            return x * y
        
        # First call should execute function
        result1 = expensive_function(5, y=2)
        assert result1 == 10
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = expensive_function(5, y=2)
        assert result2 == 10
        assert call_count == 1  # No additional call
        
        # Different args should execute function
        result3 = expensive_function(3, y=4)
        assert result3 == 12
        assert call_count == 2
    
    def test_thread_safety(self):
        """Test thread-safe cache operations."""
        def cache_worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"value_{i}"
                self.cache.put(key, value)
                retrieved = self.cache.get(key)
                assert retrieved == value
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Cache should have items from all threads (up to size limit)
        assert len(self.cache._cache) <= 10


class TestParallelProcessor:
    """Test parallel processing functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.processor = ParallelProcessor(max_workers=2, use_processes=False)
    
    def test_parallel_processing(self):
        """Test basic parallel processing."""
        def square(x):
            return x * x
        
        items = list(range(10))
        results = self.processor.process_parallel(square, items)
        
        assert len(results) == 10
        assert results == [x * x for x in items]
    
    def test_error_handling(self):
        """Test error handling in parallel processing."""
        def failing_function(x):
            if x == 5:
                raise ValueError("Test error")
            return x * 2
        
        items = list(range(10))
        results = self.processor.process_parallel(failing_function, items)
        
        # Should have results for all items (None for failed)
        assert len(results) == 10
        assert results[5] is None  # Failed item
        assert results[0] == 0     # Successful item
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        def simple_func(x):
            return x
        
        items = list(range(5))
        self.processor.process_parallel(simple_func, items, progress_callback)
        
        # Should have received progress callbacks
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Final call shows completion
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        def simple_func(x):
            time.sleep(0.01)  # Small delay
            return x
        
        items = list(range(5))
        self.processor.process_parallel(simple_func, items)
        
        stats = self.processor.get_performance_stats()
        
        assert 'avg_execution_time' in stats
        assert 'avg_throughput' in stats
        assert stats['total_executions'] >= 1


class TestMemoryOptimizer:
    """Test memory optimization functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.optimizer = MemoryOptimizer()
    
    @pytest.mark.skipif(not hasattr(psutil, 'Process'), reason="psutil not available")
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        @self.optimizer.optimize_memory_usage
        def memory_intensive_function():
            # Simulate memory usage
            data = [i for i in range(100000)]
            return len(data)
        
        result = memory_intensive_function()
        assert result == 100000
        
        stats = self.optimizer.get_memory_stats()
        assert 'current_mb' in stats
        assert 'peak_mb' in stats
        assert stats['current_mb'] > 0
    
    def test_profile_performance_decorator(self):
        """Test performance profiling decorator."""
        @profile_performance("test_function")
        def test_function(delay):
            time.sleep(delay)
            return "done"
        
        result = test_function(0.01)
        assert result == "done"
        
        # Check that profiling information was logged (would need to capture logs to verify)


class TestBatchOptimizer:
    """Test batch size optimization."""
    
    def setup_method(self):
        """Setup for each test."""
        self.optimizer = BatchOptimizer()
    
    def test_batch_size_optimization(self):
        """Test optimal batch size finding."""
        def simple_batch_processor(batch):
            # Simulate processing time based on batch size
            time.sleep(0.001 * len(batch))
            return [item * 2 for item in batch]
        
        sample_data = list(range(50))
        
        optimal_size = self.optimizer.find_optimal_batch_size(
            simple_batch_processor,
            sample_data,
            "test_workload",
            max_batch_size=20,
            min_batch_size=1
        )
        
        assert 1 <= optimal_size <= 20
        assert "test_workload" in self.optimizer.batch_performance


@pytest.mark.skipif(not hasattr(torch, 'cuda') or not torch.cuda.is_available(), 
                    reason="CUDA not available")
class TestGPUOptimization:
    """Test GPU optimization functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_manager = GPUMemoryManager(self.device)
    
    def test_gpu_memory_manager(self):
        """Test GPU memory management."""
        if self.device.type == 'cpu':
            pytest.skip("GPU not available")
        
        import torch
        
        with self.gpu_manager.memory_scope():
            # Allocate some GPU memory
            tensor = torch.randn(1000, 1000, device=self.device)
            assert tensor.is_cuda
        
        # Memory should be managed properly
        self.gpu_manager.log_memory_stats()
    
    def test_large_sequence_optimization(self):
        """Test optimization for large protein sequences."""
        optimizations = self.gpu_manager.optimize_for_large_sequences(2000)
        
        assert 'batch_size' in optimizations
        assert 'chunk_size' in optimizations
        assert 'use_checkpointing' in optimizations
        assert optimizations['use_checkpointing'] is True
    
    def test_efficient_attention(self):
        """Test efficient attention implementations."""
        import torch
        
        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        
        # Test sparse attention
        output = EfficientAttention.sparse_attention(query, key, value)
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        
        # Test chunked attention for longer sequences
        long_seq_len = 512
        long_query = torch.randn(batch_size, num_heads, long_seq_len, head_dim, device=self.device)
        long_key = torch.randn(batch_size, num_heads, long_seq_len, head_dim, device=self.device)
        long_value = torch.randn(batch_size, num_heads, long_seq_len, head_dim, device=self.device)
        
        long_output = EfficientAttention._chunked_attention(
            long_query, long_key, long_value, chunk_size=128
        )
        assert long_output.shape == (batch_size, num_heads, long_seq_len, head_dim)


class TestIntegrationPerformance:
    """Integration tests for performance optimizations."""
    
    def test_end_to_end_optimization(self):
        """Test end-to-end performance optimization pipeline."""
        # Mock a protein processing pipeline
        def mock_sequence_processor(sequence):
            # Simulate processing time
            time.sleep(0.001)
            return f"processed_{sequence}"
        
        # Use parallel processor with caching
        processor = ParallelProcessor(max_workers=2)
        
        @adaptive_cache(max_size=100)
        def cached_processor(seq):
            return mock_sequence_processor(seq)
        
        sequences = [f"seq_{i}" for i in range(20)]
        
        # First run (no cache)
        start_time = time.time()
        results1 = processor.process_parallel(cached_processor, sequences)
        first_run_time = time.time() - start_time
        
        # Second run (with cache)
        start_time = time.time()
        results2 = processor.process_parallel(cached_processor, sequences)
        second_run_time = time.time() - start_time
        
        # Results should be identical
        assert results1 == results2
        
        # Second run should be faster due to caching
        assert second_run_time < first_run_time
    
    def test_memory_performance_under_load(self):
        """Test memory performance under heavy load."""
        optimizer = MemoryOptimizer()
        
        @optimizer.optimize_memory_usage
        def memory_intensive_task():
            # Create large data structures
            data = []
            for i in range(10):
                data.append([j for j in range(10000)])
            return len(data)
        
        # Run multiple times to test memory management
        results = []
        for _ in range(5):
            result = memory_intensive_task()
            results.append(result)
        
        assert all(r == 10 for r in results)
        
        # Check memory stats
        stats = optimizer.get_memory_stats()
        assert stats['current_mb'] > 0


class TestStressPerformance:
    """Stress tests for performance optimization."""
    
    @pytest.mark.slow
    def test_cache_under_high_load(self):
        """Test cache performance under high concurrent load."""
        cache = AdaptiveCache(max_size=1000, ttl=10.0)
        
        def worker(worker_id):
            results = []
            for i in range(100):
                key = f"worker_{worker_id}_item_{i}"
                value = f"value_{i}"
                
                cache.put(key, value)
                retrieved = cache.get(key)
                results.append(retrieved == value)
            
            return all(results)
        
        # Run many concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # All workers should succeed
        assert all(results)
        
        # Cache should maintain reasonable hit rate
        stats = cache.get_stats()
        assert stats['hit_rate'] > 0.5
    
    @pytest.mark.slow
    def test_parallel_processor_scalability(self):
        """Test parallel processor scalability."""
        def cpu_intensive_task(x):
            # Simulate CPU-intensive work
            total = 0
            for i in range(x * 1000):
                total += i
            return total
        
        items = list(range(1, 21))  # 20 items
        
        # Test with different worker counts
        for num_workers in [1, 2, 4]:
            processor = ParallelProcessor(max_workers=num_workers, use_processes=False)
            
            start_time = time.time()
            results = processor.process_parallel(cpu_intensive_task, items)
            execution_time = time.time() - start_time
            
            # Verify results are correct
            assert len(results) == 20
            assert all(isinstance(r, int) for r in results)
            
            # More workers should generally be faster (though overhead may vary)
            print(f"Workers: {num_workers}, Time: {execution_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])