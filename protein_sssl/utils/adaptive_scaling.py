"""
Adaptive Scaling System
Dynamic resource allocation based on workload patterns and performance metrics
"""

import time
import threading
import multiprocessing
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import torch
import gc

from .monitoring import MetricsCollector
from .performance_optimizer import PerformanceOptimizer
from .error_handling import ProteinSSLError


@dataclass
class WorkloadPattern:
    """Workload pattern analysis"""
    avg_request_rate: float
    peak_request_rate: float
    avg_processing_time: float
    peak_processing_time: float
    memory_usage_pattern: str  # "stable", "growing", "fluctuating"
    gpu_utilization: float
    prediction_complexity: float  # based on sequence length
    
    
@dataclass
class ScalingDecision:
    """Scaling decision with rationale"""
    action: str  # "scale_up", "scale_down", "maintain", "optimize"
    target_workers: int
    target_memory_limit: int  # MB
    target_cache_size: int  # MB
    rationale: str
    confidence: float
    expected_improvement: float


class AdaptiveScalingEngine:
    """
    Intelligent scaling engine that adapts resources based on workload patterns
    """
    
    def __init__(self,
                 min_workers: int = 1,
                 max_workers: int = None,
                 scaling_interval: float = 60.0,
                 memory_threshold: float = 0.85,
                 gpu_threshold: float = 0.90):
        
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.scaling_interval = scaling_interval
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        
        # Current state
        self.current_workers = min_workers
        self.current_memory_limit = 4096  # MB
        self.current_cache_size = 1024  # MB
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Historical data
        self.workload_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=1000)
        
        # Scaling control
        self._scaling_active = False
        self._scaling_thread = None
        self._scaling_lock = threading.Lock()
        
        # Executors
        self._thread_executor = None
        self._process_executor = None
        self._gpu_workers = []
        
        # Prediction models for scaling
        self._workload_predictor = WorkloadPredictor()
        
    def start_adaptive_scaling(self):
        """Start adaptive scaling system"""
        if self._scaling_active:
            return
            
        self._scaling_active = True
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        self._scaling_thread.start()
        
        # Initialize executors
        self._initialize_executors()
        
    def stop_adaptive_scaling(self):
        """Stop adaptive scaling system"""
        self._scaling_active = False
        
        if self._scaling_thread:
            self._scaling_thread.join(timeout=10.0)
            
        self._cleanup_executors()
        
    def _initialize_executors(self):
        """Initialize worker executors"""
        with self._scaling_lock:
            # Thread pool for I/O bound tasks
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.current_workers * 2,
                thread_name_prefix="protein_ssl_thread"
            )
            
            # Process pool for CPU intensive tasks
            self._process_executor = ProcessPoolExecutor(
                max_workers=self.current_workers,
                max_tasks_per_child=100
            )
            
            # GPU workers if available
            if torch.cuda.is_available():
                self._initialize_gpu_workers()
                
    def _initialize_gpu_workers(self):
        """Initialize GPU workers"""
        num_gpus = torch.cuda.device_count()
        
        for gpu_id in range(num_gpus):
            worker = GPUWorker(gpu_id)
            self._gpu_workers.append(worker)
            
    def _cleanup_executors(self):
        """Clean up executors"""
        if self._thread_executor:
            self._thread_executor.shutdown(wait=True)
            
        if self._process_executor:
            self._process_executor.shutdown(wait=True)
            
        for worker in self._gpu_workers:
            worker.cleanup()
            
        self._gpu_workers.clear()
        
    def _scaling_loop(self):
        """Main scaling decision loop"""
        while self._scaling_active:
            try:
                # Analyze current workload
                workload_pattern = self._analyze_workload_pattern()
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(workload_pattern)
                
                # Apply scaling if needed
                if scaling_decision.action != "maintain":
                    self._apply_scaling_decision(scaling_decision)
                    
                # Record for analysis
                self._record_scaling_decision(workload_pattern, scaling_decision)
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                print(f"Scaling loop error: {e}")
                time.sleep(self.scaling_interval)
                
    def _analyze_workload_pattern(self) -> WorkloadPattern:
        """Analyze current workload patterns"""
        
        # Collect recent metrics
        metrics = self.metrics_collector.get_system_metrics()
        
        # Calculate request rate (estimate from performance history)
        recent_perf = list(self.performance_history)[-60:]  # Last 60 measurements
        
        if len(recent_perf) >= 10:
            request_times = [p.get("processing_time", 1.0) for p in recent_perf]
            avg_processing_time = np.mean(request_times)
            peak_processing_time = np.percentile(request_times, 95)
            
            # Estimate request rate
            avg_request_rate = 60.0 / max(avg_processing_time, 0.1)  # requests per minute
            peak_request_rate = 60.0 / max(np.percentile(request_times, 5), 0.1)
        else:
            avg_processing_time = 1.0
            peak_processing_time = 2.0
            avg_request_rate = 10.0
            peak_request_rate = 30.0
            
        # Memory usage pattern
        memory_usage = metrics.get("memory_usage_percent", 50.0)
        memory_pattern = self._classify_memory_pattern()
        
        # GPU utilization
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            try:
                gpu_utilization = torch.cuda.utilization() / 100.0
            except:
                gpu_utilization = 0.5  # Assume moderate usage
                
        # Prediction complexity (from sequence lengths)
        prediction_complexity = self._estimate_prediction_complexity()
        
        pattern = WorkloadPattern(
            avg_request_rate=avg_request_rate,
            peak_request_rate=peak_request_rate,
            avg_processing_time=avg_processing_time,
            peak_processing_time=peak_processing_time,
            memory_usage_pattern=memory_pattern,
            gpu_utilization=gpu_utilization,
            prediction_complexity=prediction_complexity
        )
        
        self.workload_history.append(pattern)
        return pattern
        
    def _classify_memory_pattern(self) -> str:
        """Classify memory usage pattern"""
        if len(self.workload_history) < 10:
            return "stable"
            
        recent_patterns = list(self.workload_history)[-10:]
        memory_values = [p.memory_usage_pattern for p in recent_patterns 
                        if hasattr(p, 'memory_usage_value')]
        
        if not memory_values:
            return "stable"
            
        trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        variance = np.var(memory_values)
        
        if trend > 0.02:
            return "growing"
        elif variance > 0.1:
            return "fluctuating"
        else:
            return "stable"
            
    def _estimate_prediction_complexity(self) -> float:
        """Estimate current prediction complexity"""
        # This would be based on actual sequence lengths being processed
        # For now, return a reasonable default
        return 0.5  # Medium complexity
        
    def _make_scaling_decision(self, pattern: WorkloadPattern) -> ScalingDecision:
        """Make intelligent scaling decision"""
        
        # Get current performance metrics
        current_metrics = self.metrics_collector.get_system_metrics()
        memory_usage = current_metrics.get("memory_usage_percent", 50.0) / 100.0
        cpu_usage = current_metrics.get("cpu_usage_percent", 50.0) / 100.0
        
        # Predict future workload
        predicted_load = self._workload_predictor.predict_load(self.workload_history)
        
        # Decision factors
        factors = {
            "memory_pressure": memory_usage > self.memory_threshold,
            "cpu_pressure": cpu_usage > 0.80,
            "gpu_pressure": pattern.gpu_utilization > self.gpu_threshold,
            "high_request_rate": pattern.avg_request_rate > 50,
            "slow_processing": pattern.avg_processing_time > 2.0,
            "predicted_increase": predicted_load > pattern.avg_request_rate * 1.5,
            "growing_memory": pattern.memory_usage_pattern == "growing",
            "high_complexity": pattern.prediction_complexity > 0.7
        }
        
        # Calculate scaling score
        scale_up_score = sum([
            factors["memory_pressure"] * 3,
            factors["cpu_pressure"] * 2,
            factors["gpu_pressure"] * 2,
            factors["high_request_rate"] * 2,
            factors["slow_processing"] * 2,
            factors["predicted_increase"] * 1,
            factors["high_complexity"] * 1
        ])
        
        scale_down_score = sum([
            (not factors["memory_pressure"]) * 1,
            (not factors["cpu_pressure"]) * 1,
            (not factors["high_request_rate"]) * 1,
            (pattern.avg_request_rate < 10) * 2,
            (memory_usage < 0.5) * 1
        ])
        
        # Make decision
        if scale_up_score >= 5 and self.current_workers < self.max_workers:
            action = "scale_up"
            target_workers = min(self.current_workers + 1, self.max_workers)
            target_memory_limit = int(self.current_memory_limit * 1.2)
            target_cache_size = int(self.current_cache_size * 1.1)
            rationale = f"High load detected (score: {scale_up_score})"
            confidence = min(scale_up_score / 10.0, 1.0)
            expected_improvement = 0.2
            
        elif scale_down_score >= 4 and self.current_workers > self.min_workers:
            action = "scale_down"
            target_workers = max(self.current_workers - 1, self.min_workers)
            target_memory_limit = int(self.current_memory_limit * 0.9)
            target_cache_size = int(self.current_cache_size * 0.95)
            rationale = f"Low load detected (score: {scale_down_score})"
            confidence = min(scale_down_score / 8.0, 1.0)
            expected_improvement = 0.1
            
        elif factors["growing_memory"] or memory_usage > 0.95:
            action = "optimize"
            target_workers = self.current_workers
            target_memory_limit = self.current_memory_limit
            target_cache_size = int(self.current_cache_size * 0.8)
            rationale = "Memory optimization needed"
            confidence = 0.8
            expected_improvement = 0.15
            
        else:
            action = "maintain"
            target_workers = self.current_workers
            target_memory_limit = self.current_memory_limit
            target_cache_size = self.current_cache_size
            rationale = "Current configuration optimal"
            confidence = 0.6
            expected_improvement = 0.0
            
        return ScalingDecision(
            action=action,
            target_workers=target_workers,
            target_memory_limit=target_memory_limit,
            target_cache_size=target_cache_size,
            rationale=rationale,
            confidence=confidence,
            expected_improvement=expected_improvement
        )
        
    def _apply_scaling_decision(self, decision: ScalingDecision):
        """Apply scaling decision"""
        try:
            with self._scaling_lock:
                if decision.action == "scale_up":
                    self._scale_up(decision)
                elif decision.action == "scale_down":
                    self._scale_down(decision)
                elif decision.action == "optimize":
                    self._optimize_resources(decision)
                    
                print(f"🔄 Scaling action: {decision.action} - {decision.rationale}")
                
        except Exception as e:
            print(f"Failed to apply scaling decision: {e}")
            
    def _scale_up(self, decision: ScalingDecision):
        """Scale up resources"""
        # Update worker counts
        if self._thread_executor:
            self._thread_executor._max_workers = decision.target_workers * 2
            
        # Update memory limits
        self.current_memory_limit = decision.target_memory_limit
        self.current_cache_size = decision.target_cache_size
        
        # Expand caches
        self.performance_optimizer.expand_cache_limits(decision.target_cache_size)
        
        # Initialize additional GPU workers if needed
        if torch.cuda.is_available() and len(self._gpu_workers) < decision.target_workers:
            self._add_gpu_workers(decision.target_workers - len(self._gpu_workers))
            
        self.current_workers = decision.target_workers
        
    def _scale_down(self, decision: ScalingDecision):
        """Scale down resources"""
        # Update worker counts (gradual reduction)
        if self._thread_executor:
            self._thread_executor._max_workers = decision.target_workers * 2
            
        # Reduce memory allocations
        self.current_memory_limit = decision.target_memory_limit
        self.current_cache_size = decision.target_cache_size
        
        # Reduce cache sizes
        self.performance_optimizer.reduce_cache_limits(decision.target_cache_size)
        
        # Remove excess GPU workers
        while len(self._gpu_workers) > decision.target_workers:
            worker = self._gpu_workers.pop()
            worker.cleanup()
            
        self.current_workers = decision.target_workers
        
    def _optimize_resources(self, decision: ScalingDecision):
        """Optimize resource usage without scaling workers"""
        # Clear memory
        self.performance_optimizer.clear_old_caches()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Optimize cache sizes
        self.current_cache_size = decision.target_cache_size
        self.performance_optimizer.optimize_cache_sizes()
        
    def _add_gpu_workers(self, count: int):
        """Add GPU workers"""
        for i in range(count):
            gpu_id = i % torch.cuda.device_count()
            worker = GPUWorker(gpu_id)
            self._gpu_workers.append(worker)
            
    def _record_scaling_decision(self, pattern: WorkloadPattern, decision: ScalingDecision):
        """Record scaling decision for analysis"""
        record = {
            "timestamp": time.time(),
            "pattern": pattern,
            "decision": decision,
            "system_metrics": self.metrics_collector.get_system_metrics()
        }
        
        self.scaling_history.append(record)
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        return {
            "current_workers": self.current_workers,
            "current_memory_limit": self.current_memory_limit,
            "current_cache_size": self.current_cache_size,
            "gpu_workers": len(self._gpu_workers),
            "scaling_active": self._scaling_active,
            "recent_decisions": [
                {
                    "action": record["decision"].action,
                    "rationale": record["decision"].rationale,
                    "timestamp": record["timestamp"]
                }
                for record in list(self.scaling_history)[-5:]
            ]
        }
        
    def get_performance_prediction(self) -> Dict[str, float]:
        """Get performance predictions"""
        return self._workload_predictor.get_predictions(self.workload_history)


class WorkloadPredictor:
    """Predicts future workload based on historical patterns"""
    
    def __init__(self):
        self.prediction_window = 300  # 5 minutes
        
    def predict_load(self, history: deque) -> float:
        """Predict future load based on history"""
        if len(history) < 10:
            return 10.0  # Default prediction
            
        recent_loads = [p.avg_request_rate for p in list(history)[-20:]]
        
        # Simple trend analysis
        if len(recent_loads) >= 5:
            # Linear trend
            x = np.arange(len(recent_loads))
            trend = np.polyfit(x, recent_loads, 1)[0]
            
            # Seasonal pattern (if enough data)
            baseline = np.mean(recent_loads)
            
            # Predict based on trend
            predicted = baseline + trend * 5  # 5 steps ahead
            
            return max(predicted, 0.0)
        else:
            return np.mean(recent_loads)
            
    def get_predictions(self, history: deque) -> Dict[str, float]:
        """Get various predictions"""
        if len(history) < 5:
            return {"load": 10.0, "memory": 0.5, "processing_time": 1.0}
            
        recent = list(history)[-10:]
        
        return {
            "load": self.predict_load(history),
            "memory": np.mean([0.5 for _ in recent]),  # Placeholder
            "processing_time": np.mean([p.avg_processing_time for p in recent])
        }


class GPUWorker:
    """GPU worker for handling model inference"""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.active = True
        
    def cleanup(self):
        """Clean up GPU worker"""
        self.active = False
        if torch.cuda.is_available():
            with torch.cuda.device(self.gpu_id):
                torch.cuda.empty_cache()


# Global scaling engine
_global_scaling_engine = None

def get_scaling_engine(**kwargs) -> AdaptiveScalingEngine:
    """Get global scaling engine"""
    global _global_scaling_engine
    
    if _global_scaling_engine is None:
        _global_scaling_engine = AdaptiveScalingEngine(**kwargs)
        
    return _global_scaling_engine

def start_adaptive_scaling(**kwargs) -> AdaptiveScalingEngine:
    """Start adaptive scaling system"""
    engine = get_scaling_engine(**kwargs)
    engine.start_adaptive_scaling()
    return engine

def get_scaling_status() -> Dict[str, Any]:
    """Get current scaling status"""
    engine = get_scaling_engine()
    return engine.get_scaling_status()