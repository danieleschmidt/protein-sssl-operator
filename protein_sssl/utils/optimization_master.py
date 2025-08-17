"""
Master Optimization System for Protein-SSL Operator
Coordinates all optimization subsystems for maximum performance and scalability
"""

import time
import threading
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .logging_config import setup_logging
from .monitoring import MetricsCollector

# Import all optimization subsystems
from .advanced_caching import get_cache, MultiTierCache
from .memory_optimization import get_memory_optimizer, MemoryOptimizer
from .compute_optimization import get_compute_optimizer, ComputeOptimizer
from .parallel_processing import get_parallel_processor, ParallelProcessor
from .advanced_autoscaling import get_autoscaler, AdvancedAutoScaler
from .distributed_computing import get_distributed_coordinator, DistributedCoordinator
from .storage_optimization import get_storage_optimizer, StorageOptimizer
from .network_optimization import get_network_optimizer, NetworkOptimizer
from .resource_management import get_resource_manager, ResourceManager
from .benchmarking_suite import get_benchmark_suite, BenchmarkSuite

logger = setup_logging(__name__)


@dataclass
class OptimizationProfile:
    """Optimization profile configuration"""
    name: str
    description: str
    
    # Subsystem configurations
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    enable_compute_optimization: bool = True
    enable_parallel_processing: bool = True
    enable_auto_scaling: bool = True
    enable_distributed_computing: bool = False
    enable_storage_optimization: bool = True
    enable_network_optimization: bool = True
    enable_resource_management: bool = True
    
    # Performance targets
    target_latency_ms: float = 100.0
    target_throughput_ops_sec: float = 1000.0
    max_memory_usage_gb: float = 8.0
    max_cpu_utilization: float = 0.8
    
    # Custom parameters
    custom_config: Dict[str, Any] = None


class OptimizationMaster:
    """Master coordinator for all optimization systems"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # Subsystem managers
        self.subsystems = {}
        self.subsystem_status = {}
        
        # Master state
        self.optimization_active = False
        self.initialization_complete = False
        
        # Performance monitoring
        self.metrics_collector = MetricsCollector()
        self.performance_history = []
        
        # Coordination
        self.coordination_thread = None
        self.coordination_interval = 60.0  # 1 minute
        
        # Current optimization profile
        self.current_profile = None
        
        # Built-in profiles
        self.optimization_profiles = self._create_builtin_profiles()
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load master configuration"""
        default_config = {
            'coordination_interval': 60.0,
            'auto_initialization': True,
            'performance_monitoring': True,
            'subsystem_health_checking': True,
            'optimization_profiles': {}
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def _create_builtin_profiles(self) -> Dict[str, OptimizationProfile]:
        """Create built-in optimization profiles"""
        profiles = {}
        
        # Development profile - lightweight optimization
        profiles['development'] = OptimizationProfile(
            name='development',
            description='Lightweight optimization for development',
            enable_auto_scaling=False,
            enable_distributed_computing=False,
            target_latency_ms=500.0,
            target_throughput_ops_sec=100.0,
            max_memory_usage_gb=4.0,
            max_cpu_utilization=0.6
        )
        
        # Production profile - full optimization
        profiles['production'] = OptimizationProfile(
            name='production',
            description='Full optimization for production workloads',
            enable_caching=True,
            enable_memory_optimization=True,
            enable_compute_optimization=True,
            enable_parallel_processing=True,
            enable_auto_scaling=True,
            enable_distributed_computing=True,
            enable_storage_optimization=True,
            enable_network_optimization=True,
            enable_resource_management=True,
            target_latency_ms=50.0,
            target_throughput_ops_sec=10000.0,
            max_memory_usage_gb=32.0,
            max_cpu_utilization=0.9
        )
        
        # High-throughput profile - optimized for throughput
        profiles['high_throughput'] = OptimizationProfile(
            name='high_throughput',
            description='Optimized for maximum throughput',
            enable_caching=True,
            enable_memory_optimization=True,
            enable_compute_optimization=True,
            enable_parallel_processing=True,
            enable_auto_scaling=True,
            enable_distributed_computing=True,
            enable_storage_optimization=True,
            enable_network_optimization=True,
            enable_resource_management=True,
            target_latency_ms=200.0,
            target_throughput_ops_sec=50000.0,
            max_memory_usage_gb=64.0,
            max_cpu_utilization=0.95
        )
        
        # Low-latency profile - optimized for latency
        profiles['low_latency'] = OptimizationProfile(
            name='low_latency',
            description='Optimized for minimum latency',
            enable_caching=True,
            enable_memory_optimization=True,
            enable_compute_optimization=True,
            enable_parallel_processing=True,
            enable_auto_scaling=False,  # Avoid scaling delays
            enable_distributed_computing=False,  # Avoid network latency
            enable_storage_optimization=True,
            enable_network_optimization=True,
            enable_resource_management=True,
            target_latency_ms=10.0,
            target_throughput_ops_sec=1000.0,
            max_memory_usage_gb=16.0,
            max_cpu_utilization=0.7
        )
        
        # Resource-constrained profile - minimal resource usage
        profiles['resource_constrained'] = OptimizationProfile(
            name='resource_constrained',
            description='Optimized for minimal resource usage',
            enable_caching=True,
            enable_memory_optimization=True,
            enable_compute_optimization=False,
            enable_parallel_processing=True,
            enable_auto_scaling=False,
            enable_distributed_computing=False,
            enable_storage_optimization=True,
            enable_network_optimization=False,
            enable_resource_management=True,
            target_latency_ms=1000.0,
            target_throughput_ops_sec=50.0,
            max_memory_usage_gb=2.0,
            max_cpu_utilization=0.5
        )
        
        return profiles
    
    def initialize(self, profile_name: str = 'production') -> bool:
        """Initialize optimization master with specified profile"""
        if self.initialization_complete:
            logger.warning("Optimization master already initialized")
            return True
        
        logger.info(f"Initializing optimization master with profile: {profile_name}")
        
        # Set optimization profile
        if profile_name not in self.optimization_profiles:
            logger.error(f"Unknown optimization profile: {profile_name}")
            return False
        
        self.current_profile = self.optimization_profiles[profile_name]
        
        # Initialize subsystems based on profile
        success = self._initialize_subsystems()
        
        if success:
            self.initialization_complete = True
            logger.info("Optimization master initialization completed successfully")
        else:
            logger.error("Optimization master initialization failed")
        
        return success
    
    def _initialize_subsystems(self) -> bool:
        """Initialize optimization subsystems based on current profile"""
        profile = self.current_profile
        initialization_results = {}
        
        try:
            # Initialize caching system
            if profile.enable_caching:
                cache = get_cache()
                self.subsystems['cache'] = cache
                self.subsystem_status['cache'] = 'active'
                initialization_results['cache'] = True
                logger.debug("Cache system initialized")
            
            # Initialize memory optimization
            if profile.enable_memory_optimization:
                memory_optimizer = get_memory_optimizer()
                memory_optimizer.start_optimization()
                self.subsystems['memory'] = memory_optimizer
                self.subsystem_status['memory'] = 'active'
                initialization_results['memory'] = True
                logger.debug("Memory optimization initialized")
            
            # Initialize compute optimization
            if profile.enable_compute_optimization:
                compute_optimizer = get_compute_optimizer()
                self.subsystems['compute'] = compute_optimizer
                self.subsystem_status['compute'] = 'active'
                initialization_results['compute'] = True
                logger.debug("Compute optimization initialized")
            
            # Initialize parallel processing
            if profile.enable_parallel_processing:
                parallel_processor = get_parallel_processor()
                parallel_processor.start()
                self.subsystems['parallel'] = parallel_processor
                self.subsystem_status['parallel'] = 'active'
                initialization_results['parallel'] = True
                logger.debug("Parallel processing initialized")
            
            # Initialize auto-scaling
            if profile.enable_auto_scaling:
                autoscaler = get_autoscaler()
                autoscaler.start_auto_scaling()
                self.subsystems['autoscaling'] = autoscaler
                self.subsystem_status['autoscaling'] = 'active'
                initialization_results['autoscaling'] = True
                logger.debug("Auto-scaling initialized")
            
            # Initialize distributed computing
            if profile.enable_distributed_computing:
                try:
                    distributed_coordinator = get_distributed_coordinator()
                    distributed_coordinator.start()
                    self.subsystems['distributed'] = distributed_coordinator
                    self.subsystem_status['distributed'] = 'active'
                    initialization_results['distributed'] = True
                    logger.debug("Distributed computing initialized")
                except Exception as e:
                    logger.warning(f"Distributed computing initialization failed: {e}")
                    initialization_results['distributed'] = False
            
            # Initialize storage optimization
            if profile.enable_storage_optimization:
                storage_optimizer = get_storage_optimizer()
                storage_optimizer.start_optimization()
                self.subsystems['storage'] = storage_optimizer
                self.subsystem_status['storage'] = 'active'
                initialization_results['storage'] = True
                logger.debug("Storage optimization initialized")
            
            # Initialize network optimization
            if profile.enable_network_optimization:
                network_optimizer = get_network_optimizer()
                network_optimizer.start_optimization()
                self.subsystems['network'] = network_optimizer
                self.subsystem_status['network'] = 'active'
                initialization_results['network'] = True
                logger.debug("Network optimization initialized")
            
            # Initialize resource management
            if profile.enable_resource_management:
                resource_manager = get_resource_manager()
                resource_manager.start_management()
                self.subsystems['resource'] = resource_manager
                self.subsystem_status['resource'] = 'active'
                initialization_results['resource'] = True
                logger.debug("Resource management initialized")
            
            # Check initialization results
            failed_subsystems = [name for name, success in initialization_results.items() if not success]
            
            if failed_subsystems:
                logger.warning(f"Some subsystems failed to initialize: {failed_subsystems}")
                return len(failed_subsystems) == 0
            
            logger.info(f"Successfully initialized {len(initialization_results)} subsystems")
            return True
            
        except Exception as e:
            logger.error(f"Subsystem initialization failed: {e}")
            return False
    
    def start_optimization(self) -> bool:
        """Start coordinated optimization"""
        if not self.initialization_complete:
            logger.error("Cannot start optimization - master not initialized")
            return False
        
        if self.optimization_active:
            logger.warning("Optimization already active")
            return True
        
        logger.info("Starting coordinated optimization")
        
        self.optimization_active = True
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(
            target=self._coordination_loop, daemon=True
        )
        self.coordination_thread.start()
        
        # Start performance monitoring
        if self.config.get('performance_monitoring', True):
            self._start_performance_monitoring()
        
        logger.info("Coordinated optimization started successfully")
        return True
    
    def stop_optimization(self) -> bool:
        """Stop coordinated optimization"""
        if not self.optimization_active:
            return True
        
        logger.info("Stopping coordinated optimization")
        
        self.optimization_active = False
        
        # Stop coordination thread
        if self.coordination_thread:
            self.coordination_thread.join(timeout=10.0)
        
        # Stop all subsystems
        self._stop_subsystems()
        
        logger.info("Coordinated optimization stopped")
        return True
    
    def _stop_subsystems(self):
        """Stop all active subsystems"""
        for name, subsystem in self.subsystems.items():
            try:
                if name == 'memory' and hasattr(subsystem, 'stop_optimization'):
                    subsystem.stop_optimization()
                elif name == 'parallel' and hasattr(subsystem, 'stop'):
                    subsystem.stop()
                elif name == 'autoscaling' and hasattr(subsystem, 'stop_auto_scaling'):
                    subsystem.stop_auto_scaling()
                elif name == 'distributed' and hasattr(subsystem, 'stop'):
                    subsystem.stop()
                elif name == 'storage' and hasattr(subsystem, 'stop_optimization'):
                    subsystem.stop_optimization()
                elif name == 'network' and hasattr(subsystem, 'stop_optimization'):
                    subsystem.stop_optimization()
                elif name == 'resource' and hasattr(subsystem, 'stop_management'):
                    subsystem.stop_management()
                
                self.subsystem_status[name] = 'stopped'
                logger.debug(f"Stopped {name} subsystem")
                
            except Exception as e:
                logger.warning(f"Error stopping {name} subsystem: {e}")
                self.subsystem_status[name] = 'error'
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.optimization_active:
            try:
                self._perform_coordination_cycle()
                time.sleep(self.coordination_interval)
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                time.sleep(self.coordination_interval)
    
    def _perform_coordination_cycle(self):
        """Perform coordination cycle"""
        logger.debug("Performing optimization coordination cycle")
        
        # Collect metrics from all subsystems
        system_metrics = self._collect_system_metrics()
        
        # Analyze performance against targets
        performance_analysis = self._analyze_performance(system_metrics)
        
        # Make optimization decisions
        optimization_decisions = self._make_optimization_decisions(performance_analysis)
        
        # Apply optimizations
        self._apply_optimizations(optimization_decisions)
        
        # Record performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': system_metrics,
            'analysis': performance_analysis,
            'decisions': optimization_decisions
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all subsystems"""
        metrics = {
            'timestamp': time.time(),
            'subsystem_metrics': {}
        }
        
        for name, subsystem in self.subsystems.items():
            try:
                if name == 'cache' and hasattr(subsystem, 'get_multi_tier_stats'):
                    metrics['subsystem_metrics'][name] = subsystem.get_multi_tier_stats()
                elif name == 'memory' and hasattr(subsystem, 'get_comprehensive_stats'):
                    metrics['subsystem_metrics'][name] = subsystem.get_comprehensive_stats()
                elif name == 'compute' and hasattr(subsystem, 'get_optimization_stats'):
                    metrics['subsystem_metrics'][name] = subsystem.get_optimization_stats()
                elif name == 'parallel' and hasattr(subsystem, 'get_comprehensive_stats'):
                    metrics['subsystem_metrics'][name] = subsystem.get_comprehensive_stats()
                elif name == 'autoscaling' and hasattr(subsystem, 'get_scaling_status'):
                    metrics['subsystem_metrics'][name] = subsystem.get_scaling_status()
                elif name == 'distributed' and hasattr(subsystem, 'get_cluster_status'):
                    metrics['subsystem_metrics'][name] = subsystem.get_cluster_status()
                elif name == 'storage' and hasattr(subsystem, 'get_comprehensive_stats'):
                    metrics['subsystem_metrics'][name] = subsystem.get_comprehensive_stats()
                elif name == 'network' and hasattr(subsystem, 'get_comprehensive_stats'):
                    metrics['subsystem_metrics'][name] = subsystem.get_comprehensive_stats()
                elif name == 'resource' and hasattr(subsystem, 'get_resource_status'):
                    metrics['subsystem_metrics'][name] = subsystem.get_resource_status()
                    
            except Exception as e:
                logger.warning(f"Failed to collect metrics from {name}: {e}")
                metrics['subsystem_metrics'][name] = {'error': str(e)}
        
        return metrics
    
    def _analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance against targets"""
        profile = self.current_profile
        analysis = {
            'performance_summary': {},
            'target_compliance': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Analyze against performance targets
        subsystem_metrics = metrics.get('subsystem_metrics', {})
        
        # Memory analysis
        if 'memory' in subsystem_metrics:
            memory_stats = subsystem_metrics['memory']
            system_memory = memory_stats.get('system_memory', {})
            current_memory_gb = system_memory.get('process_memory_mb', 0) / 1024
            
            analysis['target_compliance']['memory'] = current_memory_gb <= profile.max_memory_usage_gb
            if current_memory_gb > profile.max_memory_usage_gb:
                analysis['bottlenecks'].append('Memory usage exceeds target')
                analysis['recommendations'].append('Enable aggressive memory optimization')
        
        # CPU analysis
        if 'resource' in subsystem_metrics:
            resource_stats = subsystem_metrics['resource']
            current_usage = resource_stats.get('current_usage', {})
            cpu_utilization = current_usage.get('cpu_utilization', 0)
            
            analysis['target_compliance']['cpu'] = cpu_utilization <= profile.max_cpu_utilization
            if cpu_utilization > profile.max_cpu_utilization:
                analysis['bottlenecks'].append('CPU utilization exceeds target')
                analysis['recommendations'].append('Scale up or optimize compute workload')
        
        # Throughput analysis
        if 'parallel' in subsystem_metrics:
            parallel_stats = subsystem_metrics['parallel']
            performance_metrics = parallel_stats.get('performance_metrics', {})
            throughput = performance_metrics.get('throughput_tasks_per_second', 0)
            
            analysis['target_compliance']['throughput'] = throughput >= profile.target_throughput_ops_sec
            if throughput < profile.target_throughput_ops_sec:
                analysis['bottlenecks'].append('Throughput below target')
                analysis['recommendations'].append('Optimize parallel processing or scale up')
        
        # Overall performance score
        compliance_values = list(analysis['target_compliance'].values())
        if compliance_values:
            analysis['performance_summary']['compliance_score'] = sum(compliance_values) / len(compliance_values)
        else:
            analysis['performance_summary']['compliance_score'] = 0.0
        
        analysis['performance_summary']['bottleneck_count'] = len(analysis['bottlenecks'])
        analysis['performance_summary']['recommendation_count'] = len(analysis['recommendations'])
        
        return analysis
    
    def _make_optimization_decisions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make optimization decisions based on analysis"""
        decisions = {
            'actions': [],
            'priority': 'normal',
            'rationale': []
        }
        
        bottlenecks = analysis.get('bottlenecks', [])
        recommendations = analysis.get('recommendations', [])
        compliance_score = analysis.get('performance_summary', {}).get('compliance_score', 1.0)
        
        # Determine urgency
        if compliance_score < 0.5:
            decisions['priority'] = 'critical'
        elif compliance_score < 0.8:
            decisions['priority'] = 'high'
        
        # Memory optimization decisions
        if 'Memory usage exceeds target' in bottlenecks:
            decisions['actions'].append({
                'subsystem': 'memory',
                'action': 'optimize_memory',
                'parameters': {'aggressive': True}
            })
            decisions['rationale'].append('High memory usage detected')
        
        # CPU optimization decisions
        if 'CPU utilization exceeds target' in bottlenecks:
            if 'autoscaling' in self.subsystems:
                decisions['actions'].append({
                    'subsystem': 'autoscaling',
                    'action': 'scale_up',
                    'parameters': {'reason': 'high_cpu_utilization'}
                })
            decisions['rationale'].append('High CPU utilization detected')
        
        # Throughput optimization decisions
        if 'Throughput below target' in bottlenecks:
            decisions['actions'].append({
                'subsystem': 'parallel',
                'action': 'optimize_workers',
                'parameters': {'increase_workers': True}
            })
            decisions['rationale'].append('Low throughput detected')
        
        # Cache optimization
        if len(bottlenecks) > 2:  # Multiple bottlenecks
            decisions['actions'].append({
                'subsystem': 'cache',
                'action': 'optimize_cache',
                'parameters': {'clear_old_entries': True}
            })
            decisions['rationale'].append('Multiple bottlenecks suggest cache optimization needed')
        
        return decisions
    
    def _apply_optimizations(self, decisions: Dict[str, Any]):
        """Apply optimization decisions"""
        actions = decisions.get('actions', [])
        
        for action in actions:
            try:
                subsystem_name = action['subsystem']
                action_name = action['action']
                parameters = action.get('parameters', {})
                
                if subsystem_name in self.subsystems:
                    subsystem = self.subsystems[subsystem_name]
                    
                    if subsystem_name == 'memory' and action_name == 'optimize_memory':
                        subsystem.optimize_memory()
                    elif subsystem_name == 'cache' and action_name == 'optimize_cache':
                        if parameters.get('clear_old_entries'):
                            subsystem.clear(['old_entries'])
                    # Add more action implementations as needed
                    
                    logger.debug(f"Applied optimization: {subsystem_name}.{action_name}")
                else:
                    logger.warning(f"Subsystem {subsystem_name} not available for optimization")
                    
            except Exception as e:
                logger.error(f"Failed to apply optimization {action}: {e}")
    
    def _start_performance_monitoring(self):
        """Start performance monitoring"""
        # This would start background performance monitoring
        logger.debug("Performance monitoring started")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'master_status': {
                'initialization_complete': self.initialization_complete,
                'optimization_active': self.optimization_active,
                'current_profile': self.current_profile.name if self.current_profile else None,
                'coordination_interval': self.coordination_interval
            },
            'subsystem_status': dict(self.subsystem_status),
            'active_subsystems': list(self.subsystems.keys()),
            'performance_history_length': len(self.performance_history),
            'available_profiles': list(self.optimization_profiles.keys())
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        recent_entry = self.performance_history[-1]
        
        return {
            'last_updated': recent_entry['timestamp'],
            'system_metrics': recent_entry['metrics'],
            'performance_analysis': recent_entry['analysis'],
            'recent_decisions': recent_entry['decisions'],
            'profile_targets': {
                'latency_ms': self.current_profile.target_latency_ms,
                'throughput_ops_sec': self.current_profile.target_throughput_ops_sec,
                'max_memory_gb': self.current_profile.max_memory_usage_gb,
                'max_cpu_utilization': self.current_profile.max_cpu_utilization
            } if self.current_profile else {}
        }
    
    def switch_profile(self, profile_name: str) -> bool:
        """Switch optimization profile"""
        if profile_name not in self.optimization_profiles:
            logger.error(f"Unknown optimization profile: {profile_name}")
            return False
        
        logger.info(f"Switching optimization profile to: {profile_name}")
        
        # Stop current optimization
        was_active = self.optimization_active
        if was_active:
            self.stop_optimization()
        
        # Update profile
        self.current_profile = self.optimization_profiles[profile_name]
        
        # Restart with new profile
        self.initialization_complete = False
        success = self.initialize(profile_name)
        
        if success and was_active:
            self.start_optimization()
        
        return success
    
    def run_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test"""
        logger.info("Running comprehensive performance test")
        
        benchmark_suite = get_benchmark_suite()
        
        # Run quick benchmark to assess current performance
        try:
            from .benchmarking_suite import run_quick_benchmark
            results = run_quick_benchmark()
            
            logger.info("Performance test completed")
            return results
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return {'error': str(e)}
    
    def optimize_for_workload(self, workload_type: str) -> bool:
        """Optimize system for specific workload type"""
        workload_profiles = {
            'training': 'high_throughput',
            'inference': 'low_latency', 
            'batch_processing': 'high_throughput',
            'real_time': 'low_latency',
            'development': 'development',
            'production': 'production'
        }
        
        profile_name = workload_profiles.get(workload_type, 'production')
        
        logger.info(f"Optimizing for {workload_type} workload using {profile_name} profile")
        
        return self.switch_profile(profile_name)


# Global optimization master instance
_global_optimization_master = None

def get_optimization_master(**kwargs) -> OptimizationMaster:
    """Get global optimization master instance"""
    global _global_optimization_master
    
    if _global_optimization_master is None:
        _global_optimization_master = OptimizationMaster(**kwargs)
    
    return _global_optimization_master

def initialize_optimization(profile: str = 'production') -> bool:
    """Initialize optimization system with specified profile"""
    master = get_optimization_master()
    return master.initialize(profile)

def start_optimization() -> bool:
    """Start coordinated optimization"""
    master = get_optimization_master()
    return master.start_optimization()

def stop_optimization() -> bool:
    """Stop coordinated optimization"""
    master = get_optimization_master()
    return master.stop_optimization()

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    master = get_optimization_master()
    return master.get_system_status()

def optimize_for_workload(workload_type: str) -> bool:
    """Optimize system for specific workload"""
    master = get_optimization_master()
    return master.optimize_for_workload(workload_type)