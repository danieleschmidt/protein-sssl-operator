"""
Advanced Auto-Scaling Infrastructure for Protein-SSL Operator
Implements dynamic resource allocation, predictive scaling, and intelligent resource management
"""

import time
import threading
import asyncio
import json
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import psutil
import torch
import kubernetes
from kubernetes import client, config
import requests
import subprocess
from pathlib import Path

from .logging_config import setup_logging
from .monitoring import MetricsCollector
from .parallel_processing import get_parallel_processor
from .memory_optimization import get_memory_optimizer
from .compute_optimization import get_compute_optimizer

logger = setup_logging(__name__)


class ScalingTrigger(Enum):
    """Scaling trigger types"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"


class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ScalingMetric:
    """Scaling metric configuration"""
    name: str
    trigger_type: ScalingTrigger
    threshold_up: float
    threshold_down: float
    evaluation_window: int  # seconds
    cooldown_period: int   # seconds
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ScalingPolicy:
    """Scaling policy configuration"""
    name: str
    min_replicas: int
    max_replicas: int
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    max_scale_up_step: int = 5
    max_scale_down_step: int = 2
    stabilization_window: int = 300  # seconds
    metrics: List[ScalingMetric] = None


@dataclass
class ResourceAllocation:
    """Resource allocation specification"""
    cpu_cores: float
    memory_mb: int
    gpu_count: int = 0
    storage_gb: int = 10
    network_bandwidth_mbps: int = 1000
    custom_resources: Dict[str, Any] = None


@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: float
    trigger: ScalingTrigger
    direction: ScalingDirection
    old_replicas: int
    new_replicas: int
    reason: str
    success: bool
    duration: float = 0.0
    resource_allocation: ResourceAllocation = None


class PredictiveScaler:
    """Predictive scaling based on historical patterns and forecasting"""
    
    def __init__(self, history_window: int = 86400):  # 24 hours
        self.history_window = history_window
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_models = {}
        self.prediction_accuracy = defaultdict(float)
        
    def add_metric_data(self, metric_name: str, value: float, timestamp: float = None):
        """Add metric data point"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metric_history[metric_name].append((timestamp, value))
    
    def predict_metric(self, metric_name: str, horizon_seconds: int = 3600) -> Tuple[float, float]:
        """Predict metric value for given horizon with confidence"""
        history = list(self.metric_history[metric_name])
        
        if len(history) < 10:
            # Not enough data for prediction
            current_value = history[-1][1] if history else 0.0
            return current_value, 0.5
        
        # Extract time series data
        timestamps = np.array([h[0] for h in history])
        values = np.array([h[1] for h in history])
        
        # Simple forecasting methods
        prediction, confidence = self._forecast_time_series(
            timestamps, values, horizon_seconds
        )
        
        return prediction, confidence
    
    def _forecast_time_series(self, timestamps: np.ndarray, values: np.ndarray, 
                            horizon: int) -> Tuple[float, float]:
        """Forecast time series using multiple methods"""
        current_time = time.time()
        
        # Method 1: Linear trend
        trend_prediction, trend_confidence = self._linear_trend_forecast(
            timestamps, values, current_time + horizon
        )
        
        # Method 2: Seasonal decomposition
        seasonal_prediction, seasonal_confidence = self._seasonal_forecast(
            timestamps, values, current_time + horizon
        )
        
        # Method 3: Exponential smoothing
        smooth_prediction, smooth_confidence = self._exponential_smoothing_forecast(
            values, horizon
        )
        
        # Ensemble prediction
        predictions = [
            (trend_prediction, trend_confidence * 0.3),
            (seasonal_prediction, seasonal_confidence * 0.4),
            (smooth_prediction, smooth_confidence * 0.3)
        ]
        
        # Weighted average
        total_weight = sum(weight for _, weight in predictions)
        if total_weight > 0:
            prediction = sum(pred * weight for pred, weight in predictions) / total_weight
            confidence = sum(weight for _, weight in predictions) / len(predictions)
        else:
            prediction = values[-1] if len(values) > 0 else 0.0
            confidence = 0.5
        
        return prediction, min(confidence, 0.95)
    
    def _linear_trend_forecast(self, timestamps: np.ndarray, values: np.ndarray, 
                             target_time: float) -> Tuple[float, float]:
        """Linear trend forecasting"""
        if len(values) < 3:
            return values[-1] if len(values) > 0 else 0.0, 0.3
        
        # Fit linear trend
        coeffs = np.polyfit(timestamps, values, 1)
        trend = coeffs[0]
        intercept = coeffs[1]
        
        # Predict
        prediction = trend * target_time + intercept
        
        # Calculate confidence based on trend stability
        residuals = values - (trend * timestamps + intercept)
        mse = np.mean(residuals ** 2)
        confidence = max(0.1, 1.0 - min(mse / np.var(values), 1.0))
        
        return prediction, confidence
    
    def _seasonal_forecast(self, timestamps: np.ndarray, values: np.ndarray, 
                         target_time: float) -> Tuple[float, float]:
        """Simple seasonal forecasting"""
        if len(values) < 24:  # Need at least 24 data points
            return values[-1] if len(values) > 0 else 0.0, 0.3
        
        # Detect daily seasonality (86400 seconds)
        seasonal_period = 86400
        current_time = timestamps[-1]
        
        # Find similar time in the past
        time_of_day = target_time % seasonal_period
        similar_times = []
        
        for i, ts in enumerate(timestamps):
            ts_time_of_day = ts % seasonal_period
            if abs(ts_time_of_day - time_of_day) < 3600:  # Within 1 hour
                similar_times.append(values[i])
        
        if similar_times:
            prediction = np.mean(similar_times)
            confidence = min(0.8, len(similar_times) / 10.0)
        else:
            prediction = values[-1]
            confidence = 0.3
        
        return prediction, confidence
    
    def _exponential_smoothing_forecast(self, values: np.ndarray, 
                                      horizon: int) -> Tuple[float, float]:
        """Exponential smoothing forecast"""
        if len(values) < 3:
            return values[-1] if len(values) > 0 else 0.0, 0.4
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        smoothed = values[0]
        
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Prediction is the last smoothed value
        prediction = smoothed
        
        # Confidence based on recent variance
        recent_values = values[-min(10, len(values)):]
        variance = np.var(recent_values)
        confidence = max(0.2, 1.0 - min(variance / np.mean(recent_values), 1.0))
        
        return prediction, confidence
    
    def should_preemptive_scale(self, metric_name: str, current_threshold: float,
                              lookahead_minutes: int = 15) -> Tuple[bool, float, str]:
        """Determine if preemptive scaling is needed"""
        prediction, confidence = self.predict_metric(
            metric_name, lookahead_minutes * 60
        )
        
        # Decision criteria
        threshold_exceeded = prediction > current_threshold
        confidence_sufficient = confidence > 0.6
        
        if threshold_exceeded and confidence_sufficient:
            reason = (f"Predicted {metric_name} will reach {prediction:.2f} "
                     f"(threshold: {current_threshold}) in {lookahead_minutes} minutes "
                     f"with {confidence:.1%} confidence")
            return True, prediction, reason
        
        return False, prediction, "No preemptive scaling needed"


class KubernetesScaler:
    """Kubernetes-native auto-scaling implementation"""
    
    def __init__(self, namespace: str = "protein-sssl"):
        self.namespace = namespace
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        self.k8s_autoscaling_v2 = None
        
        try:
            # Try to load in-cluster config first
            config.load_incluster_config()
        except config.ConfigException:
            try:
                # Fallback to local kubeconfig
                config.load_kube_config()
            except config.ConfigException:
                logger.warning("Could not load Kubernetes config, scaling disabled")
                return
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_autoscaling_v2 = client.AutoscalingV2Api()
    
    def scale_deployment(self, deployment_name: str, target_replicas: int) -> bool:
        """Scale Kubernetes deployment"""
        if not self.k8s_apps_v1:
            logger.warning("Kubernetes API not available")
            return False
        
        try:
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = target_replicas
            
            # Apply update
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment {deployment_name} to {target_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    def get_current_replicas(self, deployment_name: str) -> Optional[int]:
        """Get current replica count for deployment"""
        if not self.k8s_apps_v1:
            return None
        
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            return deployment.status.replicas
        except Exception as e:
            logger.warning(f"Failed to get replica count for {deployment_name}: {e}")
            return None
    
    def create_hpa(self, deployment_name: str, scaling_policy: ScalingPolicy) -> bool:
        """Create Horizontal Pod Autoscaler"""
        if not self.k8s_autoscaling_v2:
            return False
        
        hpa_name = f"{deployment_name}-hpa"
        
        # Build HPA spec
        hpa_spec = client.V2HorizontalPodAutoscalerSpec(
            scale_target_ref=client.V2CrossVersionObjectReference(
                api_version="apps/v1",
                kind="Deployment",
                name=deployment_name
            ),
            min_replicas=scaling_policy.min_replicas,
            max_replicas=scaling_policy.max_replicas,
            metrics=self._build_hpa_metrics(scaling_policy.metrics),
            behavior=self._build_hpa_behavior(scaling_policy)
        )
        
        hpa = client.V2HorizontalPodAutoscaler(
            metadata=client.V1ObjectMeta(
                name=hpa_name,
                namespace=self.namespace
            ),
            spec=hpa_spec
        )
        
        try:
            self.k8s_autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            logger.info(f"Created HPA {hpa_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create HPA {hpa_name}: {e}")
            return False
    
    def _build_hpa_metrics(self, metrics: List[ScalingMetric]) -> List[client.V2MetricSpec]:
        """Build HPA metrics from scaling metrics"""
        hpa_metrics = []
        
        for metric in metrics:
            if not metric.enabled:
                continue
            
            if metric.trigger_type == ScalingTrigger.CPU_UTILIZATION:
                hpa_metric = client.V2MetricSpec(
                    type="Resource",
                    resource=client.V2ResourceMetricSource(
                        name="cpu",
                        target=client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=int(metric.threshold_up * 100)
                        )
                    )
                )
                hpa_metrics.append(hpa_metric)
            
            elif metric.trigger_type == ScalingTrigger.MEMORY_UTILIZATION:
                hpa_metric = client.V2MetricSpec(
                    type="Resource",
                    resource=client.V2ResourceMetricSource(
                        name="memory",
                        target=client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=int(metric.threshold_up * 100)
                        )
                    )
                )
                hpa_metrics.append(hpa_metric)
        
        return hpa_metrics
    
    def _build_hpa_behavior(self, policy: ScalingPolicy) -> client.V2HorizontalPodAutoscalerBehavior:
        """Build HPA behavior from scaling policy"""
        scale_up_behavior = client.V2HPAScalingRules(
            stabilization_window_seconds=policy.stabilization_window,
            policies=[
                client.V2HPAScalingPolicy(
                    type="Percent",
                    value=int((policy.scale_up_factor - 1) * 100),
                    period_seconds=60
                ),
                client.V2HPAScalingPolicy(
                    type="Pods",
                    value=policy.max_scale_up_step,
                    period_seconds=60
                )
            ]
        )
        
        scale_down_behavior = client.V2HPAScalingRules(
            stabilization_window_seconds=policy.stabilization_window * 2,
            policies=[
                client.V2HPAScalingPolicy(
                    type="Percent",
                    value=int((1 - policy.scale_down_factor) * 100),
                    period_seconds=120
                ),
                client.V2HPAScalingPolicy(
                    type="Pods",
                    value=policy.max_scale_down_step,
                    period_seconds=120
                )
            ]
        )
        
        return client.V2HorizontalPodAutoscalerBehavior(
            scale_up=scale_up_behavior,
            scale_down=scale_down_behavior
        )


class AdvancedAutoScaler:
    """Advanced auto-scaling coordinator with predictive capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Components
        self.predictive_scaler = PredictiveScaler()
        self.k8s_scaler = KubernetesScaler()
        self.metrics_collector = MetricsCollector()
        
        # Scaling state
        self.current_replicas = {}
        self.scaling_events = deque(maxlen=1000)
        self.last_scaling_time = {}
        self.scaling_policies = {}
        
        # Control
        self.scaling_enabled = True
        self.scaling_thread = None
        self.scaling_active = False
        self.scaling_interval = 30.0  # seconds
        
        # Performance tracking
        self.scaling_effectiveness = defaultdict(list)
        
        # Load scaling policies from config
        self._load_scaling_policies()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load auto-scaling configuration"""
        default_config = {
            'scaling_interval': 30,
            'predictive_enabled': True,
            'kubernetes_enabled': True,
            'metrics_retention': 86400,  # 24 hours
            'default_policy': {
                'min_replicas': 1,
                'max_replicas': 10,
                'scale_up_factor': 1.5,
                'scale_down_factor': 0.7,
                'stabilization_window': 300
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"Loaded auto-scaling config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _load_scaling_policies(self):
        """Load scaling policies from configuration"""
        policies_config = self.config.get('policies', {})
        
        for policy_name, policy_config in policies_config.items():
            metrics = []
            
            for metric_config in policy_config.get('metrics', []):
                metric = ScalingMetric(
                    name=metric_config['name'],
                    trigger_type=ScalingTrigger(metric_config['trigger_type']),
                    threshold_up=metric_config['threshold_up'],
                    threshold_down=metric_config['threshold_down'],
                    evaluation_window=metric_config.get('evaluation_window', 300),
                    cooldown_period=metric_config.get('cooldown_period', 300),
                    weight=metric_config.get('weight', 1.0),
                    enabled=metric_config.get('enabled', True)
                )
                metrics.append(metric)
            
            policy = ScalingPolicy(
                name=policy_name,
                min_replicas=policy_config.get('min_replicas', 1),
                max_replicas=policy_config.get('max_replicas', 10),
                scale_up_factor=policy_config.get('scale_up_factor', 1.5),
                scale_down_factor=policy_config.get('scale_down_factor', 0.7),
                max_scale_up_step=policy_config.get('max_scale_up_step', 5),
                max_scale_down_step=policy_config.get('max_scale_down_step', 2),
                stabilization_window=policy_config.get('stabilization_window', 300),
                metrics=metrics
            )
            
            self.scaling_policies[policy_name] = policy
            logger.info(f"Loaded scaling policy: {policy_name}")
    
    def start_auto_scaling(self):
        """Start auto-scaling system"""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Advanced auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling system"""
        self.scaling_active = False
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        logger.info("Advanced auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling loop"""
        while self.scaling_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                
                # Update predictive models
                self._update_predictive_models(current_metrics)
                
                # Evaluate scaling decisions for each policy
                for policy_name, policy in self.scaling_policies.items():
                    self._evaluate_scaling_decision(policy_name, policy, current_metrics)
                
                # Sleep until next evaluation
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(self.scaling_interval)
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {}
        
        # System metrics
        metrics['cpu_utilization'] = psutil.cpu_percent() / 100.0
        metrics['memory_utilization'] = psutil.virtual_memory().percent / 100.0
        
        # GPU metrics if available
        if torch.cuda.is_available():
            try:
                metrics['gpu_utilization'] = torch.cuda.utilization() / 100.0
                metrics['gpu_memory_utilization'] = (
                    torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                )
            except Exception:
                metrics['gpu_utilization'] = 0.0
                metrics['gpu_memory_utilization'] = 0.0
        
        # Parallel processing metrics
        try:
            processor = get_parallel_processor()
            stats = processor.get_comprehensive_stats()
            
            metrics['queue_length'] = stats['task_statistics']['pending']
            metrics['processing_rate'] = stats['performance_metrics']['throughput_tasks_per_second']
            
            # Calculate average response time
            if stats['performance_metrics']['avg_task_completion_time'] > 0:
                metrics['response_time'] = stats['performance_metrics']['avg_task_completion_time']
            else:
                metrics['response_time'] = 1.0
                
        except Exception as e:
            logger.debug(f"Could not collect parallel processing metrics: {e}")
            metrics['queue_length'] = 0
            metrics['processing_rate'] = 1.0
            metrics['response_time'] = 1.0
        
        # Request rate (simplified)
        metrics['request_rate'] = metrics.get('processing_rate', 1.0)
        
        return metrics
    
    def _update_predictive_models(self, current_metrics: Dict[str, float]):
        """Update predictive scaling models"""
        if not self.config.get('predictive_enabled', True):
            return
        
        current_time = time.time()
        
        for metric_name, value in current_metrics.items():
            self.predictive_scaler.add_metric_data(metric_name, value, current_time)
    
    def _evaluate_scaling_decision(self, policy_name: str, policy: ScalingPolicy, 
                                 current_metrics: Dict[str, float]):
        """Evaluate scaling decision for a policy"""
        if not self.scaling_enabled:
            return
        
        # Get current replica count
        current_replicas = self.current_replicas.get(policy_name, policy.min_replicas)
        
        # Check cooldown period
        last_scaling = self.last_scaling_time.get(policy_name, 0)
        if time.time() - last_scaling < policy.stabilization_window:
            return
        
        # Evaluate each metric
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0
        scaling_reasons = []
        
        for metric in policy.metrics:
            if not metric.enabled:
                continue
            
            metric_value = current_metrics.get(metric.trigger_type.value, 0.0)
            total_weight += metric.weight
            
            # Check thresholds
            if metric_value > metric.threshold_up:
                scale_up_votes += metric.weight
                scaling_reasons.append(
                    f"{metric.name}: {metric_value:.2f} > {metric.threshold_up:.2f}"
                )
            elif metric_value < metric.threshold_down:
                scale_down_votes += metric.weight
                scaling_reasons.append(
                    f"{metric.name}: {metric_value:.2f} < {metric.threshold_down:.2f}"
                )
            
            # Predictive scaling check
            if self.config.get('predictive_enabled', True):
                should_scale, prediction, reason = self.predictive_scaler.should_preemptive_scale(
                    metric.trigger_type.value, metric.threshold_up
                )
                if should_scale:
                    scale_up_votes += metric.weight * 0.5  # Reduced weight for predictions
                    scaling_reasons.append(f"Predictive: {reason}")
        
        # Make scaling decision
        if total_weight == 0:
            return
        
        scale_up_ratio = scale_up_votes / total_weight
        scale_down_ratio = scale_down_votes / total_weight
        
        # Determine scaling action
        target_replicas = current_replicas
        direction = ScalingDirection.NONE
        
        if scale_up_ratio > 0.5 and current_replicas < policy.max_replicas:
            # Scale up
            scale_factor = min(policy.scale_up_factor, 1.0 + scale_up_ratio)
            target_replicas = min(
                policy.max_replicas,
                max(current_replicas + 1, int(current_replicas * scale_factor))
            )
            target_replicas = min(target_replicas, current_replicas + policy.max_scale_up_step)
            direction = ScalingDirection.UP
            
        elif scale_down_ratio > 0.5 and current_replicas > policy.min_replicas:
            # Scale down
            scale_factor = max(policy.scale_down_factor, 1.0 - scale_down_ratio)
            target_replicas = max(
                policy.min_replicas,
                max(current_replicas - 1, int(current_replicas * scale_factor))
            )
            target_replicas = max(target_replicas, current_replicas - policy.max_scale_down_step)
            direction = ScalingDirection.DOWN
        
        # Execute scaling if needed
        if target_replicas != current_replicas:
            success = self._execute_scaling(
                policy_name, current_replicas, target_replicas, 
                direction, scaling_reasons
            )
            
            if success:
                self.current_replicas[policy_name] = target_replicas
                self.last_scaling_time[policy_name] = time.time()
    
    def _execute_scaling(self, policy_name: str, current_replicas: int, 
                        target_replicas: int, direction: ScalingDirection,
                        reasons: List[str]) -> bool:
        """Execute scaling action"""
        start_time = time.time()
        
        # Determine deployment name (simplified)
        deployment_name = policy_name.replace('_', '-')
        
        # Try Kubernetes scaling first
        success = False
        if self.config.get('kubernetes_enabled', True):
            success = self.k8s_scaler.scale_deployment(deployment_name, target_replicas)
        
        # Fallback to local scaling (for development/testing)
        if not success:
            success = self._execute_local_scaling(policy_name, target_replicas)
        
        # Record scaling event
        scaling_event = ScalingEvent(
            timestamp=start_time,
            trigger=ScalingTrigger.CUSTOM_METRIC,  # Simplified
            direction=direction,
            old_replicas=current_replicas,
            new_replicas=target_replicas,
            reason="; ".join(reasons),
            success=success,
            duration=time.time() - start_time
        )
        
        self.scaling_events.append(scaling_event)
        
        if success:
            logger.info(
                f"Scaled {policy_name}: {current_replicas} -> {target_replicas} "
                f"({direction.value}). Reason: {scaling_event.reason}"
            )
        else:
            logger.error(f"Failed to scale {policy_name}")
        
        return success
    
    def _execute_local_scaling(self, policy_name: str, target_replicas: int) -> bool:
        """Execute local scaling (for development/testing)"""
        try:
            # This would interface with local resource management
            # For now, just simulate scaling
            logger.info(f"Local scaling simulation: {policy_name} -> {target_replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"Local scaling failed: {e}")
            return False
    
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add or update scaling policy"""
        self.scaling_policies[policy.name] = policy
        
        # Create Kubernetes HPA if enabled
        if self.config.get('kubernetes_enabled', True):
            deployment_name = policy.name.replace('_', '-')
            self.k8s_scaler.create_hpa(deployment_name, policy)
        
        logger.info(f"Added scaling policy: {policy.name}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status"""
        recent_events = list(self.scaling_events)[-10:]
        
        return {
            'scaling_enabled': self.scaling_enabled,
            'scaling_active': self.scaling_active,
            'policies_count': len(self.scaling_policies),
            'current_replicas': dict(self.current_replicas),
            'recent_events': [asdict(event) for event in recent_events],
            'predictive_enabled': self.config.get('predictive_enabled', True),
            'kubernetes_enabled': self.config.get('kubernetes_enabled', True),
            'scaling_effectiveness': dict(self.scaling_effectiveness),
            'next_evaluation_in': self.scaling_interval - (time.time() % self.scaling_interval)
        }
    
    def simulate_load_scenario(self, scenario_name: str, duration_minutes: int = 60):
        """Simulate load scenario for testing scaling behavior"""
        logger.info(f"Starting load simulation: {scenario_name} for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        scenario_events = []
        
        while time.time() < end_time and self.scaling_active:
            # Generate simulated metrics based on scenario
            simulated_metrics = self._generate_scenario_metrics(
                scenario_name, time.time() - start_time
            )
            
            # Update predictive models with simulated data
            self._update_predictive_models(simulated_metrics)
            
            # Force evaluation with simulated metrics
            for policy_name, policy in self.scaling_policies.items():
                self._evaluate_scaling_decision(policy_name, policy, simulated_metrics)
            
            # Record scenario event
            scenario_events.append({
                'timestamp': time.time(),
                'metrics': simulated_metrics,
                'replicas': dict(self.current_replicas)
            })
            
            time.sleep(30)  # Simulate every 30 seconds
        
        logger.info(f"Load simulation {scenario_name} completed")
        return scenario_events
    
    def _generate_scenario_metrics(self, scenario_name: str, elapsed_seconds: float) -> Dict[str, float]:
        """Generate simulated metrics for load scenarios"""
        base_metrics = {
            'cpu_utilization': 0.3,
            'memory_utilization': 0.4,
            'gpu_utilization': 0.2,
            'request_rate': 10.0,
            'queue_length': 5,
            'response_time': 1.0
        }
        
        if scenario_name == 'spike_load':
            # Sudden spike at 25% through scenario
            if 0.25 * 3600 <= elapsed_seconds <= 0.75 * 3600:
                multiplier = 3.0
            else:
                multiplier = 1.0
        
        elif scenario_name == 'gradual_increase':
            # Gradual increase over time
            progress = elapsed_seconds / 3600
            multiplier = 1.0 + (progress * 2.0)
        
        elif scenario_name == 'oscillating_load':
            # Oscillating load pattern
            import math
            multiplier = 1.0 + 1.5 * math.sin(elapsed_seconds / 600)  # 10-minute cycles
        
        else:  # steady_load
            multiplier = 1.0
        
        # Apply multiplier to relevant metrics
        simulated_metrics = {}
        for metric, base_value in base_metrics.items():
            if metric in ['cpu_utilization', 'memory_utilization', 'gpu_utilization']:
                simulated_metrics[metric] = min(0.95, base_value * multiplier)
            else:
                simulated_metrics[metric] = base_value * multiplier
        
        return simulated_metrics


# Global auto-scaler instance
_global_autoscaler = None

def get_autoscaler(config_path: Optional[str] = None) -> AdvancedAutoScaler:
    """Get global auto-scaler instance"""
    global _global_autoscaler
    
    if _global_autoscaler is None:
        _global_autoscaler = AdvancedAutoScaler(config_path)
    
    return _global_autoscaler

def start_auto_scaling(config_path: Optional[str] = None) -> AdvancedAutoScaler:
    """Start global auto-scaling"""
    autoscaler = get_autoscaler(config_path)
    autoscaler.start_auto_scaling()
    return autoscaler

def create_scaling_policy(name: str, min_replicas: int = 1, max_replicas: int = 10,
                         **kwargs) -> ScalingPolicy:
    """Create a scaling policy with default CPU and memory metrics"""
    
    default_metrics = [
        ScalingMetric(
            name="cpu_utilization",
            trigger_type=ScalingTrigger.CPU_UTILIZATION,
            threshold_up=0.7,
            threshold_down=0.3,
            evaluation_window=300,
            cooldown_period=300,
            weight=1.0
        ),
        ScalingMetric(
            name="memory_utilization",
            trigger_type=ScalingTrigger.MEMORY_UTILIZATION,
            threshold_up=0.8,
            threshold_down=0.4,
            evaluation_window=300,
            cooldown_period=300,
            weight=1.0
        )
    ]
    
    return ScalingPolicy(
        name=name,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        metrics=kwargs.get('metrics', default_metrics),
        **{k: v for k, v in kwargs.items() if k != 'metrics'}
    )