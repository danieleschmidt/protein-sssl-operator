"""
Autonomous Quality Enhancement System
Self-improving quality monitoring and optimization
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import pickle

from .monitoring import MetricsCollector
from .performance_optimizer import PerformanceOptimizer
from .error_handling import ProteinSSLError


@dataclass
class QualityMetric:
    """Quality metric measurement"""
    name: str
    value: float
    threshold: float
    trend: str  # "improving", "degrading", "stable"
    timestamp: float
    context: Dict[str, Any]


@dataclass
class QualityIssue:
    """Identified quality issue"""
    issue_id: str
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "performance", "memory", "accuracy", "reliability"
    description: str
    suggested_action: str
    auto_fixable: bool
    timestamp: float


class AutonomousQualityEnhancer:
    """
    Autonomous system for continuous quality monitoring and improvement
    """
    
    def __init__(self, 
                 metrics_window: int = 1000,
                 quality_threshold: float = 0.95,
                 auto_fix_enabled: bool = True):
        self.metrics_window = metrics_window
        self.quality_threshold = quality_threshold
        self.auto_fix_enabled = auto_fix_enabled
        
        # Quality tracking
        self.quality_history = defaultdict(lambda: deque(maxlen=metrics_window))
        self.issues_log = deque(maxlen=10000)
        self.fixes_applied = []
        
        # Performance monitoring
        self.metrics_collector = MetricsCollector()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Continuous monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Quality baselines
        self.quality_baselines = {
            "prediction_accuracy": 0.90,
            "inference_speed": 1.0,  # seconds per prediction
            "memory_efficiency": 0.85,
            "error_rate": 0.05,
            "cache_hit_rate": 0.80,
            "throughput": 10.0  # predictions per minute
        }
        
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous quality monitoring"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_quality_metrics()
                
                # Analyze quality trends
                issues = self._analyze_quality_trends(current_metrics)
                
                # Apply autonomous fixes if enabled
                if self.auto_fix_enabled:
                    self._apply_autonomous_fixes(issues)
                    
                # Log significant changes
                self._log_quality_changes(current_metrics, issues)
                
                time.sleep(interval)
                
            except Exception as e:
                # Log monitoring errors but continue
                print(f"Quality monitoring error: {e}")
                time.sleep(interval)
                
    def _collect_quality_metrics(self) -> List[QualityMetric]:
        """Collect current quality metrics"""
        metrics = []
        timestamp = time.time()
        
        try:
            # System performance metrics
            system_metrics = self.metrics_collector.get_system_metrics()
            
            # Memory efficiency
            memory_metric = QualityMetric(
                name="memory_efficiency",
                value=1.0 - system_metrics.get("memory_usage_percent", 0) / 100,
                threshold=self.quality_baselines["memory_efficiency"],
                trend=self._calculate_trend("memory_efficiency"),
                timestamp=timestamp,
                context={"memory_mb": system_metrics.get("memory_usage_mb", 0)}
            )
            metrics.append(memory_metric)
            
            # Cache performance
            cache_stats = self.performance_optimizer.get_cache_stats()
            cache_hit_rate = cache_stats.get("hit_rate", 0.0)
            
            cache_metric = QualityMetric(
                name="cache_hit_rate",
                value=cache_hit_rate,
                threshold=self.quality_baselines["cache_hit_rate"],
                trend=self._calculate_trend("cache_hit_rate"),
                timestamp=timestamp,
                context=cache_stats
            )
            metrics.append(cache_metric)
            
            # Error rate (from recent operations)
            error_rate = self._calculate_recent_error_rate()
            error_metric = QualityMetric(
                name="error_rate",
                value=error_rate,
                threshold=self.quality_baselines["error_rate"],
                trend=self._calculate_trend("error_rate"),
                timestamp=timestamp,
                context={"recent_errors": len(self.issues_log)}
            )
            metrics.append(error_metric)
            
            # Update quality history
            for metric in metrics:
                self.quality_history[metric.name].append(metric.value)
                
        except Exception as e:
            raise ProteinSSLError(f"Failed to collect quality metrics: {e}")
            
        return metrics
        
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate quality trend for a metric"""
        history = self.quality_history[metric_name]
        
        if len(history) < 10:
            return "stable"
            
        # Compare recent values to older values
        recent_avg = sum(list(history)[-5:]) / 5
        older_avg = sum(list(history)[-10:-5]) / 5
        
        change_ratio = (recent_avg - older_avg) / max(older_avg, 1e-6)
        
        if change_ratio > 0.05:
            return "improving"
        elif change_ratio < -0.05:
            return "degrading"
        else:
            return "stable"
            
    def _calculate_recent_error_rate(self) -> float:
        """Calculate error rate from recent operations"""
        recent_issues = [
            issue for issue in self.issues_log
            if time.time() - issue.timestamp < 3600  # Last hour
        ]
        
        # Estimate based on issue severity
        error_weight = sum(
            4 if issue.severity == "critical" else
            3 if issue.severity == "high" else
            2 if issue.severity == "medium" else 1
            for issue in recent_issues
        )
        
        # Normalize to rate (assuming 100 operations per hour baseline)
        return min(error_weight / 100.0, 1.0)
        
    def _analyze_quality_trends(self, metrics: List[QualityMetric]) -> List[QualityIssue]:
        """Analyze quality trends and identify issues"""
        issues = []
        
        for metric in metrics:
            # Check threshold violations
            if metric.value < metric.threshold:
                severity = self._determine_severity(metric)
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(metric.name, metric.timestamp),
                    severity=severity,
                    category=self._categorize_metric(metric.name),
                    description=f"{metric.name} below threshold: {metric.value:.3f} < {metric.threshold:.3f}",
                    suggested_action=self._suggest_action(metric),
                    auto_fixable=self._is_auto_fixable(metric.name),
                    timestamp=metric.timestamp
                )
                issues.append(issue)
                
            # Check negative trends
            if metric.trend == "degrading":
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(f"{metric.name}_trend", metric.timestamp),
                    severity="medium",
                    category=self._categorize_metric(metric.name),
                    description=f"{metric.name} showing degrading trend",
                    suggested_action=f"Monitor {metric.name} closely and consider optimization",
                    auto_fixable=False,
                    timestamp=metric.timestamp
                )
                issues.append(issue)
                
        # Add to issues log
        self.issues_log.extend(issues)
        
        return issues
        
    def _determine_severity(self, metric: QualityMetric) -> str:
        """Determine issue severity based on metric"""
        deviation = (metric.threshold - metric.value) / metric.threshold
        
        if deviation > 0.5:
            return "critical"
        elif deviation > 0.3:
            return "high"
        elif deviation > 0.1:
            return "medium"
        else:
            return "low"
            
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric by type"""
        if "memory" in metric_name:
            return "memory"
        elif "cache" in metric_name or "speed" in metric_name:
            return "performance"
        elif "error" in metric_name:
            return "reliability"
        elif "accuracy" in metric_name:
            return "accuracy"
        else:
            return "general"
            
    def _suggest_action(self, metric: QualityMetric) -> str:
        """Suggest action based on metric"""
        actions = {
            "memory_efficiency": "Clear caches, optimize memory usage, restart workers",
            "cache_hit_rate": "Warm up caches, adjust cache size, optimize access patterns",
            "error_rate": "Review error logs, improve error handling, validate inputs",
            "prediction_accuracy": "Retrain model, validate data quality, tune hyperparameters",
            "inference_speed": "Optimize model, enable GPU acceleration, batch processing",
            "throughput": "Scale workers, optimize bottlenecks, load balancing"
        }
        
        return actions.get(metric.name, "Monitor metric and investigate root cause")
        
    def _is_auto_fixable(self, metric_name: str) -> bool:
        """Check if metric issue can be automatically fixed"""
        auto_fixable_metrics = {
            "memory_efficiency",
            "cache_hit_rate"
        }
        
        return metric_name in auto_fixable_metrics
        
    def _apply_autonomous_fixes(self, issues: List[QualityIssue]):
        """Apply autonomous fixes for eligible issues"""
        for issue in issues:
            if not issue.auto_fixable or issue.severity == "low":
                continue
                
            try:
                fix_applied = False
                
                if issue.category == "memory":
                    fix_applied = self._fix_memory_issue(issue)
                elif issue.category == "performance":
                    fix_applied = self._fix_performance_issue(issue)
                    
                if fix_applied:
                    self.fixes_applied.append({
                        "issue_id": issue.issue_id,
                        "fix_type": issue.category,
                        "timestamp": time.time(),
                        "description": f"Auto-fixed {issue.description}"
                    })
                    
            except Exception as e:
                print(f"Failed to auto-fix issue {issue.issue_id}: {e}")
                
    def _fix_memory_issue(self, issue: QualityIssue) -> bool:
        """Attempt to fix memory-related issues"""
        try:
            # Clear performance optimizer caches
            self.performance_optimizer.clear_all_caches()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return True
            
        except Exception:
            return False
            
    def _fix_performance_issue(self, issue: QualityIssue) -> bool:
        """Attempt to fix performance-related issues"""
        try:
            # Optimize cache configurations
            self.performance_optimizer.optimize_cache_sizes()
            
            # Warm up critical caches
            self.performance_optimizer.warm_up_caches()
            
            return True
            
        except Exception:
            return False
            
    def _log_quality_changes(self, metrics: List[QualityMetric], issues: List[QualityIssue]):
        """Log significant quality changes"""
        # Log critical issues
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            print(f"🚨 CRITICAL QUALITY ISSUES DETECTED: {len(critical_issues)}")
            for issue in critical_issues:
                print(f"   - {issue.description}")
                
        # Log significant improvements
        improving_metrics = [m for m in metrics if m.trend == "improving" and m.value > m.threshold * 1.1]
        if improving_metrics:
            print(f"📈 QUALITY IMPROVEMENTS: {len(improving_metrics)}")
            for metric in improving_metrics:
                print(f"   - {metric.name}: {metric.value:.3f}")
                
    def _generate_issue_id(self, base: str, timestamp: float) -> str:
        """Generate unique issue ID"""
        data = f"{base}_{timestamp}".encode()
        return hashlib.md5(data).hexdigest()[:8]
        
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            "timestamp": time.time(),
            "overall_quality_score": self._calculate_overall_quality(),
            "metrics_summary": self._summarize_metrics(),
            "recent_issues": [asdict(issue) for issue in list(self.issues_log)[-10:]],
            "fixes_applied": self.fixes_applied[-10:],
            "trends": self._analyze_trends(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
        
    def _calculate_overall_quality(self) -> float:
        """Calculate overall quality score"""
        if not self.quality_history:
            return 1.0
            
        scores = []
        for metric_name, threshold in self.quality_baselines.items():
            if metric_name in self.quality_history:
                history = self.quality_history[metric_name]
                if history:
                    current_value = history[-1]
                    score = min(current_value / threshold, 1.0)
                    scores.append(score)
                    
        return sum(scores) / len(scores) if scores else 1.0
        
    def _summarize_metrics(self) -> Dict[str, Any]:
        """Summarize current metrics"""
        summary = {}
        
        for metric_name, history in self.quality_history.items():
            if history:
                summary[metric_name] = {
                    "current": history[-1],
                    "average": sum(history) / len(history),
                    "min": min(history),
                    "max": max(history),
                    "trend": self._calculate_trend(metric_name)
                }
                
        return summary
        
    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze overall trends"""
        trends = {}
        
        for metric_name in self.quality_history:
            trends[metric_name] = self._calculate_trend(metric_name)
            
        return trends
        
    def _generate_recommendations(self) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Analyze recent issues
        recent_critical = [
            issue for issue in self.issues_log
            if issue.severity == "critical" and time.time() - issue.timestamp < 3600
        ]
        
        if recent_critical:
            recommendations.append("Address critical quality issues immediately")
            
        # Check for patterns
        degrading_metrics = [
            name for name, trend in self._analyze_trends().items()
            if trend == "degrading"
        ]
        
        if len(degrading_metrics) > 2:
            recommendations.append("Multiple metrics degrading - comprehensive system review needed")
            
        # Memory recommendations
        if "memory_efficiency" in degrading_metrics:
            recommendations.append("Optimize memory usage - consider increasing cache limits")
            
        # Performance recommendations
        if "cache_hit_rate" in degrading_metrics:
            recommendations.append("Review cache configuration and access patterns")
            
        return recommendations
        
    def save_quality_state(self, filepath: Path):
        """Save quality monitoring state"""
        state = {
            "quality_history": dict(self.quality_history),
            "issues_log": list(self.issues_log),
            "fixes_applied": self.fixes_applied,
            "quality_baselines": self.quality_baselines
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
            
    def load_quality_state(self, filepath: Path):
        """Load quality monitoring state"""
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)
                
            self.quality_history = defaultdict(lambda: deque(maxlen=self.metrics_window))
            for name, history in state["quality_history"].items():
                self.quality_history[name] = deque(history, maxlen=self.metrics_window)
                
            self.issues_log = deque(state["issues_log"], maxlen=10000)
            self.fixes_applied = state["fixes_applied"]
            self.quality_baselines.update(state["quality_baselines"])
            
        except Exception as e:
            print(f"Failed to load quality state: {e}")


# Global instance for easy access
_global_quality_enhancer = None

def get_quality_enhancer(**kwargs) -> AutonomousQualityEnhancer:
    """Get global quality enhancer instance"""
    global _global_quality_enhancer
    
    if _global_quality_enhancer is None:
        _global_quality_enhancer = AutonomousQualityEnhancer(**kwargs)
        
    return _global_quality_enhancer


def start_autonomous_quality_monitoring(interval: float = 30.0, **kwargs):
    """Start autonomous quality monitoring"""
    enhancer = get_quality_enhancer(**kwargs)
    enhancer.start_monitoring(interval=interval)
    return enhancer


def get_quality_report() -> Dict[str, Any]:
    """Get current quality report"""
    enhancer = get_quality_enhancer()
    return enhancer.get_quality_report()