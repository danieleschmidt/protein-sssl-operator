"""
Autonomous Learning Framework for Protein Structure Prediction
Advanced SDLC Enhancement - Self-Improving Systems
"""
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import hashlib


@dataclass
class LearningMetrics:
    """Metrics for autonomous learning tracking"""
    accuracy_trend: List[float]
    confidence_calibration: float
    processing_speed_trend: List[float]
    model_uncertainty: float
    adaptive_learning_rate: float
    knowledge_retention: float
    prediction_consistency: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptationDecision:
    """Decision made by autonomous adaptation system"""
    decision_type: str
    confidence: float
    reasoning: str
    parameters_changed: Dict[str, Any]
    expected_improvement: float
    timestamp: float


class AutonomousLearningFramework:
    """
    Framework for autonomous learning and adaptation in protein structure prediction
    """
    
    def __init__(self, 
                 adaptation_threshold: float = 0.85,
                 learning_memory_size: int = 10000,
                 adaptation_frequency: int = 100):
        """
        Initialize autonomous learning framework
        
        Args:
            adaptation_threshold: Threshold for triggering adaptations
            learning_memory_size: Size of learning history to maintain
            adaptation_frequency: How often to evaluate for adaptations
        """
        self.adaptation_threshold = adaptation_threshold
        self.learning_memory_size = learning_memory_size
        self.adaptation_frequency = adaptation_frequency
        
        # Learning history storage
        self.prediction_history = deque(maxlen=learning_memory_size)
        self.performance_history = deque(maxlen=1000)
        self.adaptation_history = []
        
        # Current model state
        self.current_parameters = self._initialize_parameters()
        self.learning_metrics = self._initialize_metrics()
        
        # Adaptation state
        self.predictions_since_adaptation = 0
        self.current_adaptation_phase = "exploration"
        
        # Knowledge base
        self.sequence_knowledge_base = {}
        self.pattern_library = {}
        
    def _initialize_parameters(self) -> Dict[str, Any]:
        """Initialize adaptive parameters"""
        return {
            'confidence_calibration_factor': 1.0,
            'uncertainty_threshold': 0.7,
            'energy_weight': 1.0,
            'dynamics_sensitivity': 1.0,
            'function_prediction_threshold': 0.8,
            'batch_processing_size': 32,
            'quality_gate_strictness': 0.85
        }
    
    def _initialize_metrics(self) -> LearningMetrics:
        """Initialize learning metrics"""
        return LearningMetrics(
            accuracy_trend=[0.8],
            confidence_calibration=0.0,
            processing_speed_trend=[1.0],
            model_uncertainty=0.5,
            adaptive_learning_rate=0.01,
            knowledge_retention=1.0,
            prediction_consistency=0.8
        )
    
    def record_prediction(self, 
                         sequence: str,
                         prediction_result: Dict[str, Any],
                         ground_truth: Optional[Dict[str, Any]] = None):
        """
        Record a prediction for autonomous learning
        
        Args:
            sequence: Input protein sequence
            prediction_result: Prediction output
            ground_truth: Optional ground truth for supervised learning
        """
        
        # Create prediction record
        prediction_record = {
            'timestamp': time.time(),
            'sequence': sequence,
            'sequence_hash': hashlib.md5(sequence.encode()).hexdigest()[:8],
            'sequence_length': len(sequence),
            'prediction': prediction_result,
            'ground_truth': ground_truth,
            'model_parameters': dict(self.current_parameters),
            'processing_time': prediction_result.get('processing_time', 0)
        }
        
        # Store in history
        self.prediction_history.append(prediction_record)
        
        # Update knowledge base
        self._update_knowledge_base(sequence, prediction_result)
        
        # Increment prediction counter
        self.predictions_since_adaptation += 1
        
        # Check if adaptation is needed
        if self.predictions_since_adaptation >= self.adaptation_frequency:
            self._evaluate_adaptation_needs()
            self.predictions_since_adaptation = 0
    
    def _update_knowledge_base(self, sequence: str, prediction: Dict[str, Any]):
        """Update sequence knowledge base"""
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:8]
        
        # Store sequence-specific knowledge
        self.sequence_knowledge_base[seq_hash] = {
            'sequence': sequence,
            'length': len(sequence),
            'confidence': prediction.get('confidence', []),
            'plddt': prediction.get('plddt', 0),
            'energy': prediction.get('energy', 0),
            'last_predicted': time.time(),
            'prediction_count': self.sequence_knowledge_base.get(seq_hash, {}).get('prediction_count', 0) + 1
        }
        
        # Extract and store patterns
        self._extract_patterns(sequence, prediction)
    
    def _extract_patterns(self, sequence: str, prediction: Dict[str, Any]):
        """Extract structural patterns for learning"""
        
        # Extract local patterns (3-mers, 5-mers)
        for k in [3, 5]:
            for i in range(len(sequence) - k + 1):
                pattern = sequence[i:i+k]
                
                if pattern not in self.pattern_library:
                    self.pattern_library[pattern] = {
                        'occurrences': 0,
                        'avg_confidence': 0.0,
                        'avg_energy_contribution': 0.0,
                        'structural_context': []
                    }
                
                # Update pattern statistics
                pattern_data = self.pattern_library[pattern]
                pattern_data['occurrences'] += 1
                
                # Update average confidence for this pattern
                if 'confidence' in prediction and len(prediction['confidence']) > i:
                    local_conf = np.mean(prediction['confidence'][i:i+k])
                    pattern_data['avg_confidence'] = (
                        (pattern_data['avg_confidence'] * (pattern_data['occurrences'] - 1) + local_conf) /
                        pattern_data['occurrences']
                    )
    
    def _evaluate_adaptation_needs(self):
        """Evaluate if model adaptation is needed"""
        if len(self.prediction_history) < 50:  # Need sufficient history
            return
        
        # Calculate recent performance metrics
        recent_predictions = list(self.prediction_history)[-50:]
        recent_performance = self._calculate_performance_metrics(recent_predictions)
        
        # Update learning metrics
        self.learning_metrics = self._update_learning_metrics(recent_performance)
        
        # Determine if adaptation is needed
        adaptation_needed = self._should_adapt(recent_performance)
        
        if adaptation_needed:
            decision = self._make_adaptation_decision(recent_performance)
            self._apply_adaptation(decision)
    
    def _calculate_performance_metrics(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from recent predictions"""
        
        processing_times = [p['processing_time'] for p in predictions if p['processing_time'] > 0]
        confidences = []
        plddts = []
        energies = []
        
        for p in predictions:
            pred = p['prediction']
            if 'confidence' in pred:
                confidences.extend(pred['confidence'] if isinstance(pred['confidence'], list) 
                                 else [pred['confidence']])
            if 'plddt' in pred:
                plddts.append(pred['plddt'])
            if 'energy' in pred:
                energies.append(pred['energy'])
        
        metrics = {
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'avg_plddt': np.mean(plddts) if plddts else 0,
            'avg_energy': np.mean(energies) if energies else 0,
            'prediction_consistency': self._calculate_consistency(predictions)
        }
        
        return metrics
    
    def _calculate_consistency(self, predictions: List[Dict]) -> float:
        """Calculate prediction consistency across similar sequences"""
        if len(predictions) < 10:
            return 0.8  # Default consistency
        
        # Group similar sequences (by length ranges)
        length_groups = {}
        for p in predictions:
            length = p['sequence_length']
            group = (length // 50) * 50  # Group by 50-residue ranges
            
            if group not in length_groups:
                length_groups[group] = []
            length_groups[group].append(p)
        
        # Calculate consistency within groups
        consistencies = []
        for group_predictions in length_groups.values():
            if len(group_predictions) < 3:
                continue
            
            plddts = [p['prediction'].get('plddt', 0) for p in group_predictions]
            if plddts:
                # Consistency = 1 - (std / mean)
                consistency = max(0, 1 - (np.std(plddts) / (np.mean(plddts) + 1e-6)))
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.8
    
    def _update_learning_metrics(self, performance: Dict[str, float]) -> LearningMetrics:
        """Update learning metrics based on performance"""
        
        # Update accuracy trend (using pLDDT as proxy)
        current_accuracy = performance.get('avg_plddt', 50) / 100
        self.learning_metrics.accuracy_trend.append(current_accuracy)
        if len(self.learning_metrics.accuracy_trend) > 100:
            self.learning_metrics.accuracy_trend = self.learning_metrics.accuracy_trend[-100:]
        
        # Update processing speed trend
        speed_metric = 1.0 / max(performance.get('avg_processing_time', 1), 0.001)
        self.learning_metrics.processing_speed_trend.append(speed_metric)
        if len(self.learning_metrics.processing_speed_trend) > 100:
            self.learning_metrics.processing_speed_trend = self.learning_metrics.processing_speed_trend[-100:]
        
        # Calculate confidence calibration
        confidence = performance.get('avg_confidence', 0.5)
        accuracy = current_accuracy
        self.learning_metrics.confidence_calibration = abs(confidence - accuracy)
        
        # Update model uncertainty
        self.learning_metrics.model_uncertainty = performance.get('confidence_std', 0.2)
        
        # Update prediction consistency
        self.learning_metrics.prediction_consistency = performance.get('prediction_consistency', 0.8)
        
        # Adaptive learning rate based on improvement trend
        if len(self.learning_metrics.accuracy_trend) >= 2:
            recent_trend = (self.learning_metrics.accuracy_trend[-1] - 
                          self.learning_metrics.accuracy_trend[-2])
            if recent_trend > 0:
                self.learning_metrics.adaptive_learning_rate *= 0.98  # Slow down if improving
            else:
                self.learning_metrics.adaptive_learning_rate *= 1.02  # Speed up if declining
            
            # Bound learning rate
            self.learning_metrics.adaptive_learning_rate = np.clip(
                self.learning_metrics.adaptive_learning_rate, 0.001, 0.1
            )
        
        return self.learning_metrics
    
    def _should_adapt(self, performance: Dict[str, float]) -> bool:
        """Determine if adaptation is needed"""
        
        # Check if performance is below threshold
        avg_accuracy = performance.get('avg_plddt', 50) / 100
        if avg_accuracy < self.adaptation_threshold:
            return True
        
        # Check if confidence calibration is poor
        if self.learning_metrics.confidence_calibration > 0.3:
            return True
        
        # Check if processing speed is declining
        if len(self.learning_metrics.processing_speed_trend) >= 10:
            recent_speeds = self.learning_metrics.processing_speed_trend[-10:]
            if np.mean(recent_speeds[-5:]) < np.mean(recent_speeds[:5]) * 0.8:
                return True
        
        # Check prediction consistency
        if self.learning_metrics.prediction_consistency < 0.6:
            return True
        
        return False
    
    def _make_adaptation_decision(self, performance: Dict[str, float]) -> AdaptationDecision:
        """Make autonomous adaptation decision"""
        
        # Analyze what needs improvement
        issues = []
        if performance.get('avg_plddt', 50) / 100 < self.adaptation_threshold:
            issues.append('accuracy')
        if self.learning_metrics.confidence_calibration > 0.3:
            issues.append('calibration')
        if performance.get('avg_processing_time', 1) > 2.0:
            issues.append('speed')
        if self.learning_metrics.prediction_consistency < 0.6:
            issues.append('consistency')
        
        # Determine adaptation strategy
        primary_issue = issues[0] if issues else 'optimization'
        
        if primary_issue == 'accuracy':
            decision = self._decide_accuracy_improvement()
        elif primary_issue == 'calibration':
            decision = self._decide_calibration_improvement()
        elif primary_issue == 'speed':
            decision = self._decide_speed_improvement()
        elif primary_issue == 'consistency':
            decision = self._decide_consistency_improvement()
        else:
            decision = self._decide_general_optimization()
        
        return decision
    
    def _decide_accuracy_improvement(self) -> AdaptationDecision:
        """Decide on accuracy improvement adaptations"""
        return AdaptationDecision(
            decision_type='accuracy_improvement',
            confidence=0.8,
            reasoning='Low accuracy detected, adjusting confidence calibration and energy weighting',
            parameters_changed={
                'confidence_calibration_factor': self.current_parameters['confidence_calibration_factor'] * 1.1,
                'energy_weight': self.current_parameters['energy_weight'] * 0.9,
                'quality_gate_strictness': min(0.95, self.current_parameters['quality_gate_strictness'] * 1.05)
            },
            expected_improvement=0.15,
            timestamp=time.time()
        )
    
    def _decide_calibration_improvement(self) -> AdaptationDecision:
        """Decide on calibration improvement adaptations"""
        return AdaptationDecision(
            decision_type='calibration_improvement',
            confidence=0.9,
            reasoning='Poor confidence calibration, adjusting calibration factor',
            parameters_changed={
                'confidence_calibration_factor': 0.8 if self.learning_metrics.confidence_calibration > 0 else 1.2,
                'uncertainty_threshold': max(0.5, self.current_parameters['uncertainty_threshold'] - 0.1)
            },
            expected_improvement=0.2,
            timestamp=time.time()
        )
    
    def _decide_speed_improvement(self) -> AdaptationDecision:
        """Decide on speed improvement adaptations"""
        return AdaptationDecision(
            decision_type='speed_improvement',
            confidence=0.7,
            reasoning='Slow processing detected, optimizing batch size and reducing complexity',
            parameters_changed={
                'batch_processing_size': min(64, self.current_parameters['batch_processing_size'] * 1.5),
                'dynamics_sensitivity': self.current_parameters['dynamics_sensitivity'] * 0.8,
                'function_prediction_threshold': min(0.9, self.current_parameters['function_prediction_threshold'] + 0.1)
            },
            expected_improvement=0.3,
            timestamp=time.time()
        )
    
    def _decide_consistency_improvement(self) -> AdaptationDecision:
        """Decide on consistency improvement adaptations"""
        return AdaptationDecision(
            decision_type='consistency_improvement',
            confidence=0.85,
            reasoning='Low prediction consistency, stabilizing parameters',
            parameters_changed={
                'confidence_calibration_factor': 1.0,  # Reset to stable value
                'quality_gate_strictness': 0.85,      # Standardize
                'uncertainty_threshold': 0.7          # Standard threshold
            },
            expected_improvement=0.25,
            timestamp=time.time()
        )
    
    def _decide_general_optimization(self) -> AdaptationDecision:
        """Decide on general optimization"""
        return AdaptationDecision(
            decision_type='general_optimization',
            confidence=0.6,
            reasoning='General optimization based on learning trends',
            parameters_changed={
                'confidence_calibration_factor': self.current_parameters['confidence_calibration_factor'] * 1.02,
                'energy_weight': self.current_parameters['energy_weight'] * 0.98
            },
            expected_improvement=0.05,
            timestamp=time.time()
        )
    
    def _apply_adaptation(self, decision: AdaptationDecision):
        """Apply adaptation decision"""
        
        # Update parameters
        for param, value in decision.parameters_changed.items():
            if param in self.current_parameters:
                self.current_parameters[param] = value
        
        # Record adaptation
        self.adaptation_history.append(decision)
        
        # Keep adaptation history manageable
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
        
        print(f"Autonomous Adaptation Applied: {decision.decision_type}")
        print(f"  Reasoning: {decision.reasoning}")
        print(f"  Expected Improvement: {decision.expected_improvement:.1%}")
        print(f"  Parameters Changed: {list(decision.parameters_changed.keys())}")
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get current autonomous learning status"""
        return {
            'learning_metrics': self.learning_metrics.to_dict(),
            'current_parameters': dict(self.current_parameters),
            'adaptation_history_size': len(self.adaptation_history),
            'predictions_recorded': len(self.prediction_history),
            'knowledge_base_size': len(self.sequence_knowledge_base),
            'pattern_library_size': len(self.pattern_library),
            'current_phase': self.current_adaptation_phase,
            'predictions_since_adaptation': self.predictions_since_adaptation,
            'last_adaptation': self.adaptation_history[-1].timestamp if self.adaptation_history else None
        }
    
    def export_knowledge(self, filepath: str):
        """Export learned knowledge for reuse"""
        knowledge_export = {
            'version': '1.0',
            'timestamp': time.time(),
            'learning_metrics': self.learning_metrics.to_dict(),
            'parameters': dict(self.current_parameters),
            'sequence_knowledge': dict(self.sequence_knowledge_base),
            'pattern_library': dict(self.pattern_library),
            'adaptation_history': [asdict(a) for a in self.adaptation_history[-50:]]
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge_export, f, indent=2)
        
        print(f"Knowledge exported to {filepath}")
    
    def import_knowledge(self, filepath: str):
        """Import previously learned knowledge"""
        try:
            with open(filepath, 'r') as f:
                knowledge_data = json.load(f)
            
            # Import compatible knowledge
            if 'parameters' in knowledge_data:
                for key, value in knowledge_data['parameters'].items():
                    if key in self.current_parameters:
                        self.current_parameters[key] = value
            
            if 'pattern_library' in knowledge_data:
                self.pattern_library.update(knowledge_data['pattern_library'])
            
            print(f"Knowledge imported from {filepath}")
            
        except Exception as e:
            print(f"Failed to import knowledge: {e}")


# Factory function for autonomous learning integration
def create_autonomous_learner(config: Optional[Dict] = None) -> AutonomousLearningFramework:
    """Create autonomous learning framework with configuration"""
    if config is None:
        config = {}
    
    return AutonomousLearningFramework(
        adaptation_threshold=config.get('adaptation_threshold', 0.85),
        learning_memory_size=config.get('learning_memory_size', 10000),
        adaptation_frequency=config.get('adaptation_frequency', 100)
    )


# Demonstration of autonomous learning
if __name__ == "__main__":
    # Create autonomous learner
    learner = create_autonomous_learner()
    
    # Simulate some predictions for demonstration
    test_sequences = [
        "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVID",
        "MKLYNVFHVKLYNVFHVKLYNVFHVKLYNVFHVKLYNVFHV"
    ]
    
    for seq in test_sequences:
        # Simulate prediction result
        result = {
            'confidence': np.random.uniform(0.6, 0.9, len(seq)).tolist(),
            'plddt': np.random.uniform(70, 90),
            'energy': np.random.uniform(-100, -50),
            'processing_time': np.random.uniform(0.5, 2.0)
        }
        
        # Record prediction
        learner.record_prediction(seq, result)
    
    # Get status
    status = learner.get_autonomous_status()
    print("Autonomous Learning Framework Status:")
    for key, value in status.items():
        if key != 'learning_metrics':
            print(f"  {key}: {value}")
    
    # Export knowledge
    learner.export_knowledge("autonomous_knowledge.json")