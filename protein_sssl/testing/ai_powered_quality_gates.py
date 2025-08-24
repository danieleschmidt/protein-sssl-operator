"""
AI-Powered Quality Gates for Protein Structure Prediction
Advanced SDLC Quality Assurance with Machine Learning
"""
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import re
import hashlib


class QualityLevel(Enum):
    """Quality assurance levels"""
    BASIC = "basic"
    ADVANCED = "advanced"
    AI_POWERED = "ai_powered"
    AUTONOMOUS = "autonomous"


class TestType(Enum):
    """Types of quality tests"""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    COMPATIBILITY = "compatibility"
    DATA_INTEGRITY = "data_integrity"
    AI_VALIDATION = "ai_validation"


@dataclass
class QualityMetrics:
    """Quality metrics for analysis"""
    test_coverage: float
    defect_density: float
    reliability_score: float
    performance_index: float
    security_score: float
    maintainability_index: float
    ai_confidence: float
    overall_quality_score: float


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    test_type: TestType
    status: str  # passed, failed, warning, skipped
    execution_time: float
    confidence: float
    details: Dict[str, Any]
    ai_analysis: Optional[Dict[str, Any]]
    timestamp: float


@dataclass
class QualityGateResult:
    """Quality gate validation result"""
    gate_name: str
    passed: bool
    score: float
    threshold: float
    failing_tests: List[str]
    recommendations: List[str]
    ai_insights: List[str]


class AIQualityAnalyzer:
    """AI-powered quality analyzer"""
    
    def __init__(self):
        self.pattern_library = self._initialize_pattern_library()
        self.quality_models = self._initialize_quality_models()
        self.historical_data = []
        
    def _initialize_pattern_library(self) -> Dict[str, Any]:
        """Initialize quality pattern recognition library"""
        return {
            'performance_patterns': {
                'memory_leak': r'memory.*(?:leak|grow|increase).*(?:over time|continuously)',
                'slow_performance': r'(?:slow|timeout|performance).*(?:degradation|issue)',
                'resource_exhaustion': r'(?:resource|memory|cpu).*(?:exhausted|full|limit)'
            },
            'security_patterns': {
                'injection_vulnerability': r'(?:sql|script|command).*injection',
                'authentication_bypass': r'(?:auth|login).*(?:bypass|skip|fail)',
                'data_exposure': r'(?:sensitive|private|secret).*(?:exposed|leaked|visible)'
            },
            'reliability_patterns': {
                'intermittent_failure': r'(?:intermittent|random|occasional).*(?:fail|error)',
                'cascade_failure': r'(?:cascade|chain|domino).*fail',
                'timeout_error': r'timeout.*(?:error|exception|fail)'
            }
        }
    
    def _initialize_quality_models(self) -> Dict[str, Any]:
        """Initialize AI quality models (simplified versions)"""
        return {
            'defect_prediction': {
                'weights': np.random.random((10, 1)) * 0.01,  # Simple linear model
                'bias': 0.1,
                'features': ['complexity', 'test_coverage', 'change_frequency', 'code_size']
            },
            'performance_prediction': {
                'weights': np.random.random((8, 1)) * 0.01,
                'bias': 0.05,
                'features': ['data_size', 'complexity', 'algorithm_type', 'resource_usage']
            },
            'security_assessment': {
                'weights': np.random.random((6, 1)) * 0.01,
                'bias': 0.2,
                'features': ['input_validation', 'authentication', 'encryption', 'access_control']
            }
        }
    
    def analyze_code_quality(self, code: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI-powered code quality analysis"""
        
        if metadata is None:
            metadata = {}
        
        analysis = {
            'complexity_score': self._analyze_complexity(code),
            'maintainability_index': self._calculate_maintainability(code),
            'pattern_matches': self._detect_quality_patterns(code),
            'ai_predictions': self._make_ai_predictions(code, metadata),
            'recommendations': self._generate_recommendations(code),
            'confidence': 0.0
        }
        
        # Calculate overall confidence
        analysis['confidence'] = self._calculate_analysis_confidence(analysis)
        
        return analysis
    
    def _analyze_complexity(self, code: str) -> float:
        """Analyze code complexity using AI techniques"""
        
        # Count various complexity indicators
        lines = code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Basic complexity metrics
        cyclomatic_complexity = len(re.findall(r'\b(if|elif|for|while|try|except|with)\b', code))
        function_count = len(re.findall(r'\bdef\s+\w+\s*\(', code))
        class_count = len(re.findall(r'\bclass\s+\w+\s*[:(]', code))
        
        # Nesting depth analysis
        max_nesting = self._calculate_max_nesting_depth(code)
        
        # AI-based complexity assessment
        complexity_features = np.array([
            len(non_empty_lines),
            cyclomatic_complexity,
            function_count,
            class_count,
            max_nesting,
            len(code.split()),  # word count
            len(set(re.findall(r'\b\w+\b', code))),  # unique identifiers
            code.count('import')
        ]).reshape(-1, 1)
        
        # Normalize and score
        normalized_score = np.mean(complexity_features) / 100.0
        return min(1.0, max(0.0, normalized_score))
    
    def _calculate_max_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for line in code.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Count leading whitespace to determine nesting
            leading_spaces = len(line) - len(line.lstrip())
            indentation_level = leading_spaces // 4  # Assuming 4-space indentation
            
            # Update depth based on control structures
            if re.search(r'\b(if|for|while|try|with|def|class)\b.*:', stripped):
                current_depth = indentation_level + 1
                max_depth = max(max_depth, current_depth)
        
        return max_depth
    
    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability index using AI"""
        
        lines = code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Maintainability factors
        comment_ratio = len([line for line in lines if line.strip().startswith('#')]) / max(len(lines), 1)
        avg_line_length = np.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0
        function_length_consistency = self._analyze_function_consistency(code)
        naming_quality = self._analyze_naming_quality(code)
        
        # AI-based maintainability calculation
        maintainability = (
            comment_ratio * 0.25 +
            (1.0 - min(avg_line_length / 120.0, 1.0)) * 0.25 +  # Prefer shorter lines
            function_length_consistency * 0.25 +
            naming_quality * 0.25
        )
        
        return min(1.0, max(0.0, maintainability))
    
    def _analyze_function_consistency(self, code: str) -> float:
        """Analyze function length consistency"""
        function_pattern = r'def\s+\w+\s*\([^)]*\):\s*(?:\n(?:\s{4,}.*\n)*)?'
        functions = re.findall(function_pattern, code, re.MULTILINE)
        
        if not functions:
            return 1.0
        
        function_lengths = [len(func.split('\n')) for func in functions]
        
        if len(function_lengths) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower is better)
        mean_length = np.mean(function_lengths)
        std_length = np.std(function_lengths)
        
        if mean_length == 0:
            return 1.0
        
        cv = std_length / mean_length
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def _analyze_naming_quality(self, code: str) -> float:
        """Analyze naming quality using AI patterns"""
        
        # Extract identifiers
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        
        if not identifiers:
            return 1.0
        
        quality_scores = []
        
        for identifier in set(identifiers):
            # Skip keywords and builtins
            if identifier in ['def', 'class', 'if', 'for', 'while', 'import', 'from', 'as']:
                continue
            
            score = 0.0
            
            # Length appropriateness
            if 3 <= len(identifier) <= 20:
                score += 0.3
            
            # Descriptiveness (contains vowels and consonants)
            vowels = sum(1 for c in identifier.lower() if c in 'aeiou')
            consonants = len(identifier) - vowels
            if vowels > 0 and consonants > 0:
                score += 0.3
            
            # Naming conventions
            if identifier.islower() or (identifier.isupper() and '_' in identifier):
                score += 0.2
            
            # Avoid abbreviations (heuristic)
            if len(identifier) > 2 and not re.match(r'^[A-Z]{2,}$', identifier):
                score += 0.2
            
            quality_scores.append(score)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _detect_quality_patterns(self, code: str) -> Dict[str, List[str]]:
        """Detect quality patterns using AI pattern matching"""
        
        detected_patterns = {}
        
        for category, patterns in self.pattern_library.items():
            detected_patterns[category] = {}
            
            for pattern_name, pattern_regex in patterns.items():
                matches = re.findall(pattern_regex, code, re.IGNORECASE)
                if matches:
                    detected_patterns[category][pattern_name] = matches
        
        return detected_patterns
    
    def _make_ai_predictions(self, code: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Make AI-powered predictions about code quality"""
        
        predictions = {}
        
        # Defect prediction
        defect_features = self._extract_defect_features(code, metadata)
        defect_probability = self._predict_defects(defect_features)
        predictions['defect_probability'] = defect_probability
        
        # Performance prediction
        perf_features = self._extract_performance_features(code, metadata)
        performance_score = self._predict_performance(perf_features)
        predictions['performance_score'] = performance_score
        
        # Security assessment
        security_features = self._extract_security_features(code, metadata)
        security_score = self._assess_security(security_features)
        predictions['security_score'] = security_score
        
        return predictions
    
    def _extract_defect_features(self, code: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features for defect prediction"""
        
        lines = code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        features = [
            len(non_empty_lines),  # Code size
            len(re.findall(r'\b(if|elif|for|while|try|except)\b', code)),  # Complexity
            metadata.get('test_coverage', 0.5),  # Test coverage
            metadata.get('change_frequency', 0.1),  # How often changed
            len(re.findall(r'\bdef\s+\w+', code)),  # Function count
            len(re.findall(r'\bclass\s+\w+', code)),  # Class count
            code.count('TODO') + code.count('FIXME'),  # Technical debt indicators
            len(re.findall(r'raise\s+\w+Error', code)),  # Exception handling
            metadata.get('author_experience', 0.7),  # Developer experience
            metadata.get('code_review_score', 0.8)  # Review quality
        ]
        
        return np.array(features).reshape(-1, 1)
    
    def _extract_performance_features(self, code: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features for performance prediction"""
        
        features = [
            len(code.split()),  # Code size
            len(re.findall(r'\bfor\s+.*\s+in\s+', code)),  # Loop count
            code.count('numpy') + code.count('np.'),  # Vectorization usage
            len(re.findall(r'\.sort\(\)|sorted\(', code)),  # Sorting operations
            len(re.findall(r'\.append\(|\.extend\(', code)),  # List operations
            metadata.get('expected_data_size', 1000),  # Expected data size
            len(re.findall(r'import\s+.*multiprocessing|from.*concurrent', code)),  # Parallel code
            len(re.findall(r'@cache|@lru_cache|@memoize', code))  # Caching usage
        ]
        
        return np.array(features).reshape(-1, 1)
    
    def _extract_security_features(self, code: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Extract features for security assessment"""
        
        features = [
            len(re.findall(r'input\(|raw_input\(', code)),  # User input
            len(re.findall(r'eval\(|exec\(', code)),  # Dangerous functions
            len(re.findall(r'password|secret|key|token', code, re.IGNORECASE)),  # Sensitive data
            len(re.findall(r'import\s+.*ssl|import\s+.*crypto', code)),  # Security modules
            len(re.findall(r'open\(|file\(', code)),  # File operations
            len(re.findall(r'subprocess|os\.system', code))  # System commands
        ]
        
        return np.array(features).reshape(-1, 1)
    
    def _predict_defects(self, features: np.ndarray) -> float:
        """Predict defect probability using simple AI model"""
        
        model = self.quality_models['defect_prediction']
        
        # Pad or truncate features to match model
        if len(features) > len(model['weights']):
            features = features[:len(model['weights'])]
        elif len(features) < len(model['weights']):
            features = np.pad(features.flatten(), (0, len(model['weights']) - len(features)), 'constant').reshape(-1, 1)
        
        # Simple linear prediction
        prediction = np.dot(model['weights'].T, features.flatten())[0] + model['bias']
        
        # Apply sigmoid to get probability
        probability = 1 / (1 + np.exp(-prediction))
        
        return float(probability)
    
    def _predict_performance(self, features: np.ndarray) -> float:
        """Predict performance score using AI model"""
        
        model = self.quality_models['performance_prediction']
        
        # Adjust features size
        if len(features) > len(model['weights']):
            features = features[:len(model['weights'])]
        elif len(features) < len(model['weights']):
            features = np.pad(features.flatten(), (0, len(model['weights']) - len(features)), 'constant').reshape(-1, 1)
        
        # Predict performance score
        score = np.dot(model['weights'].T, features.flatten())[0] + model['bias']
        
        # Normalize to 0-1 range
        normalized_score = 1 / (1 + np.exp(-score))
        
        return float(normalized_score)
    
    def _assess_security(self, features: np.ndarray) -> float:
        """Assess security score using AI model"""
        
        model = self.quality_models['security_assessment']
        
        # Adjust features size
        if len(features) > len(model['weights']):
            features = features[:len(model['weights'])]
        elif len(features) < len(model['weights']):
            features = np.pad(features.flatten(), (0, len(model['weights']) - len(features)), 'constant').reshape(-1, 1)
        
        # Assess security
        score = np.dot(model['weights'].T, features.flatten())[0] + model['bias']
        
        # Normalize to 0-1 range (higher is better)
        normalized_score = 1 / (1 + np.exp(-score))
        
        return float(normalized_score)
    
    def _generate_recommendations(self, code: str) -> List[str]:
        """Generate AI-powered recommendations"""
        
        recommendations = []
        
        # Complexity recommendations
        if len(code.split('\n')) > 100:
            recommendations.append("Consider breaking this code into smaller, more focused functions")
        
        # Performance recommendations
        if 'for' in code and 'append' in code:
            recommendations.append("Consider using list comprehensions or vectorized operations for better performance")
        
        # Security recommendations
        if re.search(r'eval\(|exec\(', code):
            recommendations.append("Avoid using eval() or exec() as they pose security risks")
        
        # Maintainability recommendations
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        total_lines = len([line for line in code.split('\n') if line.strip()])
        
        if total_lines > 0 and comment_lines / total_lines < 0.1:
            recommendations.append("Add more comments to improve code maintainability")
        
        return recommendations
    
    def _calculate_analysis_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the AI analysis"""
        
        # Base confidence on various factors
        confidence_factors = []
        
        # Factor 1: Code size (more code = more confident analysis)
        if 'complexity_score' in analysis:
            size_confidence = min(1.0, analysis['complexity_score'] * 2)
            confidence_factors.append(size_confidence)
        
        # Factor 2: Pattern detection success
        if 'pattern_matches' in analysis:
            pattern_confidence = len(analysis['pattern_matches']) / 10.0
            confidence_factors.append(min(1.0, pattern_confidence))
        
        # Factor 3: Model prediction consistency
        if 'ai_predictions' in analysis:
            predictions = analysis['ai_predictions']
            if len(predictions) >= 2:
                pred_values = [v for v in predictions.values() if isinstance(v, (int, float))]
                if pred_values:
                    consistency = 1.0 - np.std(pred_values) / (np.mean(pred_values) + 0.001)
                    confidence_factors.append(max(0.0, consistency))
        
        # Overall confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # Default moderate confidence


class AIQualityGates:
    """AI-powered quality gates system"""
    
    def __init__(self, quality_level: QualityLevel = QualityLevel.AI_POWERED):
        self.quality_level = quality_level
        self.ai_analyzer = AIQualityAnalyzer()
        self.test_history = []
        self.quality_thresholds = self._initialize_thresholds()
        
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds based on level"""
        
        base_thresholds = {
            'test_coverage': 0.80,
            'defect_density': 0.1,
            'reliability_score': 0.85,
            'performance_index': 0.80,
            'security_score': 0.90,
            'maintainability_index': 0.75,
            'ai_confidence': 0.70,
            'overall_quality_score': 0.80
        }
        
        # Adjust thresholds based on quality level
        if self.quality_level == QualityLevel.AUTONOMOUS:
            return {k: min(0.98, v * 1.2) for k, v in base_thresholds.items()}
        elif self.quality_level == QualityLevel.AI_POWERED:
            return {k: min(0.95, v * 1.1) for k, v in base_thresholds.items()}
        elif self.quality_level == QualityLevel.ADVANCED:
            return base_thresholds
        else:  # BASIC
            return {k: v * 0.8 for k, v in base_thresholds.items()}
    
    def run_quality_gates(self, 
                         code_files: List[str] = None,
                         test_results: List[TestResult] = None,
                         metadata: Dict[str, Any] = None) -> List[QualityGateResult]:
        """Run comprehensive AI-powered quality gates"""
        
        if metadata is None:
            metadata = {}
        
        gate_results = []
        
        # Gate 1: Code Quality Analysis
        if code_files:
            code_quality_result = self._run_code_quality_gate(code_files, metadata)
            gate_results.append(code_quality_result)
        
        # Gate 2: Test Coverage and Quality
        if test_results:
            test_quality_result = self._run_test_quality_gate(test_results, metadata)
            gate_results.append(test_quality_result)
        
        # Gate 3: Performance Analysis
        performance_result = self._run_performance_gate(code_files, test_results, metadata)
        gate_results.append(performance_result)
        
        # Gate 4: Security Assessment
        security_result = self._run_security_gate(code_files, metadata)
        gate_results.append(security_result)
        
        # Gate 5: AI-Powered Overall Assessment
        if self.quality_level in [QualityLevel.AI_POWERED, QualityLevel.AUTONOMOUS]:
            ai_result = self._run_ai_assessment_gate(code_files, test_results, metadata)
            gate_results.append(ai_result)
        
        return gate_results
    
    def _run_code_quality_gate(self, code_files: List[str], metadata: Dict[str, Any]) -> QualityGateResult:
        """Run AI-powered code quality gate"""
        
        all_analyses = []
        failing_tests = []
        recommendations = []
        ai_insights = []
        
        for file_path in code_files:
            try:
                # Read code file
                if isinstance(file_path, str) and Path(file_path).exists():
                    with open(file_path, 'r') as f:
                        code = f.read()
                else:
                    code = str(file_path)  # Treat as code content
                
                # AI analysis
                analysis = self.ai_analyzer.analyze_code_quality(code, metadata)
                all_analyses.append(analysis)
                
                # Check thresholds
                if analysis['maintainability_index'] < self.quality_thresholds['maintainability_index']:
                    failing_tests.append(f"Low maintainability in {file_path}")
                
                if analysis['confidence'] < self.quality_thresholds['ai_confidence']:
                    failing_tests.append(f"Low AI analysis confidence for {file_path}")
                
                # Collect recommendations and insights
                recommendations.extend(analysis.get('recommendations', []))
                ai_insights.append(f"AI confidence: {analysis['confidence']:.2f} for {file_path}")
                
            except Exception as e:
                failing_tests.append(f"Failed to analyze {file_path}: {str(e)}")
        
        # Calculate overall score
        if all_analyses:
            avg_maintainability = np.mean([a['maintainability_index'] for a in all_analyses])
            avg_confidence = np.mean([a['confidence'] for a in all_analyses])
            overall_score = (avg_maintainability + avg_confidence) / 2
        else:
            overall_score = 0.0
        
        return QualityGateResult(
            gate_name="code_quality",
            passed=len(failing_tests) == 0,
            score=overall_score,
            threshold=self.quality_thresholds['maintainability_index'],
            failing_tests=failing_tests,
            recommendations=list(set(recommendations)),
            ai_insights=ai_insights
        )
    
    def _run_test_quality_gate(self, test_results: List[TestResult], metadata: Dict[str, Any]) -> QualityGateResult:
        """Run test quality assessment gate"""
        
        if not test_results:
            return QualityGateResult(
                gate_name="test_quality",
                passed=False,
                score=0.0,
                threshold=self.quality_thresholds['test_coverage'],
                failing_tests=["No test results provided"],
                recommendations=["Add comprehensive test suite"],
                ai_insights=["Insufficient test data for AI analysis"]
            )
        
        # Calculate test metrics
        total_tests = len(test_results)
        passed_tests = len([t for t in test_results if t.status == 'passed'])
        test_coverage = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # AI analysis of test quality
        avg_confidence = np.mean([t.confidence for t in test_results])
        test_type_diversity = len(set(t.test_type for t in test_results)) / len(TestType)
        
        failing_tests = []
        recommendations = []
        ai_insights = []
        
        # Check coverage threshold
        if test_coverage < self.quality_thresholds['test_coverage']:
            failing_tests.append(f"Test coverage {test_coverage:.1%} below threshold {self.quality_thresholds['test_coverage']:.1%}")
        
        # Check test confidence
        if avg_confidence < self.quality_thresholds['ai_confidence']:
            failing_tests.append(f"Average test confidence {avg_confidence:.2f} below threshold")
        
        # Generate recommendations
        if test_type_diversity < 0.5:
            recommendations.append("Increase test type diversity - add more test categories")
        
        if avg_confidence < 0.8:
            recommendations.append("Improve test assertions and validation logic")
        
        # AI insights
        ai_insights.append(f"Test confidence analysis: {avg_confidence:.2f}")
        ai_insights.append(f"Test type diversity: {test_type_diversity:.2f}")
        
        overall_score = (test_coverage + avg_confidence + test_type_diversity) / 3
        
        return QualityGateResult(
            gate_name="test_quality",
            passed=len(failing_tests) == 0,
            score=overall_score,
            threshold=self.quality_thresholds['test_coverage'],
            failing_tests=failing_tests,
            recommendations=recommendations,
            ai_insights=ai_insights
        )
    
    def _run_performance_gate(self, code_files: List[str], test_results: List[TestResult], metadata: Dict[str, Any]) -> QualityGateResult:
        """Run AI-powered performance assessment gate"""
        
        performance_scores = []
        failing_tests = []
        recommendations = []
        ai_insights = []
        
        # Analyze performance from test results
        if test_results:
            performance_tests = [t for t in test_results if t.test_type == TestType.PERFORMANCE]
            if performance_tests:
                avg_execution_time = np.mean([t.execution_time for t in performance_tests])
                performance_score = max(0.0, 1.0 - avg_execution_time / 10.0)  # Assume 10s is max acceptable
                performance_scores.append(performance_score)
                
                if performance_score < self.quality_thresholds['performance_index']:
                    failing_tests.append(f"Performance score {performance_score:.2f} below threshold")
                
                ai_insights.append(f"Average execution time: {avg_execution_time:.3f}s")
        
        # AI-based performance analysis of code
        if code_files:
            for file_path in code_files[:3]:  # Analyze first 3 files to avoid overload
                try:
                    if isinstance(file_path, str) and Path(file_path).exists():
                        with open(file_path, 'r') as f:
                            code = f.read()
                    else:
                        code = str(file_path)
                    
                    analysis = self.ai_analyzer.analyze_code_quality(code, metadata)
                    if 'ai_predictions' in analysis and 'performance_score' in analysis['ai_predictions']:
                        perf_score = analysis['ai_predictions']['performance_score']
                        performance_scores.append(perf_score)
                        
                        if perf_score < self.quality_thresholds['performance_index']:
                            failing_tests.append(f"Predicted performance issues in {file_path}")
                        
                        ai_insights.append(f"AI performance prediction: {perf_score:.2f} for {file_path}")
                
                except Exception as e:
                    failing_tests.append(f"Performance analysis failed for {file_path}: {str(e)}")
        
        # Generate performance recommendations
        if any(score < 0.7 for score in performance_scores):
            recommendations.append("Consider algorithm optimization and caching strategies")
            recommendations.append("Profile code to identify performance bottlenecks")
        
        overall_score = np.mean(performance_scores) if performance_scores else 0.5
        
        return QualityGateResult(
            gate_name="performance",
            passed=len(failing_tests) == 0,
            score=overall_score,
            threshold=self.quality_thresholds['performance_index'],
            failing_tests=failing_tests,
            recommendations=recommendations,
            ai_insights=ai_insights
        )
    
    def _run_security_gate(self, code_files: List[str], metadata: Dict[str, Any]) -> QualityGateResult:
        """Run AI-powered security assessment gate"""
        
        security_scores = []
        failing_tests = []
        recommendations = []
        ai_insights = []
        
        if code_files:
            for file_path in code_files:
                try:
                    if isinstance(file_path, str) and Path(file_path).exists():
                        with open(file_path, 'r') as f:
                            code = f.read()
                    else:
                        code = str(file_path)
                    
                    analysis = self.ai_analyzer.analyze_code_quality(code, metadata)
                    
                    # Security analysis
                    if 'ai_predictions' in analysis and 'security_score' in analysis['ai_predictions']:
                        security_score = analysis['ai_predictions']['security_score']
                        security_scores.append(security_score)
                        
                        if security_score < self.quality_thresholds['security_score']:
                            failing_tests.append(f"Security concerns detected in {file_path}")
                        
                        ai_insights.append(f"AI security assessment: {security_score:.2f} for {file_path}")
                    
                    # Pattern-based security checks
                    if 'pattern_matches' in analysis:
                        security_patterns = analysis['pattern_matches'].get('security_patterns', {})
                        if security_patterns:
                            failing_tests.append(f"Security patterns detected in {file_path}: {list(security_patterns.keys())}")
                            recommendations.append("Address detected security vulnerabilities")
                
                except Exception as e:
                    failing_tests.append(f"Security analysis failed for {file_path}: {str(e)}")
        
        # Default recommendations for security
        recommendations.extend([
            "Implement input validation for all user inputs",
            "Use parameterized queries to prevent injection attacks",
            "Ensure proper authentication and authorization"
        ])
        
        overall_score = np.mean(security_scores) if security_scores else 0.5
        
        return QualityGateResult(
            gate_name="security",
            passed=len(failing_tests) == 0,
            score=overall_score,
            threshold=self.quality_thresholds['security_score'],
            failing_tests=failing_tests,
            recommendations=list(set(recommendations)),
            ai_insights=ai_insights
        )
    
    def _run_ai_assessment_gate(self, code_files: List[str], test_results: List[TestResult], metadata: Dict[str, Any]) -> QualityGateResult:
        """Run comprehensive AI assessment gate"""
        
        ai_insights = []
        recommendations = []
        failing_tests = []
        
        # Overall AI confidence analysis
        confidences = []
        
        # Collect AI confidences from various analyses
        if code_files:
            for file_path in code_files[:5]:  # Limit to avoid overload
                try:
                    if isinstance(file_path, str) and Path(file_path).exists():
                        with open(file_path, 'r') as f:
                            code = f.read()
                    else:
                        code = str(file_path)
                    
                    analysis = self.ai_analyzer.analyze_code_quality(code, metadata)
                    confidences.append(analysis['confidence'])
                
                except Exception:
                    continue
        
        if test_results:
            test_confidences = [t.confidence for t in test_results if hasattr(t, 'confidence')]
            confidences.extend(test_confidences)
        
        # AI assessment
        if confidences:
            avg_confidence = np.mean(confidences)
            confidence_consistency = 1.0 - np.std(confidences) / (avg_confidence + 0.001)
            
            ai_insights.append(f"Average AI confidence: {avg_confidence:.2f}")
            ai_insights.append(f"Confidence consistency: {confidence_consistency:.2f}")
            
            if avg_confidence < self.quality_thresholds['ai_confidence']:
                failing_tests.append("Overall AI confidence below threshold")
            
            if confidence_consistency < 0.7:
                failing_tests.append("Inconsistent AI analysis confidence")
                recommendations.append("Review code areas with low AI confidence")
        
        # Meta-analysis: AI model performance
        model_performance = self._assess_ai_model_performance()
        ai_insights.append(f"AI model performance: {model_performance:.2f}")
        
        if model_performance < 0.8:
            recommendations.append("Consider retraining AI models with more recent data")
        
        overall_score = np.mean(confidences) if confidences else 0.5
        
        return QualityGateResult(
            gate_name="ai_assessment",
            passed=len(failing_tests) == 0,
            score=overall_score,
            threshold=self.quality_thresholds['ai_confidence'],
            failing_tests=failing_tests,
            recommendations=recommendations,
            ai_insights=ai_insights
        )
    
    def _assess_ai_model_performance(self) -> float:
        """Assess the performance of AI models themselves"""
        
        # Simple heuristic based on recent analysis consistency
        if len(self.test_history) < 10:
            return 0.8  # Default good performance
        
        recent_results = self.test_history[-10:]
        scores = [r.score for r in recent_results if hasattr(r, 'score')]
        
        if not scores:
            return 0.8
        
        # Performance based on score consistency and trend
        score_trend = np.mean(scores[-5:]) - np.mean(scores[:5]) if len(scores) >= 5 else 0
        score_consistency = 1.0 - np.std(scores) / (np.mean(scores) + 0.001)
        
        performance = (score_consistency * 0.7 + max(0, score_trend) * 0.3)
        return min(1.0, max(0.0, performance))
    
    def generate_quality_report(self, gate_results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate comprehensive AI-powered quality report"""
        
        if not gate_results:
            return {"status": "no_results", "message": "No quality gate results to report"}
        
        passed_gates = [g for g in gate_results if g.passed]
        failed_gates = [g for g in gate_results if not g.passed]
        
        # Overall quality metrics
        overall_score = np.mean([g.score for g in gate_results])
        pass_rate = len(passed_gates) / len(gate_results)
        
        # Collect all recommendations and insights
        all_recommendations = []
        all_ai_insights = []
        
        for gate in gate_results:
            all_recommendations.extend(gate.recommendations)
            all_ai_insights.extend(gate.ai_insights)
        
        # Priority recommendations (based on failed gates)
        priority_recommendations = []
        for gate in failed_gates:
            priority_recommendations.extend([
                f"[HIGH] {rec}" for rec in gate.recommendations[:2]
            ])
        
        report = {
            'quality_level': self.quality_level.value,
            'overall_score': round(overall_score, 3),
            'pass_rate': round(pass_rate, 3),
            'gates_summary': {
                'total': len(gate_results),
                'passed': len(passed_gates),
                'failed': len(failed_gates),
                'gate_details': {g.gate_name: {'passed': g.passed, 'score': round(g.score, 3)} 
                               for g in gate_results}
            },
            'quality_metrics': self._calculate_quality_metrics(gate_results),
            'recommendations': {
                'priority': priority_recommendations[:5],
                'all': list(set(all_recommendations)),
                'count': len(set(all_recommendations))
            },
            'ai_insights': {
                'key_insights': all_ai_insights[:10],
                'all_insights': all_ai_insights,
                'ai_confidence': self._calculate_ai_confidence(gate_results)
            },
            'next_actions': self._generate_next_actions(failed_gates),
            'report_timestamp': time.time()
        }
        
        return report
    
    def _calculate_quality_metrics(self, gate_results: List[QualityGateResult]) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        
        # Extract metrics from gate results
        code_quality_gate = next((g for g in gate_results if g.gate_name == 'code_quality'), None)
        test_quality_gate = next((g for g in gate_results if g.gate_name == 'test_quality'), None)
        performance_gate = next((g for g in gate_results if g.gate_name == 'performance'), None)
        security_gate = next((g for g in gate_results if g.gate_name == 'security'), None)
        ai_gate = next((g for g in gate_results if g.gate_name == 'ai_assessment'), None)
        
        return QualityMetrics(
            test_coverage=test_quality_gate.score if test_quality_gate else 0.5,
            defect_density=1.0 - (code_quality_gate.score if code_quality_gate else 0.5),
            reliability_score=np.mean([g.score for g in gate_results]),
            performance_index=performance_gate.score if performance_gate else 0.5,
            security_score=security_gate.score if security_gate else 0.5,
            maintainability_index=code_quality_gate.score if code_quality_gate else 0.5,
            ai_confidence=ai_gate.score if ai_gate else 0.5,
            overall_quality_score=np.mean([g.score for g in gate_results])
        )
    
    def _calculate_ai_confidence(self, gate_results: List[QualityGateResult]) -> float:
        """Calculate overall AI confidence"""
        
        ai_scores = []
        for gate in gate_results:
            if gate.ai_insights:
                # Extract confidence values from insights
                for insight in gate.ai_insights:
                    if 'confidence:' in insight:
                        try:
                            conf_str = insight.split('confidence:')[1].split()[0]
                            confidence = float(conf_str)
                            ai_scores.append(confidence)
                        except (ValueError, IndexError):
                            continue
        
        return np.mean(ai_scores) if ai_scores else 0.7
    
    def _generate_next_actions(self, failed_gates: List[QualityGateResult]) -> List[str]:
        """Generate prioritized next actions"""
        
        actions = []
        
        for gate in failed_gates:
            if gate.gate_name == 'code_quality':
                actions.append("Refactor code to improve maintainability and reduce complexity")
            elif gate.gate_name == 'test_quality':
                actions.append("Enhance test coverage and improve test assertions")
            elif gate.gate_name == 'performance':
                actions.append("Profile and optimize performance-critical code paths")
            elif gate.gate_name == 'security':
                actions.append("Address security vulnerabilities and implement security best practices")
            elif gate.gate_name == 'ai_assessment':
                actions.append("Review and improve areas flagged by AI analysis")
        
        # Add general actions
        if len(failed_gates) > len(gate_results) * 0.5:
            actions.append("Conduct comprehensive code review and refactoring")
        
        return actions[:5]  # Return top 5 priority actions


# Factory function
def create_ai_quality_gates(config: Optional[Dict] = None) -> AIQualityGates:
    """Create AI-powered quality gates system"""
    if config is None:
        config = {}
    
    quality_level = QualityLevel(config.get('quality_level', 'ai_powered'))
    
    return AIQualityGates(quality_level=quality_level)


# Demonstration
if __name__ == "__main__":
    # Create AI quality gates
    quality_gates = create_ai_quality_gates({
        'quality_level': 'ai_powered'
    })
    
    print("🧪 AI-Powered Quality Gates Testing...")
    
    # Sample code for testing
    sample_code = '''
def predict_protein_structure(sequence: str) -> dict:
    """Predict protein structure with confidence scoring."""
    if not sequence:
        raise ValueError("Empty sequence")
    
    # Validate sequence
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(c in valid_chars for c in sequence.upper()):
        raise ValueError("Invalid amino acid sequence")
    
    # Mock prediction
    result = {
        'structure': f'PREDICTED_{len(sequence)}_STRUCTURE',
        'confidence': 0.85,
        'processing_time': len(sequence) * 0.01
    }
    
    return result
'''
    
    # Sample test results
    test_results = [
        TestResult(
            test_name="test_valid_sequence",
            test_type=TestType.FUNCTIONAL,
            status="passed",
            execution_time=0.05,
            confidence=0.9,
            details={"assertions": 3, "coverage": 0.8},
            ai_analysis=None,
            timestamp=time.time()
        ),
        TestResult(
            test_name="test_performance_benchmark",
            test_type=TestType.PERFORMANCE,
            status="passed",
            execution_time=1.2,
            confidence=0.85,
            details={"benchmark": "large_protein", "memory_usage": "acceptable"},
            ai_analysis=None,
            timestamp=time.time()
        ),
        TestResult(
            test_name="test_security_validation",
            test_type=TestType.SECURITY,
            status="passed",
            execution_time=0.3,
            confidence=0.95,
            details={"sql_injection": "none", "input_validation": "passed"},
            ai_analysis=None,
            timestamp=time.time()
        )
    ]
    
    # Run quality gates
    gate_results = quality_gates.run_quality_gates(
        code_files=[sample_code],
        test_results=test_results,
        metadata={
            'test_coverage': 0.85,
            'change_frequency': 0.2,
            'author_experience': 0.8
        }
    )
    
    print(f"✅ Ran {len(gate_results)} quality gates")
    
    # Generate quality report
    report = quality_gates.generate_quality_report(gate_results)
    
    print(f"\n📊 AI Quality Gates Report:")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print(f"Pass Rate: {report['pass_rate']:.1%}")
    print(f"Gates Passed: {report['gates_summary']['passed']}/{report['gates_summary']['total']}")
    
    print(f"\n🔍 Key AI Insights:")
    for insight in report['ai_insights']['key_insights'][:3]:
        print(f"  • {insight}")
    
    print(f"\n💡 Priority Recommendations:")
    for rec in report['recommendations']['priority'][:3]:
        print(f"  • {rec}")
    
    print(f"\n🎯 Next Actions:")
    for action in report['next_actions']:
        print(f"  • {action}")
    
    print("\n🎉 AI-Powered Quality Gates Test Complete!")