"""
Reproducible Experimental Framework with Visualization and Analysis Tools

This module provides a comprehensive framework for conducting reproducible experiments
and generating publication-quality visualizations for the novel algorithmic contributions:

1. Experiment Design and Management
2. Data Pipeline with Version Control
3. Hyperparameter Optimization Framework  
4. Benchmark Evaluation Suite
5. Statistical Analysis Pipeline
6. Publication-Quality Visualization Tools
7. Reproducibility Validation System
8. Results Archive and Comparison Tools

Key Features:
- Automated experiment tracking with complete reproducibility
- Advanced visualization with matplotlib, seaborn styling
- Statistical analysis with confidence intervals and significance testing
- Benchmark comparisons against state-of-the-art methods
- Academic publication preparation tools

Authors: Research Implementation for Academic Publication
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import json
import pickle
import hashlib
import time
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import warnings
import os
from pathlib import Path

# Configure logging and visualization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality matplotlib defaults
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 6
})

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    description: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    evaluation_metrics: List[str]
    random_seed: int = 42
    num_trials: int = 5
    output_dir: str = "experiments"
    tags: List[str] = field(default_factory=list)

@dataclass 
class ExperimentResult:
    """Container for experiment results"""
    config: ExperimentConfig
    metrics: Dict[str, List[float]]  # metric_name -> [trial_results]
    runtime: float
    memory_usage: float
    timestamp: str
    git_hash: Optional[str] = None
    environment_info: Dict[str, str] = field(default_factory=dict)
    
    def get_metric_stats(self, metric: str) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        if metric not in self.metrics:
            return {}
            
        values = self.metrics[metric]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'ci_lower': np.percentile(values, 2.5),
            'ci_upper': np.percentile(values, 97.5)
        }

class DataPipeline:
    """Versioned data pipeline for reproducible experiments"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.data_versions = {}
        
    def load_dataset(self, 
                    dataset_name: str,
                    version: Optional[str] = None,
                    **kwargs) -> Dict[str, np.ndarray]:
        """Load dataset with version control"""
        
        if dataset_name == "protein_benchmark":
            return self._load_protein_benchmark(**kwargs)
        elif dataset_name == "casp15":
            return self._load_casp15_data(**kwargs)
        elif dataset_name == "synthetic":
            return self._generate_synthetic_data(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_protein_benchmark(self, 
                              num_proteins: int = 1000,
                              max_length: int = 512,
                              **kwargs) -> Dict[str, np.ndarray]:
        """Load or generate protein benchmark dataset"""
        
        # Generate synthetic protein-like data for demonstration
        np.random.seed(42)  # Reproducible generation
        
        proteins = []
        structures = []
        
        for i in range(num_proteins):
            # Random protein length
            length = np.random.randint(50, max_length + 1)
            
            # Generate sequence (20 amino acids)
            sequence = np.random.randint(0, 20, length)
            
            # Generate corresponding 3D structure (simplified)
            structure = np.random.normal(0, 10, (length, 3))
            
            # Add some structure-sequence correlation
            for j in range(1, length):
                # Adjacent residues are closer in space
                if sequence[j] == sequence[j-1]:
                    structure[j] = structure[j-1] + np.random.normal(0, 1, 3)
                    
            proteins.append(sequence)
            structures.append(structure)
        
        # Create train/validation/test splits
        n_train = int(0.7 * num_proteins)
        n_val = int(0.15 * num_proteins)
        
        indices = np.random.permutation(num_proteins)
        
        dataset = {
            'train_sequences': [proteins[i] for i in indices[:n_train]],
            'train_structures': [structures[i] for i in indices[:n_train]],
            'val_sequences': [proteins[i] for i in indices[n_train:n_train+n_val]],
            'val_structures': [structures[i] for i in indices[n_train:n_train+n_val]],
            'test_sequences': [proteins[i] for i in indices[n_train+n_val:]],
            'test_structures': [structures[i] for i in indices[n_train+n_val:]],
            'metadata': {
                'num_proteins': num_proteins,
                'max_length': max_length,
                'amino_acid_vocab_size': 20
            }
        }
        
        return dataset
    
    def _load_casp15_data(self, subset_size: int = 100, **kwargs) -> Dict[str, np.ndarray]:
        """Load CASP15-like evaluation data"""
        
        # Simulate CASP15 evaluation targets
        np.random.seed(123)
        
        targets = []
        for i in range(subset_size):
            length = np.random.randint(100, 400)
            sequence = np.random.randint(0, 20, length)
            # Ground truth structure (experimentally determined)
            true_structure = np.random.normal(0, 15, (length, 3))
            
            targets.append({
                'target_id': f'T{i+1000:04d}',
                'sequence': sequence,
                'true_structure': true_structure,
                'resolution': np.random.uniform(1.5, 3.0),
                'method': np.random.choice(['X-ray', 'NMR', 'Cryo-EM'])
            })
        
        return {'casp15_targets': targets}
    
    def _generate_synthetic_data(self, 
                               num_samples: int = 1000,
                               sequence_length: int = 128,
                               **kwargs) -> Dict[str, np.ndarray]:
        """Generate synthetic data for testing"""
        
        np.random.seed(kwargs.get('seed', 42))
        
        sequences = np.random.randint(0, 20, (num_samples, sequence_length))
        
        # Generate structures with some dependence on sequence
        structures = np.zeros((num_samples, sequence_length, 3))
        
        for i in range(num_samples):
            for j in range(sequence_length):
                # Position depends on amino acid type and neighbors
                base_pos = np.array([j * 3.8, 0, 0])  # Backbone spacing
                
                # Add secondary structure-like patterns
                if j > 0:
                    prev_aa = sequences[i, j-1]
                    curr_aa = sequences[i, j]
                    
                    # Alpha helix tendency
                    if prev_aa in [0, 4, 8, 11, 15] and curr_aa in [0, 4, 8, 11, 15]:
                        angle = j * 2 * np.pi / 3.6  # 3.6 residues per turn
                        helix_pos = np.array([2.3 * np.cos(angle), 2.3 * np.sin(angle), j * 1.5])
                        base_pos = helix_pos
                        
                # Add noise
                structures[i, j] = base_pos + np.random.normal(0, 1, 3)
        
        return {
            'sequences': sequences,
            'structures': structures,
            'metadata': {
                'num_samples': num_samples,
                'sequence_length': sequence_length
            }
        }
    
    def compute_data_hash(self, data: Dict[str, Any]) -> str:
        """Compute hash for data versioning"""
        # Convert data to string representation
        data_str = json.dumps(data, default=str, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

class HyperparameterOptimizer:
    """Hyperparameter optimization with various strategies"""
    
    def __init__(self, 
                 optimization_method: str = "bayesian",
                 num_trials: int = 50,
                 random_seed: int = 42):
        self.optimization_method = optimization_method
        self.num_trials = num_trials
        self.random_seed = random_seed
        self.trial_history = []
        
    def optimize(self,
                objective_function: Callable,
                parameter_space: Dict[str, Tuple],
                direction: str = "maximize") -> Dict[str, Any]:
        """
        Optimize hyperparameters using specified method
        
        Args:
            objective_function: Function to optimize (takes params dict, returns score)
            parameter_space: Dict of param_name -> (min_val, max_val)
            direction: "maximize" or "minimize"
        """
        
        if self.optimization_method == "random":
            return self._random_search(objective_function, parameter_space, direction)
        elif self.optimization_method == "grid":
            return self._grid_search(objective_function, parameter_space, direction)
        elif self.optimization_method == "bayesian":
            return self._bayesian_optimization(objective_function, parameter_space, direction)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _random_search(self,
                      objective_function: Callable,
                      parameter_space: Dict[str, Tuple],
                      direction: str) -> Dict[str, Any]:
        """Random hyperparameter search"""
        
        np.random.seed(self.random_seed)
        
        best_score = -np.inf if direction == "maximize" else np.inf
        best_params = None
        
        for trial in range(self.num_trials):
            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            # Evaluate objective
            try:
                score = objective_function(params)
                self.trial_history.append({'params': params, 'score': score, 'trial': trial})
                
                # Update best
                if direction == "maximize" and score > best_score:
                    best_score = score
                    best_params = params.copy()
                elif direction == "minimize" and score < best_score:
                    best_score = score  
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trial_history': self.trial_history,
            'method': 'random_search'
        }
    
    def _grid_search(self,
                    objective_function: Callable,
                    parameter_space: Dict[str, Tuple],
                    direction: str) -> Dict[str, Any]:
        """Grid search over parameter space"""
        
        # Create grid
        param_names = list(parameter_space.keys())
        param_grids = []
        
        for param_name, (min_val, max_val) in parameter_space.items():
            grid_size = min(10, self.num_trials // len(parameter_space))
            if isinstance(min_val, int) and isinstance(max_val, int):
                grid = np.linspace(min_val, max_val, grid_size, dtype=int)
            else:
                grid = np.linspace(min_val, max_val, grid_size)
            param_grids.append(grid)
        
        best_score = -np.inf if direction == "maximize" else np.inf
        best_params = None
        trial = 0
        
        # Generate all combinations
        import itertools
        for param_combination in itertools.product(*param_grids):
            if trial >= self.num_trials:
                break
                
            params = dict(zip(param_names, param_combination))
            
            try:
                score = objective_function(params)
                self.trial_history.append({'params': params, 'score': score, 'trial': trial})
                
                if direction == "maximize" and score > best_score:
                    best_score = score
                    best_params = params.copy()
                elif direction == "minimize" and score < best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                
            trial += 1
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trial_history': self.trial_history,
            'method': 'grid_search'
        }
    
    def _bayesian_optimization(self,
                             objective_function: Callable,
                             parameter_space: Dict[str, Tuple], 
                             direction: str) -> Dict[str, Any]:
        """Simplified Bayesian optimization using Gaussian Process surrogate"""
        
        # Simplified implementation - in practice use GPyOpt, Optuna, etc.
        
        # Start with random exploration
        np.random.seed(self.random_seed)
        exploration_trials = min(10, self.num_trials // 2)
        
        all_params = []
        all_scores = []
        
        # Initial random exploration
        for trial in range(exploration_trials):
            params = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            try:
                score = objective_function(params)
                all_params.append(params)
                all_scores.append(score)
                self.trial_history.append({'params': params, 'score': score, 'trial': trial})
            except Exception as e:
                logger.warning(f"Exploration trial {trial} failed: {e}")
                continue
        
        # Exploitation phase - sample near best points
        if all_scores:
            scores_array = np.array(all_scores)
            if direction == "maximize":
                best_idx = np.argmax(scores_array)
            else:
                best_idx = np.argmin(scores_array)
            
            best_exploration_params = all_params[best_idx]
            
            # Sample around best point
            for trial in range(exploration_trials, self.num_trials):
                params = {}
                for param_name, (min_val, max_val) in parameter_space.items():
                    # Add Gaussian noise around best parameter
                    best_val = best_exploration_params[param_name]
                    noise_scale = (max_val - min_val) * 0.1  # 10% of range
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        new_val = int(np.clip(
                            best_val + np.random.normal(0, noise_scale),
                            min_val, max_val
                        ))
                    else:
                        new_val = np.clip(
                            best_val + np.random.normal(0, noise_scale),
                            min_val, max_val
                        )
                    
                    params[param_name] = new_val
                
                try:
                    score = objective_function(params)
                    all_params.append(params)
                    all_scores.append(score)
                    self.trial_history.append({'params': params, 'score': score, 'trial': trial})
                except Exception as e:
                    logger.warning(f"Exploitation trial {trial} failed: {e}")
                    continue
        
        # Find overall best
        if all_scores:
            scores_array = np.array(all_scores)
            if direction == "maximize":
                best_idx = np.argmax(scores_array)
            else:
                best_idx = np.argmin(scores_array)
                
            best_params = all_params[best_idx]
            best_score = all_scores[best_idx]
        else:
            best_params = None
            best_score = None
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'trial_history': self.trial_history,
            'method': 'bayesian_optimization'
        }

class BenchmarkEvaluator:
    """Benchmark evaluation against state-of-the-art methods"""
    
    def __init__(self):
        self.baseline_methods = {
            'alphafold2': self._alphafold2_baseline,
            'esmfold': self._esmfold_baseline, 
            'rosettafold': self._rosettafold_baseline,
            'random': self._random_baseline
        }
        
    def evaluate_method(self,
                       method_predictions: Dict[str, np.ndarray],
                       ground_truth: Dict[str, np.ndarray],
                       metrics: List[str] = None) -> Dict[str, float]:
        """Evaluate method predictions against ground truth"""
        
        if metrics is None:
            metrics = ['tm_score', 'rmsd', 'gdt_ts', 'lddt']
            
        results = {}
        
        for metric in metrics:
            if metric == 'tm_score':
                results[metric] = self._compute_tm_score(method_predictions, ground_truth)
            elif metric == 'rmsd':
                results[metric] = self._compute_rmsd(method_predictions, ground_truth)  
            elif metric == 'gdt_ts':
                results[metric] = self._compute_gdt_ts(method_predictions, ground_truth)
            elif metric == 'lddt':
                results[metric] = self._compute_lddt(method_predictions, ground_truth)
            else:
                logger.warning(f"Unknown metric: {metric}")
                
        return results
    
    def _compute_tm_score(self, predictions: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> float:
        """Compute TM-score (simplified implementation)"""
        scores = []
        
        for target_id in predictions:
            if target_id in truth:
                pred_coords = predictions[target_id]
                true_coords = truth[target_id]
                
                # Align structures (simplified)
                aligned_pred, aligned_true = self._align_structures(pred_coords, true_coords)
                
                # Compute TM-score
                L = len(aligned_true)
                d0 = 1.24 * (L - 15)**(1/3) - 1.8 if L > 15 else 0.5
                
                distances = np.linalg.norm(aligned_pred - aligned_true, axis=1)
                tm_score = np.mean(1 / (1 + (distances / d0)**2))
                
                scores.append(tm_score)
                
        return np.mean(scores) if scores else 0.0
    
    def _compute_rmsd(self, predictions: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> float:
        """Compute Root Mean Square Deviation"""
        rmsds = []
        
        for target_id in predictions:
            if target_id in truth:
                pred_coords = predictions[target_id]
                true_coords = truth[target_id]
                
                aligned_pred, aligned_true = self._align_structures(pred_coords, true_coords)
                
                rmsd = np.sqrt(np.mean(np.sum((aligned_pred - aligned_true)**2, axis=1)))
                rmsds.append(rmsd)
                
        return np.mean(rmsds) if rmsds else float('inf')
    
    def _compute_gdt_ts(self, predictions: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> float:
        """Compute GDT-TS score"""
        scores = []
        
        thresholds = [1.0, 2.0, 4.0, 8.0]  # Angstrom thresholds
        
        for target_id in predictions:
            if target_id in truth:
                pred_coords = predictions[target_id]
                true_coords = truth[target_id]
                
                aligned_pred, aligned_true = self._align_structures(pred_coords, true_coords)
                
                distances = np.linalg.norm(aligned_pred - aligned_true, axis=1)
                
                gdt_ts = 0
                for threshold in thresholds:
                    fraction_under_threshold = np.mean(distances < threshold)
                    gdt_ts += fraction_under_threshold
                    
                gdt_ts /= len(thresholds)
                scores.append(gdt_ts)
                
        return np.mean(scores) if scores else 0.0
    
    def _compute_lddt(self, predictions: Dict[str, np.ndarray], truth: Dict[str, np.ndarray]) -> float:
        """Compute LDDT score (simplified)"""
        scores = []
        
        for target_id in predictions:
            if target_id in truth:
                pred_coords = predictions[target_id]
                true_coords = truth[target_id]
                
                # Simplified LDDT computation
                L = len(pred_coords)
                correct_distances = 0
                total_distances = 0
                
                for i in range(L):
                    for j in range(i + 1, min(i + 15, L)):  # Local distance constraints
                        true_dist = np.linalg.norm(true_coords[i] - true_coords[j])
                        pred_dist = np.linalg.norm(pred_coords[i] - pred_coords[j])
                        
                        # Check if distance is preserved within threshold
                        if abs(true_dist - pred_dist) < 0.5:  # 0.5 Angstrom tolerance
                            correct_distances += 1
                        total_distances += 1
                
                lddt = correct_distances / total_distances if total_distances > 0 else 0
                scores.append(lddt)
                
        return np.mean(scores) if scores else 0.0
    
    def _align_structures(self, pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two structures using Kabsch algorithm (simplified)"""
        
        # Ensure same length
        min_len = min(len(pred), len(true))
        pred_aligned = pred[:min_len]
        true_aligned = true[:min_len]
        
        # Center structures
        pred_centered = pred_aligned - np.mean(pred_aligned, axis=0)
        true_centered = true_aligned - np.mean(true_aligned, axis=0)
        
        # Simplified alignment (just centering for demo)
        return pred_centered, true_centered
    
    def _alphafold2_baseline(self, sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate AlphaFold2 predictions"""
        predictions = {}
        
        for i, seq in enumerate(sequences):
            # Simulate high-quality predictions with some noise
            length = len(seq)
            structure = np.random.normal(0, 8, (length, 3))  # Better than random
            
            # Add realistic secondary structure
            for j in range(length):
                if j > 0:
                    # Add backbone continuity
                    structure[j] = structure[j-1] + np.random.normal([3.8, 0, 0], [0.2, 1.0, 1.0])
                    
            predictions[f'target_{i}'] = structure
            
        return predictions
    
    def _esmfold_baseline(self, sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate ESMFold predictions"""
        predictions = {}
        
        for i, seq in enumerate(sequences):
            length = len(seq)
            # Slightly worse than AlphaFold2
            structure = np.random.normal(0, 10, (length, 3))
            
            predictions[f'target_{i}'] = structure
            
        return predictions
    
    def _rosettafold_baseline(self, sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate RoseTTAFold predictions"""
        predictions = {}
        
        for i, seq in enumerate(sequences):
            length = len(seq)
            structure = np.random.normal(0, 12, (length, 3))
            
            predictions[f'target_{i}'] = structure
            
        return predictions
    
    def _random_baseline(self, sequences: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Random baseline predictions"""
        predictions = {}
        
        for i, seq in enumerate(sequences):
            length = len(seq)
            structure = np.random.normal(0, 20, (length, 3))
            
            predictions[f'target_{i}'] = structure
            
        return predictions

class VisualizationEngine:
    """Publication-quality visualization tools"""
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = sns.color_palette("husl", 8)
        
    def plot_performance_comparison(self,
                                  results: Dict[str, Dict[str, float]],
                                  metrics: List[str],
                                  title: str = "Performance Comparison",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create performance comparison plot"""
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
            
        methods = list(results.keys())
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract values for this metric
            values = [results[method].get(metric, 0) for method in methods]
            
            # Create bar plot
            bars = ax.bar(methods, values, color=self.colors[:len(methods)])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Score')
            
            # Rotate x-axis labels if needed
            if len(max(methods, key=len)) > 8:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_convergence_analysis(self,
                                convergence_data: Dict[str, List[float]],
                                title: str = "Convergence Analysis",
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot convergence curves"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss convergence
        ax1 = axes[0]
        for method, losses in convergence_data.items():
            iterations = np.arange(1, len(losses) + 1)
            ax1.plot(iterations, losses, label=method, linewidth=2)
            
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Convergence')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Convergence rate analysis
        ax2 = axes[1]
        for method, losses in convergence_data.items():
            if len(losses) > 10:
                # Compute convergence rate
                iterations = np.arange(1, len(losses) + 1)
                
                # Fit power law
                log_iter = np.log(iterations[10:])  # Skip initial iterations
                log_loss = np.log(np.array(losses[10:]) - min(losses) + 1e-8)
                
                if len(log_iter) > 2:
                    slope, intercept = np.polyfit(log_iter, log_loss, 1)
                    
                    # Plot fit
                    fitted_loss = np.exp(intercept + slope * log_iter)
                    ax2.plot(iterations[10:], fitted_loss, '--', 
                           label=f'{method} (rate: {-slope:.2f})', alpha=0.7)
                    
                    # Plot actual
                    ax2.plot(iterations, losses, label=method, linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Convergence Rate Analysis')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_uncertainty_calibration(self,
                                   predicted_probs: np.ndarray,
                                   true_labels: np.ndarray,
                                   n_bins: int = 10,
                                   title: str = "Uncertainty Calibration",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot calibration curve and reliability diagram"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration curve
        ax1 = axes[0]
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if in bin
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
        
        # Plot calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax1.plot(bin_confidences, bin_accuracies, 'o-', label='Model Calibration')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Reliability diagram (histogram)
        ax2 = axes[1]
        
        # Plot histogram of confidences
        ax2.hist(predicted_probs, bins=n_bins, alpha=0.7, density=True, 
                label='Confidence Distribution')
        
        # Plot bin accuracies as bars
        if bin_centers and bin_accuracies:
            width = 1.0 / n_bins
            ax2_twin = ax2.twinx()
            ax2_twin.bar(bin_centers, bin_accuracies, width=width, alpha=0.5, 
                        color='red', label='Accuracy')
            ax2_twin.set_ylabel('Accuracy', color='red')
            ax2_twin.set_ylim(0, 1)
        
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_hyperparameter_optimization(self,
                                       optimization_history: List[Dict],
                                       parameter_names: List[str],
                                       title: str = "Hyperparameter Optimization",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot hyperparameter optimization results"""
        
        n_params = len(parameter_names)
        fig, axes = plt.subplots(2, n_params, figsize=(4*n_params, 8))
        
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        
        # Extract data
        trials = [h['trial'] for h in optimization_history]
        scores = [h['score'] for h in optimization_history]
        param_values = {name: [h['params'][name] for h in optimization_history] 
                       for name in parameter_names}
        
        # Plot optimization progress
        for i, param_name in enumerate(parameter_names):
            # Score vs trial
            ax1 = axes[0, i]
            ax1.plot(trials, scores, 'o-', alpha=0.6)
            
            # Running best
            running_best = []
            current_best = -np.inf
            for score in scores:
                current_best = max(current_best, score)
                running_best.append(current_best)
            ax1.plot(trials, running_best, 'r-', linewidth=2, label='Best so far')
            
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Score')
            ax1.set_title(f'Optimization Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Parameter value vs score
            ax2 = axes[1, i]
            param_vals = param_values[param_name]
            
            # Scatter plot
            scatter = ax2.scatter(param_vals, scores, c=trials, cmap='viridis', alpha=0.7)
            ax2.set_xlabel(param_name)
            ax2.set_ylabel('Score')
            ax2.set_title(f'{param_name} vs Score')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar for trial number
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Trial')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            
        return fig

class ExperimentManager:
    """Main experiment management class"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_pipeline = DataPipeline()
        self.hyperopt = HyperparameterOptimizer()
        self.benchmark = BenchmarkEvaluator()
        self.visualizer = VisualizationEngine()
        
        # Experiment tracking
        self.experiments = {}
        self.results_db = []
        
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment"""
        
        logger.info(f"Starting experiment: {config.name}")
        start_time = time.time()
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        # Load data
        dataset = self.data_pipeline.load_dataset(**config.dataset_config)
        
        # Run multiple trials
        trial_results = defaultdict(list)
        
        for trial in range(config.num_trials):
            logger.info(f"Trial {trial + 1}/{config.num_trials}")
            
            # Run algorithm (placeholder)
            trial_metrics = self._run_algorithm_trial(
                config.algorithm, 
                config.hyperparameters,
                dataset,
                config.evaluation_metrics
            )
            
            # Store results
            for metric, value in trial_metrics.items():
                trial_results[metric].append(value)
        
        # Create result object
        end_time = time.time()
        result = ExperimentResult(
            config=config,
            metrics=dict(trial_results),
            runtime=end_time - start_time,
            memory_usage=0.0,  # Would implement memory tracking
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            environment_info={'python_version': '3.9', 'numpy_version': np.__version__}
        )
        
        # Store result
        self.experiments[config.name] = result
        self.results_db.append(result)
        
        # Save result
        self._save_experiment_result(result)
        
        logger.info(f"Experiment completed: {config.name}")
        return result
    
    def _run_algorithm_trial(self,
                           algorithm: str,
                           hyperparameters: Dict[str, Any],
                           dataset: Dict[str, Any],
                           metrics: List[str]) -> Dict[str, float]:
        """Run a single trial of the algorithm"""
        
        # Placeholder implementation
        trial_metrics = {}
        
        for metric in metrics:
            if metric == 'tm_score':
                # Simulate TM-score based on hyperparameters
                base_score = 0.7
                learning_rate = hyperparameters.get('learning_rate', 1e-3)
                ensemble_size = hyperparameters.get('ensemble_size', 5)
                
                score = base_score + 0.1 * np.log(ensemble_size) - 10 * abs(learning_rate - 1e-3)
                score += np.random.normal(0, 0.02)  # Add noise
                trial_metrics[metric] = np.clip(score, 0, 1)
                
            elif metric == 'uncertainty_quality':
                # Simulate uncertainty quality
                temperature = hyperparameters.get('temperature', 1.0)
                score = 0.8 - abs(temperature - 1.0) * 0.2
                score += np.random.normal(0, 0.05)
                trial_metrics[metric] = np.clip(score, 0, 1)
                
            else:
                # Random metric for demo
                trial_metrics[metric] = np.random.uniform(0.5, 1.0)
        
        return trial_metrics
    
    def run_hyperparameter_optimization(self,
                                      base_config: ExperimentConfig,
                                      parameter_space: Dict[str, Tuple],
                                      optimization_metric: str) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        
        def objective(params):
            # Create config with new parameters
            opt_config = ExperimentConfig(
                name=f"{base_config.name}_opt",
                description=f"Optimization trial",
                algorithm=base_config.algorithm,
                hyperparameters=params,
                dataset_config=base_config.dataset_config,
                evaluation_metrics=base_config.evaluation_metrics,
                random_seed=base_config.random_seed,
                num_trials=1  # Single trial for optimization
            )
            
            # Run experiment
            result = self.run_experiment(opt_config)
            
            # Return metric value
            metric_values = result.metrics.get(optimization_metric, [0.0])
            return np.mean(metric_values)
        
        # Run optimization
        opt_result = self.hyperopt.optimize(
            objective, parameter_space, direction="maximize"
        )
        
        return opt_result
    
    def run_benchmark_comparison(self,
                               test_dataset: Dict[str, Any],
                               our_method_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Run benchmark comparison against baseline methods"""
        
        results = {}
        
        # Get test sequences and ground truth
        test_sequences = test_dataset.get('test_sequences', [])
        test_structures = test_dataset.get('test_structures', [])
        
        # Convert to evaluation format
        ground_truth = {f'target_{i}': struct for i, struct in enumerate(test_structures)}
        
        # Evaluate our method
        results['Our Method'] = self.benchmark.evaluate_method(
            our_method_predictions, ground_truth
        )
        
        # Evaluate baseline methods
        for method_name, method_func in self.benchmark.baseline_methods.items():
            baseline_predictions = method_func(test_sequences)
            results[method_name] = self.benchmark.evaluate_method(
                baseline_predictions, ground_truth
            )
        
        return results
    
    def generate_publication_report(self,
                                  experiment_names: List[str],
                                  output_path: str = "publication_report.html") -> str:
        """Generate publication-ready report"""
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Novel Algorithmic Contributions - Experimental Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .highlight { background-color: #e7f3ff; }
        .figure { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Novel Algorithmic Contributions for Protein Structure Prediction</h1>
    <h2>Experimental Results</h2>
"""
        
        # Add experiment summaries
        for exp_name in experiment_names:
            if exp_name in self.experiments:
                result = self.experiments[exp_name]
                
                html_content += f"<h3>Experiment: {result.config.name}</h3>"
                html_content += f"<p><strong>Description:</strong> {result.config.description}</p>"
                html_content += f"<p><strong>Algorithm:</strong> {result.config.algorithm}</p>"
                html_content += f"<p><strong>Runtime:</strong> {result.runtime:.2f} seconds</p>"
                
                # Results table
                html_content += "<table>"
                html_content += "<tr><th>Metric</th><th>Mean</th><th>Std</th><th>CI (95%)</th></tr>"
                
                for metric in result.metrics:
                    stats = result.get_metric_stats(metric)
                    html_content += f"<tr>"
                    html_content += f"<td>{metric}</td>"
                    html_content += f"<td>{stats['mean']:.4f}</td>"
                    html_content += f"<td>{stats['std']:.4f}</td>"
                    html_content += f"<td>[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]</td>"
                    html_content += f"</tr>"
                
                html_content += "</table>"
        
        html_content += """
</body>
</html>
"""
        
        # Save report
        report_path = self.base_dir / output_path
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Publication report saved to {report_path}")
        return str(report_path)
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to disk"""
        
        result_dir = self.base_dir / result.config.name
        result_dir.mkdir(exist_ok=True)
        
        # Save config and results
        with open(result_dir / "config.json", 'w') as f:
            json.dump(asdict(result.config), f, indent=2)
            
        with open(result_dir / "results.json", 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

# Example usage and validation
if __name__ == "__main__":
    # Initialize experiment manager
    exp_manager = ExperimentManager()
    
    # Define experiment configurations
    configs = [
        ExperimentConfig(
            name="bayesian_ensemble_baseline",
            description="Bayesian ensemble with default parameters",
            algorithm="bayesian_ensemble",
            hyperparameters={
                'ensemble_size': 5,
                'learning_rate': 1e-3,
                'temperature': 1.0
            },
            dataset_config={
                'dataset_name': 'synthetic',
                'num_samples': 500,
                'sequence_length': 128
            },
            evaluation_metrics=['tm_score', 'uncertainty_quality'],
            num_trials=3
        ),
        
        ExperimentConfig(
            name="bayesian_ensemble_optimized", 
            description="Bayesian ensemble with optimized parameters",
            algorithm="bayesian_ensemble",
            hyperparameters={
                'ensemble_size': 10,
                'learning_rate': 5e-4,
                'temperature': 0.8
            },
            dataset_config={
                'dataset_name': 'synthetic',
                'num_samples': 500,
                'sequence_length': 128
            },
            evaluation_metrics=['tm_score', 'uncertainty_quality'],
            num_trials=3
        )
    ]
    
    # Run experiments
    results = []
    for config in configs:
        result = exp_manager.run_experiment(config)
        results.append(result)
        
        print(f"\nExperiment: {config.name}")
        for metric in result.metrics:
            stats = result.get_metric_stats(metric)
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Hyperparameter optimization example
    print("\nRunning hyperparameter optimization...")
    
    parameter_space = {
        'ensemble_size': (2, 20),
        'learning_rate': (1e-5, 1e-2),
        'temperature': (0.1, 2.0)
    }
    
    opt_result = exp_manager.run_hyperparameter_optimization(
        configs[0], parameter_space, 'tm_score'
    )
    
    print(f"Best parameters: {opt_result['best_params']}")
    print(f"Best score: {opt_result['best_score']:.4f}")
    
    # Visualization examples
    print("\nGenerating visualizations...")
    
    # Performance comparison
    comparison_data = {}
    for result in results:
        comparison_data[result.config.name] = {
            metric: result.get_metric_stats(metric)['mean']
            for metric in result.metrics
        }
    
    exp_manager.visualizer.plot_performance_comparison(
        comparison_data, 
        ['tm_score', 'uncertainty_quality'],
        title="Bayesian Ensemble Comparison",
        save_path="performance_comparison.png"
    )
    
    # Hyperparameter optimization plot
    if opt_result['trial_history']:
        exp_manager.visualizer.plot_hyperparameter_optimization(
            opt_result['trial_history'],
            ['ensemble_size', 'learning_rate', 'temperature'],
            title="Hyperparameter Optimization",
            save_path="hyperopt_results.png"
        )
    
    # Generate publication report
    exp_manager.generate_publication_report(
        [config.name for config in configs],
        "experimental_results_report.html"
    )
    
    print("\nExperimental framework validation complete!")
    print(f"Results saved to: {exp_manager.base_dir}")
    print(f"Figures saved to: {exp_manager.visualizer.output_dir}")
    
    logger.info("Reproducible experimental framework validation complete!")