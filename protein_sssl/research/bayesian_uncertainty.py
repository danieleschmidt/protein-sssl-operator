"""
Novel Bayesian Deep Ensemble Uncertainty Quantification for Protein Folding

This module implements a novel Bayesian deep ensemble method that goes beyond existing 
uncertainty quantification approaches by incorporating:
1. Hierarchical Bayesian modeling with structured priors
2. Epistemic and aleatoric uncertainty decomposition 
3. Information-theoretic uncertainty measures
4. Calibrated confidence prediction with temperature scaling
5. Novel ensemble diversity regularization

Mathematical Framework:
- p(y|x,D) = ∫ p(y|x,θ) p(θ|D) dθ  (Bayesian predictive distribution)
- H[y|x] = H[E[y|x,θ]] + E[H[y|x,θ]]  (Total = Epistemic + Aleatoric)
- I(y;θ|x) = H[y|x] - E[H[y|x,θ]]  (Mutual information epistemic measure)

Authors: Research Implementation for Academic Publication
License: MIT
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UncertaintyComponents:
    """Container for different types of uncertainty"""
    epistemic: np.ndarray  # Model uncertainty
    aleatoric: np.ndarray  # Data uncertainty  
    total: np.ndarray      # Combined uncertainty
    mutual_information: np.ndarray  # Information-theoretic measure
    confidence: np.ndarray  # Calibrated confidence scores
    
class BayesianPrior(ABC):
    """Abstract base class for Bayesian priors"""
    
    @abstractmethod
    def log_prob(self, params: np.ndarray) -> float:
        """Compute log probability of parameters under prior"""
        pass
    
    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """Sample from prior distribution"""
        pass

class HierarchicalGaussianPrior(BayesianPrior):
    """Hierarchical Gaussian prior with learnable hyperparameters"""
    
    def __init__(self, param_dim: int, tau_alpha: float = 1.0, tau_beta: float = 1.0):
        self.param_dim = param_dim
        self.tau_alpha = tau_alpha  # Gamma prior shape for precision
        self.tau_beta = tau_beta    # Gamma prior rate for precision
        
        # Initialize hyperparameters
        self.mu_0 = np.zeros(param_dim)
        self.tau = np.random.gamma(tau_alpha, 1/tau_beta)  # Precision parameter
        
    def log_prob(self, params: np.ndarray) -> float:
        """Log probability under hierarchical Gaussian prior"""
        # Gaussian likelihood
        gaussian_logprob = -0.5 * self.tau * np.sum((params - self.mu_0)**2)
        gaussian_logprob -= 0.5 * self.param_dim * np.log(2 * np.pi / self.tau)
        
        # Gamma prior on precision
        gamma_logprob = (self.tau_alpha - 1) * np.log(self.tau) - self.tau_beta * self.tau
        gamma_logprob += self.tau_alpha * np.log(self.tau_beta) - stats.loggamma(self.tau_alpha)
        
        return gaussian_logprob + gamma_logprob
    
    def sample(self, size: int) -> np.ndarray:
        """Sample from hierarchical prior"""
        # Sample precision from gamma prior
        tau_samples = np.random.gamma(self.tau_alpha, 1/self.tau_beta, size)
        
        # Sample parameters from Gaussian with sampled precision
        samples = []
        for tau in tau_samples:
            sigma = 1.0 / np.sqrt(tau)
            sample = np.random.normal(self.mu_0, sigma)
            samples.append(sample)
            
        return np.array(samples)
    
    def update_hyperparameters(self, params_ensemble: List[np.ndarray]):
        """Update hyperparameters based on ensemble parameters (empirical Bayes)"""
        if not params_ensemble:
            return
            
        params_array = np.array(params_ensemble)
        
        # Update mean (empirical mean)
        self.mu_0 = np.mean(params_array, axis=0)
        
        # Update precision (empirical precision with shrinkage)
        empirical_var = np.var(params_array, axis=0) + 1e-8
        self.tau = 1.0 / np.mean(empirical_var)

class ProteinStructureLikelihood:
    """Likelihood function for protein structure prediction"""
    
    def __init__(self, noise_model: str = "heteroscedastic"):
        self.noise_model = noise_model
        
    def log_likelihood(self, 
                      predictions: np.ndarray, 
                      targets: np.ndarray,
                      noise_params: Optional[np.ndarray] = None) -> float:
        """Compute log likelihood of predictions given targets"""
        
        if self.noise_model == "homoscedastic":
            # Constant noise variance
            sigma = 1.0 if noise_params is None else noise_params[0]
            log_lik = -0.5 * np.sum((predictions - targets)**2) / (sigma**2)
            log_lik -= 0.5 * len(targets) * np.log(2 * np.pi * sigma**2)
            
        elif self.noise_model == "heteroscedastic":
            # Position-dependent noise
            if noise_params is None:
                sigma = np.ones_like(targets)
            else:
                sigma = np.maximum(noise_params, 1e-6)  # Prevent numerical issues
                
            log_lik = -0.5 * np.sum((predictions - targets)**2 / (sigma**2))
            log_lik -= 0.5 * np.sum(np.log(2 * np.pi * sigma**2))
            
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")
            
        return log_lik

class NovelBayesianEnsemble:
    """
    Novel Bayesian Deep Ensemble with advanced uncertainty quantification
    
    Key innovations:
    1. Hierarchical Bayesian modeling with structured priors
    2. Information-theoretic epistemic uncertainty measures
    3. Temperature scaling for calibrated confidence
    4. Diversity regularization for ensemble quality
    5. Adaptive ensemble weighting based on local accuracy
    """
    
    def __init__(self,
                 ensemble_size: int = 10,
                 prior_type: str = "hierarchical_gaussian",
                 likelihood_model: str = "heteroscedastic",
                 temperature_scaling: bool = True,
                 diversity_regularization: float = 0.1,
                 confidence_intervals: List[float] = [0.68, 0.95, 0.99]):
        
        self.ensemble_size = ensemble_size
        self.temperature_scaling = temperature_scaling
        self.diversity_regularization = diversity_regularization  
        self.confidence_intervals = confidence_intervals
        
        # Initialize prior
        if prior_type == "hierarchical_gaussian":
            self.prior = HierarchicalGaussianPrior(param_dim=100)  # Will be updated
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
            
        # Initialize likelihood
        self.likelihood = ProteinStructureLikelihood(noise_model=likelihood_model)
        
        # Ensemble components
        self.ensemble_weights = []
        self.ensemble_predictions = []
        self.temperature = 1.0
        
        # Calibration data
        self.calibration_data = {"predictions": [], "targets": [], "confidences": []}
        
    def compute_epistemic_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """
        Compute epistemic uncertainty using mutual information
        I(y;θ|x) = H[y|x] - E[H[y|x,θ]]
        """
        if len(predictions.shape) != 2:
            raise ValueError("Predictions should be (ensemble_size, n_samples)")
            
        # Total entropy H[y|x]
        mean_pred = np.mean(predictions, axis=0)
        total_entropy = -mean_pred * np.log(mean_pred + 1e-8) - (1-mean_pred) * np.log(1-mean_pred + 1e-8)
        
        # Expected conditional entropy E[H[y|x,θ]]
        conditional_entropies = []
        for pred in predictions:
            h = -pred * np.log(pred + 1e-8) - (1-pred) * np.log(1-pred + 1e-8)
            conditional_entropies.append(h)
        expected_conditional_entropy = np.mean(conditional_entropies, axis=0)
        
        # Mutual information (epistemic uncertainty)
        epistemic = total_entropy - expected_conditional_entropy
        return np.maximum(epistemic, 0)  # Ensure non-negative
    
    def compute_aleatoric_uncertainty(self, predictions: np.ndarray, 
                                    noise_estimates: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute aleatoric (data) uncertainty"""
        if noise_estimates is not None:
            return noise_estimates
        else:
            # Estimate from prediction variance (simplified)
            return np.var(predictions, axis=0)
    
    def compute_mutual_information(self, predictions: np.ndarray) -> np.ndarray:
        """Compute mutual information I(y;θ|x) as epistemic uncertainty measure"""
        return self.compute_epistemic_uncertainty(predictions)
    
    def diversity_loss(self, predictions: np.ndarray) -> float:
        """
        Compute diversity regularization term to encourage diverse ensemble members
        Diversity = -E[correlation(p_i, p_j)] for i≠j
        """
        if len(predictions) < 2:
            return 0.0
            
        correlations = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0,1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    
        if not correlations:
            return 0.0
            
        # Return negative mean correlation (encourage diversity)
        return -np.mean(correlations)
    
    def fit_temperature_scaling(self, 
                               validation_predictions: np.ndarray,
                               validation_targets: np.ndarray) -> float:
        """
        Fit temperature parameter for calibrated confidence prediction
        T* = argmin NLL(softmax(logits/T), targets)
        """
        def nll_loss(temperature):
            if temperature <= 0:
                return np.inf
            calibrated_probs = self._apply_temperature(validation_predictions, temperature)
            nll = -np.sum(validation_targets * np.log(calibrated_probs + 1e-8))
            return nll
        
        # Grid search for optimal temperature  
        temperatures = np.logspace(-2, 2, 100)
        best_temp = 1.0
        best_nll = np.inf
        
        for temp in temperatures:
            nll = nll_loss(temp)
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
                
        self.temperature = best_temp
        logger.info(f"Optimal temperature: {best_temp:.4f}")
        return best_temp
    
    def _apply_temperature(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to logits"""
        scaled_logits = logits / temperature
        return self._softmax(scaled_logits)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def compute_confidence_intervals(self, 
                                   predictions: np.ndarray,
                                   confidence_levels: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """Compute confidence intervals at specified levels"""
        if confidence_levels is None:
            confidence_levels = self.confidence_intervals
            
        intervals = {}
        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            lower = np.percentile(predictions, lower_percentile, axis=0)
            upper = np.percentile(predictions, upper_percentile, axis=0)
            
            intervals[f"{level:.0%}"] = {"lower": lower, "upper": upper}
            
        return intervals
    
    def predict_with_uncertainty(self, 
                               input_data: np.ndarray,
                               model_ensemble: List[Callable],
                               return_all_components: bool = True) -> UncertaintyComponents:
        """
        Make predictions with comprehensive uncertainty quantification
        
        Args:
            input_data: Input features for prediction
            model_ensemble: List of trained models
            return_all_components: Whether to return all uncertainty components
            
        Returns:
            UncertaintyComponents object with all uncertainty measures
        """
        # Generate ensemble predictions
        predictions = []
        noise_estimates = []
        
        for model in model_ensemble:
            pred = model(input_data)
            predictions.append(pred)
            
            # Estimate aleatoric uncertainty (model-dependent)
            if hasattr(model, 'predict_noise'):
                noise = model.predict_noise(input_data)
                noise_estimates.append(noise)
        
        predictions = np.array(predictions)
        
        # Compute uncertainty components
        epistemic = self.compute_epistemic_uncertainty(predictions)
        
        if noise_estimates:
            aleatoric = np.mean(noise_estimates, axis=0)
        else:
            aleatoric = self.compute_aleatoric_uncertainty(predictions)
            
        total = epistemic + aleatoric
        mutual_info = self.compute_mutual_information(predictions)
        
        # Calibrated confidence
        mean_pred = np.mean(predictions, axis=0)
        if self.temperature_scaling:
            calibrated_probs = self._apply_temperature(mean_pred, self.temperature)
            confidence = np.max(calibrated_probs, axis=-1)
        else:
            confidence = 1.0 - total  # Simple confidence measure
            
        return UncertaintyComponents(
            epistemic=epistemic,
            aleatoric=aleatoric, 
            total=total,
            mutual_information=mutual_info,
            confidence=confidence
        )
    
    def evaluate_calibration(self, 
                           predictions: np.ndarray,
                           targets: np.ndarray,
                           confidences: np.ndarray,
                           n_bins: int = 10) -> Dict[str, float]:
        """
        Evaluate calibration quality using Expected Calibration Error (ECE)
        and other metrics
        """
        # Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1] 
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences_binned = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                accuracies.append(accuracy_in_bin)
                confidences_binned.append(avg_confidence_in_bin)
                bin_sizes.append(prop_in_bin)
            else:
                accuracies.append(0)
                confidences_binned.append(0) 
                bin_sizes.append(0)
        
        # Expected Calibration Error
        ece = np.sum([bin_sizes[i] * abs(accuracies[i] - confidences_binned[i]) 
                     for i in range(n_bins)])
        
        # Maximum Calibration Error
        mce = max([abs(accuracies[i] - confidences_binned[i]) for i in range(n_bins)])
        
        # Average Confidence
        avg_confidence = np.mean(confidences)
        
        # Average Accuracy
        avg_accuracy = np.mean(predictions == targets)
        
        return {
            "ece": ece,
            "mce": mce, 
            "avg_confidence": avg_confidence,
            "avg_accuracy": avg_accuracy,
            "reliability_gap": avg_confidence - avg_accuracy
        }
    
    def adaptive_ensemble_weighting(self, 
                                  predictions: np.ndarray,
                                  local_accuracies: np.ndarray) -> np.ndarray:
        """
        Compute adaptive weights for ensemble members based on local accuracy
        w_i = softmax(β * accuracy_i) where β controls sharpness
        """
        beta = 2.0  # Sharpness parameter
        weights = np.exp(beta * local_accuracies)
        weights = weights / np.sum(weights)
        return weights
    
    def save_ensemble(self, filepath: str):
        """Save ensemble state for reproducibility"""
        state = {
            "ensemble_size": self.ensemble_size,
            "temperature": self.temperature,
            "prior_state": self.prior.__dict__,
            "calibration_data": self.calibration_data,
            "diversity_regularization": self.diversity_regularization
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.ensemble_size = state["ensemble_size"]
        self.temperature = state["temperature"]
        self.calibration_data = state["calibration_data"]
        self.diversity_regularization = state["diversity_regularization"]
        
        # Restore prior state
        for key, value in state["prior_state"].items():
            setattr(self.prior, key, value)
            
        logger.info(f"Ensemble loaded from {filepath}")

class UncertaintyValidation:
    """Statistical validation framework for uncertainty quantification"""
    
    @staticmethod
    def coverage_test(predictions: np.ndarray,
                     targets: np.ndarray, 
                     confidence_intervals: Dict[str, Dict[str, np.ndarray]],
                     alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
        """
        Test if confidence intervals have correct empirical coverage
        H0: empirical_coverage = nominal_coverage
        """
        results = {}
        
        for level_name, interval in confidence_intervals.items():
            nominal_coverage = float(level_name.strip('%')) / 100
            
            # Check if targets fall within intervals
            in_interval = (targets >= interval["lower"]) & (targets <= interval["upper"])
            empirical_coverage = np.mean(in_interval)
            
            # Binomial test for coverage
            n = len(targets)
            p_value = 2 * min(
                stats.binom.cdf(np.sum(in_interval), n, nominal_coverage),
                1 - stats.binom.cdf(np.sum(in_interval) - 1, n, nominal_coverage)
            )
            
            results[level_name] = {
                "nominal_coverage": nominal_coverage,
                "empirical_coverage": empirical_coverage,
                "p_value": p_value,
                "significant_miscalibration": p_value < alpha
            }
            
        return results
    
    @staticmethod  
    def sharpness_test(confidence_intervals: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate interval sharpness (average width)
        Sharper intervals are better (assuming correct coverage)
        """
        sharpness = {}
        
        for level_name, interval in confidence_intervals.items():
            width = interval["upper"] - interval["lower"]
            avg_width = np.mean(width)
            sharpness[level_name] = avg_width
            
        return sharpness
    
    @staticmethod
    def proper_scoring_rules(predictions: np.ndarray,
                           targets: np.ndarray,
                           uncertainties: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions using proper scoring rules
        - Continuous Ranked Probability Score (CRPS)
        - Interval Score (IS)
        - Logarithmic Score
        """
        scores = {}
        
        # Logarithmic Score (for probabilistic predictions)
        # LS = -log(p(y|x)) where p is predictive density
        log_scores = []
        for i in range(len(predictions)):
            # Assume Gaussian predictive distribution
            mean = predictions[i]
            std = uncertainties[i] 
            log_score = -stats.norm.logpdf(targets[i], mean, std)
            log_scores.append(log_score)
        scores["logarithmic_score"] = np.mean(log_scores)
        
        # Continuous Ranked Probability Score (simplified)
        # CRPS = E[|Y - X|] - 0.5 * E[|X - X'|]
        crps_scores = []
        for i in range(len(predictions)):
            # Simplified CRPS for Gaussian distribution
            mean = predictions[i]
            std = uncertainties[i]
            normalized_target = (targets[i] - mean) / std
            crps = std * (normalized_target * (2 * stats.norm.cdf(normalized_target) - 1) + 
                         2 * stats.norm.pdf(normalized_target) - 1/np.sqrt(np.pi))
            crps_scores.append(crps)
        scores["crps"] = np.mean(crps_scores)
        
        return scores

# Example usage and validation
if __name__ == "__main__":
    # Initialize novel Bayesian ensemble
    ensemble = NovelBayesianEnsemble(
        ensemble_size=10,
        diversity_regularization=0.1,
        temperature_scaling=True
    )
    
    # Simulate protein structure prediction data
    n_samples = 1000
    n_features = 512
    
    # Mock ensemble predictions 
    mock_predictions = []
    for i in range(ensemble.ensemble_size):
        pred = np.random.normal(0.5, 0.1, (n_samples,))
        mock_predictions.append(pred)
    
    mock_predictions = np.array(mock_predictions)
    mock_targets = np.random.binomial(1, 0.5, n_samples)
    
    # Compute uncertainty components
    epistemic = ensemble.compute_epistemic_uncertainty(mock_predictions)
    aleatoric = ensemble.compute_aleatoric_uncertainty(mock_predictions)
    mutual_info = ensemble.compute_mutual_information(mock_predictions)
    
    print(f"Mean epistemic uncertainty: {np.mean(epistemic):.4f}")
    print(f"Mean aleatoric uncertainty: {np.mean(aleatoric):.4f}")
    print(f"Mean mutual information: {np.mean(mutual_info):.4f}")
    
    # Evaluate calibration
    confidences = 1.0 - (epistemic + aleatoric)
    mean_predictions = np.mean(mock_predictions, axis=0)
    binary_predictions = (mean_predictions > 0.5).astype(int)
    
    calibration_metrics = ensemble.evaluate_calibration(
        binary_predictions, mock_targets, confidences
    )
    print("\nCalibration Metrics:")
    for metric, value in calibration_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Statistical validation
    validation = UncertaintyValidation()
    
    # Generate confidence intervals
    confidence_intervals = ensemble.compute_confidence_intervals(mock_predictions)
    
    # Coverage test
    coverage_results = validation.coverage_test(
        mean_predictions, mock_targets.astype(float), confidence_intervals
    )
    print("\nCoverage Test Results:")
    for level, results in coverage_results.items():
        print(f"{level}: Empirical={results['empirical_coverage']:.3f}, "
              f"p-value={results['p_value']:.4f}")
    
    # Proper scoring rules
    scoring_results = validation.proper_scoring_rules(
        mean_predictions, mock_targets.astype(float), 
        epistemic + aleatoric
    )
    print("\nProper Scoring Rules:")
    for rule, score in scoring_results.items():
        print(f"{rule}: {score:.4f}")
    
    logger.info("Novel Bayesian ensemble uncertainty quantification validation complete!")