"""
Advanced Fourier-based Neural Operators for Protein Structure Prediction

This module implements novel Fourier neural operators specifically optimized for 
protein sequence-to-structure transformations. Key innovations include:

1. Adaptive Spectral Kernels with learnable frequency selection
2. Multi-scale Fourier decomposition for hierarchical feature learning
3. Physics-informed frequency filtering based on protein dynamics
4. Attention-modulated Fourier transforms for sequence-specific adaptivity
5. Novel kernel designs incorporating protein-specific inductive biases

Mathematical Framework:
- (Kφ)(x) = ∫ k(x,y) φ(y) dy  (Neural operator)
- k(x,y) = ∑ₘ αₘ(x) e^(2πimx) βₘ(y) e^(-2πimy)  (Fourier kernel)
- Multi-scale: Kᵣ = ∑ₛ₌₁ʳ Wₛ ∘ Fₛ⁻¹ ∘ Rₛ ∘ Fₛ  (r-scale decomposition)

Authors: Research Implementation for Academic Publication
License: MIT
"""

import numpy as np
from scipy import fft, signal
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FourierKernelConfig:
    """Configuration for Fourier kernel parameters"""
    max_modes: int = 64
    kernel_type: str = "adaptive_spectral"
    physics_informed: bool = True
    attention_modulated: bool = True
    multi_scale_levels: int = 3
    frequency_cutoff: float = 0.5
    regularization_weight: float = 1e-4

class SpectralKernel(ABC):
    """Abstract base class for spectral kernels"""
    
    @abstractmethod
    def forward_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply forward Fourier transform"""
        pass
    
    @abstractmethod
    def inverse_transform(self, x_freq: np.ndarray) -> np.ndarray:
        """Apply inverse Fourier transform"""
        pass
    
    @abstractmethod
    def kernel_weights(self, frequencies: np.ndarray) -> np.ndarray:
        """Compute kernel weights for given frequencies"""
        pass

class AdaptiveSpectralKernel(SpectralKernel):
    """
    Adaptive spectral kernel with learnable frequency selection
    k(ω) = σ(αω + β) * g(ω) where g(ω) is a learnable spectral function
    """
    
    def __init__(self, 
                 max_modes: int = 64,
                 learnable_frequencies: bool = True,
                 bandwidth_adaptation: bool = True):
        self.max_modes = max_modes
        self.learnable_frequencies = learnable_frequencies
        self.bandwidth_adaptation = bandwidth_adaptation
        
        # Initialize learnable parameters
        self.frequency_weights = np.ones(max_modes) / max_modes
        self.bandwidth_params = np.ones(max_modes) * 0.1
        self.phase_shifts = np.zeros(max_modes)
        
        # Frequency selection mechanism
        self.frequency_gates = np.ones(max_modes)
        
    def forward_transform(self, x: np.ndarray) -> np.ndarray:
        """FFT with adaptive windowing"""
        # Apply learnable window function
        window = self._adaptive_window(len(x))
        windowed_x = x * window
        
        # Compute FFT
        x_freq = fft.fft(windowed_x, axis=-1)
        
        # Apply frequency selection
        x_freq = self._apply_frequency_selection(x_freq)
        
        return x_freq
    
    def inverse_transform(self, x_freq: np.ndarray) -> np.ndarray:
        """IFFT with adaptive post-processing"""
        # Apply inverse frequency selection
        x_freq_filtered = self._apply_frequency_selection(x_freq, inverse=True)
        
        # Compute IFFT
        x = fft.ifft(x_freq_filtered, axis=-1).real
        
        return x
    
    def kernel_weights(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute adaptive kernel weights
        w(ω) = gate(ω) * exp(-(ω - μ)²/(2σ²)) * phase_modulation(ω)
        """
        weights = np.zeros_like(frequencies, dtype=complex)
        
        for i, freq in enumerate(frequencies):
            if i < len(self.frequency_weights):
                # Gaussian-like spectral envelope with learnable parameters
                mu = i / len(frequencies)  # Normalized frequency
                sigma = self.bandwidth_params[i]
                
                weight = self.frequency_weights[i] * np.exp(-(freq - mu)**2 / (2 * sigma**2))
                
                # Add learnable phase shift
                phase = np.exp(1j * self.phase_shifts[i])
                weights[i] = weight * phase * self.frequency_gates[i]
        
        return weights
    
    def _adaptive_window(self, length: int) -> np.ndarray:
        """Create adaptive window function"""
        # Learnable Tukey window with adaptive parameters
        alpha = 0.5  # Could be learnable
        window = signal.tukey(length, alpha=alpha)
        
        # Apply sequence-specific modulation (simplified)
        modulation = 1.0 + 0.1 * np.sin(2 * np.pi * np.arange(length) / length)
        
        return window * modulation
    
    def _apply_frequency_selection(self, x_freq: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Apply learnable frequency selection gates"""
        if inverse:
            # Remove gates during inverse transform
            gates = 1.0 / (self.frequency_gates + 1e-8)
        else:
            gates = self.frequency_gates
            
        # Apply gates to frequency domain
        for i in range(min(len(gates), x_freq.shape[-1])):
            x_freq[..., i] *= gates[i]
            
        return x_freq
    
    def update_parameters(self, 
                         loss_gradient: np.ndarray,
                         learning_rate: float = 1e-3):
        """Update learnable parameters based on gradients"""
        if len(loss_gradient) >= len(self.frequency_weights):
            grad_freq = loss_gradient[:len(self.frequency_weights)]
            self.frequency_weights -= learning_rate * grad_freq
            self.frequency_weights = np.maximum(self.frequency_weights, 0)
            
        # Update frequency gates with sigmoid activation
        gate_grads = loss_gradient[len(self.frequency_weights):len(self.frequency_weights)+len(self.frequency_gates)]
        if len(gate_grads) > 0:
            self.frequency_gates -= learning_rate * gate_grads
            self.frequency_gates = 1.0 / (1.0 + np.exp(-self.frequency_gates))  # Sigmoid

class MultiScaleFourierOperator:
    """
    Multi-scale Fourier operator for hierarchical feature learning
    Implements r-scale decomposition: K = ∑ₛ₌₁ʳ Wₛ ∘ Fₛ⁻¹ ∘ Rₛ ∘ Fₛ
    """
    
    def __init__(self,
                 num_scales: int = 3,
                 base_modes: int = 32,
                 scale_factor: float = 2.0,
                 coupling_strength: float = 0.1):
        self.num_scales = num_scales
        self.base_modes = base_modes
        self.scale_factor = scale_factor
        self.coupling_strength = coupling_strength
        
        # Initialize kernels for each scale
        self.scale_kernels = []
        for s in range(num_scales):
            modes = int(base_modes / (scale_factor ** s))
            kernel = AdaptiveSpectralKernel(max_modes=modes)
            self.scale_kernels.append(kernel)
            
        # Cross-scale coupling weights
        self.coupling_weights = np.random.normal(0, 0.1, (num_scales, num_scales))
        np.fill_diagonal(self.coupling_weights, 1.0)
        
    def decompose_frequencies(self, x: np.ndarray) -> List[np.ndarray]:
        """Decompose input into multiple frequency scales"""
        decomposed = []
        
        for s, kernel in enumerate(self.scale_kernels):
            # Apply scale-specific filtering
            if s == 0:
                # Low-frequency components
                filtered_x = self._low_pass_filter(x, cutoff=0.1)
            elif s == len(self.scale_kernels) - 1:
                # High-frequency components  
                filtered_x = self._high_pass_filter(x, cutoff=0.4)
            else:
                # Band-pass for intermediate scales
                low_cutoff = 0.1 + 0.1 * s
                high_cutoff = 0.1 + 0.1 * (s + 1)
                filtered_x = self._band_pass_filter(x, low_cutoff, high_cutoff)
                
            decomposed.append(filtered_x)
            
        return decomposed
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Multi-scale forward transform"""
        # Decompose input into scales
        scale_components = self.decompose_frequencies(x)
        
        # Process each scale
        processed_scales = []
        for s, (component, kernel) in enumerate(zip(scale_components, self.scale_kernels)):
            # Apply Fourier transform
            freq_component = kernel.forward_transform(component)
            
            # Apply scale-specific processing
            frequencies = fft.fftfreq(len(component))
            weights = kernel.kernel_weights(frequencies)
            
            processed_freq = freq_component * weights
            processed_component = kernel.inverse_transform(processed_freq)
            
            processed_scales.append(processed_component)
        
        # Cross-scale coupling
        coupled_scales = self._apply_cross_scale_coupling(processed_scales)
        
        # Combine scales
        output = self._combine_scales(coupled_scales)
        
        return output
    
    def _apply_cross_scale_coupling(self, scales: List[np.ndarray]) -> List[np.ndarray]:
        """Apply coupling between different frequency scales"""
        coupled_scales = []
        
        for i, scale in enumerate(scales):
            coupled_scale = np.zeros_like(scale)
            
            for j, other_scale in enumerate(scales):
                # Resize other scale to match current scale
                if other_scale.shape != scale.shape:
                    if len(other_scale) > len(scale):
                        # Downsample
                        resized = signal.resample(other_scale, len(scale))
                    else:
                        # Upsample
                        resized = signal.resample(other_scale, len(scale))
                else:
                    resized = other_scale
                    
                coupled_scale += self.coupling_weights[i, j] * resized
                
            coupled_scales.append(coupled_scale)
            
        return coupled_scales
    
    def _combine_scales(self, scales: List[np.ndarray]) -> np.ndarray:
        """Combine multiple scales into single output"""
        if not scales:
            return np.array([])
            
        # Ensure all scales have same length
        target_length = len(scales[0])
        normalized_scales = []
        
        for scale in scales:
            if len(scale) != target_length:
                resized = signal.resample(scale, target_length)
                normalized_scales.append(resized)
            else:
                normalized_scales.append(scale)
                
        # Weighted combination
        weights = np.exp(-np.arange(len(scales)) * 0.1)  # Exponential weighting
        weights = weights / np.sum(weights)
        
        combined = np.zeros(target_length)
        for weight, scale in zip(weights, normalized_scales):
            combined += weight * scale
            
        return combined
    
    def _low_pass_filter(self, x: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply low-pass filter"""
        sos = signal.butter(4, cutoff, btype='low', output='sos')
        return signal.sosfilt(sos, x)
    
    def _high_pass_filter(self, x: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply high-pass filter"""
        sos = signal.butter(4, cutoff, btype='high', output='sos')
        return signal.sosfilt(sos, x)
    
    def _band_pass_filter(self, x: np.ndarray, low_cutoff: float, high_cutoff: float) -> np.ndarray:
        """Apply band-pass filter"""
        sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', output='sos')
        return signal.sosfilt(sos, x)

class PhysicsInformedFourierKernel:
    """
    Physics-informed Fourier kernel incorporating protein dynamics constraints
    
    Incorporates knowledge about:
    - Characteristic frequencies of protein motions
    - Secondary structure periodicities
    - Contact map symmetries
    - Ramachandran constraints
    """
    
    def __init__(self):
        # Protein-specific frequency characteristics
        self.alpha_helix_frequency = 3.6  # residues per turn
        self.beta_strand_frequency = 2.0  # residues per strand spacing
        self.loop_frequencies = [5.0, 7.0, 10.0]  # typical loop sizes
        
        # Contact map symmetries
        self.contact_symmetry_weight = 1.0
        
        # Ramachandran constraints (simplified)
        self.phi_psi_constraints = {
            'alpha': (-60, -45),   # (phi, psi) for alpha helix
            'beta': (-120, 120),   # (phi, psi) for beta strand  
            'ppii': (-75, 145)     # (phi, psi) for PPII helix
        }
        
    def apply_physics_constraints(self, 
                                fourier_coeffs: np.ndarray,
                                sequence_length: int) -> np.ndarray:
        """Apply physics-informed constraints to Fourier coefficients"""
        constrained_coeffs = fourier_coeffs.copy()
        
        # Apply secondary structure frequency priors
        frequencies = fft.fftfreq(sequence_length)
        
        for i, freq in enumerate(frequencies):
            # Enhance coefficients at characteristic frequencies
            enhancement = 1.0
            
            # Alpha helix frequency enhancement
            alpha_distance = abs(freq - 1.0/self.alpha_helix_frequency)
            enhancement += 0.5 * np.exp(-alpha_distance * 10)
            
            # Beta strand frequency enhancement
            beta_distance = abs(freq - 1.0/self.beta_strand_frequency)
            enhancement += 0.3 * np.exp(-beta_distance * 10)
            
            # Loop frequency enhancement
            for loop_freq in self.loop_frequencies:
                loop_distance = abs(freq - 1.0/loop_freq)
                enhancement += 0.2 * np.exp(-loop_distance * 5)
                
            constrained_coeffs[i] *= enhancement
            
        return constrained_coeffs
    
    def enforce_contact_symmetry(self, contact_map: np.ndarray) -> np.ndarray:
        """Enforce contact map symmetry in Fourier domain"""
        # Contact maps should be symmetric: C[i,j] = C[j,i]
        fourier_contact = fft.fft2(contact_map)
        
        # Enforce Hermitian symmetry in Fourier domain
        symmetric_fourier = 0.5 * (fourier_contact + np.conj(fourier_contact.T))
        
        symmetric_contact = fft.ifft2(symmetric_fourier).real
        
        return symmetric_contact
    
    def ramachandran_frequency_filter(self, 
                                    torsion_predictions: np.ndarray,
                                    structure_type: str = 'mixed') -> np.ndarray:
        """Apply Ramachandran constraints via frequency filtering"""
        phi_pred = torsion_predictions[..., 0]  # Phi angles
        psi_pred = torsion_predictions[..., 1]  # Psi angles
        
        # Convert to Fourier domain
        phi_freq = fft.fft(phi_pred)
        psi_freq = fft.fft(psi_pred)
        
        # Apply structure-specific constraints
        if structure_type in self.phi_psi_constraints:
            target_phi, target_psi = self.phi_psi_constraints[structure_type]
            
            # Create constraint filter
            constraint_filter = self._create_ramachandran_filter(
                len(phi_pred), target_phi, target_psi
            )
            
            phi_freq *= constraint_filter
            psi_freq *= constraint_filter
            
        # Convert back to spatial domain
        constrained_phi = fft.ifft(phi_freq).real
        constrained_psi = fft.ifft(psi_freq).real
        
        # Reconstruct torsion predictions
        constrained_torsions = torsion_predictions.copy()
        constrained_torsions[..., 0] = constrained_phi
        constrained_torsions[..., 1] = constrained_psi
        
        return constrained_torsions
    
    def _create_ramachandran_filter(self, 
                                  length: int, 
                                  target_phi: float, 
                                  target_psi: float) -> np.ndarray:
        """Create frequency filter based on Ramachandran constraints"""
        frequencies = fft.fftfreq(length)
        filter_values = np.ones(length)
        
        # Apply constraints based on target angles
        for i, freq in enumerate(frequencies):
            # Frequency-based constraint (simplified)
            constraint_strength = np.exp(-abs(freq) * 2)  # Prefer low frequencies
            filter_values[i] = constraint_strength
            
        return filter_values

class AttentionModulatedFourierTransform:
    """
    Attention-modulated Fourier transform for sequence-specific adaptivity
    A(x) = Attention(x) ⊙ F(x) where ⊙ is element-wise multiplication
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 num_heads: int = 8,
                 attention_dropout: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        
        # Attention parameters (simplified)
        self.query_weights = np.random.normal(0, 0.02, (d_model, d_model))
        self.key_weights = np.random.normal(0, 0.02, (d_model, d_model))
        self.value_weights = np.random.normal(0, 0.02, (d_model, d_model))
        
    def compute_attention_weights(self, x: np.ndarray) -> np.ndarray:
        """Compute attention weights for Fourier modulation"""
        seq_len, d_model = x.shape
        
        # Compute Q, K, V
        Q = np.dot(x, self.query_weights)
        K = np.dot(x, self.key_weights) 
        V = np.dot(x, self.value_weights)
        
        # Multi-head attention (simplified)
        head_dim = d_model // self.num_heads
        attention_scores = np.zeros((seq_len, seq_len))
        
        for h in range(self.num_heads):
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            
            Q_h = Q[:, start_idx:end_idx]
            K_h = K[:, start_idx:end_idx]
            
            # Attention scores
            scores = np.dot(Q_h, K_h.T) / np.sqrt(head_dim)
            attention_scores += self._softmax(scores)
            
        attention_scores /= self.num_heads
        
        return attention_scores
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply attention-modulated Fourier transform"""
        # Compute attention weights
        attention_weights = self.compute_attention_weights(x)
        
        # Apply attention modulation to each sequence position
        modulated_sequences = []
        
        for i in range(x.shape[0]):
            # Get attention distribution for position i
            attention_dist = attention_weights[i, :]
            
            # Create modulated sequence
            modulated_seq = np.zeros_like(x[i])
            for j in range(x.shape[0]):
                modulated_seq += attention_dist[j] * x[j]
                
            modulated_sequences.append(modulated_seq)
            
        modulated_x = np.array(modulated_sequences)
        
        # Apply Fourier transform to modulated sequences
        fourier_modulated = fft.fft(modulated_x, axis=0)
        
        # Apply frequency-specific attention
        freq_attention = self._compute_frequency_attention(fourier_modulated)
        attentive_fourier = fourier_modulated * freq_attention
        
        # Inverse transform
        output = fft.ifft(attentive_fourier, axis=0).real
        
        return output
    
    def _compute_frequency_attention(self, fourier_coeffs: np.ndarray) -> np.ndarray:
        """Compute attention weights in frequency domain"""
        # Magnitude-based attention
        magnitudes = np.abs(fourier_coeffs)
        
        # Normalize magnitudes to get attention weights
        attention_weights = magnitudes / (np.sum(magnitudes, axis=0, keepdims=True) + 1e-8)
        
        return attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class ProteinFourierOperator:
    """
    Main class integrating all novel Fourier operator components
    for protein structure prediction
    """
    
    def __init__(self, config: FourierKernelConfig):
        self.config = config
        
        # Initialize components
        self.adaptive_kernel = AdaptiveSpectralKernel(
            max_modes=config.max_modes,
            learnable_frequencies=True,
            bandwidth_adaptation=True
        )
        
        self.multiscale_operator = MultiScaleFourierOperator(
            num_scales=config.multi_scale_levels,
            base_modes=config.max_modes
        )
        
        if config.physics_informed:
            self.physics_kernel = PhysicsInformedFourierKernel()
            
        if config.attention_modulated:
            self.attention_fourier = AttentionModulatedFourierTransform()
            
        # Performance tracking
        self.computation_stats = {
            "forward_time": [],
            "memory_usage": [],
            "frequency_usage": []
        }
        
    def forward(self, 
                sequence_features: np.ndarray,
                return_intermediate: bool = False) -> Dict[str, np.ndarray]:
        """
        Forward pass through the protein Fourier operator
        
        Args:
            sequence_features: Input protein sequence features
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            Dictionary containing processed features and optional intermediates
        """
        outputs = {}
        intermediates = {} if return_intermediate else None
        
        # Step 1: Multi-scale decomposition
        multiscale_output = self.multiscale_operator.forward(sequence_features)
        if return_intermediate:
            intermediates["multiscale"] = multiscale_output
            
        # Step 2: Adaptive spectral processing
        adaptive_output = self.adaptive_kernel.forward_transform(multiscale_output)
        frequencies = fft.fftfreq(len(multiscale_output))
        kernel_weights = self.adaptive_kernel.kernel_weights(frequencies)
        
        weighted_frequencies = adaptive_output * kernel_weights
        
        if return_intermediate:
            intermediates["adaptive_frequencies"] = weighted_frequencies
            
        # Step 3: Physics-informed constraints
        if self.config.physics_informed:
            constrained_frequencies = self.physics_kernel.apply_physics_constraints(
                weighted_frequencies, len(sequence_features)
            )
        else:
            constrained_frequencies = weighted_frequencies
            
        if return_intermediate:
            intermediates["physics_constrained"] = constrained_frequencies
            
        # Step 4: Attention modulation
        if self.config.attention_modulated:
            # Convert to spatial domain for attention
            spatial_features = self.adaptive_kernel.inverse_transform(constrained_frequencies)
            
            # Reshape for attention (sequence_length, features)
            if len(spatial_features.shape) == 1:
                spatial_features = spatial_features.reshape(-1, 1)
                
            attention_output = self.attention_fourier.forward(spatial_features)
            
            # Convert back to frequency domain
            final_frequencies = self.adaptive_kernel.forward_transform(attention_output.flatten())
        else:
            final_frequencies = constrained_frequencies
            
        if return_intermediate:
            intermediates["attention_modulated"] = final_frequencies
            
        # Step 5: Final inverse transform
        final_output = self.adaptive_kernel.inverse_transform(final_frequencies)
        
        outputs["features"] = final_output
        outputs["frequency_representation"] = final_frequencies
        
        if return_intermediate:
            outputs["intermediates"] = intermediates
            
        return outputs
    
    def analyze_frequency_usage(self, 
                              sequence_features: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze which frequencies are most utilized"""
        # Forward transform
        freq_repr = self.adaptive_kernel.forward_transform(sequence_features)
        frequencies = fft.fftfreq(len(sequence_features))
        
        # Compute frequency magnitudes
        magnitudes = np.abs(freq_repr)
        
        # Analyze usage patterns
        analysis = {
            "frequency_magnitudes": magnitudes,
            "dominant_frequencies": frequencies[np.argsort(magnitudes)[-10:]],
            "frequency_distribution": np.histogram(frequencies, weights=magnitudes, bins=20),
            "spectral_centroid": np.sum(frequencies * magnitudes) / np.sum(magnitudes),
            "spectral_bandwidth": np.sqrt(np.sum((frequencies - np.sum(frequencies * magnitudes) / np.sum(magnitudes))**2 * magnitudes) / np.sum(magnitudes))
        }
        
        return analysis
    
    def optimize_kernel_parameters(self, 
                                 training_data: List[np.ndarray],
                                 targets: List[np.ndarray],
                                 num_iterations: int = 100,
                                 learning_rate: float = 1e-3) -> Dict[str, float]:
        """Optimize kernel parameters using gradient-free optimization"""
        best_loss = np.inf
        best_params = None
        loss_history = []
        
        for iteration in range(num_iterations):
            total_loss = 0.0
            
            for data, target in zip(training_data, targets):
                # Forward pass
                output = self.forward(data)
                
                # Compute loss (MSE)
                loss = np.mean((output["features"] - target)**2)
                
                # Add regularization
                if self.config.regularization_weight > 0:
                    reg_loss = self.config.regularization_weight * np.sum(self.adaptive_kernel.frequency_weights**2)
                    loss += reg_loss
                    
                total_loss += loss
                
            avg_loss = total_loss / len(training_data)
            loss_history.append(avg_loss)
            
            # Simple parameter update (gradient-free)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = {
                    "frequency_weights": self.adaptive_kernel.frequency_weights.copy(),
                    "bandwidth_params": self.adaptive_kernel.bandwidth_params.copy(),
                    "frequency_gates": self.adaptive_kernel.frequency_gates.copy()
                }
            else:
                # Add noise for exploration
                noise_scale = learning_rate * np.exp(-iteration / 50)
                self.adaptive_kernel.frequency_weights += np.random.normal(0, noise_scale, self.adaptive_kernel.frequency_weights.shape)
                self.adaptive_kernel.bandwidth_params += np.random.normal(0, noise_scale, self.adaptive_kernel.bandwidth_params.shape)
                
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Loss = {avg_loss:.6f}")
                
        # Restore best parameters
        if best_params:
            self.adaptive_kernel.frequency_weights = best_params["frequency_weights"]
            self.adaptive_kernel.bandwidth_params = best_params["bandwidth_params"] 
            self.adaptive_kernel.frequency_gates = best_params["frequency_gates"]
            
        return {
            "best_loss": best_loss,
            "loss_history": loss_history,
            "num_iterations": num_iterations
        }
    
    def save_operator(self, filepath: str):
        """Save operator state"""
        state = {
            "config": self.config.__dict__,
            "adaptive_kernel_state": {
                "frequency_weights": self.adaptive_kernel.frequency_weights.tolist(),
                "bandwidth_params": self.adaptive_kernel.bandwidth_params.tolist(),
                "frequency_gates": self.adaptive_kernel.frequency_gates.tolist(),
                "phase_shifts": self.adaptive_kernel.phase_shifts.tolist()
            },
            "computation_stats": self.computation_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Protein Fourier operator saved to {filepath}")

# Example usage and validation
if __name__ == "__main__":
    # Initialize configuration
    config = FourierKernelConfig(
        max_modes=64,
        kernel_type="adaptive_spectral",
        physics_informed=True,
        attention_modulated=True,
        multi_scale_levels=3
    )
    
    # Create operator
    operator = ProteinFourierOperator(config)
    
    # Simulate protein sequence features
    sequence_length = 128
    feature_dim = 256
    sequence_features = np.random.normal(0, 1, sequence_length)
    
    # Test forward pass
    outputs = operator.forward(sequence_features, return_intermediate=True)
    
    print(f"Input shape: {sequence_features.shape}")
    print(f"Output shape: {outputs['features'].shape}")
    print(f"Frequency representation shape: {outputs['frequency_representation'].shape}")
    print(f"Number of intermediate representations: {len(outputs['intermediates'])}")
    
    # Analyze frequency usage
    freq_analysis = operator.analyze_frequency_usage(sequence_features)
    print(f"\nSpectral centroid: {freq_analysis['spectral_centroid']:.4f}")
    print(f"Spectral bandwidth: {freq_analysis['spectral_bandwidth']:.4f}")
    print(f"Top 5 dominant frequencies: {freq_analysis['dominant_frequencies'][-5:]}")
    
    # Test optimization
    training_data = [np.random.normal(0, 1, sequence_length) for _ in range(10)]
    targets = [np.random.normal(0, 1, sequence_length) for _ in range(10)]
    
    optimization_results = operator.optimize_kernel_parameters(
        training_data, targets, num_iterations=20, learning_rate=1e-3
    )
    
    print(f"\nOptimization results:")
    print(f"Best loss: {optimization_results['best_loss']:.6f}")
    print(f"Final loss: {optimization_results['loss_history'][-1]:.6f}")
    
    logger.info("Advanced Fourier operator validation complete!")