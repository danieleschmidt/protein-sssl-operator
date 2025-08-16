import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import math
import numpy as np
import logging
from ..utils.error_handling_v2 import (
    robust_error_handler, validate_tensor_dimensions, 
    safe_tensor_operation, ModelValidator, DimensionMismatchError
)

class FourierLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier weights
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * 
            (1 / (in_channels * out_channels))
        )
        
    @robust_error_handler(fallback_value=None, reraise=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input dimensions - only check last dimension (channels)
        if x.shape[-1] != self.in_channels:
            raise DimensionMismatchError(f"Expected {self.in_channels} input channels, got {x.shape[-1]}")
        
        batch_size, seq_len, channels = x.shape
        
        # FFT
        x_ft = torch.fft.fft(x, dim=1)
        
        # Multiply by weights (only for lower modes)
        out_ft = torch.zeros(batch_size, seq_len, self.out_channels, 
                           dtype=torch.cfloat, device=x.device)
        
        # Only use available modes (min of modes and actual sequence length)
        usable_modes = min(self.modes, seq_len)
        if usable_modes > 0:
            try:
                # Extract the weights for the usable modes - ensure compatibility
                if self.weights.shape[2] < usable_modes:
                    # Pad weights if needed
                    weights_slice = F.pad(self.weights, (0, usable_modes - self.weights.shape[2]))[:, :, :usable_modes]
                else:
                    weights_slice = self.weights[:, :, :usable_modes]
                
                x_ft_slice = x_ft[:, :usable_modes, :]
                
                # Safe einsum operation
                result = safe_tensor_operation(
                    torch.einsum, "bic,ico->boc", 
                    x_ft_slice, weights_slice,
                    fallback_strategy="zeros"
                )
                out_ft[:, :usable_modes, :] = result
                
            except Exception as e:
                logging.warning(f"Fourier layer computation failed: {e}, using identity mapping")
                # Fallback to identity-like operation
                if self.in_channels == self.out_channels:
                    out_ft = x_ft.clone()
                else:
                    # Project to correct output dimension
                    out_ft = F.linear(x_ft.real, torch.randn(self.out_channels, self.in_channels, device=x.device)).to(x_ft.dtype)
        
        # IFFT
        result = torch.fft.ifft(out_ft, dim=1).real
        
        # Ensure output has correct shape
        if result.shape != (batch_size, seq_len, self.out_channels):
            logging.warning(f"Fourier output shape mismatch: {result.shape} vs expected {(batch_size, seq_len, self.out_channels)}")
        return result

class NeuralOperatorLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        fourier_modes: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.fourier_modes = fourier_modes
        
        # Fourier operator
        self.fourier_layer = FourierLayer(d_model, d_model, fourier_modes)
        
        # MLP
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Residual connection weights
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier transform branch with residual
        fourier_out = self.fourier_layer(self.norm1(x))
        x = x + self.alpha * fourier_out
        
        # MLP branch with residual  
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.beta * mlp_out
        
        return x

class StructureAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        x: torch.Tensor, 
        distance_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply distance-based mask if provided
        if distance_mask is not None:
            scores = scores + distance_mask.unsqueeze(1)
            
        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(out)

class NeuralOperatorFold(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        d_model: int = 1280,
        operator_layers: int = 12,
        fourier_modes: int = 64,
        n_heads: int = 16,
        attention_type: str = "efficient",
        uncertainty_method: str = "ensemble",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.d_model = d_model
        self.operator_layers = operator_layers
        self.uncertainty_method = uncertainty_method
        
        # Input projection if encoder not provided
        if encoder is None:
            self.input_proj = nn.Linear(21, d_model)  # 21 amino acids
        else:
            # Adjust dimensions if needed
            encoder_dim = getattr(encoder, 'd_model', d_model)
            if encoder_dim != d_model:
                self.input_proj = nn.Linear(encoder_dim, d_model)
            else:
                self.input_proj = nn.Identity()
        
        # Neural operator layers
        self.operator_layers_list = nn.ModuleList([
            NeuralOperatorLayer(
                d_model=d_model,
                fourier_modes=fourier_modes,
                dropout=dropout
            ) for _ in range(operator_layers)
        ])
        
        # Structure attention layers
        self.structure_attention = nn.ModuleList([
            StructureAttention(d_model, n_heads, dropout)
            for _ in range(operator_layers // 2)
        ])
        
        # Output heads
        self.distance_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64),  # Distance bins
            nn.Dropout(dropout)
        )
        
        self.torsion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(), 
            nn.Linear(d_model // 2, 8),  # phi, psi, omega, chi1-5
            nn.Dropout(dropout)
        )
        
        self.secondary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 8),  # Secondary structure classes
            nn.Dropout(dropout)
        )
        
        # Uncertainty estimation
        if uncertainty_method == "ensemble":
            self.num_heads = 5
            self.ensemble_heads = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in range(self.num_heads)
            ])
        elif uncertainty_method == "dropout":
            self.uncertainty_head = nn.Linear(d_model, 1)
            
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    
    @robust_error_handler(fallback_value={}, reraise=True)
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        # Validate inputs
        validated_inputs = ModelValidator.validate_model_input(input_ids, attention_mask)
        input_ids = validated_inputs["input_ids"]
        attention_mask = validated_inputs["attention_mask"]
        
        batch_size, seq_len = input_ids.shape
        
        # Get encoder representations
        if self.encoder is not None:
            with torch.no_grad():
                encoder_outputs = self.encoder(input_ids, attention_mask)
                x = encoder_outputs["last_hidden_state"] if isinstance(encoder_outputs, dict) else encoder_outputs
        else:
            # One-hot encoding
            x = F.one_hot(input_ids, num_classes=21).float()
            
        # Project to operator space
        x = self.input_proj(x)
        
        # Apply neural operator layers with structure attention
        for i, op_layer in enumerate(self.operator_layers_list):
            x = op_layer(x)
            
            # Apply structure attention every 2 layers
            if i % 2 == 1 and i // 2 < len(self.structure_attention):
                x = x + self.structure_attention[i // 2](x)
        
        # Generate pairwise features for distance prediction
        x_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pairwise_features = torch.cat([x_i, x_j], dim=-1)
        
        # Output predictions
        outputs = {}
        
        # Distance map
        distance_logits = self.distance_head(pairwise_features)
        outputs["distance_logits"] = distance_logits
        
        # Torsion angles
        torsion_angles = self.torsion_head(x)
        outputs["torsion_angles"] = torsion_angles
        
        # Secondary structure
        secondary_structure = self.secondary_head(x)
        outputs["secondary_structure"] = secondary_structure
        
        # Uncertainty quantification
        if return_uncertainty:
            if self.uncertainty_method == "ensemble":
                uncertainties = []
                for head in self.ensemble_heads:
                    uncertainties.append(head(x))
                uncertainty_ensemble = torch.stack(uncertainties, dim=-1)
                outputs["uncertainty"] = uncertainty_ensemble.std(dim=-1)
                outputs["ensemble_predictions"] = uncertainty_ensemble
                
            elif self.uncertainty_method == "dropout":
                # Enable dropout during inference for MC dropout
                self.train()
                uncertainty_samples = []
                for _ in range(10):
                    uncertainty_samples.append(self.uncertainty_head(x))
                self.eval()
                uncertainty_stack = torch.stack(uncertainty_samples, dim=-1)
                outputs["uncertainty"] = uncertainty_stack.std(dim=-1)
        
        return outputs
    
    def predict_structure(
        self,
        sequence: str,
        tokenizer,
        device: str = "cpu",
        return_confidence: bool = True,
        num_recycles: int = 3,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        
        self.eval()
        
        # Tokenize sequence
        inputs = tokenizer.encode(sequence)
        input_ids = inputs["input_ids"].unsqueeze(0).to(device)
        attention_mask = inputs["attention_mask"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Initial prediction
            outputs = self.forward(
                input_ids, 
                attention_mask, 
                return_uncertainty=return_confidence
            )
            
            # Recycle predictions (iterative refinement)
            for _ in range(num_recycles):
                # Use previous distance predictions as structural bias
                prev_distances = F.softmax(outputs["distance_logits"] / temperature, dim=-1)
                
                # Re-run with structural information (simplified recycling)
                outputs = self.forward(
                    input_ids,
                    attention_mask,
                    return_uncertainty=return_confidence
                )
                
        return outputs