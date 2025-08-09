"""
Performance optimization utilities for protein-sssl-operator
Provides memory optimization, computation acceleration, and efficiency improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from functools import wraps, lru_cache
from contextlib import contextmanager
import warnings
import gc
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization settings"""
    use_flash_attention: bool = False
    use_memory_efficient_attention: bool = True
    use_torch_compile: bool = False
    compile_mode: str = "default"
    use_channels_last: bool = False
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    fuse_kernels: bool = True
    optimize_for_inference: bool = False

class MemoryOptimizer:
    """Memory optimization utilities"""
    
    def __init__(self):
        self.memory_stats = {}
        self.peak_memory = 0
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache and perform garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @contextmanager
    def track_memory(self, operation_name: str):
        """Context manager to track memory usage"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
            peak_start = torch.cuda.max_memory_allocated()
        else:
            start_memory = 0
            peak_start = 0
        
        start_time = time.time()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                peak_end = torch.cuda.max_memory_allocated()
                
                memory_diff = end_memory - start_memory
                peak_diff = peak_end - peak_start
                
                self.memory_stats[operation_name] = {
                    'memory_delta': memory_diff / (1024**2),  # MB
                    'peak_memory': peak_diff / (1024**2),     # MB
                    'duration': time.time() - start_time
                }
                
                logger.debug(f"Memory tracking - {operation_name}: "
                           f"Δ{memory_diff/(1024**2):.1f}MB, "
                           f"Peak: {peak_diff/(1024**2):.1f}MB")
    
    @staticmethod
    def optimize_dataloader(dataloader, pin_memory: bool = True, 
                           persistent_workers: bool = True):
        """Optimize DataLoader settings"""
        # These optimizations would need to be applied during DataLoader creation
        return {
            'pin_memory': pin_memory and torch.cuda.is_available(),
            'persistent_workers': persistent_workers and dataloader.num_workers > 0,
            'prefetch_factor': 2 if dataloader.num_workers > 0 else 2,
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        summary = {'operations': self.memory_stats}
        
        if torch.cuda.is_available():
            summary.update({
                'current_allocated': torch.cuda.memory_allocated() / (1024**2),
                'max_allocated': torch.cuda.max_memory_allocated() / (1024**2),
                'reserved': torch.cuda.memory_reserved() / (1024**2),
                'max_reserved': torch.cuda.max_memory_reserved() / (1024**2)
            })
        
        return summary

class AttentionOptimizer:
    """Optimized attention mechanisms"""
    
    @staticmethod
    def scaled_dot_product_attention_optimized(query: torch.Tensor,
                                             key: torch.Tensor,
                                             value: torch.Tensor,
                                             attn_mask: Optional[torch.Tensor] = None,
                                             dropout_p: float = 0.0,
                                             is_causal: bool = False) -> torch.Tensor:
        """Optimized scaled dot-product attention"""
        
        # Use PyTorch 2.0+ optimized attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            try:
                return F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal
                )
            except Exception as e:
                logger.debug(f"Failed to use optimized attention, falling back: {e}")
        
        # Fallback implementation with memory optimization
        return AttentionOptimizer._attention_fallback(
            query, key, value, attn_mask, dropout_p
        )
    
    @staticmethod
    def _attention_fallback(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          attn_mask: Optional[torch.Tensor] = None,
                          dropout_p: float = 0.0) -> torch.Tensor:
        """Memory-efficient attention fallback"""
        
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = head_dim ** -0.5
        
        # Chunk computation for large sequences
        if seq_len > 1024:
            return AttentionOptimizer._chunked_attention(
                query, key, value, attn_mask, dropout_p, scale
            )
        
        # Standard attention
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
        
        output = torch.matmul(attn_weights, value)
        return output
    
    @staticmethod
    def _chunked_attention(query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         attn_mask: Optional[torch.Tensor] = None,
                         dropout_p: float = 0.0,
                         scale: float = 1.0,
                         chunk_size: int = 512) -> torch.Tensor:
        """Chunked attention for memory efficiency"""
        
        batch_size, num_heads, seq_len, head_dim = query.shape
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            query_chunk = query[:, :, i:end_i]
            
            # Compute attention for this chunk
            scores = torch.matmul(query_chunk, key.transpose(-2, -1)) * scale
            
            if attn_mask is not None:
                mask_chunk = attn_mask[:, :, i:end_i]
                scores = scores.masked_fill(mask_chunk == 0, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
            output[:, :, i:end_i] = torch.matmul(attn_weights, value)
        
        return output

class ModelOptimizer:
    """Model-level optimizations"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive model optimizations"""
        
        # Convert to channels last if specified
        if self.config.use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Fuse operations where possible
        if self.config.fuse_kernels:
            model = self._fuse_model_operations(model)
        
        # Apply torch.compile if available and requested
        if self.config.use_torch_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(
                    model,
                    mode=self.config.compile_mode,
                    dynamic=not self.config.optimize_for_inference
                )
                logger.info(f"Applied torch.compile with mode: {self.config.compile_mode}")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        
        return model
    
    def _fuse_model_operations(self, model: nn.Module) -> nn.Module:
        """Fuse compatible operations"""
        
        # Fuse conv-bn, linear-relu, etc.
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                # Look for fusion opportunities
                self._fuse_sequential(module)
        
        return model
    
    def _fuse_sequential(self, seq_module: nn.Sequential):
        """Fuse operations in sequential modules"""
        
        modules = list(seq_module.children())
        fused_modules = []
        i = 0
        
        while i < len(modules):
            current = modules[i]
            
            # Fuse Linear + ReLU
            if (isinstance(current, nn.Linear) and 
                i + 1 < len(modules) and 
                isinstance(modules[i + 1], nn.ReLU)):
                
                fused_modules.append(nn.Sequential(current, modules[i + 1]))
                i += 2
            else:
                fused_modules.append(current)
                i += 1
        
        # Update the sequential module
        for idx, module in enumerate(fused_modules):
            seq_module[idx] = module
    
    def optimize_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model specifically for inference"""
        
        model.eval()
        
        # Convert BatchNorm to running stats
        self._freeze_batch_norm(model)
        
        # Remove dropout layers
        self._remove_dropout(model)
        
        # Optimize for inference
        if self.config.optimize_for_inference:
            with torch.no_grad():
                model = torch.jit.optimize_for_inference(model)
        
        return model
    
    def _freeze_batch_norm(self, model: nn.Module):
        """Freeze batch normalization layers"""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.requires_grad_(False)
    
    def _remove_dropout(self, model: nn.Module):
        """Remove dropout layers for inference"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

class GradientOptimizer:
    """Gradient computation optimizations"""
    
    @staticmethod
    def clip_gradients(model: nn.Module, 
                      max_norm: float = 1.0,
                      norm_type: float = 2.0) -> float:
        """Optimized gradient clipping"""
        
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if not parameters:
            return 0.0
        
        # Use torch.nn.utils.clip_grad_norm_ for efficiency
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, norm_type=norm_type
        )
        
        return total_norm.item()
    
    @staticmethod
    def accumulate_gradients(model: nn.Module, 
                           accumulation_steps: int,
                           current_step: int):
        """Efficient gradient accumulation"""
        
        if current_step % accumulation_steps != 0:
            # Scale gradients by accumulation steps
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data /= accumulation_steps
    
    @staticmethod
    def zero_gradients_efficiently(optimizer: torch.optim.Optimizer):
        """More efficient gradient zeroing"""
        
        # Use set_to_none for better performance
        optimizer.zero_grad(set_to_none=True)

class ComputationOptimizer:
    """General computation optimizations"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def create_attention_mask(seq_len: int, 
                            batch_size: int,
                            device: torch.device) -> torch.Tensor:
        """Cached attention mask creation"""
        return torch.ones(batch_size, seq_len, seq_len, device=device)
    
    @staticmethod
    def optimize_tensor_operations(func: Callable) -> Callable:
        """Decorator to optimize tensor operations"""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Enable optimized tensor operations
            with torch.backends.cudnn.flags(
                enabled=True,
                benchmark=True,
                deterministic=False,
                allow_tf32=True
            ):
                return func(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def vectorize_computation(data: List[torch.Tensor]) -> torch.Tensor:
        """Vectorize batch computations"""
        
        # Stack tensors for batch processing
        if len(data) > 1:
            max_len = max(t.shape[0] for t in data)
            
            # Pad tensors to same length
            padded_data = []
            for tensor in data:
                if tensor.shape[0] < max_len:
                    pad_size = max_len - tensor.shape[0]
                    padding = torch.zeros(pad_size, *tensor.shape[1:], 
                                        device=tensor.device, dtype=tensor.dtype)
                    padded_data.append(torch.cat([tensor, padding], dim=0))
                else:
                    padded_data.append(tensor)
            
            return torch.stack(padded_data)
        
        return data[0].unsqueeze(0)

class CacheOptimizer:
    """Caching optimizations for frequently computed values"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.caches = {}
    
    def get_or_compute(self, 
                      cache_key: str,
                      compute_func: Callable,
                      *args,
                      **kwargs) -> Any:
        """Get cached result or compute and cache"""
        
        if cache_key not in self.caches:
            self.caches[cache_key] = {}
        
        cache = self.caches[cache_key]
        key = hash((args, tuple(sorted(kwargs.items()))))
        
        if key in cache:
            return cache[key]
        
        # Compute result
        result = compute_func(*args, **kwargs)
        
        # Cache with size limit
        if len(cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU approximation)
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        
        cache[key] = result
        return result
    
    def clear_cache(self, cache_name: Optional[str] = None):
        """Clear specific cache or all caches"""
        if cache_name:
            self.caches[cache_name] = {}
        else:
            self.caches = {}

class ProfilerOptimizer:
    """Profiling and optimization utilities"""
    
    def __init__(self):
        self.profiler = None
        self.profiling_data = {}
    
    @contextmanager
    def profile(self, 
               activities: List[str] = None,
               record_shapes: bool = True,
               profile_memory: bool = True):
        """Context manager for profiling"""
        
        activities = activities or ['cpu', 'cuda']
        
        if torch.cuda.is_available():
            torch_activities = []
            if 'cpu' in activities:
                torch_activities.append(torch.profiler.ProfilerActivity.CPU)
            if 'cuda' in activities:
                torch_activities.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            torch_activities = [torch.profiler.ProfilerActivity.CPU]
        
        with torch.profiler.profile(
            activities=torch_activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=True
        ) as prof:
            self.profiler = prof
            yield prof
            self.profiler = None
    
    def analyze_profile(self, sort_by: str = "cuda_time_total") -> str:
        """Analyze profiling results"""
        
        if not self.profiler:
            return "No profiling data available"
        
        return self.profiler.key_averages().table(
            sort_by=sort_by,
            row_limit=20
        )
    
    def export_profile(self, filename: str):
        """Export profiling results"""
        
        if self.profiler:
            self.profiler.export_chrome_trace(filename)

# Global optimization utilities
_memory_optimizer = MemoryOptimizer()
_cache_optimizer = CacheOptimizer()

def optimize_model(model: nn.Module, 
                  config: Optional[OptimizationConfig] = None) -> nn.Module:
    """Global model optimization function"""
    optimizer = ModelOptimizer(config)
    return optimizer.optimize_model(model)

def track_memory(operation_name: str):
    """Global memory tracking context manager"""
    return _memory_optimizer.track_memory(operation_name)

def clear_memory():
    """Global memory clearing function"""
    MemoryOptimizer.clear_cache()

def get_memory_summary() -> Dict[str, Any]:
    """Get global memory summary"""
    return _memory_optimizer.get_memory_summary()

def cached_computation(cache_key: str):
    """Decorator for caching computations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return _cache_optimizer.get_or_compute(
                cache_key, func, *args, **kwargs
            )
        return wrapper
    return decorator

class AutoOptimizer:
    """Automatic optimization based on model characteristics"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model_stats = self._analyze_model()
    
    def _analyze_model(self) -> Dict[str, Any]:
        """Analyze model characteristics"""
        
        stats = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'has_attention': False,
            'has_conv': False,
            'has_linear': False,
            'max_sequence_length': 0
        }
        
        for module in self.model.modules():
            if hasattr(module, 'attention') or 'attention' in str(type(module)).lower():
                stats['has_attention'] = True
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                stats['has_conv'] = True
            elif isinstance(module, nn.Linear):
                stats['has_linear'] = True
        
        return stats
    
    def get_optimal_config(self) -> OptimizationConfig:
        """Generate optimal configuration based on model analysis"""
        
        config = OptimizationConfig()
        
        # Large models benefit from gradient checkpointing
        if self.model_stats['total_params'] > 100_000_000:
            config.gradient_checkpointing = True
            config.use_mixed_precision = True
        
        # Attention models benefit from optimized attention
        if self.model_stats['has_attention']:
            config.use_memory_efficient_attention = True
        
        # Conv models benefit from channels last
        if self.model_stats['has_conv']:
            config.use_channels_last = True
        
        return config
    
    def apply_optimizations(self) -> nn.Module:
        """Apply automatic optimizations"""
        
        config = self.get_optimal_config()
        optimizer = ModelOptimizer(config)
        
        return optimizer.optimize_model(self.model)