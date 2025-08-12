"""
GPU-specific optimization utilities for protein structure prediction.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import functools
import time
from contextlib import contextmanager
import warnings

from .logging_config import setup_logging

logger = setup_logging(__name__)


class GPUMemoryManager:
    """Advanced GPU memory management for large-scale protein prediction."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_stats = {'peak_allocated': 0, 'peak_reserved': 0}
        
        # Memory optimization settings
        self.gradient_checkpointing = True
        self.mixed_precision = True
        self.memory_fraction = 0.9  # Use 90% of available GPU memory max
        
        if self.device.type == 'cuda':
            self._setup_cuda_memory()
    
    def _setup_cuda_memory(self):
        """Setup CUDA memory optimization."""
        try:
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory pooling for better allocation
            torch.cuda.empty_cache()
            
            logger.info(f"GPU memory manager initialized on {self.device}")
            self.log_memory_stats()
            
        except Exception as e:
            logger.warning(f"CUDA memory setup failed: {e}")
    
    @contextmanager
    def memory_scope(self, clear_cache: bool = True):
        """Context manager for memory-conscious operations."""
        if self.device.type == 'cuda':
            initial_memory = torch.cuda.memory_allocated()
            
        try:
            yield
            
        finally:
            if self.device.type == 'cuda':
                final_memory = torch.cuda.memory_allocated()
                memory_diff = (final_memory - initial_memory) / 1024**2
                
                if memory_diff > 100:  # Log if >100MB difference
                    logger.debug(f"Memory delta: {memory_diff:.1f}MB")
                
                if clear_cache and memory_diff > 500:  # Clear cache if >500MB
                    torch.cuda.empty_cache()
                    logger.debug("GPU cache cleared due to high memory usage")
    
    def optimize_for_large_sequences(self, sequence_length: int) -> Dict[str, Any]:
        """Optimize settings for large protein sequences."""
        optimizations = {
            'use_checkpointing': True,
            'batch_size': 1,  # Start conservative
            'chunk_size': min(512, sequence_length // 4),
            'mixed_precision': True,
            'attention_type': 'sparse'
        }
        
        if self.device.type == 'cuda':
            # GPU-specific optimizations
            available_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_mb = available_memory / 1024**2
            
            # Adjust batch size based on available memory and sequence length
            memory_per_residue_mb = 2.5  # Empirical estimate
            estimated_memory = sequence_length * memory_per_residue_mb
            
            if estimated_memory < available_mb * 0.7:
                optimizations['batch_size'] = min(4, int(available_mb * 0.7 / estimated_memory))
            
            # Use larger chunks for longer sequences if memory allows
            if estimated_memory < available_mb * 0.5:
                optimizations['chunk_size'] = min(1024, sequence_length // 2)
            
            logger.info(
                f"GPU optimizations for sequence length {sequence_length}: "
                f"batch_size={optimizations['batch_size']}, "
                f"chunk_size={optimizations['chunk_size']}"
            )
        
        return optimizations
    
    def log_memory_stats(self):
        """Log current GPU memory statistics."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            
            # Update peak stats
            self.memory_stats['peak_allocated'] = max(
                self.memory_stats['peak_allocated'], allocated
            )
            self.memory_stats['peak_reserved'] = max(
                self.memory_stats['peak_reserved'], reserved
            )
            
            logger.debug(
                f"GPU Memory - Allocated: {allocated:.1f}MB, "
                f"Reserved: {reserved:.1f}MB, "
                f"Peak: {max_allocated:.1f}MB"
            )


class EfficientAttention:
    """Efficient attention implementations for long protein sequences."""
    
    @staticmethod
    def sparse_attention(
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sparsity_factor: int = 4
    ) -> torch.Tensor:
        """Sparse attention for reduced memory usage."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Use sliding window attention for very long sequences
        if seq_len > 2048:
            return EfficientAttention._sliding_window_attention(
                query, key, value, attention_mask, window_size=512
            )
        
        # Standard sparse attention
        # Create sparse attention pattern
        sparse_mask = torch.zeros(seq_len, seq_len, device=query.device, dtype=torch.bool)
        
        # Local attention (diagonal band)
        band_width = seq_len // sparsity_factor
        for i in range(seq_len):
            start = max(0, i - band_width)
            end = min(seq_len, i + band_width + 1)
            sparse_mask[i, start:end] = True
        
        # Global attention (every sparsity_factor-th position)
        global_indices = torch.arange(0, seq_len, sparsity_factor, device=query.device)
        sparse_mask[global_indices, :] = True
        sparse_mask[:, global_indices] = True
        
        # Apply sparse mask
        if attention_mask is not None:
            sparse_mask = sparse_mask & attention_mask.squeeze()
        
        # Compute attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        scores.masked_fill_(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_probs, value)
        
        return output
    
    @staticmethod
    def _sliding_window_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        window_size: int = 512
    ) -> torch.Tensor:
        """Sliding window attention for very long sequences."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Initialize output
        output = torch.zeros_like(query)
        
        # Process in overlapping windows
        step_size = window_size // 2
        
        for start in range(0, seq_len, step_size):
            end = min(start + window_size, seq_len)
            
            # Extract window
            q_window = query[:, :, start:end, :]
            k_window = key[:, :, start:end, :]
            v_window = value[:, :, start:end, :]
            
            # Compute attention for window
            scores = torch.matmul(q_window, k_window.transpose(-2, -1)) / (head_dim ** 0.5)
            
            if attention_mask is not None:
                mask_window = attention_mask[:, start:end, start:end]
                scores.masked_fill_(~mask_window.unsqueeze(1), float('-inf'))
            
            attn_probs = F.softmax(scores, dim=-1)
            window_output = torch.matmul(attn_probs, v_window)
            
            # Blend overlapping regions
            if start > 0:
                # Average with previous window for overlap
                overlap_start = start
                overlap_end = min(start + step_size, end)
                alpha = 0.5  # Simple blending
                output[:, :, overlap_start:overlap_end, :] = (
                    alpha * output[:, :, overlap_start:overlap_end, :] +
                    (1 - alpha) * window_output[:, :, overlap_start - start:overlap_end - start, :]
                )
                
                # Set non-overlapping region
                if overlap_end < end:
                    output[:, :, overlap_end:end, :] = window_output[:, :, overlap_end - start:, :]
            else:
                output[:, :, start:end, :] = window_output
        
        return output
    
    @staticmethod
    def flash_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """FlashAttention-style memory-efficient attention (simplified)."""
        try:
            # Try to use actual FlashAttention if available
            import flash_attn
            return flash_attn.flash_attn_func(query, key, value, causal=False)
        except ImportError:
            # Fallback to chunked attention
            return EfficientAttention._chunked_attention(query, key, value, attention_mask)
    
    @staticmethod
    def _chunked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 1024
    ) -> torch.Tensor:
        """Chunked attention computation to reduce memory usage."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        if seq_len <= chunk_size:
            # Use standard attention for short sequences
            scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
            if attention_mask is not None:
                scores.masked_fill_(~attention_mask.unsqueeze(1), float('-inf'))
            attn_probs = F.softmax(scores, dim=-1)
            return torch.matmul(attn_probs, value)
        
        # Process in chunks
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = query[:, :, i:end_i, :]
            
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                k_chunk = key[:, :, j:end_j, :]
                v_chunk = value[:, :, j:end_j, :]
                
                # Compute attention for chunk pair
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (head_dim ** 0.5)
                
                if attention_mask is not None:
                    mask_chunk = attention_mask[:, i:end_i, j:end_j]
                    scores.masked_fill_(~mask_chunk.unsqueeze(1), float('-inf'))
                
                attn_probs = F.softmax(scores, dim=-1)
                chunk_output = torch.matmul(attn_probs, v_chunk)
                
                # Accumulate output
                if j == 0:
                    output[:, :, i:end_i, :] = chunk_output
                else:
                    output[:, :, i:end_i, :] += chunk_output
        
        return output


class ModelOptimizer:
    """Model-level optimizations for protein structure prediction."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.gpu_memory_manager = GPUMemoryManager(device)
        
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply inference-time optimizations to model."""
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable inference mode optimizations
        if hasattr(torch, 'jit') and hasattr(model, 'forward'):
            try:
                # Try to JIT compile for better performance
                example_input = self._create_example_input(model)
                model = torch.jit.trace(model, example_input)
                logger.info("Model successfully JIT compiled")
            except Exception as e:
                logger.debug(f"JIT compilation failed, using eager mode: {e}")
        
        # Apply CUDA-specific optimizations
        if self.device.type == 'cuda':
            model = model.half()  # Use FP16 for inference
            logger.info("Model converted to FP16 for inference")
        
        return model
    
    def _create_example_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Create example input for model tracing."""
        # This is a simplified example - in practice, you'd need to match your model's input format
        batch_size = 1
        seq_len = 64
        vocab_size = 21
        
        example_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        return example_input
    
    def optimize_batch_processing(
        self,
        model: torch.nn.Module,
        sequences: List[str],
        max_batch_size: int = 16
    ) -> List[torch.Tensor]:
        """Optimize batch processing for multiple sequences."""
        # Sort sequences by length for efficient batching
        seq_lengths = [(i, len(seq)) for i, seq in enumerate(sequences)]
        seq_lengths.sort(key=lambda x: x[1])
        
        results = [None] * len(sequences)
        
        # Process in length-grouped batches
        i = 0
        while i < len(seq_lengths):
            # Find sequences of similar length
            current_length = seq_lengths[i][1]
            batch_indices = []
            batch_sequences = []
            
            while (i < len(seq_lengths) and 
                   len(batch_indices) < max_batch_size and
                   abs(seq_lengths[i][1] - current_length) <= current_length * 0.1):
                idx, length = seq_lengths[i]
                batch_indices.append(idx)
                batch_sequences.append(sequences[idx])
                i += 1
            
            # Process batch
            with self.gpu_memory_manager.memory_scope():
                batch_results = self._process_sequence_batch(model, batch_sequences)
                
                # Store results in original order
                for batch_idx, result in enumerate(batch_results):
                    original_idx = batch_indices[batch_idx]
                    results[original_idx] = result
        
        return results
    
    def _process_sequence_batch(
        self,
        model: torch.nn.Module,
        sequences: List[str]
    ) -> List[torch.Tensor]:
        """Process a batch of sequences of similar length."""
        # This is a simplified implementation
        # In practice, you'd need proper tokenization and padding
        max_length = max(len(seq) for seq in sequences)
        
        # Tokenize and pad sequences (simplified)
        batch_tokens = []
        for seq in sequences:
            # Simple amino acid to index mapping
            tokens = [ord(c) - ord('A') for c in seq.upper() if 'A' <= c <= 'Z']
            tokens = tokens[:max_length]  # Truncate if too long
            tokens.extend([0] * (max_length - len(tokens)))  # Pad
            batch_tokens.append(tokens)
        
        # Convert to tensor
        input_tensor = torch.tensor(batch_tokens, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            if self.device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
                with torch.cuda.amp.autocast():
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)
        
        # Split batch results
        return [outputs[i] for i in range(len(sequences))]


def optimize_tensor_operations():
    """Apply global tensor operation optimizations."""
    # Enable TensorFloat-32 for better performance on modern GPUs
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable optimized attention if available
        torch.backends.cuda.enable_flash_sdp(True)
        
        logger.info("Enabled CUDA optimizations (TF32, FlashSDP)")
    
    # Optimize CPU operations
    torch.set_num_threads(torch.get_num_threads())  # Use all available cores
    
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
        logger.debug("Enabled MKL-DNN optimizations")


@contextmanager
def inference_mode():
    """Context manager for optimized inference."""
    # Store original settings
    original_grad_enabled = torch.is_grad_enabled()
    
    try:
        # Enable inference optimizations
        torch.set_grad_enabled(False)
        
        if torch.cuda.is_available():
            # Use inference mode for additional optimizations
            with torch.inference_mode():
                yield
        else:
            yield
            
    finally:
        # Restore original settings
        torch.set_grad_enabled(original_grad_enabled)


def gpu_memory_efficient(func):
    """Decorator for GPU memory efficient execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            # Clear cache before execution
            torch.cuda.empty_cache()
            
            # Track memory usage
            start_memory = torch.cuda.memory_allocated()
            
        try:
            result = func(*args, **kwargs)
            
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_used = (end_memory - start_memory) / 1024**2  # MB
                
                if memory_used > 100:  # Log significant memory usage
                    logger.debug(f"Function {func.__name__} used {memory_used:.1f}MB GPU memory")
            
            return result
            
        finally:
            if torch.cuda.is_available():
                # Clean up memory after execution
                torch.cuda.empty_cache()
    
    return wrapper


# Global GPU optimization setup
if torch.cuda.is_available():
    optimize_tensor_operations()
    gpu_memory_manager = GPUMemoryManager()
    logger.info(f"GPU optimizations initialized for {torch.cuda.device_count()} device(s)")
else:
    gpu_memory_manager = None
    logger.info("GPU not available, using CPU optimizations only")