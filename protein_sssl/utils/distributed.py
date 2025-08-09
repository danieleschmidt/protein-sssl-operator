"""
Distributed training utilities for protein-sssl-operator
Supports multi-GPU and multi-node training with advanced optimizations
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
import time
import socket
from typing import Optional, Dict, Any, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str = "env://"
    master_addr: str = "localhost"
    master_port: str = "12355"
    timeout_minutes: int = 30
    use_apex: bool = False
    gradient_predivide_factor: float = 1.0
    bucket_cap_mb: int = 25

class DistributedManager:
    """Manages distributed training setup and coordination"""
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.is_initialized = False
        self.process_group = None
        
    def setup_distributed(self, 
                         rank: Optional[int] = None,
                         world_size: Optional[int] = None,
                         backend: Optional[str] = None):
        """Setup distributed training environment"""
        
        # Update config if parameters provided
        if rank is not None:
            self.config.rank = rank
        if world_size is not None:
            self.config.world_size = world_size
        if backend is not None:
            self.config.backend = backend
            
        # Set environment variables
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.rank)
        
        if torch.cuda.is_available():
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            torch.cuda.set_device(self.config.local_rank)
        
        try:
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=torch.distributed.default_pg_timeout
            )
            
            self.is_initialized = True
            self.process_group = dist.group.WORLD
            
            logger.info(f"Initialized distributed training: rank={self.config.rank}, "
                       f"world_size={self.config.world_size}, backend={self.config.backend}")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.is_initialized:
            try:
                dist.destroy_process_group()
                self.is_initialized = False
                logger.info("Cleaned up distributed resources")
            except Exception as e:
                logger.warning(f"Error during distributed cleanup: {e}")
    
    def is_master(self) -> bool:
        """Check if current process is master (rank 0)"""
        return self.config.rank == 0
    
    def get_rank(self) -> int:
        """Get current process rank"""
        return self.config.rank
    
    def get_world_size(self) -> int:
        """Get total world size"""
        return self.config.world_size
    
    def barrier(self):
        """Synchronization barrier"""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """All-reduce operation across all processes"""
        if self.is_initialized and self.config.world_size > 1:
            dist.all_reduce(tensor, op=op)
            return tensor / self.config.world_size
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """All-gather operation"""
        if not self.is_initialized or self.config.world_size == 1:
            return [tensor]
        
        gather_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(gather_list, tensor)
        return gather_list
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from source to all processes"""
        if self.is_initialized and self.config.world_size > 1:
            dist.broadcast(tensor, src=src)
        return tensor

class DistributedDataLoader:
    """Distributed data loader with advanced features"""
    
    def __init__(self,
                 dataset,
                 batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 distributed_config: Optional[DistributedConfig] = None):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.config = distributed_config or DistributedConfig()
        
        # Create distributed sampler if in distributed mode
        if self.config.world_size > 1:
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=shuffle,
                drop_last=drop_last
            )
            shuffle = False  # Sampler handles shuffling
        else:
            self.sampler = None
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0
        )
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

class DistributedModel:
    """Wrapper for distributed model with optimizations"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 distributed_config: Optional[DistributedConfig] = None,
                 find_unused_parameters: bool = False,
                 static_graph: bool = False):
        
        self.config = distributed_config or DistributedConfig()
        self.original_model = model
        self.find_unused_parameters = find_unused_parameters
        self.static_graph = static_graph
        
        # Move model to appropriate device
        if torch.cuda.is_available() and self.config.world_size > 1:
            device = torch.device(f"cuda:{self.config.local_rank}")
            model = model.to(device)
        
        # Wrap with DDP if distributed
        if self.config.world_size > 1:
            self.model = DDP(
                model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb,
                static_graph=static_graph
            )
        else:
            self.model = model
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()
    
    def state_dict(self):
        """Get state dict, handling DDP wrapper"""
        if isinstance(self.model, DDP):
            return self.model.module.state_dict()
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict, handling DDP wrapper"""
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

class GradientSynchronizer:
    """Advanced gradient synchronization utilities"""
    
    def __init__(self, model: torch.nn.Module, distributed_config: DistributedConfig):
        self.model = model
        self.config = distributed_config
        self.gradient_buffers = {}
        
    def sync_gradients(self):
        """Manually synchronize gradients across processes"""
        if self.config.world_size <= 1:
            return
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # All-reduce gradients
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.config.world_size
    
    def average_gradients(self):
        """Average gradients across all processes"""
        if self.config.world_size <= 1:
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.config.world_size

class DistributedCheckpointing:
    """Distributed checkpointing with coordination"""
    
    def __init__(self, distributed_config: DistributedConfig):
        self.config = distributed_config
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       step: int,
                       filepath: str,
                       metadata: Optional[Dict[str, Any]] = None):
        """Save checkpoint (only on master process)"""
        
        if not self.config.rank == 0:
            # Wait for master to finish saving
            if self.config.world_size > 1:
                dist.barrier()
            return
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metadata': metadata or {},
            'distributed_config': {
                'world_size': self.config.world_size,
                'backend': self.config.backend
            }
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
        finally:
            # Synchronize all processes
            if self.config.world_size > 1:
                dist.barrier()
    
    def load_checkpoint(self,
                       filepath: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       strict: bool = True) -> Dict[str, Any]:
        """Load checkpoint on all processes"""
        
        try:
            checkpoint = torch.load(filepath, map_location=f"cuda:{self.config.local_rank}")
            
            # Load model state
            if isinstance(model, DDP):
                model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Checkpoint loaded: {filepath}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

@contextmanager
def distributed_context(rank: int, 
                       world_size: int,
                       backend: str = "nccl",
                       cleanup_on_exit: bool = True):
    """Context manager for distributed training"""
    
    manager = DistributedManager(DistributedConfig(
        rank=rank,
        world_size=world_size,
        backend=backend
    ))
    
    try:
        manager.setup_distributed()
        yield manager
    finally:
        if cleanup_on_exit:
            manager.cleanup()

def find_free_port() -> str:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)

def get_local_rank() -> int:
    """Get local rank from environment"""
    return int(os.environ.get('LOCAL_RANK', '0'))

def get_global_rank() -> int:
    """Get global rank from environment"""
    return int(os.environ.get('RANK', '0'))

def get_world_size() -> int:
    """Get world size from environment"""
    return int(os.environ.get('WORLD_SIZE', '1'))

def is_distributed() -> bool:
    """Check if distributed training is enabled"""
    return get_world_size() > 1

def is_master() -> bool:
    """Check if current process is master"""
    return get_global_rank() == 0

class MultiNodeLauncher:
    """Utility for launching multi-node training jobs"""
    
    def __init__(self,
                 script_path: str,
                 nodes: List[str],
                 gpus_per_node: int = 8,
                 master_port: int = 29500):
        
        self.script_path = script_path
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node
        self.master_port = master_port
        self.world_size = len(nodes) * gpus_per_node
    
    def launch_slurm(self,
                    partition: str,
                    job_name: str,
                    time_limit: str = "24:00:00",
                    additional_args: Optional[List[str]] = None) -> str:
        """Launch job using SLURM"""
        
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={len(self.nodes)}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_PORT={self.master_port}
export WORLD_SIZE={self.world_size}

srun python -u {self.script_path} {' '.join(additional_args or [])}
"""
        
        # Save SLURM script
        script_file = f"{job_name}.sbatch"
        with open(script_file, 'w') as f:
            f.write(slurm_script)
        
        # Submit job
        result = subprocess.run(['sbatch', script_file], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"SLURM job submitted: {job_id}")
            return job_id
        else:
            raise RuntimeError(f"SLURM job submission failed: {result.stderr}")
    
    def launch_torchrun(self,
                       additional_args: Optional[List[str]] = None) -> None:
        """Launch using torchrun"""
        
        cmd = [
            'torchrun',
            f'--nproc_per_node={self.gpus_per_node}',
            f'--nnodes={len(self.nodes)}',
            f'--node_rank=0',  # This would need to be set per node
            f'--master_addr={self.nodes[0]}',
            f'--master_port={self.master_port}',
            self.script_path
        ]
        
        if additional_args:
            cmd.extend(additional_args)
        
        logger.info(f"Launching with torchrun: {' '.join(cmd)}")
        subprocess.run(cmd)

class PerformanceOptimizer:
    """Distributed training performance optimizations"""
    
    def __init__(self, model: torch.nn.Module, distributed_config: DistributedConfig):
        self.model = model
        self.config = distributed_config
    
    def optimize_ddp(self):
        """Optimize DDP settings"""
        if isinstance(self.model, DDP):
            # Enable gradient bucketing optimization
            self.model._set_static_graph()
            
            # Set find_unused_parameters to False if possible
            self.model.find_unused_parameters = False
    
    def setup_gradient_compression(self):
        """Setup gradient compression for bandwidth optimization"""
        # This would integrate with libraries like Horovod or FP16 compression
        logger.info("Gradient compression setup (placeholder)")
    
    def optimize_communication(self):
        """Optimize communication patterns"""
        # Setup optimal process group topologies
        if self.config.world_size > 1:
            # Could implement hierarchical all-reduce, etc.
            logger.info("Communication optimization setup (placeholder)")

def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce tensor across all processes"""
    if world_size <= 1:
        return tensor
        
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def gather_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather tensor from all processes"""
    if world_size <= 1:
        return tensor
        
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list, dim=0)