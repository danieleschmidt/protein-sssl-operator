import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import wandb
from tqdm import tqdm
import os
import json
from pathlib import Path
import time
import math

class SSLTrainer:
    """Self-Supervised Learning Trainer for protein sequences"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = False,
        mixed_precision: bool = True,
        accumulation_steps: int = 1,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        ssl_loss_weights: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        
        # SSL loss weights
        if ssl_loss_weights is None:
            ssl_loss_weights = {
                "masked_lm": 1.0,
                "contrastive": 0.5,
                "distance_prediction": 0.3
            }
        self.ssl_loss_weights = ssl_loss_weights
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(optimizer_type)
        self.scheduler = None
        self.scheduler_type = scheduler_type
        
        # Mixed precision
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # Metrics tracking
        self.training_metrics = {
            'step': 0,
            'epoch': 0,
            'total_loss': [],
            'masked_lm_loss': [],
            'contrastive_loss': [],
            'distance_loss': [],
            'learning_rate': [],
            'grad_norm': []
        }
        
        # Device
        self.device = next(model.parameters()).device
        
    def _setup_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """Setup optimizer with parameter groups"""
        
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if optimizer_type.lower() == "adamw":
            return optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        elif optimizer_type.lower() == "adam":
            return optim.Adam(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
            
    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler"""
        
        if self.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.learning_rate * 0.1
            )
        elif self.scheduler_type == "linear":
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / self.warmup_steps
                else:
                    return max(0.1, (num_training_steps - step) / (num_training_steps - self.warmup_steps))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.scheduler_type == "constant":
            def lr_lambda(step):
                return 1.0 if step >= self.warmup_steps else step / self.warmup_steps
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
    def compute_ssl_losses(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute self-supervised learning losses"""
        
        losses = {}
        
        # Masked Language Model Loss
        if "masked_lm_logits" in outputs and "labels" in batch:
            masked_lm_loss = nn.CrossEntropyLoss(ignore_index=-100)
            logits = outputs["masked_lm_logits"].view(-1, outputs["masked_lm_logits"].size(-1))
            labels = batch["labels"].view(-1)
            losses["masked_lm"] = masked_lm_loss(logits, labels)
            
        # Contrastive Loss
        if "contrastive_features" in outputs and "contrastive_input" in batch:
            losses["contrastive"] = self._compute_contrastive_loss(
                outputs["contrastive_features"], 
                batch
            )
            
        # Distance Prediction Loss
        if "distance_logits" in outputs and "distance_targets" in batch:
            losses["distance_prediction"] = self._compute_distance_loss(
                outputs["distance_logits"],
                batch["distance_targets"]
            )
            
        return losses
        
    def _compute_contrastive_loss(
        self, 
        features: torch.Tensor, 
        batch: Dict[str, torch.Tensor],
        temperature: float = 0.07
    ) -> torch.Tensor:
        """Compute contrastive loss (InfoNCE)"""
        
        batch_size = features.shape[0]
        
        # Normalize features
        features = nn.functional.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / temperature
        
        # Create labels (positive pairs are diagonal)
        labels = torch.arange(batch_size, device=features.device)
        
        # InfoNCE loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(similarity, labels)
        
        return loss
        
    def _compute_distance_loss(
        self, 
        distance_logits: torch.Tensor, 
        distance_targets: torch.Tensor,
        distance_bins: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute distance prediction loss"""
        
        if distance_bins is None:
            distance_bins = torch.linspace(2.0, 22.0, 64, device=distance_logits.device)
            
        # Convert continuous targets to binned targets
        distances = distance_targets.unsqueeze(-1)  # [B, L, L, 1]
        bins = distance_bins.view(1, 1, 1, -1)     # [1, 1, 1, 64]
        
        # Find closest bin for each distance
        bin_indices = torch.argmin(torch.abs(distances - bins), dim=-1)
        
        # Cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            distance_logits.reshape(-1, distance_logits.size(-1)),
            bin_indices.reshape(-1)
        )
        
        return loss
        
    def pretrain(
        self,
        dataset,
        epochs: int = 10,
        batch_size: int = 128,
        num_gpus: int = 1,
        save_dir: str = "./checkpoints",
        eval_every: int = 1000,
        save_every: int = 5000,
        log_every: int = 100,
        use_wandb: bool = True,
        project_name: str = "protein-ssl"
    ):
        """Pre-train the model with SSL objectives"""
        
        # Setup distributed training if multiple GPUs
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            
        # Setup data loader
        from ..data.sequence_dataset import ProteinDataLoader
        data_loader = ProteinDataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            dynamic_batching=True
        ).get_dataloader()
        
        # Calculate total training steps
        total_steps = len(data_loader) * epochs
        self._setup_scheduler(total_steps)
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                config={
                    "learning_rate": self.learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "model_params": sum(p.numel() for p in self.model.parameters()),
                    "ssl_objectives": list(self.ssl_loss_weights.keys())
                }
            )
            
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_losses = {key: 0.0 for key in self.ssl_loss_weights.keys()}
            
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch["input_ids"],
                            attention_mask=batch.get("attention_mask", None)
                        )
                        
                        # Compute losses
                        ssl_losses = self.compute_ssl_losses(outputs, batch)
                        
                        # Weighted total loss
                        total_loss = sum(
                            self.ssl_loss_weights.get(key, 0.0) * loss 
                            for key, loss in ssl_losses.items()
                        )
                        
                    # Backward pass
                    self.scaler.scale(total_loss / self.accumulation_steps).backward()
                    
                else:
                    outputs = self.model(
                        batch["input_ids"],
                        attention_mask=batch.get("attention_mask", None)
                    )
                    
                    ssl_losses = self.compute_ssl_losses(outputs, batch)
                    
                    total_loss = sum(
                        self.ssl_loss_weights.get(key, 0.0) * loss 
                        for key, loss in ssl_losses.items()
                    )
                    
                    (total_loss / self.accumulation_steps).backward()
                
                # Gradient accumulation
                if (step + 1) % self.accumulation_steps == 0:
                    if self.mixed_precision:
                        # Gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                        
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                    
                    if self.scheduler is not None:
                        self.scheduler.step()
                        
                    # Update metrics
                    self.training_metrics['step'] += 1
                    
                # Logging
                if step % log_every == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f"{total_loss.item():.4f}",
                        'LR': f"{current_lr:.2e}",
                        'GradNorm': f"{grad_norm:.2f}" if 'grad_norm' in locals() else "N/A"
                    })
                    
                    # Log to wandb
                    if use_wandb:
                        log_dict = {
                            'train/total_loss': total_loss.item(),
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch,
                            'train/step': self.training_metrics['step']
                        }
                        
                        # Add individual SSL losses
                        for key, loss in ssl_losses.items():
                            log_dict[f'train/{key}_loss'] = loss.item()
                            
                        if 'grad_norm' in locals():
                            log_dict['train/grad_norm'] = grad_norm.item()
                            
                        wandb.log(log_dict)
                
                # Save checkpoint
                if step % save_every == 0 and step > 0:
                    self.save_checkpoint(
                        os.path.join(save_dir, f"checkpoint_step_{self.training_metrics['step']}.pt"),
                        epoch, step, total_loss.item()
                    )
                
                # Update epoch metrics
                epoch_loss += total_loss.item()
                for key, loss in ssl_losses.items():
                    epoch_losses[key] += loss.item()
                    
            # End of epoch
            self.training_metrics['epoch'] = epoch + 1
            avg_epoch_loss = epoch_loss / len(data_loader)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            for key, loss in epoch_losses.items():
                avg_loss = loss / len(data_loader)
                print(f"  {key} Loss: {avg_loss:.4f}")
                
            # Save epoch checkpoint
            self.save_checkpoint(
                os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                epoch, step, avg_epoch_loss
            )
            
        # Save final model
        final_save_path = os.path.join(save_dir, "final_model")
        if isinstance(self.model, nn.DataParallel):
            self.model.module.save_pretrained(final_save_path)
        else:
            self.model.save_pretrained(final_save_path)
            
        print(f"Training completed. Final model saved to {final_save_path}")
        
        if use_wandb:
            wandb.finish()
            
    def save_checkpoint(self, filepath: str, epoch: int, step: int, loss: float):
        """Save training checkpoint"""
        
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'step': step, 
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'training_metrics': self.training_metrics,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.training_metrics = checkpoint['training_metrics']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Step: {checkpoint['step']}")
        print(f"  Loss: {checkpoint['loss']:.4f}")