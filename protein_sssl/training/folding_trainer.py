import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple
import wandb
from tqdm import tqdm
import os
import json
from pathlib import Path
import time
import math
import logging
from sklearn.metrics import accuracy_score, mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoldingTrainer:
    """Trainer for protein folding with comprehensive error handling and validation"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        loss_weights: Optional[Dict[str, float]] = None,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine",
        gradient_checkpointing: bool = False,
        mixed_precision: bool = True,
        max_grad_norm: float = 1.0,
        accumulation_steps: int = 1,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4,
        validation_metric: str = "total_loss"
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.validation_metric = validation_metric
        
        # Loss weights
        if loss_weights is None:
            loss_weights = {
                "distance_map": 1.0,
                "torsion_angles": 0.5,
                "secondary_structure": 0.3,
                "uncertainty": 0.2
            }
        self.loss_weights = loss_weights
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(optimizer_type)
        self.scheduler = None
        self.scheduler_type = scheduler_type
        
        # Mixed precision
        if mixed_precision:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except:
                logger.warning("Mixed precision not available, falling back to FP32")
                self.mixed_precision = False
                self.scaler = None
        else:
            self.scaler = None
            
        # Metrics tracking
        self.training_metrics = {
            'step': 0,
            'epoch': 0,
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'grad_norms': [],
            'best_val_loss': float('inf'),
            'early_stopping_counter': 0
        }
        
        # Device
        self.device = next(model.parameters()).device
        
        # Loss functions
        self.distance_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.torsion_loss_fn = nn.MSELoss()
        self.secondary_loss_fn = nn.CrossEntropyLoss()
        self.uncertainty_loss_fn = nn.MSELoss()
        
    def _setup_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """Setup optimizer with proper parameter grouping"""
        
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # Filter out empty parameter groups
        optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if group["params"]]
        
        if not optimizer_grouped_parameters:
            raise ValueError("No trainable parameters found in the model")
        
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
        elif self.scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.learning_rate * 0.01
            )
        elif self.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=num_training_steps // 3,
                gamma=0.1
            )
            
    def compute_folding_losses(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute protein folding losses with error handling"""
        
        losses = {}
        
        try:
            # Distance map loss
            if "distance_logits" in outputs and "distance_targets" in batch:
                distance_logits = outputs["distance_logits"]
                distance_targets = batch["distance_targets"]
                
                # Ensure shapes match
                if distance_logits.shape[:3] == distance_targets.shape:
                    losses["distance_map"] = self.distance_loss_fn(
                        distance_logits.reshape(-1, distance_logits.size(-1)),
                        distance_targets.reshape(-1)
                    )
                else:
                    logger.warning(f"Distance shape mismatch: logits {distance_logits.shape}, targets {distance_targets.shape}")
                    
        except Exception as e:
            logger.error(f"Error computing distance loss: {e}")
            
        try:
            # Torsion angles loss
            if "torsion_angles" in outputs and "torsion_angles" in batch:
                pred_torsions = outputs["torsion_angles"]
                true_torsions = batch["torsion_angles"]
                
                if pred_torsions.shape == true_torsions.shape:
                    # Mask for valid torsion angles
                    mask = ~torch.isnan(true_torsions).any(dim=-1)
                    if mask.any():
                        losses["torsion_angles"] = self.torsion_loss_fn(
                            pred_torsions[mask], 
                            true_torsions[mask]
                        )
                else:
                    logger.warning(f"Torsion shape mismatch: pred {pred_torsions.shape}, true {true_torsions.shape}")
                    
        except Exception as e:
            logger.error(f"Error computing torsion loss: {e}")
            
        try:
            # Secondary structure loss
            if "secondary_structure" in outputs and "secondary_structure" in batch:
                pred_ss = outputs["secondary_structure"]
                true_ss = batch["secondary_structure"]
                
                if pred_ss.shape[:2] == true_ss.shape:
                    losses["secondary_structure"] = self.secondary_loss_fn(
                        pred_ss.reshape(-1, pred_ss.size(-1)),
                        true_ss.reshape(-1)
                    )
                    
        except Exception as e:
            logger.error(f"Error computing secondary structure loss: {e}")
            
        try:
            # Uncertainty loss (if available)
            if "uncertainty" in outputs and "uncertainty_targets" in batch:
                pred_uncertainty = outputs["uncertainty"]
                true_uncertainty = batch["uncertainty_targets"]
                
                if pred_uncertainty.shape == true_uncertainty.shape:
                    losses["uncertainty"] = self.uncertainty_loss_fn(
                        pred_uncertainty, 
                        true_uncertainty
                    )
                    
        except Exception as e:
            logger.error(f"Error computing uncertainty loss: {e}")
            
        # Ensure we have at least one loss
        if not losses:
            logger.warning("No valid losses computed, using dummy loss")
            losses["dummy"] = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return losses
        
    def compute_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        
        metrics = {}
        
        try:
            # Distance accuracy (within 1Å)
            if "distance_logits" in outputs and "distance_targets" in batch:
                pred_distances = torch.argmax(outputs["distance_logits"], dim=-1).float()
                true_distances = batch["distance_targets"].float()
                
                # Convert bins back to distances (simplified)
                distance_bins = torch.linspace(2.0, 22.0, 64, device=pred_distances.device)
                
                pred_dist_values = distance_bins[pred_distances.long().clamp(0, len(distance_bins)-1)]
                true_dist_values = distance_bins[true_distances.long().clamp(0, len(distance_bins)-1)]
                
                accuracy = (torch.abs(pred_dist_values - true_dist_values) < 1.0).float().mean()
                metrics["distance_accuracy"] = accuracy.item()
                
        except Exception as e:
            logger.debug(f"Error computing distance metrics: {e}")
            
        try:
            # Secondary structure accuracy
            if "secondary_structure" in outputs and "secondary_structure" in batch:
                pred_ss = torch.argmax(outputs["secondary_structure"], dim=-1)
                true_ss = batch["secondary_structure"]
                
                if pred_ss.shape == true_ss.shape:
                    accuracy = (pred_ss == true_ss).float().mean()
                    metrics["ss_accuracy"] = accuracy.item()
                    
        except Exception as e:
            logger.debug(f"Error computing SS metrics: {e}")
            
        try:
            # Torsion RMSE
            if "torsion_angles" in outputs and "torsion_angles" in batch:
                pred_torsions = outputs["torsion_angles"]
                true_torsions = batch["torsion_angles"]
                
                if pred_torsions.shape == true_torsions.shape:
                    mask = ~torch.isnan(true_torsions).any(dim=-1)
                    if mask.any():
                        mse = torch.mean((pred_torsions[mask] - true_torsions[mask]) ** 2)
                        metrics["torsion_rmse"] = torch.sqrt(mse).item()
                        
        except Exception as e:
            logger.debug(f"Error computing torsion metrics: {e}")
            
        return metrics
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model with comprehensive error handling"""
        
        self.model.eval()
        
        total_losses = {key: 0.0 for key in self.loss_weights.keys()}
        total_losses["total"] = 0.0
        total_metrics = {}
        num_batches = 0
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                try:
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    outputs = self.model(
                        batch["input_ids"],
                        attention_mask=batch.get("attention_mask", None),
                        return_uncertainty=True
                    )
                    
                    # Compute losses
                    losses = self.compute_folding_losses(outputs, batch)
                    
                    # Compute total loss
                    total_loss = sum(
                        self.loss_weights.get(key, 0.0) * loss 
                        for key, loss in losses.items()
                    )
                    
                    # Accumulate losses
                    for key, loss in losses.items():
                        if key in total_losses:
                            total_losses[key] += loss.item()
                    total_losses["total"] += total_loss.item()
                    
                    # Compute metrics
                    batch_metrics = self.compute_metrics(outputs, batch)
                    for key, value in batch_metrics.items():
                        if key not in total_metrics:
                            total_metrics[key] = 0.0
                        total_metrics[key] += value
                        
                    num_valid_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Error in validation batch {num_batches}: {e}")
                    
                num_batches += 1
                
        # Average losses and metrics
        if num_valid_batches > 0:
            for key in total_losses:
                total_losses[key] /= num_valid_batches
            for key in total_metrics:
                total_metrics[key] /= num_valid_batches
        else:
            logger.error("No valid validation batches!")
            
        self.model.train()
        
        return {**total_losses, **total_metrics}
        
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device with error handling"""
        
        moved_batch = {}
        for key, value in batch.items():
            try:
                if isinstance(value, torch.Tensor):
                    moved_batch[key] = value.to(self.device, non_blocking=True)
                else:
                    moved_batch[key] = value
            except Exception as e:
                logger.warning(f"Failed to move {key} to device: {e}")
                moved_batch[key] = value
                
        return moved_batch
        
    def fit(
        self,
        dataset,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.1,
        save_dir: str = "./folding_checkpoints",
        eval_every: int = 500,
        save_every: int = 2000,
        log_every: int = 50,
        use_wandb: bool = True,
        project_name: str = "protein-folding",
        resume_from_checkpoint: Optional[str] = None
    ):
        """Train the folding model with robust error handling"""
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if provided
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            self.training_metrics.update(checkpoint.get('training_metrics', {}))
            
        # Split dataset
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        logger.info(f"Training set: {len(train_dataset)} samples")
        logger.info(f"Validation set: {len(val_dataset)} samples")
        
        # Create data loaders with error handling
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            # Fallback to single-threaded loading
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Setup scheduler
        total_steps = len(train_loader) * epochs
        self._setup_scheduler(total_steps)
        
        # Initialize wandb
        if use_wandb:
            try:
                wandb.init(
                    project=project_name,
                    config={
                        "learning_rate": self.learning_rate,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "model_params": sum(p.numel() for p in self.model.parameters()),
                        "loss_weights": self.loss_weights
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                use_wandb = False
                
        # Training loop
        self.model.train()
        
        for epoch in range(start_epoch, epochs):
            epoch_losses = {key: 0.0 for key in self.loss_weights.keys()}
            epoch_losses["total"] = 0.0
            num_batches = 0
            num_valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                try:
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass with mixed precision
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                batch["input_ids"],
                                attention_mask=batch.get("attention_mask", None),
                                return_uncertainty=True
                            )
                            
                            # Compute losses
                            losses = self.compute_folding_losses(outputs, batch)
                            
                            # Weighted total loss
                            total_loss = sum(
                                self.loss_weights.get(key, 0.0) * loss 
                                for key, loss in losses.items()
                            )
                            
                        # Backward pass
                        self.scaler.scale(total_loss / self.accumulation_steps).backward()
                        
                    else:
                        outputs = self.model(
                            batch["input_ids"],
                            attention_mask=batch.get("attention_mask", None),
                            return_uncertainty=True
                        )
                        
                        losses = self.compute_folding_losses(outputs, batch)
                        
                        total_loss = sum(
                            self.loss_weights.get(key, 0.0) * loss 
                            for key, loss in losses.items()
                        )
                        
                        (total_loss / self.accumulation_steps).backward()
                    
                    # Gradient accumulation and update
                    if (step + 1) % self.accumulation_steps == 0:
                        grad_norm = self._optimizer_step()
                        
                        # Update learning rate
                        if self.scheduler and self.scheduler_type != "reduce_on_plateau":
                            self.scheduler.step()
                            
                        self.training_metrics['step'] += 1
                    
                    # Accumulate losses
                    for key, loss in losses.items():
                        if key in epoch_losses:
                            epoch_losses[key] += loss.item()
                    epoch_losses["total"] += total_loss.item()
                    num_valid_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Error in training batch {num_batches}: {e}")
                    
                num_batches += 1
                
                # Logging
                if step % log_every == 0:
                    self._log_training_progress(
                        progress_bar, total_loss, losses, 
                        grad_norm if 'grad_norm' in locals() else None,
                        use_wandb, epoch
                    )
                    
                # Validation
                if step % eval_every == 0 and step > 0:
                    val_metrics = self.validate(val_loader)
                    self._handle_validation_results(val_metrics, use_wandb, epoch, step)
                    
                    # Early stopping check
                    if self._check_early_stopping(val_metrics):
                        logger.info("Early stopping triggered")
                        break
                        
                # Save checkpoint
                if step % save_every == 0 and step > 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
                    self.save_checkpoint(checkpoint_path, epoch, step)
                    
            else:
                # End of epoch validation
                val_metrics = self.validate(val_loader)
                self._handle_validation_results(val_metrics, use_wandb, epoch, len(train_loader))
                
                # Update scheduler if reduce_on_plateau
                if self.scheduler and self.scheduler_type == "reduce_on_plateau":
                    self.scheduler.step(val_metrics.get(self.validation_metric, val_metrics.get("total", 0)))
                    
                # Save epoch checkpoint
                epoch_checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(epoch_checkpoint_path, epoch+1, len(train_loader))
                
                # Early stopping check
                if self._check_early_stopping(val_metrics):
                    logger.info("Early stopping triggered")
                    break
                    
                continue
                
            # If we broke out of the inner loop (early stopping)
            break
            
        # Save final model
        final_save_path = os.path.join(save_dir, "final_model.pt")
        self.save_checkpoint(final_save_path, epochs, total_steps)
        
        logger.info(f"Training completed. Final model saved to {final_save_path}")
        
        if use_wandb:
            wandb.finish()
            
    def _optimizer_step(self) -> float:
        """Perform optimizer step with gradient clipping"""
        
        grad_norm = 0.0
        
        try:
            if self.mixed_precision:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
            
            if self.mixed_precision:
                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
        except Exception as e:
            logger.warning(f"Error in optimizer step: {e}")
            self.optimizer.zero_grad()  # Clear gradients anyway
            
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        
    def _log_training_progress(
        self, progress_bar, total_loss, losses, grad_norm, use_wandb, epoch
    ):
        """Log training progress"""
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Update progress bar
        postfix = {
            'Loss': f"{total_loss.item():.4f}",
            'LR': f"{current_lr:.2e}"
        }
        if grad_norm is not None:
            postfix['GradNorm'] = f"{grad_norm:.2f}"
            
        progress_bar.set_postfix(postfix)
        
        # Log to wandb
        if use_wandb:
            try:
                log_dict = {
                    'train/total_loss': total_loss.item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                    'train/step': self.training_metrics['step']
                }
                
                # Add individual losses
                for key, loss in losses.items():
                    log_dict[f'train/{key}_loss'] = loss.item()
                    
                if grad_norm is not None:
                    log_dict['train/grad_norm'] = grad_norm
                    
                wandb.log(log_dict)
            except Exception as e:
                logger.debug(f"Error logging to wandb: {e}")
                
    def _handle_validation_results(
        self, val_metrics: Dict[str, float], use_wandb: bool, epoch: int, step: int
    ):
        """Handle validation results"""
        
        logger.info(f"Validation results (Epoch {epoch}, Step {step}):")
        for key, value in val_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
        # Log to wandb
        if use_wandb:
            try:
                log_dict = {f'val/{key}': value for key, value in val_metrics.items()}
                log_dict['val/epoch'] = epoch
                log_dict['val/step'] = step
                wandb.log(log_dict)
            except Exception as e:
                logger.debug(f"Error logging validation to wandb: {e}")
                
        # Update metrics
        self.training_metrics['val_losses'].append(val_metrics)
        
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check early stopping condition"""
        
        current_metric = val_metrics.get(self.validation_metric, val_metrics.get("total", float('inf')))
        
        if current_metric < self.training_metrics['best_val_loss'] - self.min_delta:
            self.training_metrics['best_val_loss'] = current_metric
            self.training_metrics['early_stopping_counter'] = 0
        else:
            self.training_metrics['early_stopping_counter'] += 1
            
        return self.training_metrics['early_stopping_counter'] >= self.early_stopping_patience
        
    def save_checkpoint(self, filepath: str, epoch: int, step: int):
        """Save training checkpoint"""
        
        try:
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'training_metrics': self.training_metrics,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'loss_weights': self.loss_weights
            }
            
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")