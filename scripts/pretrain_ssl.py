#!/usr/bin/env python3

import argparse
import torch
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from protein_sssl import SequenceStructureSSL, ProteinDataset, SSLTrainer

def main():
    parser = argparse.ArgumentParser(description="Pre-train protein SSL model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to FASTA file with protein sequences")
    parser.add_argument("--max_sequences", type=int, default=None,
                       help="Maximum number of sequences to use")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=1280,
                       help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=33,
                       help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=20,
                       help="Number of attention heads")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000,
                       help="Warmup steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                       help="Use mixed precision training")
    
    # Infrastructure arguments
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="protein-ssl-pretraining",
                       help="W&B project name")
    
    # SSL objectives
    parser.add_argument("--ssl_objectives", nargs="+", 
                       default=["masked_modeling", "contrastive", "distance_prediction"],
                       help="SSL objectives to use")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = ProteinDataset.from_fasta(
        args.data_path,
        max_length=args.max_length,
        max_sequences=args.max_sequences
    )
    print(f"Loaded {len(dataset)} protein sequences")
    
    # Initialize model
    print("Initializing SSL model...")
    model = SequenceStructureSSL(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_length=args.max_length,
        ssl_objectives=args.ssl_objectives
    )
    
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = SSLTrainer(
        model=model,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        mixed_precision=args.mixed_precision
    )
    
    # Start pre-training
    print("Starting pre-training...")
    trainer.pretrain(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        project_name=args.project_name
    )
    
    print("Pre-training completed!")

if __name__ == "__main__":
    main()