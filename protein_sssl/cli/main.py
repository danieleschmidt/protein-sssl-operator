#!/usr/bin/env python3
"""
Command-line interface for protein-sssl-operator.
"""
import argparse
import sys
from pathlib import Path

from .predict import predict_command
from .train import train_command
from .evaluate import evaluate_command


def create_parser():
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="protein-sssl",
        description="Protein Structure Prediction with Self-Supervised Neural Operators"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict protein structure")
    predict_parser.add_argument("sequence", help="Protein sequence or FASTA file path")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--output", "-o", default="prediction.pdb", 
                               help="Output PDB file path")
    predict_parser.add_argument("--confidence", action="store_true",
                               help="Include confidence scores in B-factors")
    predict_parser.add_argument("--device", default="auto", 
                               help="Device to use (cuda, cpu, or auto)")
    predict_parser.add_argument("--num-recycles", type=int, default=3,
                               help="Number of recycling iterations")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train protein folding model")
    train_parser.add_argument("--config", required=True, help="Training configuration file")
    train_parser.add_argument("--data", required=True, help="Training data directory")
    train_parser.add_argument("--output", "-o", default="./models", 
                             help="Output directory for trained models")
    train_parser.add_argument("--resume", help="Path to checkpoint to resume from")
    train_parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    # Evaluate subcommand  
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--test-data", required=True, help="Test data directory")
    eval_parser.add_argument("--output", "-o", default="evaluation_results.json",
                            help="Output file for results")
    eval_parser.add_argument("--metrics", nargs="+", 
                            default=["tm_score", "gdt_ts", "lddt"],
                            help="Metrics to compute")
    
    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "predict":
            predict_command(args)
        elif args.command == "train":
            train_command(args)
        elif args.command == "evaluate":
            evaluate_command(args)
        else:
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()