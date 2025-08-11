"""
Prediction command for protein-sssl CLI.
"""
import torch
from pathlib import Path
import sys

from ..models.structure_decoder import StructurePredictor
from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)


def predict_command(args):
    """Execute protein structure prediction."""
    logger.info("Starting protein structure prediction...")
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        predictor = StructurePredictor(
            model_path=args.model,
            device=device
        )
        logger.info(f"Loaded model from {args.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Handle sequence input
    sequence = args.sequence
    if Path(sequence).exists():
        # Input is a file
        try:
            with open(sequence, 'r') as f:
                content = f.read().strip()
                if content.startswith('>'):
                    # FASTA format
                    lines = content.split('\n')
                    sequence = ''.join(line for line in lines if not line.startswith('>'))
                else:
                    sequence = content
        except Exception as e:
            logger.error(f"Failed to read sequence file: {e}")
            sys.exit(1)
    
    # Clean sequence
    sequence = ''.join(c.upper() for c in sequence if c.isalpha())
    logger.info(f"Predicting structure for sequence of length {len(sequence)}")
    
    # Predict structure
    try:
        prediction = predictor.predict(
            sequence,
            return_confidence=args.confidence,
            num_recycles=args.num_recycles
        )
        
        # Save results
        prediction.save_pdb(args.output)
        logger.info(f"Structure saved to {args.output}")
        
        # Print summary
        print(f"Prediction completed:")
        print(f"  Sequence length: {len(sequence)}")
        print(f"  pLDDT score: {prediction.plddt_score:.2f}")
        print(f"  Output file: {args.output}")
        
        if hasattr(prediction, 'confidence'):
            print(f"  Overall confidence: {prediction.confidence:.2%}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)