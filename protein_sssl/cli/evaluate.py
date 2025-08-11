"""
Evaluation command for protein-sssl CLI.
"""
import json
import torch
from pathlib import Path
import sys
from typing import Dict, Any

from ..models.structure_decoder import StructurePredictor
from ..evaluation.structure_metrics import StructureEvaluator
from ..data.structure_dataset import StructureDataset
from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)


def evaluate_command(args):
    """Execute model evaluation."""
    logger.info("Starting model evaluation...")
    
    # Load model
    try:
        predictor = StructurePredictor(
            model_path=args.model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loaded model from {args.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load test dataset
    try:
        test_dataset = StructureDataset.from_pdb(
            args.test_data,
            resolution_cutoff=3.0,
            remove_redundancy=False  # Keep all test structures
        )
        logger.info(f"Loaded {len(test_dataset)} test structures")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = StructureEvaluator()
    
    # Run evaluation
    results = {}
    all_metrics = []
    
    for i, sample in enumerate(test_dataset):
        try:
            sequence = sample['sequence']
            true_coords = sample['coordinates']
            
            logger.info(f"Evaluating structure {i+1}/{len(test_dataset)}")
            
            # Predict structure
            prediction = predictor.predict(sequence, return_confidence=True)
            pred_coords = prediction.coordinates
            
            # Compute metrics
            structure_metrics = {}
            
            if 'tm_score' in args.metrics:
                structure_metrics['tm_score'] = evaluator.compute_tm_score(
                    pred_coords, true_coords
                )
            
            if 'gdt_ts' in args.metrics:
                structure_metrics['gdt_ts'] = evaluator.compute_gdt_ts(
                    pred_coords, true_coords
                )
            
            if 'lddt' in args.metrics:
                structure_metrics['lddt'] = evaluator.compute_lddt(
                    pred_coords, true_coords
                )
            
            if 'rmsd' in args.metrics:
                structure_metrics['rmsd'] = evaluator.compute_rmsd(
                    pred_coords, true_coords
                )
            
            # Add confidence metrics if available
            if hasattr(prediction, 'confidence'):
                structure_metrics['confidence'] = prediction.confidence
                structure_metrics['plddt_score'] = prediction.plddt_score
            
            structure_metrics['sequence_length'] = len(sequence)
            structure_metrics['structure_id'] = f"structure_{i:04d}"
            
            all_metrics.append(structure_metrics)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate structure {i}: {e}")
            continue
    
    # Compute summary statistics
    if all_metrics:
        results = _compute_summary_stats(all_metrics, args.metrics)
        results['individual_results'] = all_metrics
        results['total_structures'] = len(all_metrics)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {args.output}")
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"  Total structures evaluated: {len(all_metrics)}")
        for metric in args.metrics:
            if f'{metric}_mean' in results:
                mean_val = results[f'{metric}_mean']
                std_val = results[f'{metric}_std']
                print(f"  {metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")
                
    else:
        logger.error("No structures were successfully evaluated")
        sys.exit(1)


def _compute_summary_stats(all_metrics: list, requested_metrics: list) -> Dict[str, Any]:
    """Compute summary statistics from individual results."""
    import numpy as np
    
    results = {}
    
    for metric in requested_metrics:
        values = [m[metric] for m in all_metrics if metric in m and m[metric] is not None]
        
        if values:
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
            results[f'{metric}_median'] = np.median(values)
            results[f'{metric}_min'] = np.min(values)
            results[f'{metric}_max'] = np.max(values)
            results[f'{metric}_count'] = len(values)
    
    # Compute additional summary metrics
    if 'confidence' in requested_metrics:
        conf_values = [m['confidence'] for m in all_metrics if 'confidence' in m]
        if conf_values:
            results['high_confidence_count'] = sum(1 for c in conf_values if c > 0.8)
            results['medium_confidence_count'] = sum(1 for c in conf_values if 0.5 <= c <= 0.8)
            results['low_confidence_count'] = sum(1 for c in conf_values if c < 0.5)
    
    # Length-based analysis
    seq_lengths = [m['sequence_length'] for m in all_metrics if 'sequence_length' in m]
    if seq_lengths:
        results['avg_sequence_length'] = np.mean(seq_lengths)
        results['sequence_length_range'] = [np.min(seq_lengths), np.max(seq_lengths)]
    
    return results