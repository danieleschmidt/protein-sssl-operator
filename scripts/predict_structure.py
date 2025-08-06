#!/usr/bin/env python3

import argparse
import torch
import os
from pathlib import Path
import sys

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from protein_sssl import SequenceStructureSSL, NeuralOperatorFold, StructurePredictor
from protein_sssl.models.ssl_encoder import ProteinTokenizer

def main():
    parser = argparse.ArgumentParser(description="Predict protein structure")
    
    # Input arguments
    parser.add_argument("--sequence", type=str, required=True,
                       help="Protein sequence to predict")
    parser.add_argument("--sequence_file", type=str,
                       help="File containing protein sequence")
    
    # Model arguments
    parser.add_argument("--ssl_model_path", type=str, required=True,
                       help="Path to pre-trained SSL model")
    parser.add_argument("--folding_model_path", type=str,
                       help="Path to fine-tuned folding model")
    
    # Prediction arguments
    parser.add_argument("--return_confidence", action="store_true", default=True,
                       help="Return confidence scores")
    parser.add_argument("--num_recycles", type=int, default=3,
                       help="Number of recycling iterations")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./predictions",
                       help="Output directory")
    parser.add_argument("--output_prefix", type=str, default="prediction",
                       help="Output file prefix")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Get sequence
    if args.sequence_file:
        with open(args.sequence_file, 'r') as f:
            sequence = f.read().strip().replace('\n', '').replace(' ', '')
    else:
        sequence = args.sequence.replace(' ', '').replace('\n', '')
    
    # Validate sequence
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(aa.upper() in valid_aas for aa in sequence):
        print("Warning: Sequence contains invalid amino acids")
        sequence = ''.join([aa if aa.upper() in valid_aas else 'X' for aa in sequence.upper()])
    
    print(f"Predicting structure for sequence of length {len(sequence)}")
    print(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
    
    # Load pre-trained SSL model
    print(f"Loading SSL model from {args.ssl_model_path}")
    ssl_model = SequenceStructureSSL.from_pretrained(args.ssl_model_path)
    ssl_model.eval()
    
    # Create folding model
    print("Initializing folding model...")
    folding_model = NeuralOperatorFold(
        encoder=ssl_model,
        d_model=ssl_model.d_model,
        operator_layers=12,
        fourier_modes=64,
        uncertainty_method="ensemble"
    )
    
    # Load fine-tuned folding model if available
    if args.folding_model_path and os.path.exists(args.folding_model_path):
        print(f"Loading fine-tuned folding model from {args.folding_model_path}")
        checkpoint = torch.load(args.folding_model_path, map_location=device)
        folding_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No fine-tuned folding model found, using SSL encoder only")
    
    # Create predictor
    predictor = StructurePredictor(
        model=folding_model,
        device=device,
        num_ensemble_models=5 if args.return_confidence else 1,
        confidence_threshold=0.8
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Predict structure
    print("Predicting structure...")
    try:
        prediction = predictor.predict(
            sequence,
            return_confidence=args.return_confidence,
            num_recycles=args.num_recycles,
            temperature=args.temperature
        )
        
        # Print results
        print("\nPrediction Results:")
        print(f"  Overall confidence: {prediction.confidence:.2%}")
        print(f"  pLDDT score: {prediction.plddt_score:.2f}")
        print(f"  Predicted TM-score: {prediction.predicted_tm:.2f}")
        
        # Save PDB file
        pdb_path = os.path.join(args.output_dir, f"{args.output_prefix}.pdb")
        prediction.save_pdb(pdb_path)
        print(f"  Structure saved: {pdb_path}")
        
        # Save confidence plot
        if args.return_confidence:
            plot_path = os.path.join(args.output_dir, f"{args.output_prefix}_confidence.png")
            prediction.save_confidence_plot(plot_path)
            print(f"  Confidence plot saved: {plot_path}")
            
            # Analyze uncertainty
            uncertainty_analysis = predictor.analyze_uncertainty(prediction)
            print("\nUncertainty Analysis:")
            print(f"  {uncertainty_analysis['confidence_summary']}")
            if uncertainty_analysis['uncertain_regions']:
                print(f"  High uncertainty regions: {', '.join(uncertainty_analysis['uncertain_regions'])}")
            if uncertainty_analysis['stabilizing_mutations']:
                print(f"  Suggested stabilizing mutations: {', '.join(uncertainty_analysis['stabilizing_mutations'])}")
        
        # Save prediction data
        import pickle
        data_path = os.path.join(args.output_dir, f"{args.output_prefix}_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'sequence': sequence,
                'coordinates': prediction.coordinates.cpu(),
                'confidence': prediction.confidence,
                'plddt_score': prediction.plddt_score,
                'predicted_tm': prediction.predicted_tm,
                'distance_map': prediction.distance_map.cpu(),
                'torsion_angles': prediction.torsion_angles.cpu(),
                'secondary_structure': prediction.secondary_structure.cpu(),
                'uncertainty': prediction.uncertainty.cpu() if prediction.uncertainty is not None else None
            }, f)
        print(f"  Prediction data saved: {data_path}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    print("\nPrediction completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())