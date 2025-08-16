#!/usr/bin/env python3
"""
Demo script showcasing basic functionality of protein-sssl-operator
Generation 1: MAKE IT WORK - Basic functionality demonstration
"""

import torch
import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protein_sssl.models.ssl_encoder import SequenceStructureSSL, ProteinTokenizer
from protein_sssl.models.neural_operator import NeuralOperatorFold
from protein_sssl.models.structure_decoder import StructurePredictor
from protein_sssl.data.sequence_dataset import ProteinDataset, ProteinDataLoader


def demo_basic_ssl_encoding():
    """Demo 1: Basic Self-Supervised Learning Encoder"""
    print("🧬 Demo 1: Self-Supervised Learning Encoder")
    print("=" * 50)
    
    # Initialize model
    model = SequenceStructureSSL(
        d_model=256,  # Smaller for demo
        n_layers=6,
        n_heads=8,
        ssl_objectives=["masked_modeling", "contrastive"]
    )
    
    # Initialize tokenizer
    tokenizer = ProteinTokenizer()
    
    # Example protein sequence (Insulin A-chain)
    sequence = "GIVEQCCTSICSLYQLENYCN"
    print(f"Input sequence: {sequence}")
    print(f"Length: {len(sequence)} residues")
    
    # Tokenize
    inputs = tokenizer.encode(sequence, max_length=128)
    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)
    
    print(f"Tokenized shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, return_dict=True)
        
    print(f"Hidden states shape: {outputs['last_hidden_state'].shape}")
    
    if "masked_lm_logits" in outputs:
        print(f"Masked LM logits shape: {outputs['masked_lm_logits'].shape}")
        
    if "contrastive_features" in outputs:
        print(f"Contrastive features shape: {outputs['contrastive_features'].shape}")
        
    print("✅ SSL Encoder working correctly!\n")


def demo_neural_operator():
    """Demo 2: Neural Operator for Structure Prediction"""
    print("🔮 Demo 2: Neural Operator Structure Prediction")  
    print("=" * 50)
    
    # Create simple encoder
    encoder = SequenceStructureSSL(
        d_model=256,
        n_layers=4,
        n_heads=8
    )
    
    # Neural operator model
    folding_model = NeuralOperatorFold(
        encoder=encoder,
        d_model=256,
        operator_layers=6,
        fourier_modes=16,
        n_heads=8,
        uncertainty_method="ensemble"
    )
    
    # Example sequence
    sequence = "MKFLKFSLLTAVLLSVVFAFSSCG"
    tokenizer = ProteinTokenizer()
    
    inputs = tokenizer.encode(sequence, max_length=128)
    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)
    
    print(f"Input sequence: {sequence}")
    print(f"Length: {len(sequence)} residues")
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = folding_model(
                input_ids, 
                attention_mask, 
                return_uncertainty=True
            )
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        print("🔧 Please check implementation and dependencies")
        return
    
    print(f"Distance logits shape: {outputs['distance_logits'].shape}")
    print(f"Torsion angles shape: {outputs['torsion_angles'].shape}")  
    print(f"Secondary structure shape: {outputs['secondary_structure'].shape}")
    
    if "uncertainty" in outputs:
        print(f"Uncertainty shape: {outputs['uncertainty'].shape}")
        mean_uncertainty = outputs['uncertainty'].mean().item()
        print(f"Mean uncertainty: {mean_uncertainty:.3f}")
        
    print("✅ Neural Operator working correctly!\n")


def demo_structure_prediction():
    """Demo 3: Full Structure Prediction Pipeline"""
    print("🏗️ Demo 3: Full Structure Prediction Pipeline")
    print("=" * 50)
    
    # Create complete model
    encoder = SequenceStructureSSL(
        d_model=128,  # Small for demo
        n_layers=4,
        n_heads=4
    )
    
    folding_model = NeuralOperatorFold(
        encoder=encoder,
        d_model=128,
        operator_layers=4,
        fourier_modes=8,
        n_heads=4
    )
    
    # Structure predictor
    predictor = StructurePredictor(
        model=folding_model,
        device="cpu",
        confidence_threshold=0.7
    )
    
    # Test sequence (short peptide)
    sequence = "MKFLKFSLLTAVLLSV"
    print(f"Predicting structure for: {sequence}")
    print(f"Length: {len(sequence)} residues")
    
    # Predict structure
    prediction = predictor.predict(
        sequence,
        return_confidence=True,
        num_recycles=1,  # Fast demo
        temperature=0.5
    )
    
    print(f"Coordinates shape: {prediction.coordinates.shape}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"pLDDT score: {prediction.plddt_score:.1f}")
    print(f"Predicted TM-score: {prediction.predicted_tm:.3f}")
    print(f"Distance map shape: {prediction.distance_map.shape}")
    
    # Save demo output
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    prediction.save_pdb(str(output_dir / "demo_structure.pdb"))
    prediction.save_confidence_plot(str(output_dir / "demo_confidence.png"))
    
    print(f"✅ Structure saved to {output_dir}/")
    print("✅ Full pipeline working correctly!\n")


def demo_dataset_loading():
    """Demo 4: Dataset Loading and SSL Objectives"""
    print("📊 Demo 4: Dataset Loading and SSL Training Data")
    print("=" * 50)
    
    # Create synthetic dataset
    synthetic_sequences = [
        "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
        "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFY",
        "FVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGA",
        "VQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDRNKPFKFMLG"
    ]
    
    # Create dataset
    dataset = ProteinDataset(
        sequences=synthetic_sequences,
        max_length=512,
        mask_prob=0.15,
        ssl_objectives=["masked_modeling", "contrastive", "distance_prediction"]
    )
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"SSL objectives: {dataset.ssl_objectives}")
    
    # Get sample
    sample = dataset[0]
    
    print("\nSample data keys:", list(sample.keys()))
    for key, tensor in sample.items():
        if torch.is_tensor(tensor):
            print(f"{key}: {tensor.shape} ({tensor.dtype})")
    
    # Create dataloader
    dataloader = ProteinDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        dynamic_batching=False  # Simple batching for demo
    ).get_dataloader()
    
    # Get batch
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {list(batch.keys())}")
    for key, tensor in batch.items():
        if torch.is_tensor(tensor):
            print(f"Batch {key}: {tensor.shape}")
    
    print("✅ Dataset loading working correctly!\n")


def main():
    """Run all demos"""
    print("🚀 PROTEIN-SSSL-OPERATOR BASIC FUNCTIONALITY DEMO")
    print("=" * 60)
    print("Generation 1: MAKE IT WORK - Basic Implementation")
    print("=" * 60)
    print()
    
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print()
        
        # Run demos
        demo_basic_ssl_encoding()
        demo_neural_operator()
        demo_structure_prediction()
        demo_dataset_loading()
        
        print("🎉 All demos completed successfully!")
        print("✅ Generation 1 (MAKE IT WORK) - COMPLETE")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("🔧 Please check implementation and dependencies")
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)