import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from protein_sssl.models.ssl_encoder import SequenceStructureSSL, ProteinTokenizer
from protein_sssl.models.neural_operator import NeuralOperatorFold
from protein_sssl.models.structure_decoder import StructurePredictor, DistanceGeometry

class TestProteinTokenizer:
    
    def test_tokenizer_init(self):
        tokenizer = ProteinTokenizer()
        assert tokenizer.vocab_size == 21
        assert 'A' in tokenizer.aa_to_id
        assert tokenizer.aa_to_id['A'] == 0
        
    def test_encode(self):
        tokenizer = ProteinTokenizer()
        sequence = "MKFLKFSLLT"
        encoded = tokenizer.encode(sequence, max_length=20)
        
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert len(encoded['input_ids']) == 20
        assert len(encoded['attention_mask']) == 20
        assert encoded['attention_mask'][:len(sequence)].sum() == len(sequence)
        
    def test_decode(self):
        tokenizer = ProteinTokenizer()
        sequence = "ACDEF"
        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded['input_ids'][:len(sequence)])
        assert decoded == sequence
        
    def test_unknown_amino_acids(self):
        tokenizer = ProteinTokenizer()
        sequence = "MKFLZXQWER"  # Z is not a standard amino acid
        encoded = tokenizer.encode(sequence)
        # Unknown amino acids should be mapped to X (20)
        assert 20 in encoded['input_ids']

class TestSequenceStructureSSL:
    
    def test_model_init(self):
        model = SequenceStructureSSL(
            d_model=128,
            n_layers=2,
            n_heads=4,
            ssl_objectives=["masked_modeling"]
        )
        assert model.d_model == 128
        assert model.n_layers == 2
        assert model.n_heads == 4
        
    def test_forward_pass(self):
        model = SequenceStructureSSL(
            d_model=128,
            n_layers=2,
            n_heads=4,
            ssl_objectives=["masked_modeling", "contrastive"]
        )
        
        batch_size, seq_len = 2, 50
        input_ids = torch.randint(0, 21, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, 128)
        assert "masked_lm_logits" in outputs
        assert "contrastive_features" in outputs
        
    def test_get_sequence_embeddings(self):
        model = SequenceStructureSSL(d_model=128, n_layers=2, n_heads=4)
        
        input_ids = torch.randint(0, 21, (1, 30))
        embeddings = model.get_sequence_embeddings(input_ids)
        
        assert embeddings.shape == (1, 30, 128)
        
    def test_save_load_pretrained(self, tmp_path):
        model = SequenceStructureSSL(d_model=128, n_layers=2, n_heads=4)
        
        # Save
        save_dir = tmp_path / "test_model"
        save_dir.mkdir()
        model.save_pretrained(str(save_dir))
        
        # Load
        loaded_model = SequenceStructureSSL.from_pretrained(str(save_dir))
        
        assert loaded_model.d_model == 128
        assert loaded_model.n_layers == 2
        assert loaded_model.n_heads == 4

class TestNeuralOperatorFold:
    
    def test_model_init(self):
        model = NeuralOperatorFold(
            d_model=128,
            operator_layers=2,
            fourier_modes=16,
            n_heads=4
        )
        assert model.d_model == 128
        assert model.operator_layers == 2
        
    def test_forward_pass(self):
        model = NeuralOperatorFold(
            d_model=128,
            operator_layers=2,
            fourier_modes=16,
            n_heads=4
        )
        
        batch_size, seq_len = 2, 30
        input_ids = torch.randint(0, 21, (batch_size, seq_len))
        
        outputs = model(input_ids, return_uncertainty=True)
        
        assert "distance_logits" in outputs
        assert "torsion_angles" in outputs  
        assert "secondary_structure" in outputs
        assert "uncertainty" in outputs
        
        # Check shapes
        assert outputs["distance_logits"].shape == (batch_size, seq_len, seq_len, 64)
        assert outputs["torsion_angles"].shape == (batch_size, seq_len, 8)
        assert outputs["secondary_structure"].shape == (batch_size, seq_len, 8)
        
    def test_with_encoder(self):
        encoder = SequenceStructureSSL(d_model=128, n_layers=2, n_heads=4)
        model = NeuralOperatorFold(
            encoder=encoder,
            d_model=128,
            operator_layers=2
        )
        
        input_ids = torch.randint(0, 21, (1, 20))
        outputs = model(input_ids)
        
        assert "distance_logits" in outputs
        
    def test_predict_structure(self):
        tokenizer = ProteinTokenizer()
        model = NeuralOperatorFold(d_model=64, operator_layers=1, fourier_modes=8)
        
        sequence = "MKFLKFSLLT"
        outputs = model.predict_structure(
            sequence, tokenizer, device="cpu", num_recycles=1
        )
        
        assert "distance_logits" in outputs

class TestDistanceGeometry:
    
    def test_distances_to_coords(self):
        dg = DistanceGeometry()
        
        # Create fake distance probabilities
        seq_len = 10
        n_bins = 64
        distance_probs = torch.rand(seq_len, seq_len, n_bins)
        distance_probs = torch.softmax(distance_probs, dim=-1)
        
        coords = dg.distances_to_coords(distance_probs)
        
        assert coords.shape == (seq_len, 3)
        assert not torch.isnan(coords).any()

class TestStructurePredictor:
    
    def test_init_with_model(self):
        model = NeuralOperatorFold(d_model=64, operator_layers=1)
        predictor = StructurePredictor(model=model, device="cpu")
        
        assert predictor.model is not None
        assert predictor.device == "cpu"
        
    def test_predict(self):
        model = NeuralOperatorFold(d_model=64, operator_layers=1, fourier_modes=8)
        predictor = StructurePredictor(model=model, device="cpu")
        
        sequence = "MKFLKFSLLTAV"
        prediction = predictor.predict(sequence, return_confidence=True, num_recycles=1)
        
        assert prediction.coordinates.shape[0] == len(sequence)
        assert prediction.coordinates.shape[1] == 3
        assert 0 <= prediction.confidence <= 1
        assert 0 <= prediction.plddt_score <= 100
        assert prediction.sequence == sequence
        
    def test_analyze_uncertainty(self):
        model = NeuralOperatorFold(d_model=64, operator_layers=1, uncertainty_method="ensemble")
        predictor = StructurePredictor(model=model, device="cpu")
        
        sequence = "MKFLKFSLLTAV"
        prediction = predictor.predict(sequence, return_confidence=True, num_recycles=1)
        
        analysis = predictor.analyze_uncertainty(prediction)
        
        assert "uncertain_regions" in analysis
        assert "stabilizing_mutations" in analysis
        assert "confidence_summary" in analysis
        
    def test_save_pdb(self, tmp_path):
        model = NeuralOperatorFold(d_model=64, operator_layers=1)
        predictor = StructurePredictor(model=model, device="cpu")
        
        sequence = "MKFL"
        prediction = predictor.predict(sequence, num_recycles=1)
        
        pdb_path = tmp_path / "test.pdb"
        prediction.save_pdb(str(pdb_path))
        
        assert pdb_path.exists()
        
        # Check PDB content
        with open(pdb_path, 'r') as f:
            content = f.read()
            assert "HEADER" in content
            assert "ATOM" in content
            assert "CA" in content

class TestModelIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_ssl_to_folding_pipeline(self):
        # Create SSL model
        ssl_model = SequenceStructureSSL(
            d_model=64,
            n_layers=2,
            n_heads=4,
            ssl_objectives=["masked_modeling"]
        )
        
        # Create folding model with SSL encoder
        folding_model = NeuralOperatorFold(
            encoder=ssl_model,
            d_model=64,
            operator_layers=2,
            fourier_modes=8
        )
        
        # Test sequence
        sequence = "MKFLKFSLLTAVLLSVVF"
        tokenizer = ProteinTokenizer()
        
        # Encode sequence
        inputs = tokenizer.encode(sequence)
        input_ids = inputs["input_ids"].unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            outputs = folding_model(input_ids, return_uncertainty=True)
            
        assert "distance_logits" in outputs
        assert "torsion_angles" in outputs
        assert "secondary_structure" in outputs
        assert "uncertainty" in outputs
        
        # Check realistic output ranges
        distance_probs = torch.softmax(outputs["distance_logits"], dim=-1)
        assert torch.all(distance_probs >= 0)
        assert torch.allclose(distance_probs.sum(dim=-1), torch.ones_like(distance_probs.sum(dim=-1)))
        
    def test_end_to_end_prediction(self):
        # Create complete prediction pipeline
        ssl_model = SequenceStructureSSL(d_model=64, n_layers=1, n_heads=4)
        folding_model = NeuralOperatorFold(
            encoder=ssl_model,
            d_model=64,
            operator_layers=1,
            fourier_modes=8,
            uncertainty_method="ensemble"
        )
        
        predictor = StructurePredictor(
            model=folding_model,
            device="cpu",
            num_ensemble_models=2
        )
        
        # Test prediction
        sequence = "MKFLKFSLLTAVLLSVVFAFSSC"
        prediction = predictor.predict(
            sequence,
            return_confidence=True,
            num_recycles=2,
            temperature=0.5
        )
        
        # Validate prediction
        assert prediction.coordinates.shape == (len(sequence), 3)
        assert not torch.isnan(prediction.coordinates).any()
        assert 0 <= prediction.confidence <= 1
        assert 0 <= prediction.plddt_score <= 100
        assert 0 <= prediction.predicted_tm <= 1
        assert prediction.sequence == sequence
        
        # Test uncertainty analysis
        analysis = predictor.analyze_uncertainty(prediction)
        assert isinstance(analysis["uncertain_regions"], list)
        assert isinstance(analysis["stabilizing_mutations"], list)

# Performance and edge case tests
class TestEdgeCases:
    
    def test_empty_sequence(self):
        tokenizer = ProteinTokenizer()
        encoded = tokenizer.encode("", max_length=10)
        assert len(encoded['input_ids']) == 10
        assert encoded['attention_mask'].sum() == 0
        
    def test_very_long_sequence(self):
        tokenizer = ProteinTokenizer()
        long_sequence = "A" * 2000
        encoded = tokenizer.encode(long_sequence, max_length=1024)
        assert len(encoded['input_ids']) == 1024
        
    def test_invalid_characters(self):
        tokenizer = ProteinTokenizer()
        sequence = "MKFL123XYZ!@#"
        encoded = tokenizer.encode(sequence)
        # Should handle gracefully by mapping to X
        decoded = tokenizer.decode(encoded['input_ids'][:len(sequence)])
        assert 'X' in decoded
        
    def test_single_amino_acid(self):
        model = NeuralOperatorFold(d_model=32, operator_layers=1, fourier_modes=4)
        predictor = StructurePredictor(model=model, device="cpu")
        
        sequence = "M"
        prediction = predictor.predict(sequence, num_recycles=1)
        
        assert prediction.coordinates.shape == (1, 3)
        assert not torch.isnan(prediction.coordinates).any()

if __name__ == "__main__":
    pytest.main([__file__])