#!/usr/bin/env python3
"""
Integration tests for the protein-sssl-operator pipeline
Tests end-to-end functionality from data loading to structure prediction
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from protein_sssl.models.ssl_encoder import SequenceStructureSSL, ProteinTokenizer
from protein_sssl.models.neural_operator import NeuralOperatorFold
from protein_sssl.models.structure_decoder import StructurePredictor
from protein_sssl.data.sequence_dataset import ProteinDataset
from protein_sssl.data.structure_dataset import StructureDataset
from protein_sssl.training.ssl_trainer import SSLTrainer
from protein_sssl.training.folding_trainer import FoldingTrainer
from protein_sssl.evaluation.structure_metrics import StructureEvaluator, BatchStructureEvaluator
from protein_sssl.analysis.domain_analysis import DomainSegmenter, MultiScaleAnalyzer

class TestIntegrationPipeline:
    """Integration tests for the complete pipeline"""
    
    @pytest.fixture
    def sample_sequences(self):
        """Sample protein sequences for testing"""
        return [
            "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
            "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET",
            "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFPTSAJ"
        ]
    
    @pytest.fixture
    def sample_fasta_file(self, sample_sequences):
        """Create temporary FASTA file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            for i, seq in enumerate(sample_sequences):
                f.write(f">seq_{i+1}\n{seq}\n")
            return f.name
    
    @pytest.fixture
    def small_ssl_model(self):
        """Small SSL model for testing"""
        return SequenceStructureSSL(
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=21,
            max_length=256,
            ssl_objectives=["masked_modeling", "contrastive"],
            dropout=0.1
        )
    
    def test_ssl_model_creation(self, small_ssl_model):
        """Test SSL model creation and basic functionality"""
        model = small_ssl_model
        
        # Test model properties
        assert model.d_model == 128
        assert model.n_layers == 2
        assert model.n_heads == 4
        
        # Test forward pass
        tokenizer = ProteinTokenizer()
        sequence = "MKFLKFSLLTAVLLSVVFAFS"
        
        inputs = tokenizer.encode(sequence)
        input_ids = inputs["input_ids"].unsqueeze(0)
        attention_mask = inputs["attention_mask"].unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (1, len(sequence), 128)
    
    def test_dataset_loading(self, sample_fasta_file, sample_sequences):
        """Test dataset loading from FASTA file"""
        
        # Test FASTA loading
        dataset = ProteinDataset.from_fasta(
            sample_fasta_file,
            max_length=256,
            max_sequences=10
        )
        
        assert len(dataset) == len(sample_sequences)
        
        # Test data item structure
        item = dataset[0]
        required_keys = ['input_ids', 'labels', 'attention_mask']
        for key in required_keys:
            assert key in item
            assert isinstance(item[key], torch.Tensor)
        
        # Cleanup
        os.unlink(sample_fasta_file)
    
    def test_ssl_training_step(self, small_ssl_model, sample_fasta_file):
        """Test a single SSL training step"""
        
        # Create dataset
        dataset = ProteinDataset.from_fasta(
            sample_fasta_file,
            max_length=128,
            max_sequences=3
        )
        
        # Create trainer
        trainer = SSLTrainer(
            model=small_ssl_model,
            learning_rate=1e-3,
            warmup_steps=10,
            mixed_precision=False  # Disable for testing
        )
        
        # Test loss computation
        batch = dataset[0]
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        outputs = small_ssl_model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Compute losses
        losses = trainer.compute_ssl_losses(outputs, batch)
        
        assert isinstance(losses, dict)
        assert "masked_lm" in losses
        
        # Cleanup
        os.unlink(sample_fasta_file)
    
    def test_neural_operator_model(self, small_ssl_model):
        """Test neural operator folding model"""
        
        folding_model = NeuralOperatorFold(
            encoder=small_ssl_model,
            d_model=128,
            operator_layers=2,
            fourier_modes=16,
            n_heads=4,
            uncertainty_method="dropout"
        )
        
        # Test forward pass
        tokenizer = ProteinTokenizer()
        sequence = "MKFLKFSLLTAVLLSVVFAFS"
        
        inputs = tokenizer.encode(sequence)
        input_ids = inputs["input_ids"].unsqueeze(0)
        attention_mask = inputs["attention_mask"].unsqueeze(0)
        
        with torch.no_grad():
            outputs = folding_model(
                input_ids,
                attention_mask=attention_mask,
                return_uncertainty=True
            )
        
        # Check output structure
        assert "distance_logits" in outputs
        assert "torsion_angles" in outputs
        assert "secondary_structure" in outputs
        assert "uncertainty" in outputs
        
        # Check shapes
        seq_len = len(sequence)
        assert outputs["distance_logits"].shape == (1, seq_len, seq_len, 64)
        assert outputs["torsion_angles"].shape == (1, seq_len, 8)
        assert outputs["secondary_structure"].shape == (1, seq_len, 8)
    
    def test_structure_prediction_pipeline(self, small_ssl_model):
        """Test complete structure prediction pipeline"""
        
        # Create folding model
        folding_model = NeuralOperatorFold(
            encoder=small_ssl_model,
            d_model=128,
            operator_layers=2,
            fourier_modes=16,
            uncertainty_method="ensemble"
        )
        
        # Create predictor
        predictor = StructurePredictor(
            model=folding_model,
            device="cpu",
            num_ensemble_models=2
        )
        
        # Test prediction
        sequence = "MKFLKFSLLTAVLLSVVFAFS"
        
        prediction = predictor.predict(
            sequence,
            return_confidence=True,
            num_recycles=1
        )
        
        # Check prediction structure
        assert hasattr(prediction, 'coordinates')
        assert hasattr(prediction, 'confidence')
        assert hasattr(prediction, 'plddt_score')
        assert hasattr(prediction, 'predicted_tm')
        
        # Check coordinate shape
        assert prediction.coordinates.shape == (len(sequence), 3)
        
        # Check confidence values
        assert 0 <= prediction.confidence <= 1
        assert 0 <= prediction.plddt_score <= 100
        assert 0 <= prediction.predicted_tm <= 1
    
    def test_structure_evaluation(self):
        """Test structure evaluation metrics"""
        
        # Create mock structures
        seq_len = 50
        pred_coords = torch.randn(seq_len, 3)
        true_coords = pred_coords + 0.5 * torch.randn(seq_len, 3)  # Add noise
        
        evaluator = StructureEvaluator()
        
        metrics = evaluator.evaluate_structure(
            pred_coords,
            true_coords,
            seq_len
        )
        
        # Check metric types
        assert hasattr(metrics, 'tm_score')
        assert hasattr(metrics, 'gdt_ts')
        assert hasattr(metrics, 'rmsd')
        assert hasattr(metrics, 'lddt')
        
        # Check metric ranges
        assert 0 <= metrics.tm_score <= 1
        assert 0 <= metrics.gdt_ts <= 1
        assert metrics.rmsd >= 0
        assert 0 <= metrics.lddt <= 1
    
    def test_batch_structure_evaluation(self):
        """Test batch structure evaluation"""
        
        batch_size = 3
        seq_len = 30
        
        pred_coords_batch = torch.randn(batch_size, seq_len, 3)
        true_coords_batch = pred_coords_batch + 0.3 * torch.randn(batch_size, seq_len, 3)
        
        batch_evaluator = BatchStructureEvaluator()
        
        metrics_list = batch_evaluator.evaluate_batch(
            pred_coords_batch,
            true_coords_batch,
            sequence_lengths=[seq_len] * batch_size
        )
        
        assert len(metrics_list) == batch_size
        
        # Test aggregation
        aggregated = batch_evaluator.aggregate_metrics(metrics_list)
        
        assert 'mean_tm_score' in aggregated
        assert 'mean_rmsd' in aggregated
        assert 'mean_lddt' in aggregated
    
    def test_domain_segmentation(self):
        """Test domain segmentation"""
        
        segmenter = DomainSegmenter(min_domain_length=20)
        
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKVTEYKLVVVGAGGVGKSALTI"
        
        domains = segmenter.segment(
            sequence,
            min_domain_length=20,
            use_evolutionary_info=False
        )
        
        assert len(domains) >= 1
        
        for domain in domains:
            assert hasattr(domain, 'id')
            assert hasattr(domain, 'start') 
            assert hasattr(domain, 'end')
            assert hasattr(domain, 'confidence')
            assert domain.end - domain.start >= 20
    
    def test_multi_scale_analysis(self, small_ssl_model):
        """Test multi-scale protein analysis"""
        
        analyzer = MultiScaleAnalyzer(small_ssl_model)
        
        sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        
        analysis = analyzer.analyze_domain(
            sequence,
            context_sequence=sequence,
            scales=["residue", "secondary"]
        )
        
        assert hasattr(analysis, 'sequence')
        assert hasattr(analysis, 'residue_features')
        assert hasattr(analysis, 'secondary_structure')
        assert analysis.sequence == sequence
    
    def test_end_to_end_pipeline(self, sample_fasta_file):
        """Test complete end-to-end pipeline"""
        
        # 1. Load data
        dataset = ProteinDataset.from_fasta(
            sample_fasta_file,
            max_length=128,
            max_sequences=2
        )
        
        # 2. Create small model for testing
        ssl_model = SequenceStructureSSL(
            d_model=64,
            n_layers=1,
            n_heads=2,
            max_length=128,
            ssl_objectives=["masked_modeling"]
        )
        
        # 3. Test minimal training step
        trainer = SSLTrainer(
            model=ssl_model,
            learning_rate=1e-3,
            mixed_precision=False
        )
        
        # Single training step
        batch = dataset[0]
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
        
        outputs = ssl_model(batch["input_ids"], batch["attention_mask"])
        losses = trainer.compute_ssl_losses(outputs, batch)
        
        assert len(losses) > 0
        
        # 4. Create folding model
        folding_model = NeuralOperatorFold(
            encoder=ssl_model,
            d_model=64,
            operator_layers=1,
            fourier_modes=8
        )
        
        # 5. Test structure prediction
        predictor = StructurePredictor(
            model=folding_model,
            device="cpu",
            num_ensemble_models=1
        )
        
        sequence = "MKFLKFSLLTAVLLSVVFAFS"
        prediction = predictor.predict(sequence, return_confidence=False)
        
        assert prediction.coordinates.shape == (len(sequence), 3)
        
        # 6. Test evaluation
        # Create mock true coordinates
        true_coords = prediction.coordinates + 0.5 * torch.randn_like(prediction.coordinates)
        
        evaluator = StructureEvaluator()
        metrics = evaluator.evaluate_structure(
            prediction.coordinates,
            true_coords,
            len(sequence)
        )
        
        assert metrics.tm_score >= 0
        
        # Cleanup
        os.unlink(sample_fasta_file)
    
    def test_model_serialization(self, small_ssl_model):
        """Test model saving and loading"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model")
            
            # Save model
            small_ssl_model.save_pretrained(save_path)
            
            # Verify files exist
            assert os.path.exists(os.path.join(save_path, "pytorch_model.bin"))
            
            # Load model
            loaded_model = SequenceStructureSSL.from_pretrained(save_path)
            
            # Test loaded model
            tokenizer = ProteinTokenizer()
            sequence = "MKFLKFSL"
            inputs = tokenizer.encode(sequence)
            
            with torch.no_grad():
                orig_output = small_ssl_model(inputs["input_ids"].unsqueeze(0))
                loaded_output = loaded_model(inputs["input_ids"].unsqueeze(0))
            
            # Outputs should be similar (not exact due to random initialization)
            assert orig_output["last_hidden_state"].shape == loaded_output["last_hidden_state"].shape
    
    def test_tokenizer_functionality(self):
        """Test protein tokenizer"""
        
        tokenizer = ProteinTokenizer()
        
        # Test encoding
        sequence = "MKFLKFSLLT"
        inputs = tokenizer.encode(sequence, max_length=20)
        
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert len(inputs["input_ids"]) == 20  # Padded to max_length
        assert inputs["attention_mask"].sum() == len(sequence)  # Non-padded length
        
        # Test decoding
        decoded = tokenizer.decode(inputs["input_ids"][:len(sequence)])
        assert decoded == sequence
    
    def test_error_handling(self):
        """Test error handling in various components"""
        
        # Test invalid sequence
        tokenizer = ProteinTokenizer()
        invalid_sequence = "MKFLKFSLLT123XZ"  # Contains invalid characters
        
        # Should handle gracefully
        inputs = tokenizer.encode(invalid_sequence)
        assert "input_ids" in inputs
        
        # Test empty dataset
        empty_sequences = []
        dataset = ProteinDataset(empty_sequences, max_length=100)
        assert len(dataset) == 0
        
        # Test very short sequence
        short_sequence = "MK"
        predictor_inputs = tokenizer.encode(short_sequence)
        assert len(predictor_inputs["input_ids"]) > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])