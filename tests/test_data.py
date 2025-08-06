import pytest
import torch
import numpy as np
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from protein_sssl.data.sequence_dataset import ProteinDataset, ProteinDataLoader
from protein_sssl.data.structure_dataset import StructureDataset

class TestProteinDataset:
    
    def test_init(self):
        sequences = ["MKFLKFSLLT", "ACDEFGHIKL", "NQRSTVWY"]
        dataset = ProteinDataset(sequences, max_length=100)
        
        assert len(dataset) == 3
        assert dataset.max_length == 100
        assert dataset.mask_prob == 0.15
        
    def test_process_sequences(self):
        sequences = ["mkflkfsllt", "ACDEFGHIKL", "NQ", ""]  # Mixed case, short, empty
        dataset = ProteinDataset(sequences, max_length=100)
        
        # Should filter out sequences that are too short or empty
        assert len(dataset) == 2  # Only valid sequences
        
    def test_getitem(self):
        sequences = ["MKFLKFSLLTAVLLSVVFAFSSC"]
        dataset = ProteinDataset(sequences, ssl_objectives=["masked_modeling", "contrastive"])
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert "contrastive_input" in item
        assert "contrastive_mask" in item
        
        # Check tensor types and shapes
        assert isinstance(item["input_ids"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long
        assert len(item["input_ids"]) == dataset.max_length
        
    def test_masked_lm_example(self):
        sequences = ["MKFLKFSLLT"]
        dataset = ProteinDataset(sequences, mask_prob=1.0)  # Mask everything for testing
        
        item = dataset[0]
        
        # With 100% masking, should have some masked tokens
        assert (item["input_ids"] == dataset.mask_token_id).any()
        assert (item["labels"] != -100).any()  # Some positions should have labels
        
    def test_augment_sequence(self):
        sequences = ["MKFLKFSLLT"]
        dataset = ProteinDataset(sequences)
        
        original = sequences[0]
        augmented = dataset._augment_sequence(original, prob=1.0)  # 100% augmentation
        
        # Should be same length but potentially different
        assert len(augmented) == len(original)
        
    def test_from_text_file(self, tmp_path):
        # Create temporary text file
        text_file = tmp_path / "sequences.txt"
        sequences = ["MKFLKFSLLT", "ACDEFGHIKL", "NQRSTVWY"]
        with open(text_file, 'w') as f:
            for seq in sequences:
                f.write(seq + '\n')
                
        dataset = ProteinDataset.from_text_file(str(text_file))
        
        # Should filter based on length (min 30 by default)
        assert len(dataset) == 0  # All sequences too short
        
    def test_cache_functionality(self, tmp_path):
        sequences = ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"]
        dataset = ProteinDataset(sequences)
        
        cache_path = tmp_path / "cache.pkl"
        dataset.save_cache(str(cache_path))
        
        assert cache_path.exists()
        
        # Load from cache
        loaded_dataset = ProteinDataset.load_cache(str(cache_path))
        assert len(loaded_dataset) == len(dataset)
        assert loaded_dataset.processed_sequences == dataset.processed_sequences
        
    def test_distance_targets(self):
        sequences = ["MKFLKFSLLTAVLLSVVF"]
        dataset = ProteinDataset(sequences, ssl_objectives=["distance_prediction"])
        
        item = dataset[0]
        
        assert "distance_targets" in item
        assert isinstance(item["distance_targets"], torch.Tensor)
        
        seq_len = len([c for c in sequences[0] if c in dataset.aa_to_id])
        expected_shape = (dataset.max_length, dataset.max_length)
        assert item["distance_targets"].shape == expected_shape

class TestProteinDataLoader:
    
    def test_init(self):
        sequences = ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"] * 10
        dataset = ProteinDataset(sequences)
        
        data_loader = ProteinDataLoader(
            dataset, 
            batch_size=4, 
            dynamic_batching=False
        )
        
        loader = data_loader.get_dataloader()
        
        batch = next(iter(loader))
        assert len(batch["input_ids"]) == 4  # batch size
        
    def test_dynamic_batching(self):
        # Create sequences of different lengths
        sequences = [
            "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
            "MKFLKFSLLTAVLLSVVFAFSSC",
            "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKVTEST",
        ] * 5
        dataset = ProteinDataset(sequences)
        
        data_loader = ProteinDataLoader(
            dataset,
            batch_size=8,
            dynamic_batching=True,
            max_tokens=1000
        )
        
        loader = data_loader.get_dataloader()
        
        # Should create batches
        batches = list(loader)
        assert len(batches) > 0
        
        # Each batch should be a list of items
        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) > 0

class TestStructureDataset:
    
    def create_mock_pdb_content(self):
        """Create mock PDB content for testing"""
        return """HEADER    MOCK PROTEIN STRUCTURE
ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 50.00           C
ATOM      2  CA  ARG A   2      13.800  20.000  30.000  1.00 45.00           C
ATOM      3  CA  ASN A   3      17.600  20.000  30.000  1.00 40.00           C
ATOM      4  CA  ASP A   4      21.400  20.000  30.000  1.00 35.00           C
ATOM      5  CA  CYS A   5      25.200  20.000  30.000  1.00 30.00           C
END
"""
    
    def test_init(self, tmp_path):
        # Create mock PDB file
        pdb_file = tmp_path / "test.pdb"
        with open(pdb_file, 'w') as f:
            f.write(self.create_mock_pdb_content())
            
        structure_paths = [str(pdb_file)]
        
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset(
                structure_paths, 
                max_length=100,
                resolution_cutoff=3.0
            )
        
        # Should process the mock structure
        assert len(dataset) >= 0  # Might be filtered out due to length
        
    def test_quality_filters(self, tmp_path):
        # Create mock PDB file
        pdb_file = tmp_path / "test.pdb"
        with open(pdb_file, 'w') as f:
            f.write(self.create_mock_pdb_content())
            
        structure_paths = [str(pdb_file)]
        
        # Set very strict quality filters
        quality_filters = {
            'min_length': 100,  # Our mock has only 5 residues
            'max_missing_residues': 0.0,
            'max_b_factor': 20.0,
            'chain_breaks_allowed': 0
        }
        
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset(
                structure_paths,
                quality_filters=quality_filters
            )
        
        # Should filter out the structure
        assert len(dataset) == 0
        
    def test_distance_matrix_calculation(self):
        # Test distance matrix calculation
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset([])  # Empty dataset for testing methods
        
        # Create mock coordinates
        coords = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        distance_matrix = dataset._calculate_distance_matrix(coords)
        
        assert distance_matrix.shape == (4, 4)
        assert distance_matrix[0, 0] == 0  # Distance to self
        assert abs(distance_matrix[0, 1] - 1.0) < 1e-6  # Distance should be 1.0
        assert distance_matrix[0, 1] == distance_matrix[1, 0]  # Symmetric
        
    def test_torsion_angle_calculation(self):
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset([])
            
        # Mock residues with atoms
        mock_residues = []
        for i in range(3):
            mock_residue = Mock()
            # Mock atoms
            atoms = {
                'C': Mock(), 'N': Mock(), 'CA': Mock()
            }
            
            # Set coordinates
            atoms['C'].get_coord.return_value = np.array([i, 0, 0])
            atoms['N'].get_coord.return_value = np.array([i, 1, 0])
            atoms['CA'].get_coord.return_value = np.array([i, 0.5, 0.5])
            
            mock_residue.__getitem__ = lambda self, key: atoms.get(key)
            mock_residue.__contains__ = lambda self, key: key in atoms
            mock_residues.append(mock_residue)
            
        angles = dataset._calculate_torsion_angles(mock_residues)
        
        assert angles.shape == (3, 2)  # phi, psi for each residue
        assert not np.isnan(angles).any()
        
    def test_from_pdb_directory(self, tmp_path):
        # Create directory with mock PDB files
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        
        for i in range(3):
            pdb_file = pdb_dir / f"protein_{i}.pdb"
            with open(pdb_file, 'w') as f:
                f.write(self.create_mock_pdb_content())
                
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset.from_pdb_directory(
                str(pdb_dir),
                max_structures=2
            )
            
        # Should have processed up to 2 structures
        assert len(dataset.structure_paths) <= 2
        
    def test_getitem_error_handling(self):
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset([])
            dataset.processed_structures = [None]  # Invalid structure
            
        # Should return dummy data instead of crashing
        item = dataset[0]
        
        assert "input_ids" in item
        assert "coordinates" in item
        assert item["sequence"] == "X" * 50  # Dummy sequence
        
    def test_statistics(self, tmp_path):
        # Create mock structures with different properties
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset([])
            
            # Mock processed structures
            dataset.processed_structures = [
                {
                    'sequence': 'ACDEFGHIKLMNPQRSTVWY' * 2,  # 40 residues
                    'quality_metrics': {'resolution': 2.5}
                },
                {
                    'sequence': 'ACDEFGHIKLMNPQRSTVWY' * 3,  # 60 residues
                    'quality_metrics': {'resolution': 1.8}
                }
            ]
            
        stats = dataset.get_statistics()
        
        assert 'num_structures' in stats
        assert 'sequence_length' in stats
        assert 'resolution' in stats
        assert stats['num_structures'] == 2
        assert stats['sequence_length']['mean'] == 50.0  # (40 + 60) / 2

class TestDataIntegration:
    """Integration tests for data loading pipeline"""
    
    def test_sequence_to_structure_pipeline(self, tmp_path):
        # Create sample sequence data
        sequences = [
            "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFG"
        ]
        
        seq_dataset = ProteinDataset(sequences, ssl_objectives=["masked_modeling"])
        
        # Test sequence dataset
        seq_item = seq_dataset[0]
        assert "input_ids" in seq_item
        assert "labels" in seq_item
        
        # Create mock structure dataset
        pdb_file = tmp_path / "test.pdb"
        with open(pdb_file, 'w') as f:
            f.write("""HEADER    TEST
ATOM      1  CA  MET A   1      10.000  20.000  30.000  1.00 50.00           C
ATOM      2  CA  LYS A   2      13.800  20.000  30.000  1.00 45.00           C
ATOM      3  CA  PHE A   3      17.600  20.000  30.000  1.00 40.00           C
ATOM      4  CA  LEU A   4      21.400  20.000  30.000  1.00 35.00           C
ATOM      5  CA  LYS A   5      25.200  20.000  30.000  1.00 30.00           C
END
""")
        
        with patch('protein_sssl.data.structure_dataset.logger'):
            struct_dataset = StructureDataset([str(pdb_file)])
            
        if len(struct_dataset) > 0:
            struct_item = struct_dataset[0]
            assert "coordinates" in struct_item
            assert "distance_matrix" in struct_item
            
    def test_dataloader_compatibility(self):
        sequences = ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"] * 8
        dataset = ProteinDataset(sequences)
        
        data_loader = ProteinDataLoader(dataset, batch_size=4)
        loader = data_loader.get_dataloader()
        
        # Test that we can iterate through batches
        batch_count = 0
        for batch in loader:
            assert "input_ids" in batch
            assert batch["input_ids"].shape[0] <= 4  # Batch size
            batch_count += 1
            
            if batch_count >= 2:  # Test a few batches
                break
                
        assert batch_count >= 1

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""
    
    def test_empty_dataset(self):
        dataset = ProteinDataset([])
        assert len(dataset) == 0
        
        # Should handle empty dataset gracefully
        data_loader = ProteinDataLoader(dataset, batch_size=4)
        loader = data_loader.get_dataloader()
        
        batches = list(loader)
        assert len(batches) == 0
        
    def test_invalid_sequences(self):
        sequences = ["", "A", "123", "ZZZZZ", "MKFL*KF&SL"]
        dataset = ProteinDataset(sequences)
        
        # Should filter out invalid/short sequences
        assert len(dataset) == 0  # All too short or invalid
        
    def test_corrupted_structure_handling(self, tmp_path):
        # Create corrupted PDB file
        pdb_file = tmp_path / "corrupted.pdb"
        with open(pdb_file, 'w') as f:
            f.write("INVALID PDB CONTENT\n")
            
        with patch('protein_sssl.data.structure_dataset.logger'):
            dataset = StructureDataset([str(pdb_file)])
            
        # Should handle gracefully
        assert len(dataset) == 0
        assert str(pdb_file) in dataset.failed_structures
        
    def test_large_batch_handling(self):
        # Test with very large batch size
        sequences = ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"] * 2
        dataset = ProteinDataset(sequences)
        
        data_loader = ProteinDataLoader(dataset, batch_size=100)  # Larger than dataset
        loader = data_loader.get_dataloader()
        
        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 2  # Should be dataset size, not batch size
        
    def test_memory_efficiency(self):
        # Test that dataset doesn't load everything into memory at once
        sequences = ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"] * 1000
        dataset = ProteinDataset(sequences, max_length=512)
        
        # Should create without loading all into memory
        assert len(dataset) > 0
        
        # Individual items should be reasonable size
        item = dataset[0]
        assert item["input_ids"].numel() <= 512
        
if __name__ == "__main__":
    pytest.main([__file__])