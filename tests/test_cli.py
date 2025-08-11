"""
Tests for CLI functionality.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import argparse

from protein_sssl.cli.main import create_parser, main
from protein_sssl.cli.predict import predict_command
from protein_sssl.cli.train import train_command
from protein_sssl.cli.evaluate import evaluate_command


class TestCLIParser:
    """Test CLI argument parsing."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "protein-sssl"
    
    def test_predict_subcommand(self):
        """Test predict subcommand parsing."""
        parser = create_parser()
        
        args = parser.parse_args([
            "predict", 
            "MKFLKFSLLTAV", 
            "--model", "test_model.pt",
            "--output", "test.pdb",
            "--confidence",
            "--device", "cpu",
            "--num-recycles", "5"
        ])
        
        assert args.command == "predict"
        assert args.sequence == "MKFLKFSLLTAV"
        assert args.model == "test_model.pt"
        assert args.output == "test.pdb"
        assert args.confidence is True
        assert args.device == "cpu"
        assert args.num_recycles == 5
    
    def test_train_subcommand(self):
        """Test train subcommand parsing."""
        parser = create_parser()
        
        args = parser.parse_args([
            "train",
            "--config", "config.yaml",
            "--data", "data/",
            "--output", "models/",
            "--gpus", "2"
        ])
        
        assert args.command == "train"
        assert args.config == "config.yaml"
        assert args.data == "data/"
        assert args.output == "models/"
        assert args.gpus == 2
    
    def test_evaluate_subcommand(self):
        """Test evaluate subcommand parsing."""
        parser = create_parser()
        
        args = parser.parse_args([
            "evaluate",
            "--model", "model.pt",
            "--test-data", "test/",
            "--metrics", "tm_score", "lddt"
        ])
        
        assert args.command == "evaluate"
        assert args.model == "model.pt"
        assert args.test_data == "test/"
        assert args.metrics == ["tm_score", "lddt"]


class TestPredictCommand:
    """Test predict command functionality."""
    
    @patch('protein_sssl.cli.predict.StructurePredictor')
    def test_predict_sequence_string(self, mock_predictor_class):
        """Test prediction with sequence string."""
        # Setup mocks
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.plddt_score = 85.5
        mock_prediction.save_pdb = Mock()
        
        mock_predictor.predict.return_value = mock_prediction
        mock_predictor_class.return_value = mock_predictor
        
        # Create args
        args = Mock()
        args.sequence = "MKFLKFSLLTAV"
        args.model = "test_model.pt"
        args.output = "test.pdb"
        args.confidence = True
        args.device = "auto"
        args.num_recycles = 3
        
        with patch('torch.cuda.is_available', return_value=True):
            predict_command(args)
        
        # Verify calls
        mock_predictor_class.assert_called_once_with(
            model_path="test_model.pt",
            device="cuda"
        )
        mock_predictor.predict.assert_called_once_with(
            "MKFLKFSLLTAV",
            return_confidence=True,
            num_recycles=3
        )
        mock_prediction.save_pdb.assert_called_once_with("test.pdb")
    
    @patch('protein_sssl.cli.predict.StructurePredictor')
    def test_predict_fasta_file(self, mock_predictor_class):
        """Test prediction with FASTA file input."""
        # Create temporary FASTA file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as f:
            f.write(">test_protein\nMKFLKFSLLTAV\n")
            fasta_path = f.name
        
        try:
            # Setup mocks
            mock_predictor = Mock()
            mock_prediction = Mock()
            mock_prediction.plddt_score = 85.5
            mock_prediction.save_pdb = Mock()
            
            mock_predictor.predict.return_value = mock_prediction
            mock_predictor_class.return_value = mock_predictor
            
            # Create args
            args = Mock()
            args.sequence = fasta_path
            args.model = "test_model.pt"
            args.output = "test.pdb"
            args.confidence = False
            args.device = "cpu"
            args.num_recycles = 1
            
            predict_command(args)
            
            # Verify sequence was extracted from FASTA
            mock_predictor.predict.assert_called_once_with(
                "MKFLKFSLLTAV",
                return_confidence=False,
                num_recycles=1
            )
            
        finally:
            os.unlink(fasta_path)
    
    @patch('protein_sssl.cli.predict.StructurePredictor')
    def test_predict_invalid_model(self, mock_predictor_class):
        """Test prediction with invalid model path."""
        mock_predictor_class.side_effect = Exception("Model not found")
        
        args = Mock()
        args.sequence = "MKFLKFSLLTAV"
        args.model = "nonexistent.pt"
        args.device = "auto"
        
        with pytest.raises(SystemExit):
            predict_command(args)


class TestTrainCommand:
    """Test train command functionality."""
    
    @patch('protein_sssl.cli.train.OmegaConf')
    @patch('protein_sssl.cli.train._train_ssl')
    def test_train_ssl(self, mock_train_ssl, mock_omega_conf):
        """Test SSL training command."""
        # Setup config mock
        mock_config = Mock()
        mock_config.training_type = 'ssl'
        mock_omega_conf.load.return_value = mock_config
        
        # Create args
        args = Mock()
        args.config = "ssl_config.yaml"
        args.data = "data/"
        args.output = "models/"
        args.resume = None
        args.gpus = 1
        
        train_command(args)
        
        # Verify calls
        mock_omega_conf.load.assert_called_once_with("ssl_config.yaml")
        mock_train_ssl.assert_called_once_with(args, mock_config)
    
    @patch('protein_sssl.cli.train.OmegaConf')
    @patch('protein_sssl.cli.train._train_folding')
    def test_train_folding(self, mock_train_folding, mock_omega_conf):
        """Test folding training command."""
        # Setup config mock
        mock_config = Mock()
        mock_config.training_type = 'folding'
        mock_omega_conf.load.return_value = mock_config
        
        # Create args
        args = Mock()
        args.config = "folding_config.yaml"
        args.data = "data/"
        args.output = "models/"
        args.resume = None
        args.gpus = 1
        
        train_command(args)
        
        # Verify calls
        mock_omega_conf.load.assert_called_once_with("folding_config.yaml")
        mock_train_folding.assert_called_once_with(args, mock_config)
    
    @patch('protein_sssl.cli.train.OmegaConf')
    def test_train_invalid_config(self, mock_omega_conf):
        """Test training with invalid config."""
        mock_omega_conf.load.side_effect = Exception("Config error")
        
        args = Mock()
        args.config = "invalid.yaml"
        
        with pytest.raises(SystemExit):
            train_command(args)


class TestEvaluateCommand:
    """Test evaluate command functionality."""
    
    @patch('protein_sssl.cli.evaluate.StructurePredictor')
    @patch('protein_sssl.cli.evaluate.StructureDataset')
    @patch('protein_sssl.cli.evaluate.StructureEvaluator')
    def test_evaluate_basic(self, mock_evaluator_class, mock_dataset_class, mock_predictor_class):
        """Test basic evaluation functionality."""
        # Setup mocks
        mock_predictor = Mock()
        mock_prediction = Mock()
        mock_prediction.coordinates = "pred_coords"
        mock_prediction.confidence = 0.85
        mock_prediction.plddt_score = 85.0
        mock_predictor.predict.return_value = mock_prediction
        mock_predictor_class.return_value = mock_predictor
        
        mock_dataset = Mock()
        mock_dataset.__len__.return_value = 1
        mock_dataset.__iter__.return_value = iter([{
            'sequence': 'MKFLKFSLLTAV',
            'coordinates': 'true_coords'
        }])
        mock_dataset_class.from_pdb.return_value = mock_dataset
        
        mock_evaluator = Mock()
        mock_evaluator.compute_tm_score.return_value = 0.8
        mock_evaluator.compute_lddt.return_value = 0.75
        mock_evaluator_class.return_value = mock_evaluator
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name
        
        try:
            # Create args
            args = Mock()
            args.model = "test_model.pt"
            args.test_data = "test_data/"
            args.output = output_path
            args.metrics = ["tm_score", "lddt"]
            
            evaluate_command(args)
            
            # Verify calls
            mock_predictor.predict.assert_called_once()
            mock_evaluator.compute_tm_score.assert_called_once()
            mock_evaluator.compute_lddt.assert_called_once()
            
            # Check output file exists
            assert os.path.exists(output_path)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestMainFunction:
    """Test main CLI function."""
    
    @patch('sys.argv', ['protein-sssl'])
    @patch('protein_sssl.cli.main.create_parser')
    def test_main_no_command(self, mock_create_parser):
        """Test main function with no command."""
        mock_parser = Mock()
        mock_parser.parse_args.return_value = Mock(command=None)
        mock_create_parser.return_value = mock_parser
        
        with pytest.raises(SystemExit):
            main()
        
        mock_parser.print_help.assert_called_once()
    
    @patch('sys.argv', ['protein-sssl', 'predict', 'MKFL', '--model', 'test.pt'])
    @patch('protein_sssl.cli.main.predict_command')
    def test_main_predict_command(self, mock_predict_command):
        """Test main function with predict command."""
        main()
        mock_predict_command.assert_called_once()
    
    @patch('sys.argv', ['protein-sssl', 'invalid'])
    def test_main_invalid_command(self):
        """Test main function with invalid command."""
        with pytest.raises(SystemExit):
            main()


if __name__ == "__main__":
    pytest.main([__file__])