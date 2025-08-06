# API Reference

## Core Models

### SequenceStructureSSL

Self-supervised learning model for protein sequences.

```python
from protein_sssl import SequenceStructureSSL

model = SequenceStructureSSL(
    d_model=1280,           # Model dimension
    n_layers=33,            # Number of transformer layers  
    n_heads=20,             # Number of attention heads
    vocab_size=21,          # Amino acid vocabulary size
    max_length=1024,        # Maximum sequence length
    ssl_objectives=[        # Self-supervised objectives
        "masked_modeling",
        "contrastive", 
        "distance_prediction"
    ],
    dropout=0.1             # Dropout rate
)
```

#### Methods

**`forward(input_ids, attention_mask=None, return_dict=True)`**
- **Args**:
  - `input_ids` (torch.Tensor): Tokenized sequence IDs [batch, seq_len]
  - `attention_mask` (torch.Tensor): Attention mask [batch, seq_len]
  - `return_dict` (bool): Return dictionary format
- **Returns**: Model outputs with hidden states and SSL objective logits

**`get_sequence_embeddings(input_ids, attention_mask=None)`**
- **Args**: Same as forward
- **Returns**: torch.Tensor [batch, seq_len, d_model] sequence embeddings

**`save_pretrained(save_directory)`**
- **Args**: `save_directory` (str): Path to save model
- **Returns**: None

**`from_pretrained(model_path)`** (classmethod)
- **Args**: `model_path` (str): Path to saved model
- **Returns**: SequenceStructureSSL instance

### NeuralOperatorFold

Neural operator for protein folding with Fourier layers.

```python
from protein_sssl import NeuralOperatorFold

model = NeuralOperatorFold(
    encoder=ssl_model,      # Pre-trained SSL encoder (optional)
    d_model=1280,           # Model dimension
    operator_layers=12,     # Number of neural operator layers
    fourier_modes=64,       # Number of Fourier modes
    n_heads=16,             # Attention heads
    attention_type="efficient",  # Attention type
    uncertainty_method="ensemble",  # Uncertainty quantification
    dropout=0.1             # Dropout rate
)
```

#### Methods

**`forward(input_ids, attention_mask=None, return_uncertainty=False)`**
- **Args**:
  - `input_ids` (torch.Tensor): Input sequence tokens
  - `attention_mask` (torch.Tensor): Attention mask
  - `return_uncertainty` (bool): Include uncertainty estimates
- **Returns**: Dict with distance_logits, torsion_angles, secondary_structure, uncertainty

**`predict_structure(sequence, tokenizer, device="cpu", return_confidence=True, num_recycles=3)`**
- **Args**:
  - `sequence` (str): Protein sequence
  - `tokenizer`: Tokenizer instance
  - `device` (str): Compute device
  - `return_confidence` (bool): Include confidence scores
  - `num_recycles` (int): Number of recycling iterations
- **Returns**: Structure prediction dictionary

### StructurePredictor

Complete structure prediction pipeline.

```python
from protein_sssl import StructurePredictor

predictor = StructurePredictor(
    model_path="path/to/model",  # Or model instance
    device="cuda",               # Compute device
    num_ensemble_models=5,       # Ensemble size for uncertainty
    confidence_threshold=0.8     # Confidence threshold
)
```

#### Methods

**`predict(sequence, return_confidence=True, num_recycles=3, temperature=0.1)`**
- **Args**:
  - `sequence` (str): Protein sequence
  - `return_confidence` (bool): Return confidence scores
  - `num_recycles` (int): Recycling iterations
  - `temperature` (float): Sampling temperature
- **Returns**: StructurePrediction object

**`analyze_uncertainty(prediction)`**
- **Args**: `prediction` (StructurePrediction): Prediction result
- **Returns**: Dict with uncertainty analysis

### StructurePrediction

Container for structure prediction results.

#### Attributes
- `coordinates` (torch.Tensor): 3D coordinates [seq_len, 3]
- `confidence` (float): Overall confidence score
- `plddt_score` (float): pLDDT confidence score
- `predicted_tm` (float): Predicted TM-score
- `distance_map` (torch.Tensor): Distance probability map
- `torsion_angles` (torch.Tensor): Backbone torsion angles
- `secondary_structure` (torch.Tensor): Secondary structure probabilities
- `uncertainty` (torch.Tensor): Per-residue uncertainty
- `sequence` (str): Input sequence

#### Methods

**`save_pdb(filename)`**
- **Args**: `filename` (str): Output PDB file path
- **Returns**: None

**`save_confidence_plot(filename)`**
- **Args**: `filename` (str): Output plot file path
- **Returns**: None

## Data Components

### ProteinDataset

Dataset for protein sequences with SSL objectives.

```python
from protein_sssl import ProteinDataset

dataset = ProteinDataset(
    sequences=["MKFL...", "ACDE..."],  # List of sequences
    max_length=1024,                   # Maximum sequence length
    mask_prob=0.15,                    # Masking probability
    ssl_objectives=["masked_modeling"]  # SSL objectives
)

# Load from FASTA
dataset = ProteinDataset.from_fasta(
    "sequences.fasta",
    max_length=1024,
    max_sequences=10000
)
```

#### Methods

**`__getitem__(idx)`**
- **Args**: `idx` (int): Dataset index
- **Returns**: Dict with input_ids, labels, attention_mask, etc.

**`save_cache(cache_path)`**
- **Args**: `cache_path` (str): Cache file path
- **Returns**: None

### StructureDataset

Dataset for protein structures with quality filtering.

```python
from protein_sssl.data import StructureDataset

dataset = StructureDataset(
    structure_paths=["protein1.pdb", "protein2.pdb"],
    max_length=1024,
    resolution_cutoff=3.0,
    remove_redundancy=True,
    quality_filters={
        'min_length': 30,
        'max_missing_residues': 0.1,
        'max_b_factor': 100.0
    }
)

# Load from directory
dataset = StructureDataset.from_pdb_directory(
    "pdb_files/",
    resolution_cutoff=2.5,
    max_structures=1000
)
```

#### Methods

**`get_statistics()`**
- **Returns**: Dict with dataset statistics

## Training Components

### SSLTrainer

Trainer for self-supervised learning.

```python
from protein_sssl.training import SSLTrainer

trainer = SSLTrainer(
    model=ssl_model,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=10000,
    mixed_precision=True,
    ssl_loss_weights={
        "masked_lm": 1.0,
        "contrastive": 0.5,
        "distance_prediction": 0.3
    }
)
```

#### Methods

**`pretrain(dataset, epochs=10, batch_size=128, save_dir="./checkpoints")`**
- **Args**:
  - `dataset`: ProteinDataset instance
  - `epochs` (int): Number of training epochs
  - `batch_size` (int): Batch size
  - `save_dir` (str): Checkpoint directory
- **Returns**: None

### FoldingTrainer

Trainer for structure prediction fine-tuning.

```python
from protein_sssl.training import FoldingTrainer

trainer = FoldingTrainer(
    model=folding_model,
    learning_rate=1e-4,
    loss_weights={
        "distance_map": 1.0,
        "torsion_angles": 0.5,
        "secondary_structure": 0.3
    },
    early_stopping_patience=10
)
```

#### Methods

**`fit(dataset, epochs=50, batch_size=16, validation_split=0.1)`**
- **Args**:
  - `dataset`: StructureDataset instance  
  - `epochs` (int): Training epochs
  - `batch_size` (int): Batch size
  - `validation_split` (float): Validation fraction
- **Returns**: None

## Analysis Components

### DomainSegmenter

Multi-scale domain segmentation with evolutionary information.

```python
from protein_sssl.analysis import DomainSegmenter

segmenter = DomainSegmenter(
    min_domain_length=40,
    use_evolutionary_info=True,
    cache_size=500,
    num_workers=4
)
```

#### Methods

**`segment(sequence, min_domain_length=None, use_evolutionary_info=None)`**
- **Args**:
  - `sequence` (str): Protein sequence
  - `min_domain_length` (int): Minimum domain length
  - `use_evolutionary_info` (bool): Use evolutionary features
- **Returns**: List[DomainPrediction]

### MultiScaleAnalyzer

Multi-scale protein structure analysis.

```python
from protein_sssl.analysis import MultiScaleAnalyzer

analyzer = MultiScaleAnalyzer(
    model=folding_model,
    cache_size=200,
    num_workers=4,
    gpu_acceleration=True
)
```

#### Methods

**`analyze_domain(sequence, context_sequence=None, scales=None, coordinates=None)`**
- **Args**:
  - `sequence` (str): Domain sequence
  - `context_sequence` (str): Full protein context
  - `scales` (List[str]): Analysis scales
  - `coordinates` (torch.Tensor): 3D coordinates if available
- **Returns**: MultiScaleAnalysis object

## Evaluation Components

### OptimizedStructureEvaluator

High-performance structure evaluation with GPU acceleration.

```python
from protein_sssl.evaluation import OptimizedStructureEvaluator

evaluator = OptimizedStructureEvaluator(
    device="cuda",
    batch_size=32,
    num_workers=4,
    cache_size=1000
)
```

#### Methods

**`evaluate_structure(coords_pred, coords_true, sequence, **kwargs)`**
- **Args**:
  - `coords_pred` (torch.Tensor): Predicted coordinates
  - `coords_true` (torch.Tensor): True coordinates
  - `sequence` (str): Protein sequence
  - Additional optional tensors for comprehensive evaluation
- **Returns**: StructureMetrics object

**`batch_evaluate_structures(predictions, ground_truths, sequences)`**
- **Args**:
  - `predictions` (List[Dict]): Prediction data
  - `ground_truths` (List[Dict]): Ground truth data
  - `sequences` (List[str]): Sequences
- **Returns**: List[StructureMetrics]

## Utility Classes

### ProteinTokenizer

Simple tokenizer for protein sequences.

```python
from protein_sssl.models.ssl_encoder import ProteinTokenizer

tokenizer = ProteinTokenizer()
encoded = tokenizer.encode("MKFLKFSLLT", max_length=50)
decoded = tokenizer.decode(encoded['input_ids'])
```

## Configuration

All models and trainers support configuration via dictionaries or YAML files:

```python
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model from config
model = SequenceStructureSSL(**config['model'])
trainer = SSLTrainer(model=model, **config['training'])
```

## Error Handling

All components include comprehensive error handling:

- **Input Validation**: Type checking and value validation
- **Graceful Degradation**: Fallback behaviors for edge cases  
- **Resource Management**: Automatic cleanup and memory management
- **Logging**: Structured logging for debugging and monitoring

## Performance Tips

### GPU Optimization
- Use mixed precision training (`mixed_precision=True`)
- Batch sequences by similar length
- Enable gradient checkpointing for large models

### CPU Optimization  
- Use multiple workers for data loading (`num_workers=4`)
- Enable caching for repeated computations
- Use vectorized operations where possible

### Memory Management
- Use dynamic batching for variable-length sequences
- Enable gradient accumulation for large effective batch sizes
- Clear caches periodically in long-running processes