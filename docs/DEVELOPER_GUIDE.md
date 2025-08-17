# Developer Guide

## Overview

This comprehensive developer guide provides detailed information for contributors, maintainers, and developers working on the protein-sssl-operator. It covers architecture deep-dive, contribution guidelines, development processes, testing strategies, and best practices.

## Table of Contents

1. [Architecture Deep-Dive](#architecture-deep-dive)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Contribution Guidelines](#code-contribution-guidelines)
4. [Testing Strategies](#testing-strategies)
5. [Release Management](#release-management)
6. [Code Review Process](#code-review-process)
7. [Performance Engineering](#performance-engineering)
8. [Security Guidelines](#security-guidelines)
9. [Documentation Standards](#documentation-standards)
10. [Troubleshooting Development Issues](#troubleshooting-development-issues)

## Architecture Deep-Dive

### System Architecture Overview

The protein-sssl-operator follows a modular, layered architecture designed for scalability, maintainability, and extensibility:

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer                                │
├─────────────────────────────────────────────────────────────┤
│                 Business Logic Layer                        │
├─────────────────────────────────────────────────────────────┤
│                   Model Layer                               │
├─────────────────────────────────────────────────────────────┤
│                 Data Processing Layer                       │
├─────────────────────────────────────────────────────────────┤
│                Infrastructure Layer                         │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Self-Supervised Learning Framework

**Location**: `protein_sssl/models/ssl_encoder.py`

```python
class SequenceStructureSSL(nn.Module):
    """
    Core SSL model implementing multiple pre-training objectives
    
    Architecture:
    - Transformer encoder with rotary positional embeddings
    - Multi-head attention with configurable architecture
    - Multiple SSL objective heads (MLM, contrastive, distance prediction)
    """
    
    def __init__(
        self,
        d_model: int = 1280,
        n_layers: int = 33,
        n_heads: int = 20,
        vocab_size: int = 21,
        max_length: int = 1024,
        ssl_objectives: List[str] = None,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        # Core transformer architecture
        self.embeddings = ProteinEmbeddings(d_model, vocab_size, max_length)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            gradient_checkpointing=use_gradient_checkpointing
        )
        
        # SSL objective heads
        self.ssl_heads = nn.ModuleDict()
        if ssl_objectives:
            for objective in ssl_objectives:
                self.ssl_heads[objective] = self._create_ssl_head(objective, d_model)
    
    def _create_ssl_head(self, objective: str, d_model: int) -> nn.Module:
        """Factory method for SSL objective heads"""
        if objective == "masked_modeling":
            return MaskedLanguageModelingHead(d_model, self.config.vocab_size)
        elif objective == "contrastive":
            return ContrastiveLearningHead(d_model)
        elif objective == "distance_prediction":
            return DistancePredictionHead(d_model)
        else:
            raise ValueError(f"Unknown SSL objective: {objective}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ssl_labels: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass with optional SSL objective computation
        
        Args:
            input_ids: Tokenized sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            ssl_labels: Labels for SSL objectives
            return_dict: Whether to return dictionary format
            
        Returns:
            Model outputs including hidden states and SSL losses
        """
        # Embedding layer
        embeddings = self.embeddings(input_ids)
        
        # Transformer encoder
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=attention_mask
        )
        
        hidden_states = encoder_outputs.last_hidden_state
        
        # SSL objectives
        ssl_outputs = {}
        total_loss = 0
        
        if ssl_labels:
            for objective, head in self.ssl_heads.items():
                if objective in ssl_labels:
                    ssl_output = head(hidden_states, ssl_labels[objective])
                    ssl_outputs[f"{objective}_loss"] = ssl_output.loss
                    ssl_outputs[f"{objective}_logits"] = ssl_output.logits
                    total_loss += ssl_output.loss
        
        if return_dict:
            return {
                'last_hidden_state': hidden_states,
                'ssl_outputs': ssl_outputs,
                'total_loss': total_loss if ssl_labels else None
            }
        else:
            return hidden_states
```

#### 2. Neural Operator Framework

**Location**: `protein_sssl/models/neural_operator.py`

```python
class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator for protein folding
    
    Implements spectral convolution in Fourier space for modeling
    continuous transformations from sequence to structure space.
    """
    
    def __init__(
        self,
        d_model: int,
        fourier_modes: int = 64,
        n_layers: int = 4,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.fourier_modes = fourier_modes
        self.n_layers = n_layers
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv1d(d_model, d_model, fourier_modes)
            for _ in range(n_layers)
        ])
        
        # Pointwise convolutions
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 1)
            for _ in range(n_layers)
        ])
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fourier Neural Operator
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            
        Returns:
            Transformed features [batch_size, seq_len, d_model]
        """
        # Transpose for conv1d: [batch, d_model, seq_len]
        x = x.transpose(1, 2)
        
        for i in range(self.n_layers):
            # Spectral convolution
            x1 = self.fourier_layers[i](x)
            
            # Pointwise convolution
            x2 = self.pointwise_layers[i](x)
            
            # Combine and apply activation
            x = self.activation(x1 + x2)
            
            # Layer normalization (transpose back temporarily)
            x = x.transpose(1, 2)
            x = self.layer_norms[i](x)
            x = x.transpose(1, 2)
        
        # Transpose back: [batch, seq_len, d_model]
        return x.transpose(1, 2)

class SpectralConv1d(nn.Module):
    """1D Spectral Convolution using FFT"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier weights (complex-valued)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
        
        # Initialize weights
        nn.init.xavier_normal_(self.weights.real)
        nn.init.xavier_normal_(self.weights.imag)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral convolution in Fourier space
        
        Args:
            x: Input tensor [batch, channels, length]
            
        Returns:
            Convolved tensor [batch, channels, length]
        """
        batch_size, in_channels, length = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Multiply with Fourier weights
        out_ft = torch.zeros(
            batch_size, self.out_channels, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        
        # Apply weights to low-frequency modes
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :self.modes], 
            self.weights
        )
        
        # IFFT
        x = torch.fft.irfft(out_ft, n=length, dim=-1)
        
        return x
```

#### 3. Structure Prediction Pipeline

**Location**: `protein_sssl/models/structure_decoder.py`

```python
class StructureDecoder(nn.Module):
    """
    Structure decoder that converts sequence representations to 3D coordinates
    
    Uses distance geometry and iterative refinement to generate
    physically plausible protein structures.
    """
    
    def __init__(
        self,
        d_model: int,
        n_recycles: int = 3,
        distance_bins: int = 64,
        angle_bins: int = 24
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_recycles = n_recycles
        self.distance_bins = distance_bins
        self.angle_bins = angle_bins
        
        # Distance prediction head
        self.distance_head = DistancePredictor(
            d_model, distance_bins
        )
        
        # Torsion angle prediction head
        self.angle_head = AnglePredictor(
            d_model, angle_bins
        )
        
        # Secondary structure prediction
        self.ss_head = SecondaryStructurePredictor(d_model)
        
        # Coordinate generation module
        self.coord_generator = CoordinateGenerator()
        
        # Recycling module for iterative refinement
        self.recycling_module = RecyclingModule(d_model)
    
    def forward(
        self,
        sequence_features: torch.Tensor,
        prev_coords: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate 3D structure from sequence features
        
        Args:
            sequence_features: Encoded sequence [batch, seq_len, d_model]
            prev_coords: Previous coordinates for recycling
            return_intermediates: Return intermediate predictions
            
        Returns:
            Dictionary containing structure predictions
        """
        batch_size, seq_len, _ = sequence_features.shape
        
        # Initialize recycling features
        if prev_coords is not None:
            recycling_features = self.recycling_module(
                sequence_features, prev_coords
            )
            features = sequence_features + recycling_features
        else:
            features = sequence_features
        
        # Predict distances
        distance_logits = self.distance_head(features)
        distance_probs = F.softmax(distance_logits, dim=-1)
        
        # Predict torsion angles
        angle_logits = self.angle_head(features)
        
        # Predict secondary structure
        ss_logits = self.ss_head(features)
        
        # Generate coordinates from distance geometry
        coordinates = self.coord_generator(
            distance_probs=distance_probs,
            angle_logits=angle_logits,
            sequence_length=seq_len
        )
        
        outputs = {
            'coordinates': coordinates,
            'distance_logits': distance_logits,
            'angle_logits': angle_logits,
            'secondary_structure_logits': ss_logits
        }
        
        if return_intermediates:
            outputs.update({
                'distance_probs': distance_probs,
                'recycling_features': recycling_features if prev_coords else None
            })
        
        return outputs

class CoordinateGenerator(nn.Module):
    """Generate 3D coordinates from distance and angle predictions"""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-4):
        super().__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def forward(
        self,
        distance_probs: torch.Tensor,
        angle_logits: torch.Tensor,
        sequence_length: int
    ) -> torch.Tensor:
        """
        Generate coordinates using distance geometry
        
        Args:
            distance_probs: Distance probability distributions
            angle_logits: Torsion angle predictions
            sequence_length: Length of sequence
            
        Returns:
            3D coordinates [batch, seq_len, 3]
        """
        batch_size = distance_probs.shape[0]
        
        # Convert distance probabilities to distance estimates
        distance_bins = torch.linspace(2.0, 20.0, distance_probs.shape[-1])
        distance_bins = distance_bins.to(distance_probs.device)
        
        distances = torch.sum(
            distance_probs * distance_bins.view(1, 1, 1, -1),
            dim=-1
        )
        
        # Extract torsion angles
        angles = self._extract_torsion_angles(angle_logits)
        
        # Initialize coordinates
        coords = self._initialize_coordinates(batch_size, sequence_length, distance_probs.device)
        
        # Iterative optimization
        for iteration in range(self.max_iterations):
            old_coords = coords.clone()
            
            # Update coordinates based on distance constraints
            coords = self._update_coordinates_distances(coords, distances)
            
            # Update coordinates based on angle constraints
            coords = self._update_coordinates_angles(coords, angles)
            
            # Check convergence
            diff = torch.norm(coords - old_coords, dim=-1).mean()
            if diff < self.tolerance:
                break
        
        return coords
    
    def _initialize_coordinates(
        self, 
        batch_size: int, 
        seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Initialize coordinates using ideal backbone geometry"""
        # Start with extended chain
        coords = torch.zeros(batch_size, seq_len, 3, device=device)
        
        # Place residues along x-axis with ideal bond lengths
        for i in range(1, seq_len):
            coords[:, i, 0] = coords[:, i-1, 0] + 3.8  # CA-CA distance
        
        return coords
    
    def _update_coordinates_distances(
        self,
        coords: torch.Tensor,
        target_distances: torch.Tensor
    ) -> torch.Tensor:
        """Update coordinates to satisfy distance constraints"""
        batch_size, seq_len, _ = coords.shape
        
        # Compute current distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [batch, seq_len, seq_len, 3]
        current_distances = torch.norm(diff, dim=-1)  # [batch, seq_len, seq_len]
        
        # Compute distance errors
        distance_errors = target_distances - current_distances
        
        # Apply corrections (simplified gradient descent)
        learning_rate = 0.01
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                error = distance_errors[:, i, j]
                direction = diff[:, i, j] / (current_distances[:, i, j].unsqueeze(-1) + 1e-8)
                
                correction = learning_rate * error.unsqueeze(-1) * direction
                coords[:, i] += correction
                coords[:, j] -= correction
        
        return coords
    
    def _update_coordinates_angles(
        self,
        coords: torch.Tensor,
        target_angles: torch.Tensor
    ) -> torch.Tensor:
        """Update coordinates to satisfy angle constraints"""
        # Implement torsion angle optimization
        # This is a simplified version - full implementation would use
        # proper torsion angle rotation matrices
        return coords
    
    def _extract_torsion_angles(self, angle_logits: torch.Tensor) -> torch.Tensor:
        """Extract torsion angles from logits"""
        # Convert logits to angle values
        angle_probs = F.softmax(angle_logits, dim=-1)
        angle_bins = torch.linspace(-np.pi, np.pi, angle_logits.shape[-1])
        angle_bins = angle_bins.to(angle_logits.device)
        
        angles = torch.sum(
            angle_probs * angle_bins.view(1, 1, 1, -1),
            dim=-1
        )
        
        return angles
```

### Data Processing Pipeline

#### Data Ingestion and Preprocessing

**Location**: `protein_sssl/data/`

```python
class ProteinDataProcessor:
    """
    Comprehensive data processing pipeline for protein sequences and structures
    
    Handles:
    - Sequence validation and cleaning
    - Structure parsing and quality assessment
    - Data augmentation for training
    - Batch optimization
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = ProteinTokenizer()
        self.quality_filter = QualityFilter(config.quality_thresholds)
        self.augmentor = DataAugmentor(config.augmentation_params)
    
    def process_sequence_data(
        self,
        sequences: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> ProcessedSequenceData:
        """Process raw protein sequences into model-ready format"""
        
        processed_sequences = []
        valid_indices = []
        
        for i, sequence in enumerate(sequences):
            try:
                # Validate and clean sequence
                cleaned_seq = self._clean_sequence(sequence)
                
                # Quality filtering
                if not self.quality_filter.is_valid_sequence(cleaned_seq):
                    continue
                
                # Tokenization
                tokens = self.tokenizer.encode(cleaned_seq)
                
                # Add to processed data
                processed_sequences.append({
                    'sequence': cleaned_seq,
                    'tokens': tokens,
                    'length': len(cleaned_seq),
                    'metadata': metadata[i] if metadata else {}
                })
                valid_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Failed to process sequence {i}: {e}")
                continue
        
        return ProcessedSequenceData(
            sequences=processed_sequences,
            valid_indices=valid_indices,
            statistics=self._compute_statistics(processed_sequences)
        )
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and validate protein sequence"""
        # Remove whitespace and convert to uppercase
        sequence = ''.join(sequence.split()).upper()
        
        # Replace ambiguous amino acids
        replacements = {
            'X': 'A',  # Unknown -> Alanine
            'B': 'N',  # Asn or Asp -> Asn
            'Z': 'Q',  # Gln or Glu -> Gln
            'J': 'L',  # Leu or Ile -> Leu
            'U': 'C',  # Selenocysteine -> Cysteine
            'O': 'K',  # Pyrrolysine -> Lysine
        }
        
        for old, new in replacements.items():
            sequence = sequence.replace(old, new)
        
        # Validate amino acid characters
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(sequence) - valid_chars
        
        if invalid_chars:
            raise ValueError(f"Invalid characters in sequence: {invalid_chars}")
        
        return sequence

class DynamicBatchSampler:
    """
    Dynamic batch sampler that groups sequences by length for efficiency
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        max_length: int = 1024,
        length_tolerance: float = 0.1,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.length_tolerance = length_tolerance
        self.shuffle = shuffle
        
        # Group sequences by length
        self.length_groups = self._create_length_groups()
    
    def _create_length_groups(self) -> Dict[int, List[int]]:
        """Group dataset indices by sequence length"""
        groups = {}
        
        for idx, item in enumerate(self.dataset):
            length = len(item['sequence'])
            
            # Find appropriate length bucket
            bucket = self._find_length_bucket(length, groups.keys())
            
            if bucket is None:
                # Create new bucket
                bucket = length
                groups[bucket] = []
            
            groups[bucket].append(idx)
        
        return groups
    
    def _find_length_bucket(self, length: int, existing_buckets: List[int]) -> Optional[int]:
        """Find appropriate length bucket for sequence"""
        for bucket in existing_buckets:
            if abs(length - bucket) / bucket <= self.length_tolerance:
                return bucket
        return None
    
    def __iter__(self):
        """Generate batches"""
        all_batches = []
        
        for bucket_length, indices in self.length_groups.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches from this length group
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                all_batches.append(batch_indices)
        
        if self.shuffle:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
```

### Training Infrastructure

#### Distributed Training Support

**Location**: `protein_sssl/training/`

```python
class DistributedSSLTrainer:
    """
    Distributed trainer for self-supervised learning with advanced features
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        accelerator: Optional[Accelerator] = None
    ):
        self.model = model
        self.config = config
        
        # Initialize accelerator for distributed training
        self.accelerator = accelerator or Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb else None
        )
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Prepare for distributed training
        (
            self.model,
            self.optimizer,
            self.scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with layer-wise learning rate decay"""
        
        # Layer-wise learning rate decay
        param_groups = []
        
        # Embedding parameters
        param_groups.append({
            'params': self.model.embeddings.parameters(),
            'lr': self.config.learning_rate * self.config.embedding_lr_factor,
            'weight_decay': self.config.weight_decay
        })
        
        # Encoder layers (with decay)
        n_layers = len(self.model.encoder.layers)
        for i, layer in enumerate(self.model.encoder.layers):
            # Earlier layers get lower learning rates
            decay_factor = self.config.layer_lr_decay ** (n_layers - i - 1)
            param_groups.append({
                'params': layer.parameters(),
                'lr': self.config.learning_rate * decay_factor,
                'weight_decay': self.config.weight_decay
            })
        
        # SSL heads
        param_groups.append({
            'params': self.model.ssl_heads.parameters(),
            'lr': self.config.learning_rate * self.config.head_lr_factor,
            'weight_decay': self.config.weight_decay
        })
        
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer == "adafactor":
            from transformers import Adafactor
            return Adafactor(
                param_groups,
                scale_parameter=True,
                relative_step_size=True,
                warmup_init=True
            )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps,
                eta_min=self.config.min_learning_rate
            )
        elif self.config.scheduler == "cosine_warmup":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps
            )
        elif self.config.scheduler == "polynomial":
            return get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps,
                power=self.config.polynomial_decay_power
            )
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        num_epochs: int = None
    ):
        """Main training loop"""
        
        num_epochs = num_epochs or self.config.num_epochs
        
        # Initialize wandb logging
        if self.accelerator.is_main_process and self.config.use_wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
                name=self.config.run_name
            )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(train_dataloader)
            
            # Evaluation phase
            if eval_dataloader is not None:
                eval_metrics = self._eval_epoch(eval_dataloader)
            else:
                eval_metrics = {}
            
            # Logging
            self._log_metrics(train_metrics, eval_metrics, epoch)
            
            # Checkpointing
            if self.accelerator.is_main_process:
                self._save_checkpoint(epoch, eval_metrics.get('loss', train_metrics['loss']))
            
            # Early stopping
            if self.config.early_stopping_patience:
                if self._should_stop_early(eval_metrics.get('loss', train_metrics['loss'])):
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed")
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(
            dataloader,
            disable=not self.accelerator.is_local_main_process,
            desc=f"Epoch {self.epoch}"
        )
        
        for step, batch in enumerate(progress_bar):
            # Forward pass
            with self.accelerator.accumulate(self.model):
                outputs = self._forward_step(batch)
                loss = outputs['total_loss']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Metrics tracking
            self._update_metrics(epoch_metrics, outputs)
            
            # Progress bar update
            if step % self.config.logging_steps == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                    'step': self.global_step
                })
            
            self.global_step += 1
            
            # Save checkpoint
            if (self.global_step % self.config.save_steps == 0 and 
                self.accelerator.is_main_process):
                self._save_checkpoint(self.epoch, loss.item())
        
        # Aggregate metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for one batch"""
        
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        ssl_labels = {
            key: batch[key] for key in batch
            if key.endswith('_labels')
        }
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ssl_labels=ssl_labels,
            return_dict=True
        )
        
        return outputs
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        
        if loss < self.best_loss:
            self.best_loss = loss
            
            checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-best"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            self.accelerator.save_state(checkpoint_dir / "accelerator_state")
            
            # Save additional metadata
            metadata = {
                'epoch': epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'config': self.config.__dict__
            }
            
            with open(checkpoint_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved best checkpoint at epoch {epoch} with loss {loss:.4f}")
```

## Development Environment Setup

### Complete Development Environment

```bash
#!/bin/bash
# setup_dev_environment.sh

set -e

echo "🚀 Setting up protein-sssl-operator development environment..."

# Check system requirements
check_requirements() {
    echo "📋 Checking system requirements..."
    
    # Check Python version
    python_version=$(python3 --version | cut -d' ' -f2)
    required_python="3.9.0"
    
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)"; then
        echo "❌ Python 3.9+ required, found $python_version"
        exit 1
    fi
    
    # Check CUDA availability (optional)
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        echo "⚠️  No NVIDIA GPU detected (CPU-only mode)"
    fi
    
    # Check available memory
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -lt 16 ]; then
        echo "⚠️  Warning: Less than 16GB RAM detected ($total_mem GB)"
    else
        echo "✅ Memory: ${total_mem}GB"
    fi
}

# Setup Python environment
setup_python_env() {
    echo "🐍 Setting up Python environment..."
    
    # Create virtual environment
    python3 -m venv protein_ssl_dev
    source protein_ssl_dev/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install core development dependencies
    pip install -e ".[dev,gpu,optimization]"
    
    # Install additional development tools
    pip install \
        jupyter jupyterlab \
        pytest pytest-cov pytest-mock pytest-asyncio \
        black isort flake8 mypy \
        pre-commit \
        ipdb pdbpp \
        memory-profiler line-profiler \
        sphinx sphinx-rtd-theme \
        wandb tensorboard
    
    echo "✅ Python environment setup complete"
}

# Setup pre-commit hooks
setup_pre_commit() {
    echo "🔧 Setting up pre-commit hooks..."
    
    # Install pre-commit hooks
    pre-commit install
    
    # Run initial check
    pre-commit run --all-files || true
    
    echo "✅ Pre-commit hooks installed"
}

# Setup development tools
setup_dev_tools() {
    echo "🛠️ Setting up development tools..."
    
    # Create development configuration
    mkdir -p configs/development
    cat > configs/development/config.yaml << 'EOF'
development:
  log_level: DEBUG
  model:
    cache_size: 100
    batch_size: 4
    gradient_checkpointing: true
  training:
    mixed_precision: true
    gradient_accumulation_steps: 4
  monitoring:
    wandb_enabled: true
    tensorboard_enabled: true
  testing:
    run_slow_tests: false
    use_mock_api: true
EOF
    
    # Create Jupyter kernel
    python -m ipykernel install --user --name protein_ssl_dev --display-name "Protein-SSL Dev"
    
    # Setup VSCode configuration
    mkdir -p .vscode
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./protein_ssl_dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "*.egg-info": true
    }
}
EOF
    
    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "ENVIRONMENT": "development"
            }
        },
        {
            "name": "Python: Train Model",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/train.py",
            "args": ["--config", "configs/development/config.yaml"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "ENVIRONMENT": "development"
            }
        },
        {
            "name": "Python: Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "ENVIRONMENT": "testing"
            }
        }
    ]
}
EOF
    
    echo "✅ Development tools configured"
}

# Setup documentation
setup_docs() {
    echo "📚 Setting up documentation..."
    
    # Create docs structure
    mkdir -p docs/{source,build}
    
    # Sphinx configuration
    cat > docs/source/conf.py << 'EOF'
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'protein-sssl-operator'
copyright = '2024, Terragon Labs'
author = 'Daniel Schmidt'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme'
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = True
EOF
    
    echo "✅ Documentation setup complete"
}

# Run setup
main() {
    check_requirements
    setup_python_env
    setup_pre_commit
    setup_dev_tools
    setup_docs
    
    echo ""
    echo "🎉 Development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment: source protein_ssl_dev/bin/activate"
    echo "2. Run tests: pytest tests/"
    echo "3. Start Jupyter: jupyter lab"
    echo "4. Open project in VSCode with Python extension"
    echo ""
    echo "Happy coding! 🧬"
}

main "$@"
```

### Docker Development Environment

```dockerfile
# Dockerfile.dev
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip3 install --no-cache-dir -r requirements-dev.txt

# Install development tools
RUN pip3 install \
    jupyter jupyterlab \
    pytest pytest-cov \
    black isort flake8 mypy \
    pre-commit \
    wandb tensorboard

# Setup Jupyter
RUN jupyter lab --generate-config
COPY jupyter_lab_config.py /root/.jupyter/

# Copy source code
COPY . .

# Install package in development mode
RUN pip3 install -e ".[dev,gpu]"

# Expose ports
EXPOSE 8888 6006 8000

# Development command
CMD ["bash"]
```

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  protein-ssl-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/workspace
      - protein-ssl-data:/data
      - protein-ssl-cache:/cache
      - jupyter-config:/root/.jupyter
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard
      - "8000:8000"  # API server
    environment:
      - CUDA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${WANDB_API_KEY}
      - ENVIRONMENT=development
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      bash -c "
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
        tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006 &
        tail -f /dev/null
      "

volumes:
  protein-ssl-data:
  protein-ssl-cache:
  jupyter-config:
```

## Code Contribution Guidelines

### Contribution Workflow

#### 1. Setup Development Environment

```bash
# Fork and clone repository
git clone https://github.com/your-username/protein-sssl-operator.git
cd protein-sssl-operator

# Add upstream remote
git remote add upstream https://github.com/terragonlabs/protein-sssl-operator.git

# Setup development environment
./scripts/setup_dev_environment.sh

# Activate environment
source protein_ssl_dev/bin/activate
```

#### 2. Development Process

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes with proper testing
# ... development work ...

# Run tests
pytest tests/ -v

# Run code quality checks
black .
isort .
flake8 .
mypy protein_sssl/

# Run pre-commit hooks
pre-commit run --all-files

# Commit changes
git add .
git commit -m "feat: add your feature description

- Detailed description of changes
- Any breaking changes
- Fixes #issue_number"

# Push to your fork
git push origin feature/your-feature-name
```

#### 3. Pull Request Guidelines

**PR Title Format:**
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scopes: api, models, training, data, utils, docs
```

**PR Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Added docstrings for new functions/classes
- [ ] Updated CHANGELOG.md if needed
```

### Code Style Guidelines

#### Python Style Standards

```python
# Code Style Example
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for model parameters.
    
    Attributes:
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

class ProteinEncoder(nn.Module):
    """Protein sequence encoder using transformer architecture.
    
    This class implements a transformer-based encoder for protein sequences
    with support for different attention mechanisms and positional encodings.
    
    Args:
        config: Model configuration object
        vocab_size: Size of amino acid vocabulary
        max_length: Maximum sequence length
        
    Example:
        >>> config = ModelConfig(d_model=512, n_layers=6)
        >>> encoder = ProteinEncoder(config, vocab_size=21, max_length=1024)
        >>> output = encoder(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        config: ModelConfig,
        vocab_size: int,
        max_length: int = 1024
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize layers
        self.embeddings = self._create_embeddings()
        self.encoder_layers = self._create_encoder_layers()
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized ProteinEncoder with {self._count_parameters():,} parameters")
    
    def _create_embeddings(self) -> nn.Module:
        """Create embedding layers."""
        return nn.Embedding(self.vocab_size, self.config.d_model)
    
    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create transformer encoder layers."""
        layers = []
        for i in range(self.config.n_layers):
            layer = TransformerEncoderLayer(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout
            )
            layers.append(layer)
        return nn.ModuleList(layers)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through the encoder.
        
        Args:
            input_ids: Tokenized input sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_all_layers: Whether to return outputs from all layers
            
        Returns:
            Encoded representations [batch_size, seq_len, d_model]
            or tuple of all layer outputs if return_all_layers=True
            
        Raises:
            ValueError: If input sequence length exceeds maximum length
        """
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.max_length:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum "
                f"supported length ({self.max_length})"
            )
        
        # Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Encoder layers
        all_layer_outputs = []
        
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
            
            if return_all_layers:
                all_layer_outputs.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if return_all_layers:
            all_layer_outputs.append(hidden_states)
            return tuple(all_layer_outputs)
        else:
            return hidden_states
```

#### Documentation Standards

```python
def predict_structure(
    self,
    sequence: str,
    model_config: Optional[Dict[str, Any]] = None,
    return_confidence: bool = True,
    num_recycles: int = 3
) -> StructurePrediction:
    """Predict protein structure from amino acid sequence.
    
    This method performs structure prediction using the trained neural
    operator model with optional iterative refinement through recycling.
    
    Args:
        sequence: Amino acid sequence using single-letter codes.
            Must contain only standard amino acids (A-Z except B, J, O, U, X, Z).
            Length should be between 10 and 5000 residues.
        model_config: Optional model configuration overrides.
            Can specify custom parameters like temperature, beam_size, etc.
        return_confidence: Whether to compute and return confidence scores.
            If True, returns per-residue confidence estimates and global scores.
        num_recycles: Number of recycling iterations for refinement.
            Higher values may improve accuracy but increase computation time.
            Recommended range: 1-5.
            
    Returns:
        StructurePrediction object containing:
            - coordinates: 3D atomic coordinates [N_atoms, 3]
            - confidence_scores: Per-residue confidence [seq_len] if requested
            - global_confidence: Overall prediction confidence score
            - processing_time: Time taken for prediction in seconds
            - metadata: Additional prediction metadata
            
    Raises:
        ValueError: If sequence contains invalid characters or is too long/short
        RuntimeError: If prediction fails due to computational issues
        MemoryError: If sequence is too long for available memory
        
    Example:
        >>> predictor = StructurePredictor(model_path="path/to/model.pt")
        >>> sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        >>> prediction = predictor.predict_structure(
        ...     sequence=sequence,
        ...     return_confidence=True,
        ...     num_recycles=3
        ... )
        >>> print(f"Confidence: {prediction.global_confidence:.3f}")
        >>> prediction.save_pdb("predicted_structure.pdb")
        
    Note:
        For sequences longer than 1000 residues, consider using batch
        prediction or domain-based prediction for better performance.
        
    See Also:
        predict_batch: For predicting multiple sequences
        predict_domains: For domain-specific prediction
    """
```

### Testing Guidelines

#### Test Structure and Organization

```python
# tests/unit/test_models.py
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from protein_sssl.models.ssl_encoder import SequenceStructureSSL
from protein_sssl.models.neural_operator import FourierNeuralOperator
from protein_sssl.config import ModelConfig

class TestSequenceStructureSSL:
    """Test suite for SequenceStructureSSL model."""
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=21,
            max_length=64,
            dropout=0.1
        )
    
    @pytest.fixture
    def ssl_model(self, model_config):
        """Create SSL model instance for testing."""
        return SequenceStructureSSL(
            config=model_config,
            ssl_objectives=["masked_modeling", "contrastive"]
        )
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        batch_size, seq_len = 2, 32
        return {
            'input_ids': torch.randint(0, 21, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'masked_lm_labels': torch.randint(0, 21, (batch_size, seq_len)),
            'contrastive_labels': torch.randint(0, 2, (batch_size,))
        }
    
    def test_model_initialization(self, ssl_model, model_config):
        """Test model initialization."""
        assert ssl_model.config.d_model == model_config.d_model
        assert ssl_model.config.n_layers == model_config.n_layers
        assert len(ssl_model.ssl_heads) == 2
        assert "masked_modeling" in ssl_model.ssl_heads
        assert "contrastive" in ssl_model.ssl_heads
    
    def test_forward_pass_without_labels(self, ssl_model, sample_batch):
        """Test forward pass without SSL labels."""
        input_ids = sample_batch['input_ids']
        attention_mask = sample_batch['attention_mask']
        
        with torch.no_grad():
            outputs = ssl_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        batch_size, seq_len = input_ids.shape
        expected_shape = (batch_size, seq_len, ssl_model.config.d_model)
        
        assert 'last_hidden_state' in outputs
        assert outputs['last_hidden_state'].shape == expected_shape
        assert outputs['total_loss'] is None
    
    def test_forward_pass_with_ssl_labels(self, ssl_model, sample_batch):
        """Test forward pass with SSL labels."""
        ssl_labels = {
            'masked_modeling': sample_batch['masked_lm_labels'],
            'contrastive': sample_batch['contrastive_labels']
        }
        
        outputs = ssl_model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            ssl_labels=ssl_labels,
            return_dict=True
        )
        
        assert 'ssl_outputs' in outputs
        assert 'total_loss' in outputs
        assert outputs['total_loss'] is not None
        assert outputs['total_loss'].requires_grad
        
        # Check SSL objective outputs
        ssl_outputs = outputs['ssl_outputs']
        assert 'masked_modeling_loss' in ssl_outputs
        assert 'contrastive_loss' in ssl_outputs
    
    def test_gradient_flow(self, ssl_model, sample_batch):
        """Test gradient computation and flow."""
        ssl_labels = {
            'masked_modeling': sample_batch['masked_lm_labels']
        }
        
        outputs = ssl_model(
            input_ids=sample_batch['input_ids'],
            ssl_labels=ssl_labels
        )
        
        loss = outputs['total_loss']
        loss.backward()
        
        # Check that gradients are computed
        for name, param in ssl_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_model_saving_and_loading(self, ssl_model, tmp_path):
        """Test model saving and loading."""
        save_path = tmp_path / "test_model.pt"
        
        # Save model
        torch.save(ssl_model.state_dict(), save_path)
        
        # Create new model and load state
        new_model = SequenceStructureSSL(
            config=ssl_model.config,
            ssl_objectives=["masked_modeling", "contrastive"]
        )
        new_model.load_state_dict(torch.load(save_path))
        
        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            ssl_model.named_parameters(),
            new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 16),
        (4, 32),
        (8, 64)
    ])
    def test_different_input_sizes(self, ssl_model, batch_size, seq_len):
        """Test model with different input sizes."""
        input_ids = torch.randint(0, 21, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = ssl_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        expected_shape = (batch_size, seq_len, ssl_model.config.d_model)
        assert outputs['last_hidden_state'].shape == expected_shape
    
    def test_invalid_input_handling(self, ssl_model):
        """Test handling of invalid inputs."""
        # Test empty input
        with pytest.raises(RuntimeError):
            ssl_model(input_ids=torch.empty(0, 0, dtype=torch.long))
        
        # Test mismatched shapes
        input_ids = torch.randint(0, 21, (2, 32))
        attention_mask = torch.ones(2, 16)  # Wrong length
        
        with pytest.raises(RuntimeError):
            ssl_model(input_ids=input_ids, attention_mask=attention_mask)

# Integration tests
class TestEndToEndPipeline:
    """Integration tests for complete prediction pipeline."""
    
    @pytest.fixture
    def temp_model_dir(self, tmp_path):
        """Create temporary directory for model files."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        return model_dir
    
    @pytest.mark.integration
    def test_complete_prediction_pipeline(self, temp_model_dir):
        """Test complete pipeline from sequence to structure."""
        from protein_sssl import StructurePredictor
        
        # Create minimal test model
        config = ModelConfig(d_model=64, n_layers=1, n_heads=2)
        model = SequenceStructureSSL(config)
        
        # Save test model
        model_path = temp_model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Test prediction
        predictor = StructurePredictor(model_path=str(model_path))
        
        test_sequence = "MKFLKFSLLTAV"
        prediction = predictor.predict_structure(
            sequence=test_sequence,
            return_confidence=True
        )
        
        # Verify outputs
        assert prediction.sequence == test_sequence
        assert len(prediction.coordinates) == len(test_sequence)
        assert 0 <= prediction.confidence <= 1
        assert prediction.processing_time > 0
    
    @pytest.mark.slow
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        import time
        
        # Test sequence processing speed
        sequences = ["MKFL" * 50] * 10  # 10 sequences of 200 residues
        
        start_time = time.time()
        # Process sequences
        processing_time = time.time() - start_time
        
        # Should process 10 sequences in reasonable time
        assert processing_time < 60, f"Processing took {processing_time:.2f}s, expected < 60s"

# Property-based tests
class TestModelProperties:
    """Property-based tests using hypothesis."""
    
    @pytest.mark.hypothesis
    def test_model_deterministic_given_seed(self):
        """Test that model outputs are deterministic given same seed."""
        from hypothesis import given, strategies as st
        
        @given(
            batch_size=st.integers(min_value=1, max_value=4),
            seq_len=st.integers(min_value=8, max_value=32),
            seed=st.integers(min_value=0, max_value=1000)
        )
        def test_deterministic(batch_size, seq_len, seed):
            torch.manual_seed(seed)
            
            config = ModelConfig(d_model=32, n_layers=1, n_heads=2)
            model1 = SequenceStructureSSL(config)
            
            torch.manual_seed(seed)
            model2 = SequenceStructureSSL(config)
            
            input_ids = torch.randint(0, 21, (batch_size, seq_len))
            
            with torch.no_grad():
                output1 = model1(input_ids)
                output2 = model2(input_ids)
            
            assert torch.allclose(
                output1['last_hidden_state'],
                output2['last_hidden_state'],
                atol=1e-6
            )
        
        test_deterministic()
```

### Performance Testing

```python
# tests/performance/test_benchmarks.py
import pytest
import torch
import time
import psutil
import numpy as np
from contextlib import contextmanager
from typing import Dict, Any

class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    @staticmethod
    @contextmanager
    def measure_time():
        """Context manager to measure execution time."""
        start_time = time.time()
        yield
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
    
    @staticmethod
    @contextmanager
    def measure_memory():
        """Context manager to measure memory usage."""
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        yield
        
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - start_memory
        print(f"Memory usage: {end_memory:.1f} MB (Δ {memory_delta:+.1f} MB)")
    
    @staticmethod
    def measure_gpu_memory():
        """Measure GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return {'allocated': allocated, 'cached': cached}
        return {'allocated': 0, 'cached': 0}

@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for model components."""
    
    def test_ssl_encoder_throughput(self):
        """Test SSL encoder throughput."""
        config = ModelConfig(d_model=1280, n_layers=33, n_heads=20)
        model = SequenceStructureSSL(config)
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        seq_len = 512
        
        results = {}
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 21, (batch_size, seq_len))
            
            # Warmup
            with torch.no_grad():
                _ = model(input_ids)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # Average over 10 runs
                    _ = model(input_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            throughput = batch_size / avg_time  # sequences per second
            
            results[batch_size] = {
                'avg_time': avg_time,
                'throughput': throughput
            }
            
            print(f"Batch size {batch_size}: {throughput:.2f} sequences/sec")
        
        # Performance assertions
        assert results[16]['throughput'] > results[1]['throughput'], \
            "Larger batch size should have higher throughput"
    
    def test_memory_scaling(self):
        """Test memory usage scaling with sequence length."""
        config = ModelConfig(d_model=512, n_layers=6, n_heads=8)
        model = SequenceStructureSSL(config)
        model.eval()
        
        sequence_lengths = [128, 256, 512, 1024]
        batch_size = 4
        
        memory_usage = {}
        
        for seq_len in sequence_lengths:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            input_ids = torch.randint(0, 21, (batch_size, seq_len))
            
            start_memory = PerformanceBenchmark.measure_gpu_memory()
            
            with torch.no_grad():
                _ = model(input_ids)
            
            end_memory = PerformanceBenchmark.measure_gpu_memory()
            
            memory_delta = end_memory['allocated'] - start_memory['allocated']
            memory_usage[seq_len] = memory_delta
            
            print(f"Sequence length {seq_len}: {memory_delta:.3f} GB")
        
        # Memory should scale roughly quadratically with sequence length
        # (due to attention mechanism)
        ratio_512_256 = memory_usage[512] / memory_usage[256]
        assert 3.5 < ratio_512_256 < 4.5, \
            f"Memory scaling unexpected: {ratio_512_256:.2f}"
    
    @pytest.mark.gpu
    def test_gpu_utilization(self):
        """Test GPU utilization during training."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        config = ModelConfig(d_model=1024, n_layers=12, n_heads=16)
        model = SequenceStructureSSL(config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        batch_size = 8
        seq_len = 512
        
        # Simulate training step
        for _ in range(5):  # Warmup
            input_ids = torch.randint(0, 21, (batch_size, seq_len)).cuda()
            ssl_labels = {'masked_modeling': input_ids.clone()}
            
            outputs = model(input_ids, ssl_labels=ssl_labels)
            loss = outputs['total_loss']
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Measure actual training step
        torch.cuda.synchronize()
        start_time = time.time()
        
        input_ids = torch.randint(0, 21, (batch_size, seq_len)).cuda()
        ssl_labels = {'masked_modeling': input_ids.clone()}
        
        outputs = model(input_ids, ssl_labels=ssl_labels)
        loss = outputs['total_loss']
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        step_time = end_time - start_time
        print(f"Training step time: {step_time:.4f} seconds")
        
        # Assert reasonable training speed
        assert step_time < 1.0, f"Training step too slow: {step_time:.4f}s"

@pytest.mark.benchmark
def test_end_to_end_prediction_benchmark():
    """Benchmark complete prediction pipeline."""
    from protein_sssl import StructurePredictor
    
    # Create test predictor
    predictor = StructurePredictor(model_path="path/to/test/model.pt")
    
    # Test sequences of different lengths
    sequences = {
        "short": "MKFLKFSLLTAV",  # 12 residues
        "medium": "MKFLKFSLLTAV" * 10,  # 120 residues
        "long": "MKFLKFSLLTAV" * 25,   # 300 residues
    }
    
    benchmark_results = {}
    
    for name, sequence in sequences.items():
        with PerformanceBenchmark.measure_time():
            prediction = predictor.predict_structure(
                sequence=sequence,
                return_confidence=True
            )
        
        benchmark_results[name] = {
            'length': len(sequence),
            'processing_time': prediction.processing_time,
            'confidence': prediction.confidence
        }
    
    # Performance assertions
    assert benchmark_results["short"]["processing_time"] < 10, \
        "Short sequence prediction should be fast"
    
    assert benchmark_results["medium"]["processing_time"] < 60, \
        "Medium sequence prediction should complete in reasonable time"
    
    # Confidence should be reasonable
    for result in benchmark_results.values():
        assert 0.5 < result["confidence"] < 1.0, \
            f"Confidence {result['confidence']} seems unreasonable"
```

## Release Management

### Release Process

#### 1. Version Management

```python
# protein_sssl/_version.py
__version__ = "1.0.0"

def get_version():
    """Get version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    import platform
    import torch
    
    return {
        'version': __version__,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
```

#### 2. Release Automation

```bash
#!/bin/bash
# scripts/release.sh

set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.0"
    exit 1
fi

echo "🚀 Starting release process for version $VERSION"

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ Invalid version format. Use semantic versioning (e.g., 1.0.0)"
    exit 1
fi

# Check if on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Must be on main branch to release"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Uncommitted changes detected"
    exit 1
fi

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

# Update version
echo "📝 Updating version to $VERSION"
sed -i "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" protein_sssl/_version.py

# Update CHANGELOG
echo "📋 Updating CHANGELOG..."
DATE=$(date +%Y-%m-%d)
cat > temp_changelog.md << EOF
# Changelog

## [$VERSION] - $DATE

### Added
- New features added in this release

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Security
- Security improvements

$(tail -n +3 CHANGELOG.md)
EOF

mv temp_changelog.md CHANGELOG.md

# Build package
echo "📦 Building package..."
python -m build

# Test package installation
echo "🔍 Testing package installation..."
python -m pip install dist/protein_sssl_operator-$VERSION-py3-none-any.whl --force-reinstall

# Run quick smoke test
python -c "
import protein_sssl
print(f'Version: {protein_sssl.__version__}')
print('Package installed successfully')
"

# Create git tag
echo "🏷️ Creating git tag..."
git add protein_sssl/_version.py CHANGELOG.md
git commit -m "Release version $VERSION"
git tag -a "v$VERSION" -m "Release version $VERSION"

# Push changes
echo "📤 Pushing changes..."
git push origin main
git push origin "v$VERSION"

# Create GitHub release
echo "🎉 Creating GitHub release..."
gh release create "v$VERSION" \
    --title "Release $VERSION" \
    --notes-file <(grep -A 20 "## \[$VERSION\]" CHANGELOG.md | tail -n +3 | head -n -1) \
    dist/*

echo "✅ Release $VERSION completed successfully!"
echo "📦 Package uploaded to PyPI"
echo "🎉 GitHub release created"
```

#### 3. Continuous Integration

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=protein_sssl
    
    - name: Build package
      run: |
        python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        twine upload dist/*
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

---

This comprehensive developer guide provides everything needed for effective contribution to and maintenance of the protein-sssl-operator project. Regular updates ensure the documentation stays current with project evolution and best practices.