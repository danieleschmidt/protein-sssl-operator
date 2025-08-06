import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import random
from pathlib import Path
import gzip
from Bio import SeqIO
import pickle

class ProteinDataset(Dataset):
    """Dataset for protein sequences with SSL objectives"""
    
    def __init__(
        self,
        sequences: List[str],
        max_length: int = 1024,
        mask_prob: float = 0.15,
        clustering_threshold: float = 0.5,
        ssl_objectives: List[str] = None
    ):
        if ssl_objectives is None:
            ssl_objectives = ["masked_modeling", "contrastive"]
            
        self.sequences = sequences
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.clustering_threshold = clustering_threshold
        self.ssl_objectives = ssl_objectives
        
        # Amino acid vocabulary
        self.aa_to_id = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
            'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20  # X for unknown/mask
        }
        self.id_to_aa = {v: k for k, v in self.aa_to_id.items()}
        self.vocab_size = len(self.aa_to_id)
        
        # Special tokens
        self.mask_token_id = 20
        self.pad_token_id = 20
        
        # Filter and process sequences
        self.processed_sequences = self._process_sequences()
        
    def _process_sequences(self) -> List[str]:
        """Clean and filter sequences"""
        processed = []
        
        for seq in self.sequences:
            # Clean sequence
            seq = seq.upper().replace('U', 'C')  # Selenocysteine -> Cysteine
            seq = ''.join([aa if aa in self.aa_to_id else 'X' for aa in seq])
            
            # Filter by length
            if 30 <= len(seq) <= self.max_length:
                processed.append(seq)
                
        return processed
        
    @classmethod
    def from_fasta(
        cls,
        fasta_path: str,
        max_length: int = 1024,
        clustering_threshold: float = 0.5,
        max_sequences: Optional[int] = None
    ):
        """Load dataset from FASTA file"""
        sequences = []
        
        # Handle gzipped files
        if fasta_path.endswith('.gz'):
            file_handle = gzip.open(fasta_path, 'rt')
        else:
            file_handle = open(fasta_path, 'r')
            
        try:
            count = 0
            for record in SeqIO.parse(file_handle, 'fasta'):
                sequences.append(str(record.seq))
                count += 1
                
                if max_sequences and count >= max_sequences:
                    break
                    
        finally:
            file_handle.close()
            
        return cls(
            sequences=sequences,
            max_length=max_length,
            clustering_threshold=clustering_threshold
        )
        
    @classmethod  
    def from_text_file(cls, text_path: str, **kwargs):
        """Load sequences from text file (one per line)"""
        with open(text_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
            
        return cls(sequences=sequences, **kwargs)
        
    def __len__(self) -> int:
        return len(self.processed_sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.processed_sequences[idx]
        
        # Tokenize
        tokens = self._tokenize(sequence)
        
        # Create SSL training examples
        ssl_data = {}
        
        if "masked_modeling" in self.ssl_objectives:
            masked_tokens, labels = self._create_masked_lm_example(tokens)
            ssl_data.update({
                'input_ids': torch.tensor(masked_tokens, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor([1] * len(masked_tokens), dtype=torch.long)
            })
            
        if "contrastive" in self.ssl_objectives:
            # For contrastive learning, we'll create augmented versions
            augmented = self._augment_sequence(sequence)
            aug_tokens = self._tokenize(augmented)
            ssl_data['contrastive_input'] = torch.tensor(aug_tokens, dtype=torch.long)
            ssl_data['contrastive_mask'] = torch.tensor([1] * len(aug_tokens), dtype=torch.long)
            
        if "distance_prediction" in self.ssl_objectives:
            # Create synthetic distance targets (simplified)
            dist_targets = self._create_distance_targets(sequence)
            ssl_data['distance_targets'] = torch.tensor(dist_targets, dtype=torch.float)
            
        # Pad sequences to max_length
        ssl_data = self._pad_sequences(ssl_data)
        
        return ssl_data
        
    def _tokenize(self, sequence: str) -> List[int]:
        """Convert sequence to token IDs"""
        return [self.aa_to_id.get(aa, self.mask_token_id) for aa in sequence]
        
    def _create_masked_lm_example(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """Create masked language modeling example"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 ignores loss
        
        for i in range(len(tokens)):
            if random.random() < self.mask_prob:
                labels[i] = tokens[i]  # Original token
                
                # 80% mask, 10% random, 10% unchanged
                rand = random.random()
                if rand < 0.8:
                    masked_tokens[i] = self.mask_token_id
                elif rand < 0.9:
                    masked_tokens[i] = random.randint(0, self.vocab_size - 2)  # Not mask token
                    
        return masked_tokens, labels
        
    def _augment_sequence(self, sequence: str, prob: float = 0.1) -> str:
        """Create augmented version of sequence for contrastive learning"""
        augmented = []
        
        for aa in sequence:
            if random.random() < prob:
                # Simple amino acid substitution with biochemically similar amino acids
                similar_aa = {
                    'A': ['G', 'S'], 'R': ['K', 'H'], 'N': ['D', 'Q'], 'D': ['N', 'E'],
                    'C': ['S'], 'Q': ['N', 'E'], 'E': ['D', 'Q'], 'G': ['A', 'S'],
                    'H': ['R', 'K'], 'I': ['L', 'V'], 'L': ['I', 'V'], 'K': ['R', 'H'],
                    'M': ['L', 'I'], 'F': ['Y', 'W'], 'P': ['A'], 'S': ['A', 'T'],
                    'T': ['S'], 'W': ['F', 'Y'], 'Y': ['F', 'W'], 'V': ['I', 'L']
                }
                
                if aa in similar_aa:
                    augmented.append(random.choice(similar_aa[aa]))
                else:
                    augmented.append(aa)
            else:
                augmented.append(aa)
                
        return ''.join(augmented)
        
    def _create_distance_targets(self, sequence: str) -> np.ndarray:
        """Create synthetic distance targets (simplified for SSL)"""
        seq_len = len(sequence)
        
        # Simple heuristic distance matrix based on sequence separation
        distances = np.zeros((seq_len, seq_len))
        
        for i in range(seq_len):
            for j in range(seq_len):
                sep = abs(i - j)
                if sep == 0:
                    distances[i, j] = 0.0
                elif sep <= 3:
                    distances[i, j] = 3.8 + sep * 0.5  # Approximate backbone distances
                else:
                    # Random distances for distant pairs (SSL pre-training)
                    distances[i, j] = random.uniform(8.0, 22.0)
                    
        return distances
        
    def _pad_sequences(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pad sequences to max_length"""
        padded_data = {}
        
        for key, tensor in data.items():
            if tensor.dim() == 1:  # Sequence data
                current_len = len(tensor)
                if current_len < self.max_length:
                    # Pad with pad_token_id or 0
                    pad_value = self.pad_token_id if 'input' in key or 'label' in key else 0
                    pad_size = self.max_length - current_len
                    
                    if key == 'labels':
                        # Use -100 for labels (ignore in loss)
                        padding = torch.full((pad_size,), -100, dtype=tensor.dtype)
                    elif key.endswith('_mask'):
                        # Use 0 for attention masks
                        padding = torch.zeros(pad_size, dtype=tensor.dtype)
                    else:
                        padding = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                        
                    padded_tensor = torch.cat([tensor, padding])
                else:
                    # Truncate if too long
                    padded_tensor = tensor[:self.max_length]
                    
            elif tensor.dim() == 2:  # Distance matrices
                current_len = tensor.shape[0]
                if current_len < self.max_length:
                    # Pad distance matrix
                    pad_size = self.max_length - current_len
                    padded_tensor = torch.zeros((self.max_length, self.max_length), dtype=tensor.dtype)
                    padded_tensor[:current_len, :current_len] = tensor
                else:
                    padded_tensor = tensor[:self.max_length, :self.max_length]
            else:
                padded_tensor = tensor
                
            padded_data[key] = padded_tensor
            
        return padded_data
        
    def save_cache(self, cache_path: str):
        """Save processed dataset to cache"""
        cache_data = {
            'processed_sequences': self.processed_sequences,
            'max_length': self.max_length,
            'mask_prob': self.mask_prob,
            'ssl_objectives': self.ssl_objectives,
            'aa_to_id': self.aa_to_id
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
            
    @classmethod
    def load_cache(cls, cache_path: str):
        """Load dataset from cache"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            
        dataset = cls.__new__(cls)
        dataset.processed_sequences = cache_data['processed_sequences']
        dataset.max_length = cache_data['max_length']
        dataset.mask_prob = cache_data['mask_prob']
        dataset.ssl_objectives = cache_data['ssl_objectives']
        dataset.aa_to_id = cache_data['aa_to_id']
        dataset.id_to_aa = {v: k for k, v in dataset.aa_to_id.items()}
        dataset.vocab_size = len(dataset.aa_to_id)
        dataset.mask_token_id = 20
        dataset.pad_token_id = 20
        
        return dataset

class ProteinDataLoader:
    """Custom data loader with dynamic batching"""
    
    def __init__(
        self,
        dataset: ProteinDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        dynamic_batching: bool = True,
        max_tokens: int = 32000
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dynamic_batching = dynamic_batching
        self.max_tokens = max_tokens
        
    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader"""
        
        if self.dynamic_batching:
            # Group sequences by similar length for efficient batching
            return self._create_dynamic_dataloader()
        else:
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn
            )
            
    def _create_dynamic_dataloader(self) -> DataLoader:
        """Create dataloader with dynamic batching based on sequence length"""
        # Sort by length for dynamic batching
        indices_and_lengths = [
            (i, len(self.dataset.processed_sequences[i])) 
            for i in range(len(self.dataset))
        ]
        indices_and_lengths.sort(key=lambda x: x[1])
        
        # Create batches
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, seq_len in indices_and_lengths:
            batch_tokens = (len(current_batch) + 1) * max(seq_len, 
                           max([self.dataset.processed_sequences[i] 
                               for i, _ in current_batch] + [0], key=len, default=0))
                           
            if batch_tokens > self.max_tokens and current_batch:
                batches.append([i for i, _ in current_batch])
                current_batch = [(idx, seq_len)]
            else:
                current_batch.append((idx, seq_len))
                
        if current_batch:
            batches.append([i for i, _ in current_batch])
            
        if self.shuffle:
            random.shuffle(batches)
            
        # Create dataset with pre-computed batches
        class BatchDataset(Dataset):
            def __init__(self, original_dataset, batches):
                self.original_dataset = original_dataset
                self.batches = batches
                
            def __len__(self):
                return len(self.batches)
                
            def __getitem__(self, idx):
                batch_indices = self.batches[idx]
                return [self.original_dataset[i] for i in batch_indices]
                
        batch_dataset = BatchDataset(self.dataset, batches)
        
        return DataLoader(
            batch_dataset,
            batch_size=1,  # Each item is already a batch
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0]  # Just return the batch
        )
        
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        if not batch:
            return {}
            
        # Get all keys
        keys = batch[0].keys()
        
        # Stack tensors
        collated = {}
        for key in keys:
            tensors = [item[key] for item in batch]
            collated[key] = torch.stack(tensors, dim=0)
            
        return collated