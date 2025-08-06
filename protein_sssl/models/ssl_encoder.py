import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Union, Tuple
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = seq_len or x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        return cos, sin

class SequenceStructureSSL(nn.Module):
    def __init__(
        self,
        d_model: int = 1280,
        n_layers: int = 33,
        n_heads: int = 20,
        vocab_size: int = 21,  # 20 amino acids + padding
        max_length: int = 1024,
        ssl_objectives: List[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if ssl_objectives is None:
            ssl_objectives = ["masked_modeling", "contrastive", "distance_prediction"]
            
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.ssl_objectives = ssl_objectives
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = RotaryPositionalEmbedding(d_model // n_heads)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # SSL heads
        self.heads = nn.ModuleDict()
        if "masked_modeling" in ssl_objectives:
            self.heads["masked_lm"] = nn.Linear(d_model, vocab_size)
            
        if "contrastive" in ssl_objectives:
            self.heads["contrastive"] = nn.Linear(d_model, d_model)
            
        if "distance_prediction" in ssl_objectives:
            self.heads["distance"] = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)  # Distance bins
            )
            
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Apply attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Create causal mask for transformer
        src_key_padding_mask = (attention_mask == 0)
        
        # Forward pass
        hidden_states = self.transformer(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        hidden_states = self.layer_norm(hidden_states)
        
        outputs = {"last_hidden_state": hidden_states}
        
        # SSL objectives
        if "masked_modeling" in self.ssl_objectives and "masked_lm" in self.heads:
            outputs["masked_lm_logits"] = self.heads["masked_lm"](hidden_states)
            
        if "contrastive" in self.ssl_objectives and "contrastive" in self.heads:
            # Pool sequence representation
            pooled = hidden_states.mean(dim=1)  # Simple mean pooling
            outputs["contrastive_features"] = self.heads["contrastive"](pooled)
            
        if "distance_prediction" in self.ssl_objectives and "distance" in self.heads:
            # Pairwise distance prediction
            h_i = hidden_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
            h_j = hidden_states.unsqueeze(1).expand(-1, seq_len, -1, -1)
            pairwise = torch.cat([h_i, h_j], dim=-1)
            outputs["distance_logits"] = self.heads["distance"](pairwise)
            
        if not return_dict:
            return hidden_states
            
        return outputs
    
    def get_sequence_embeddings(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, return_dict=True)
            return outputs["last_hidden_state"]
            
    def save_pretrained(self, save_directory: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'ssl_objectives': self.ssl_objectives
            }
        }, f"{save_directory}/pytorch_model.bin")
        
    @classmethod
    def from_pretrained(cls, model_path: str):
        checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
        
class ProteinTokenizer:
    """Simple tokenizer for protein sequences"""
    
    def __init__(self):
        self.aa_to_id = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
            'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20  # X for unknown/padding
        }
        self.id_to_aa = {v: k for k, v in self.aa_to_id.items()}
        self.vocab_size = len(self.aa_to_id)
        
    def encode(self, sequence: str, max_length: int = 1024) -> Dict[str, torch.Tensor]:
        sequence = sequence.upper().replace('U', 'C')  # Replace selenocysteine
        
        # Convert to IDs
        ids = []
        for aa in sequence:
            ids.append(self.aa_to_id.get(aa, 20))  # Unknown -> X (20)
            
        # Truncate or pad
        if len(ids) > max_length:
            ids = ids[:max_length]
        
        attention_mask = [1] * len(ids)
        
        # Pad if necessary
        while len(ids) < max_length:
            ids.append(20)  # Padding token
            attention_mask.append(0)
            
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
    def decode(self, ids: List[int]) -> str:
        return ''.join([self.id_to_aa.get(id, 'X') for id in ids])