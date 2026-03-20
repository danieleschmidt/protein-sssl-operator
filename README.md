# Protein SSSL Operator

Self-supervised structure-sequence learning with neural operators for protein folding.

## Features
- **SequenceEncoder**: Amino acid one-hot + sinusoidal positional encoding
- **StructureEncoder**: RBF-expanded distance matrix from 3D coordinates
- **ContrastiveLoss**: NT-Xent loss for sequence-structure alignment
- **SSLTrainer**: Self-supervised pretraining loop
- **LinearProbe**: Fine-tuning for secondary structure prediction

## Install

```bash
pip install numpy
```

## Usage

```python
from sssl.sequence_encoder import SequenceEncoder
from sssl.structure_encoder import StructureEncoder
import numpy as np

enc = SequenceEncoder()
z = enc.encode("ACDEFGHIKLMNPQRSTVWY")
print(z.shape)  # (20, 85)
```

### Full pipeline

```python
import numpy as np
from sssl.sequence_encoder import SequenceEncoder
from sssl.structure_encoder import StructureEncoder
from sssl.contrastive_loss import ContrastiveLoss
from sssl.trainer import SSLTrainer
from sssl.linear_probe import LinearProbe

# Encoders
seq_enc    = SequenceEncoder()
struct_enc = StructureEncoder()
loss_fn    = ContrastiveLoss(temperature=0.07)

# Pretraining
trainer = SSLTrainer(seq_enc, struct_enc, loss_fn)
sequences   = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVK"]
coords_list = [np.random.rand(20, 3) * 10, np.random.rand(16, 3) * 10]
losses = trainer.train(sequences, coords_list, n_epochs=10)
print(losses)

# Linear probe
features = np.random.rand(50, 85).astype("float32")
labels   = np.random.randint(0, 3, 50)
probe    = LinearProbe(input_dim=85)
probe.train(features, labels, n_epochs=50)
probs = probe.predict(features)
print(probs.shape)  # (50, 3)
```

## Run Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

## Architecture

| Component | Input | Output |
|---|---|---|
| SequenceEncoder | str (amino acids) | (L, 85) |
| StructureEncoder.from_coords | (n, 3) coords | (n, n, 16) |
| StructureEncoder.flatten | (n, n, 16) | (n, 16) |
| ContrastiveLoss | (B, D), (B, D) | scalar |
| LinearProbe.predict | (n, D) | (n, 3) softmax |
