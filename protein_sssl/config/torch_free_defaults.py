"""
Torch-free default configurations for Generation 1 testing
"""

def get_default_ssl_config():
    """Get default SSL configuration without torch dependencies"""
    return {
        'model': {
            'd_model': 1280,
            'n_layers': 33,
            'n_heads': 20,
            'vocab_size': 21,
            'max_length': 1024,
            'ssl_objectives': ['masked_modeling', 'contrastive', 'distance_prediction'],
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 10,
            'warmup_steps': 10000,
            'gradient_checkpointing': True,
            'mixed_precision': True
        },
        'data': {
            'max_sequence_length': 1024,
            'clustering_threshold': 0.5,
            'augmentation': {
                'mask_probability': 0.15,
                'random_token_prob': 0.1,
                'unchanged_prob': 0.1
            }
        },
        'training_type': 'ssl'
    }

def get_default_folding_config():
    """Get default folding configuration without torch dependencies"""
    return {
        'model': {
            'd_model': 1280,
            'operator_layers': 12,
            'fourier_modes': 16,
            'attention_type': 'efficient',
            'uncertainty_method': 'ensemble'
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'epochs': 50,
            'validation_split': 0.1,
            'loss_weights': {
                'distance_map': 1.0,
                'torsion_angles': 0.5,
                'secondary_structure': 0.3,
                'uncertainty': 0.2
            }
        },
        'data': {
            'resolution_cutoff': 3.0,
            'remove_redundancy': True,
            'structure_format': 'pdb'
        },
        'training_type': 'folding'
    }