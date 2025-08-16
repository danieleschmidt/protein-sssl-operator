"""
Mock PyTorch for Generation 1 testing when torch is not available
"""

class MockTensor:
    def __init__(self, data=None, dtype=None):
        self.data = data or []
        self.dtype = dtype
        self.shape = (len(data) if isinstance(data, list) else 0,)
    
    def __getitem__(self, key):
        return self.data[key] if hasattr(self.data, '__getitem__') else MockTensor()
    
    def mean(self, dim=None):
        return MockTensor([0.5])
    
    def expand(self, *args):
        return MockTensor()
    
    def unsqueeze(self, dim):
        return MockTensor()

class MockModule:
    def __init__(self, *args, **kwargs):
        self.training = True
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, x):
        return x
        
    def state_dict(self):
        return {}
        
    def load_state_dict(self, state_dict):
        pass
        
    def modules(self):
        return [self]
        
    def parameters(self):
        return []

class MockOptim:
    def __init__(self, *args, **kwargs):
        pass
    def step(self):
        pass
    def zero_grad(self):
        pass

def create_torch_mock():
    torch = type('torch', (), {})()
    
    # Core functions
    torch.tensor = lambda data, dtype=None: MockTensor(data, dtype)
    torch.ones_like = lambda x: MockTensor([1] * (x.shape[0] if hasattr(x, 'shape') else 1))
    torch.cat = lambda tensors, dim=None: MockTensor()
    torch.arange = lambda *args, **kwargs: MockTensor(list(range(args[0] if args else 10)))
    torch.einsum = lambda *args, **kwargs: MockTensor()
    torch.save = lambda obj, path: None
    torch.load = lambda path, map_location=None: {'model_state_dict': {}, 'config': {}}
    
    # Data types
    torch.long = 'long'
    torch.float = 'float'
    torch.Tensor = MockTensor
    
    # NN module
    torch.nn = type('nn', (), {})()
    torch.nn.Module = MockModule
    torch.nn.Embedding = MockModule
    torch.nn.Linear = MockModule
    torch.nn.LayerNorm = MockModule
    torch.nn.TransformerEncoder = MockModule
    torch.nn.TransformerEncoderLayer = MockModule
    torch.nn.Sequential = MockModule
    torch.nn.ReLU = MockModule
    torch.nn.ModuleDict = dict
    torch.nn.functional = type('functional', (), {})()
    torch.nn.functional.cross_entropy = lambda *args, **kwargs: MockTensor([1.0])
    
    # Optimizers
    torch.optim = type('optim', (), {})()
    torch.optim.AdamW = MockOptim
    torch.optim.Adam = MockOptim
    
    # Init functions
    torch.nn.init = type('init', (), {})()
    torch.nn.init.normal_ = lambda tensor, mean=0.0, std=1.0: None
    torch.nn.init.zeros_ = lambda tensor: None
    
    # CUDA
    torch.cuda = type('cuda', (), {})()
    torch.cuda.is_available = lambda: False
    
    return torch

def get_mock_nn():
    """Get mock nn module"""
    torch = create_torch_mock()
    return torch.nn