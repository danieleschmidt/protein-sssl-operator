# Contributing to Protein-SSSL-Operator

Thank you for your interest in contributing to the Protein-SSSL-Operator project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/danieleschmidt/protein-sssl-operator.git
   cd protein-sssl-operator
   ```

2. **Create a development environment**:
   ```bash
   conda env create -f environment.yml
   conda activate protein-sssl
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines below

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Run linting and formatting**:
   ```bash
   black protein_sssl tests scripts
   isort protein_sssl tests scripts
   flake8 protein_sssl tests scripts
   mypy protein_sssl
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

6. **Push and create a pull request**

## 📋 Contribution Guidelines

### Code Style

We use several tools to maintain consistent code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

Configuration is in `pyproject.toml`. Pre-commit hooks will automatically run these tools.

### Commit Message Convention

We follow conventional commits:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes
- `ci:` - CI/CD changes

Example:
```
feat(models): add attention mechanism to neural operator

- Implement multi-head attention in FourierLayer
- Add attention weights visualization
- Update documentation with attention parameters
```

### Testing

We maintain high test coverage. Please include tests for new features:

#### Unit Tests
```python
def test_new_feature():
    # Test your new functionality
    result = your_function(input_data)
    assert result == expected_output
```

#### Integration Tests
```python
@pytest.mark.integration
def test_end_to_end_pipeline():
    # Test complete workflows
    pass
```

#### Performance Tests
```python
@pytest.mark.slow
def test_performance_regression():
    # Benchmark critical paths
    pass
```

Run tests with:
```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# With coverage
pytest --cov=protein_sssl --cov-report=html
```

### Documentation

#### Code Documentation
- All public functions/classes must have docstrings
- Use Google-style docstrings:

```python
def predict_structure(
    sequence: str,
    model: nn.Module,
    confidence_threshold: float = 0.8
) -> StructurePrediction:
    """Predict protein structure from sequence.
    
    Args:
        sequence: Amino acid sequence string
        model: Pre-trained folding model
        confidence_threshold: Minimum confidence for reliable prediction
        
    Returns:
        StructurePrediction object with coordinates and confidence
        
    Raises:
        ValueError: If sequence contains invalid amino acids
        
    Examples:
        >>> predictor = StructurePredictor(model_path="model.pt")
        >>> result = predictor.predict("MKFLKFSLLT")
        >>> print(f"Confidence: {result.confidence:.2%}")
    """
```

#### API Documentation
- Update `docs/API_REFERENCE.md` for new public APIs
- Include usage examples
- Document all parameters and return values

## 🎯 Priority Contribution Areas

### 1. New Self-Supervised Objectives

We welcome novel SSL objectives for protein sequences:

```python
class YourSSLObjective(nn.Module):
    """Your novel SSL objective."""
    
    def forward(self, representations, batch):
        # Implement your objective
        return loss
```

### 2. Improved Uncertainty Quantification

Contributions to uncertainty estimation:
- Bayesian neural networks
- Ensemble methods
- Calibration techniques
- Uncertainty-aware training

### 3. Integration with Experimental Data

Help integrate experimental constraints:
- NMR restraints
- Cross-linking mass spectrometry
- SAXS/SANS data
- Cryo-EM density maps

### 4. Computational Efficiency

Performance optimizations:
- GPU kernel optimizations
- Memory usage reduction
- Faster data loading
- Model compression techniques

### 5. New Evaluation Metrics

Novel structure evaluation methods:
- Physics-based metrics
- Functional assessment
- Dynamics-aware scoring
- Multi-state evaluation

## 🔬 Research Contributions

### Publishing Research

If your contribution leads to a research publication:

1. **Cite the project**:
   ```bibtex
   @software{protein_sssl_operator,
     title={Protein-SSSL-Operator: Self-Supervised Structure-Sequence Learning},
     author={Schmidt, Daniel and contributors},
     url={https://github.com/danieleschmidt/protein-sssl-operator},
     year={2025}
   }
   ```

2. **Include acknowledgment**: "This work used the Protein-SSSL-Operator framework."

3. **Share results**: We'd love to feature your research on our project page!

### Reproducibility

For research contributions:
- Include complete hyperparameter configurations
- Provide evaluation scripts and data splits
- Document computational requirements
- Share trained model weights when possible

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information**:
   ```bash
   python --version
   pip list | grep torch
   nvidia-smi  # If using GPU
   ```

2. **Minimal reproduction example**:
   ```python
   # Code that reproduces the bug
   from protein_sssl import StructurePredictor
   predictor = StructurePredictor(...)
   # Bug occurs here
   ```

3. **Error messages** and stack traces

4. **Expected vs actual behavior**

## 💡 Feature Requests

For new features:
1. **Describe the use case** - What problem does this solve?
2. **Provide examples** - How would the API look?
3. **Consider alternatives** - Are there existing solutions?
4. **Discuss implementation** - Any technical constraints?

## 🧪 Experimental Features

We encourage experimental contributions in dedicated branches:

```bash
git checkout -b experimental/your-experiment
```

Mark experimental features clearly:

```python
@experimental
def new_experimental_feature():
    """
    Warning: This is an experimental feature and may change or be removed
    in future versions.
    """
    pass
```

## 📊 Benchmarking

When adding new models or methods:

1. **Benchmark against baselines** on standard datasets
2. **Profile performance** (memory, speed, accuracy)
3. **Include ablation studies** for new components
4. **Document trade-offs** between accuracy and efficiency

Use our evaluation framework:

```python
from protein_sssl.evaluation import benchmark_model

results = benchmark_model(
    model=your_model,
    test_dataset=casp_dataset,
    metrics=["tm_score", "gdt_ts", "lddt"]
)
```

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and diverse perspectives  
- Focus on constructive feedback
- Collaborate openly and transparently

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code reviews and discussions

### Mentorship

New contributors:
- Ask questions - no question is too basic!
- Start with "good first issue" labels
- Pair programming sessions available
- Code review as learning opportunity

## 🏆 Recognition

Contributors are recognized through:

- **CONTRIBUTORS.md**: All contributors listed
- **Release notes**: Contribution highlights
- **Academic citations**: Co-authorship opportunities for significant research contributions
- **Conference presentations**: Speaking opportunities

## 🔧 Development Tips

### Local Testing

```bash
# Test specific modules
pytest tests/test_models.py -v

# Test with different Python versions
tox

# Test GPU functionality
pytest tests/ -m gpu --gpu-id=0
```

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
import protein_sssl
protein_sssl.set_log_level("DEBUG")
```

### Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

## 📦 Release Process

For maintainers and core contributors:

1. **Version bumping**: Follow semantic versioning
2. **Changelog**: Update `CHANGELOG.md`
3. **Testing**: Full test suite on multiple environments
4. **Documentation**: Update API docs and examples
5. **Release notes**: Highlight new features and breaking changes

## 💬 Questions?

- Check existing issues and discussions
- Review documentation and examples
- Ask in GitHub Discussions
- Reach out to maintainers directly for complex questions

Thank you for contributing to advancing protein structure prediction research! 🧬✨