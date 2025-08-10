#!/bin/bash
# CI/CD Pipeline Script for Protein SSSL Operator
# This script can be used locally or integrated with any CI/CD system

set -e

echo "🚀 Starting Protein SSSL Operator CI/CD Pipeline"

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .
pip install pytest pytest-cov flake8 mypy black isort bandit

# Code formatting and linting
echo "🔧 Running code formatting and linting..."
black protein_sssl/ tests/ --check
isort protein_sssl/ tests/ --check-only
flake8 protein_sssl/ tests/ --max-line-length=88 --ignore=E203,W503
mypy protein_sssl/ --ignore-missing-imports

# Security scanning
echo "🔒 Running security scan..."
bandit -r protein_sssl/ -f json -o security-report.json || true

# Run tests
echo "🧪 Running tests..."
pytest tests/ --cov=protein_sssl --cov-report=xml --cov-report=html -v

# Performance tests
echo "⚡ Running performance tests..."
python -m pytest tests/test_performance.py -v

# Integration tests
echo "🔄 Running integration tests..."
python -m pytest tests/test_integration.py -v

# Build Docker image (if Docker is available)
if command -v docker &> /dev/null; then
    echo "🐳 Building Docker image..."
    docker build -t protein-sssl-operator:latest -f docker/Dockerfile .
    echo "✅ Docker image built successfully"
else
    echo "⚠️  Docker not available, skipping image build"
fi

echo "✅ CI/CD Pipeline completed successfully!"
echo "📊 Coverage report available in htmlcov/index.html"
echo "🔒 Security report available in security-report.json"