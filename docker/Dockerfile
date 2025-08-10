# Multi-stage Docker build for protein-sssl-operator
# Production-ready container with optimizations

FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 protein_user && \
    mkdir -p /app && \
    chown protein_user:protein_user /app

WORKDIR /app

# Install Python dependencies (in separate layer for caching)
COPY pyproject.toml environment.yml ./
RUN python3.11 -m pip install --upgrade pip setuptools wheel && \
    python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    python3.11 -m pip install transformers datasets accelerate && \
    python3.11 -m pip install numpy scipy matplotlib seaborn pandas && \
    python3.11 -m pip install pytest pytest-cov black flake8 mypy && \
    python3.11 -m pip install wandb tensorboard optuna && \
    python3.11 -m pip install biotite biopython mdtraj

# Copy application code
COPY --chown=protein_user:protein_user . .

# Install the package in development mode
RUN python3.11 -m pip install -e .

# Switch to non-root user
USER protein_user

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python3.11 -c "import protein_sssl; print('OK')" || exit 1

# Default command
CMD ["python3.11", "-m", "protein_sssl.cli", "--help"]

# Production stage with optimizations
FROM base AS production

# Remove development dependencies
RUN python3.11 -m pip uninstall -y pytest pytest-cov black flake8 mypy

# Set production environment
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4

# Copy only necessary files
COPY --from=base --chown=protein_user:protein_user /app/protein_sssl /app/protein_sssl
COPY --from=base --chown=protein_user:protein_user /app/scripts /app/scripts
COPY --from=base --chown=protein_user:protein_user /app/setup.py /app/pyproject.toml /app/

EXPOSE 8000

# Production entrypoint
ENTRYPOINT ["python3.11", "-O"]
CMD ["-m", "protein_sssl.cli"]