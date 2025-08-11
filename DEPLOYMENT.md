# Deployment Guide for protein-sssl-operator

## Overview

This guide provides comprehensive instructions for deploying the protein-sssl-operator in various environments, from local development to production-scale Kubernetes clusters.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │     Staging     │    │   Production    │
│   Environment   │    │   Environment   │    │   Environment   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Local Docker  │    │ • K8s Cluster   │    │ • K8s Cluster   │
│ • Direct Python │    │ • Helm Charts   │    │ • Helm Charts   │
│ • Hot Reload    │    │ • CI/CD Pipeline│    │ • Auto-scaling  │
│ • Debug Mode    │    │ • Load Testing  │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, Recommended 8+ cores
- **Memory**: Minimum 16GB RAM, Recommended 32GB+ RAM  
- **Storage**: Minimum 100GB SSD, Recommended 500GB+ NVMe SSD
- **GPU** (Optional): NVIDIA GPU with 8GB+ VRAM for accelerated inference

### Software Dependencies

- Docker 20.10+
- Kubernetes 1.25+
- Helm 3.8+
- kubectl
- Python 3.9+

### Network Requirements

- Internet access for downloading models and dependencies
- Ingress controller for external access
- Load balancer (for production)

## Deployment Methods

### 1. Local Development Deployment

#### Option A: Direct Python Installation

```bash
# Clone repository
git clone https://github.com/danieleschmidt/protein-sssl-operator.git
cd protein-sssl-operator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]

# Run development server
protein-sssl predict "MKFLKFSLLTAV" --model models/test_model.pt
```

#### Option B: Docker Development

```bash
# Build development image
docker build -t protein-sssl:dev -f docker/Dockerfile .

# Run interactive container
docker run -it --rm \
  -v $(pwd):/app \
  -p 8000:8000 \
  protein-sssl:dev bash

# Inside container
protein-sssl --help
```

#### Option C: Docker Compose

```bash
# Start development stack
docker-compose up -d

# View logs
docker-compose logs -f

# Access services
curl http://localhost:8000/health
```

### 2. Staging Deployment

#### Prerequisites

- Kubernetes cluster access
- Helm 3.8+ installed
- kubectl configured

#### Deployment Steps

```bash
# Set environment variables
export ENVIRONMENT=staging
export NAMESPACE=protein-sssl-staging
export VERSION=v0.1.0

# Create namespace
kubectl create namespace $NAMESPACE

# Install dependencies (Prometheus, etc.)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Deploy monitoring stack
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Deploy application
helm upgrade --install protein-sssl ./helm/protein-sssl \
  --namespace $NAMESPACE \
  --set image.tag=$VERSION \
  --set environment=staging \
  --set resources.requests.memory=4Gi \
  --set resources.requests.cpu=2 \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=5 \
  --wait \
  --timeout=10m

# Verify deployment
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
```

#### Validation

```bash
# Check pod status
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=protein-sssl -n $NAMESPACE --timeout=300s

# Run health check
kubectl run health-check --rm -i --restart=Never \
  --image=curlimages/curl \
  -- curl -f http://protein-sssl.$NAMESPACE.svc.cluster.local/health

# Test prediction endpoint
kubectl run prediction-test --rm -i --restart=Never \
  --image=curlimages/curl \
  -- curl -X POST http://protein-sssl.$NAMESPACE.svc.cluster.local/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLKFSLLTAV"}'
```

### 3. Production Deployment

#### Automated Deployment (Recommended)

```bash
# Run production deployment script
./deployment/production-deploy.sh \
  --environment production \
  --version v0.1.0 \
  --namespace protein-sssl-prod
```

#### Manual Deployment Steps

```bash
# Set production variables
export ENVIRONMENT=production
export NAMESPACE=protein-sssl-prod
export VERSION=v0.1.0
export REGISTRY=ghcr.io/terragonlabs

# Create production namespace with labels
kubectl create namespace $NAMESPACE
kubectl label namespace $NAMESPACE environment=production
kubectl label namespace $NAMESPACE app.kubernetes.io/name=protein-sssl

# Configure production secrets
kubectl create secret generic protein-sssl-secrets \
  --namespace $NAMESPACE \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=redis-url="redis://redis:6379/0" \
  --from-literal=secret-key="$(openssl rand -base64 32)"

# Deploy with production values
helm upgrade --install protein-sssl ./helm/protein-sssl \
  --namespace $NAMESPACE \
  --values ./helm/protein-sssl/values-production.yaml \
  --set image.repository=$REGISTRY/protein-sssl-operator \
  --set image.tag=$VERSION \
  --set environment=production \
  --wait \
  --timeout=15m

# Apply additional production resources
kubectl apply -f kubernetes/autoscaling.yaml -n $NAMESPACE
kubectl apply -f kubernetes/monitoring.yaml -n $NAMESPACE
```

## Configuration Management

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `development` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `MODEL_PATH` | Path to model files | `/app/models` | No |
| `CACHE_SIZE` | Cache size limit | `1000` | No |
| `GPU_ENABLED` | Enable GPU acceleration | `false` | No |
| `METRICS_ENABLED` | Enable metrics collection | `true` | No |

### Configuration Files

#### Development
```yaml
# configs/development.yaml
environment: development
log_level: DEBUG
model:
  cache_size: 100
  batch_size: 1
monitoring:
  enabled: false
```

#### Production
```yaml
# configs/production.yaml
environment: production
log_level: INFO
model:
  cache_size: 10000
  batch_size: 32
monitoring:
  enabled: true
  metrics_port: 9090
autoscaling:
  enabled: true
  min_replicas: 3
  max_replicas: 50
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale deployment manually
kubectl scale deployment protein-sssl --replicas=10 -n $NAMESPACE

# Enable horizontal pod autoscaler
kubectl apply -f kubernetes/autoscaling.yaml -n $NAMESPACE

# Check HPA status
kubectl get hpa -n $NAMESPACE
```

### Vertical Scaling

```bash
# Apply VPA configuration
kubectl apply -f kubernetes/autoscaling.yaml -n $NAMESPACE

# Check VPA recommendations
kubectl describe vpa protein-sssl-vpa -n $NAMESPACE
```

### Performance Tuning

#### GPU Acceleration

```bash
# Deploy GPU-enabled version
helm upgrade protein-sssl ./helm/protein-sssl \
  --namespace $NAMESPACE \
  --set gpu.enabled=true \
  --set gpu.nodeSelector="accelerator=nvidia-tesla-v100" \
  --set resources.limits.nvidia.com/gpu=1
```

#### Memory Optimization

```bash
# Apply memory-optimized settings
helm upgrade protein-sssl ./helm/protein-sssl \
  --namespace $NAMESPACE \
  --set resources.requests.memory=8Gi \
  --set resources.limits.memory=16Gi \
  --set jvm.heap.max=12g
```

## Monitoring and Observability

### Metrics Collection

```bash
# Install Prometheus monitoring
kubectl apply -f kubernetes/monitoring.yaml -n $NAMESPACE

# Access Grafana dashboard
kubectl port-forward service/prometheus-grafana 3000:80 -n monitoring
# Visit http://localhost:3000
```

### Logging

```bash
# View application logs
kubectl logs -l app.kubernetes.io/name=protein-sssl -n $NAMESPACE -f

# Aggregate logs with stern (if installed)
stern protein-sssl -n $NAMESPACE
```

### Health Checks

```bash
# Check application health
curl http://protein-sssl.your-domain.com/health

# Detailed health check
curl http://protein-sssl.your-domain.com/health/detailed
```

## Security Configuration

### Network Policies

```bash
# Apply network security policies
kubectl apply -f kubernetes/network-policies.yaml -n $NAMESPACE
```

### Pod Security Standards

```bash
# Apply pod security policies
kubectl label namespace $NAMESPACE pod-security.kubernetes.io/enforce=restricted
kubectl label namespace $NAMESPACE pod-security.kubernetes.io/audit=restricted
kubectl label namespace $NAMESPACE pod-security.kubernetes.io/warn=restricted
```

### Secret Management

```bash
# Create TLS certificates
kubectl create secret tls protein-sssl-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n $NAMESPACE

# Create image pull secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN \
  -n $NAMESPACE
```

## Backup and Disaster Recovery

### Database Backup

```bash
# Create backup job
kubectl apply -f kubernetes/backup-job.yaml -n $NAMESPACE

# Check backup status
kubectl get jobs -n $NAMESPACE
```

### Configuration Backup

```bash
# Backup Helm release
helm get values protein-sssl -n $NAMESPACE > backup/helm-values-$(date +%Y%m%d).yaml

# Backup Kubernetes resources
kubectl get all -n $NAMESPACE -o yaml > backup/k8s-resources-$(date +%Y%m%d).yaml
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues

```bash
# Check pod events
kubectl describe pod <pod-name> -n $NAMESPACE

# Check logs
kubectl logs <pod-name> -n $NAMESPACE --previous

# Check resource constraints
kubectl top pods -n $NAMESPACE
```

#### Performance Issues

```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -n $NAMESPACE

# Check HPA metrics
kubectl describe hpa protein-sssl -n $NAMESPACE

# Profile application
kubectl exec -it <pod-name> -n $NAMESPACE -- python -m cProfile -o profile.stats app.py
```

#### Network Issues

```bash
# Test service connectivity
kubectl run debug --rm -i --restart=Never --image=busybox -- nslookup protein-sssl.$NAMESPACE.svc.cluster.local

# Check ingress configuration
kubectl describe ingress protein-sssl -n $NAMESPACE
```

### Debugging Commands

```bash
# Get into container for debugging
kubectl exec -it <pod-name> -n $NAMESPACE -- /bin/bash

# Port forward for local testing
kubectl port-forward service/protein-sssl 8000:80 -n $NAMESPACE

# Restart deployment
kubectl rollout restart deployment/protein-sssl -n $NAMESPACE
```

## Rollback Procedures

### Helm Rollback

```bash
# List releases
helm history protein-sssl -n $NAMESPACE

# Rollback to previous version
helm rollback protein-sssl -n $NAMESPACE

# Rollback to specific revision
helm rollback protein-sssl 2 -n $NAMESPACE
```

### Kubernetes Rollback

```bash
# Check rollout history
kubectl rollout history deployment/protein-sssl -n $NAMESPACE

# Rollback to previous version
kubectl rollout undo deployment/protein-sssl -n $NAMESPACE

# Rollback to specific revision
kubectl rollout undo deployment/protein-sssl --to-revision=2 -n $NAMESPACE
```

## Maintenance

### Updates

```bash
# Update to new version
helm upgrade protein-sssl ./helm/protein-sssl \
  --namespace $NAMESPACE \
  --set image.tag=v0.2.0 \
  --wait

# Verify update
kubectl rollout status deployment/protein-sssl -n $NAMESPACE
```

### Cleanup

```bash
# Remove deployment
helm uninstall protein-sssl -n $NAMESPACE

# Remove namespace
kubectl delete namespace $NAMESPACE

# Clean up persistent volumes
kubectl delete pv $(kubectl get pv | grep $NAMESPACE | awk '{print $1}')
```

## Support

### Documentation
- [API Reference](docs/API_REFERENCE.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

### Community
- GitHub Issues: https://github.com/danieleschmidt/protein-sssl-operator/issues
- Discussions: https://github.com/danieleschmidt/protein-sssl-operator/discussions

### Professional Support
- Email: support@terragonlabs.ai
- Slack: #protein-sssl-support