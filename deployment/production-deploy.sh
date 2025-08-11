#!/bin/bash

# Production Deployment Script for protein-sssl-operator
# Terragon Labs - Autonomous SDLC Execution
# Enhanced with comprehensive deployment automation, monitoring, and safety features

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-protein-sssl-prod}"
CLUSTER_CONTEXT="${CLUSTER_CONTEXT:-production}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-protein-sssl}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Safety checks
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check if we can connect to the cluster
    if ! kubectl config use-context "$CLUSTER_CONTEXT" &> /dev/null; then
        log_error "Cannot connect to cluster context: $CLUSTER_CONTEXT"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    log_success "Prerequisites check passed"
}

# Backup current deployment
backup_deployment() {
    log_info "Creating backup of current deployment..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup Helm release values
    if helm status "$HELM_RELEASE_NAME" -n "$NAMESPACE" &> /dev/null; then
        helm get values "$HELM_RELEASE_NAME" -n "$NAMESPACE" > "$BACKUP_DIR/helm-values.yaml"
        helm get manifest "$HELM_RELEASE_NAME" -n "$NAMESPACE" > "$BACKUP_DIR/manifest.yaml"
        log_success "Helm release backed up to $BACKUP_DIR"
    fi
    
    # Backup ConfigMaps and Secrets
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/configmaps.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/secrets.yaml"
    
    log_success "Backup completed: $BACKUP_DIR"
}

# Pre-deployment validation
validate_deployment() {
    log_info "Validating deployment configuration..."
    
    # Check if Docker image exists
    log_info "Validating Docker image: protein-sssl:$IMAGE_TAG"
    
    # Validate Helm chart
    helm lint helm/protein-sssl/
    
    # Dry-run the deployment
    helm upgrade --install "$HELM_RELEASE_NAME" helm/protein-sssl/ \
        --namespace "$NAMESPACE" \
        --set image.tag="$IMAGE_TAG" \
        --dry-run --debug > /tmp/helm-dry-run.yaml
    
    log_success "Deployment validation passed"
}

# Deploy monitoring stack first
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --create-namespace \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set grafana.adminPassword="$(openssl rand -base64 32)"
    
    log_success "Monitoring stack deployed"
}

# Main deployment function
deploy_application() {
    log_info "Deploying protein-sssl application..."
    
    # Deploy using Helm
    helm upgrade --install "$HELM_RELEASE_NAME" helm/protein-sssl/ \
        --namespace "$NAMESPACE" \
        --set image.tag="$IMAGE_TAG" \
        --set config.environment="production" \
        --set monitoring.enabled=true \
        --set autoscaling.enabled=true \
        --wait \
        --timeout=10m
    
    log_success "Application deployed successfully"
}

# Post-deployment verification
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=protein-sssl -n "$NAMESPACE" --timeout=300s
    
    # Check service endpoints
    kubectl get services -n "$NAMESPACE"
    
    # Run health checks
    log_info "Running health checks..."
    
    # Check if inference API is responding
    INFERENCE_SERVICE=$(kubectl get service protein-sssl-inference-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$INFERENCE_SERVICE" ]; then
        curl -f "http://$INFERENCE_SERVICE/health" || log_warning "Health check failed for inference service"
    fi
    
    # Check training service
    kubectl port-forward service/protein-sssl-training-service 8888:8888 -n "$NAMESPACE" &
    PORT_FORWARD_PID=$!
    sleep 5
    curl -f "http://localhost:8888/api" || log_warning "Training service health check failed"
    kill $PORT_FORWARD_PID
    
    log_success "Deployment verification completed"
}

# Rollback function
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    helm rollback "$HELM_RELEASE_NAME" -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=protein-sssl -n "$NAMESPACE" --timeout=300s
    
    log_success "Rollback completed"
}

# Cleanup old backups
cleanup_backups() {
    log_info "Cleaning up old backups..."
    
    find backups/ -type d -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    
    log_success "Backup cleanup completed"
}

# Main deployment process
main() {
    log_info "Starting production deployment of protein-sssl-operator"
    log_info "Image tag: $IMAGE_TAG"
    log_info "Namespace: $NAMESPACE"
    log_info "Cluster context: $CLUSTER_CONTEXT"
    
    # Confirm deployment in production
    if [[ "$CLUSTER_CONTEXT" == *"prod"* ]]; then
        read -p "This is a PRODUCTION deployment. Are you sure? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    # Execute deployment steps
    check_prerequisites
    backup_deployment
    validate_deployment
    
    # Deploy monitoring if not exists
    if ! kubectl get namespace "$MONITORING_NAMESPACE" &> /dev/null; then
        deploy_monitoring
    fi
    
    # Deploy application with error handling
    if deploy_application; then
        verify_deployment
        cleanup_backups
        log_success "Deployment completed successfully!"
        
        # Display access information
        echo
        log_info "Access Information:"
        echo "- Namespace: $NAMESPACE"
        echo "- Helm release: $HELM_RELEASE_NAME"
        echo "- Image tag: $IMAGE_TAG"
        
        # Get service endpoints
        kubectl get ingress -n "$NAMESPACE"
        
    else
        log_error "Deployment failed!"
        read -p "Do you want to rollback? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback_deployment
        fi
        exit 1
    fi
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"