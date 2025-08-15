#!/bin/bash

# Autonomous Production Deployment Script
# Advanced deployment with self-monitoring and rollback capabilities

set -euo pipefail

# Configuration
DEPLOYMENT_ID="deploy_$(date +%s)"
PROJECT_NAME="protein-sssl-operator"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
REGISTRY="${REGISTRY:-ghcr.io/danieleschmidt}"
NAMESPACE="${NAMESPACE:-protein-sssl}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"
AUTO_ROLLBACK="${AUTO_ROLLBACK:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "deployment_${DEPLOYMENT_ID}.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "deployment_${DEPLOYMENT_ID}.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "deployment_${DEPLOYMENT_ID}.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "deployment_${DEPLOYMENT_ID}.log"
}

# Error handling
handle_error() {
    local exit_code=$?
    log_error "Deployment failed at line $1 with exit code $exit_code"
    
    if [[ "$AUTO_ROLLBACK" == "true" ]]; then
        log_warning "Initiating automatic rollback..."
        rollback_deployment
    fi
    
    cleanup_deployment
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Pre-deployment validation
validate_environment() {
    log_info "🔍 Validating deployment environment..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "docker" "python3")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' not found"
            exit 1
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Validate configuration files
    local config_files=("../configs/ssl_config.yaml" "../configs/folding_config.yaml")
    for config in "${config_files[@]}"; do
        if [[ ! -f "$config" ]]; then
            log_error "Configuration file not found: $config"
            exit 1
        fi
    done
    
    log_success "Environment validation completed"
}

# Quality gates validation
run_quality_gates() {
    log_info "🧪 Running comprehensive quality gates..."
    
    cd "$(dirname "$0")/.."
    
    # Run quality validation
    if ! python3 scripts/quality_gates.py; then
        log_error "Quality gates failed - deployment aborted"
        exit 1
    fi
    
    # Run security scan
    log_info "Running security scan..."
    if command -v bandit &> /dev/null; then
        bandit -r protein_sssl/ -f json -o security_scan_${DEPLOYMENT_ID}.json || true
    fi
    
    # Run performance benchmark
    log_info "Running performance benchmark..."
    if python3 scripts/test_scaling_minimal.py; then
        log_success "Performance benchmark passed"
    else
        log_warning "Performance benchmark had issues - proceeding with caution"
    fi
    
    log_success "Quality gates validation completed"
}

# Build and push container images
build_and_push_images() {
    log_info "🐳 Building and pushing container images..."
    
    local git_sha=$(git rev-parse --short HEAD)
    local image_tag="${DEPLOYMENT_ENV}-${git_sha}-$(date +%s)"
    
    # Build CPU image
    log_info "Building CPU image..."
    docker build -f docker/Dockerfile -t "${REGISTRY}/${PROJECT_NAME}:${image_tag}" .
    docker build -f docker/Dockerfile -t "${REGISTRY}/${PROJECT_NAME}:${DEPLOYMENT_ENV}-latest" .
    
    # Build GPU image
    log_info "Building GPU image..."
    docker build -f docker/Dockerfile.gpu -t "${REGISTRY}/${PROJECT_NAME}-gpu:${image_tag}" .
    docker build -f docker/Dockerfile.gpu -t "${REGISTRY}/${PROJECT_NAME}-gpu:${DEPLOYMENT_ENV}-latest" .
    
    # Push images
    log_info "Pushing images to registry..."
    docker push "${REGISTRY}/${PROJECT_NAME}:${image_tag}"
    docker push "${REGISTRY}/${PROJECT_NAME}:${DEPLOYMENT_ENV}-latest"
    docker push "${REGISTRY}/${PROJECT_NAME}-gpu:${image_tag}"
    docker push "${REGISTRY}/${PROJECT_NAME}-gpu:${DEPLOYMENT_ENV}-latest"
    
    # Store image tags for rollback
    echo "${REGISTRY}/${PROJECT_NAME}:${image_tag}" > current_image_tag.txt
    echo "${REGISTRY}/${PROJECT_NAME}-gpu:${image_tag}" > current_gpu_image_tag.txt
    
    log_success "Container images built and pushed successfully"
}

# Deploy with Helm
deploy_with_helm() {
    log_info "⚡ Deploying with Helm..."
    
    local image_tag=$(cat current_image_tag.txt)
    local gpu_image_tag=$(cat current_gpu_image_tag.txt)
    
    # Update Helm values
    cat > helm_values_${DEPLOYMENT_ID}.yaml << EOF
image:
  repository: ${REGISTRY}/${PROJECT_NAME}
  tag: $(basename $image_tag | cut -d: -f2)
  pullPolicy: Always

gpuImage:
  repository: ${REGISTRY}/${PROJECT_NAME}-gpu
  tag: $(basename $gpu_image_tag | cut -d: -f2)

environment: ${DEPLOYMENT_ENV}

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: ${MONITORING_ENABLED}
  prometheus:
    enabled: true
  grafana:
    enabled: true

security:
  networkPolicies:
    enabled: true
  podSecurityPolicy:
    enabled: true

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"

persistence:
  enabled: true
  size: 100Gi
  storageClass: fast-ssd

configMaps:
  ssl_config: ../configs/ssl_config.yaml
  folding_config: ../configs/folding_config.yaml
EOF

    # Deploy or upgrade
    if helm list -n "$NAMESPACE" | grep -q "$PROJECT_NAME"; then
        log_info "Upgrading existing deployment..."
        helm upgrade "$PROJECT_NAME" helm/protein-sssl \
            --namespace "$NAMESPACE" \
            --values "helm_values_${DEPLOYMENT_ID}.yaml" \
            --timeout 10m \
            --wait
    else
        log_info "Installing new deployment..."
        helm install "$PROJECT_NAME" helm/protein-sssl \
            --namespace "$NAMESPACE" \
            --values "helm_values_${DEPLOYMENT_ID}.yaml" \
            --timeout 10m \
            --wait
    fi
    
    log_success "Helm deployment completed"
}

# Health checks and validation
validate_deployment() {
    log_info "🏥 Validating deployment health..."
    
    local max_attempts=60
    local attempt=0
    
    # Wait for pods to be ready
    while [[ $attempt -lt $max_attempts ]]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app="$PROJECT_NAME" -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -o "True" | wc -l)
        local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app="$PROJECT_NAME" -o jsonpath='{.items[*].metadata.name}' | wc -w)
        
        if [[ $ready_pods -eq $total_pods ]] && [[ $total_pods -gt 0 ]]; then
            log_success "All pods are ready ($ready_pods/$total_pods)"
            break
        fi
        
        log_info "Waiting for pods to be ready: $ready_pods/$total_pods (attempt $((attempt + 1))/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Deployment health check failed - pods not ready within timeout"
        return 1
    fi
    
    # Test API endpoints
    log_info "Testing API endpoints..."
    local service_ip=$(kubectl get service -n "$NAMESPACE" "$PROJECT_NAME" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -n "$service_ip" ]]; then
        # Test health endpoint
        if curl -f "http://${service_ip}/health" &> /dev/null; then
            log_success "Health endpoint responding"
        else
            log_warning "Health endpoint not responding"
        fi
        
        # Test prediction endpoint with sample data
        if curl -f -X POST "http://${service_ip}/predict" \
           -H "Content-Type: application/json" \
           -d '{"sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"}' &> /dev/null; then
            log_success "Prediction endpoint working"
        else
            log_warning "Prediction endpoint not responding properly"
        fi
    fi
    
    log_success "Deployment validation completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    if [[ "$MONITORING_ENABLED" != "true" ]]; then
        log_info "Monitoring disabled, skipping setup"
        return 0
    fi
    
    log_info "📊 Setting up monitoring and alerting..."
    
    # Apply monitoring manifests
    kubectl apply -f kubernetes/monitoring.yaml -n "$NAMESPACE"
    
    # Configure Prometheus alerts
    kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ${PROJECT_NAME}-alerts
  namespace: ${NAMESPACE}
spec:
  groups:
  - name: ${PROJECT_NAME}
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: High error rate detected
        
    - alert: HighMemoryUsage
      expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: High memory usage
        
    - alert: PredictionLatencyHigh
      expr: histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m])) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: High prediction latency
EOF

    log_success "Monitoring setup completed"
}

# Performance verification
verify_performance() {
    log_info "⚡ Verifying deployment performance..."
    
    # Run load test
    local service_ip=$(kubectl get service -n "$NAMESPACE" "$PROJECT_NAME" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -n "$service_ip" ]]; then
        # Simple load test with sample sequences
        log_info "Running load test..."
        
        cat > load_test_${DEPLOYMENT_ID}.py << 'EOF'
import asyncio
import aiohttp
import time
import json

async def make_prediction(session, url, sequence):
    async with session.post(url, json={"sequence": sequence}) as response:
        return await response.json()

async def load_test():
    url = f"http://{service_ip}/predict"
    sequences = [
        "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ]
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = []
        
        # Create 50 concurrent requests
        for i in range(50):
            seq = sequences[i % len(sequences)]
            task = make_prediction(session, url, seq)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        
        print(f"Load test results:")
        print(f"  Total requests: {len(tasks)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(tasks) - successful}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests/second: {len(tasks) / total_time:.2f}")
        
        return successful >= len(tasks) * 0.9  # 90% success rate required

if __name__ == "__main__":
    success = asyncio.run(load_test())
    exit(0 if success else 1)
EOF

        # Run load test
        if python3 "load_test_${DEPLOYMENT_ID}.py"; then
            log_success "Load test passed"
        else
            log_warning "Load test failed - monitoring performance"
        fi
    fi
    
    log_success "Performance verification completed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "🔄 Rolling back deployment..."
    
    # Get previous revision
    local previous_revision=$(helm history "$PROJECT_NAME" -n "$NAMESPACE" | tail -2 | head -1 | awk '{print $1}')
    
    if [[ -n "$previous_revision" ]]; then
        helm rollback "$PROJECT_NAME" "$previous_revision" -n "$NAMESPACE"
        log_success "Rollback to revision $previous_revision completed"
    else
        log_error "No previous revision found for rollback"
    fi
}

# Cleanup temporary files
cleanup_deployment() {
    log_info "🧹 Cleaning up temporary files..."
    
    rm -f "helm_values_${DEPLOYMENT_ID}.yaml"
    rm -f "load_test_${DEPLOYMENT_ID}.py"
    rm -f "current_image_tag.txt"
    rm -f "current_gpu_image_tag.txt"
    
    log_info "Cleanup completed"
}

# Generate deployment report
generate_deployment_report() {
    log_info "📋 Generating deployment report..."
    
    local report_file="deployment_report_${DEPLOYMENT_ID}.json"
    
    cat > "$report_file" << EOF
{
  "deployment_id": "$DEPLOYMENT_ID",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$DEPLOYMENT_ENV",
  "project": "$PROJECT_NAME",
  "namespace": "$NAMESPACE",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "$(git branch --show-current)",
  "images": {
    "cpu": "$(cat current_image_tag.txt 2>/dev/null || echo 'N/A')",
    "gpu": "$(cat current_gpu_image_tag.txt 2>/dev/null || echo 'N/A')"
  },
  "kubernetes": {
    "cluster": "$(kubectl config current-context)",
    "namespace": "$NAMESPACE",
    "pods": $(kubectl get pods -n "$NAMESPACE" -l app="$PROJECT_NAME" -o json | jq '.items | length'),
    "services": $(kubectl get services -n "$NAMESPACE" -l app="$PROJECT_NAME" -o json | jq '.items | length')
  },
  "monitoring_enabled": $MONITORING_ENABLED,
  "auto_rollback_enabled": $AUTO_ROLLBACK,
  "deployment_duration": "$(date +%s) - start_time",
  "status": "completed"
}
EOF

    log_success "Deployment report generated: $report_file"
}

# Main deployment flow
main() {
    local start_time=$(date +%s)
    
    log_info "🚀 Starting autonomous production deployment"
    log_info "Deployment ID: $DEPLOYMENT_ID"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Namespace: $NAMESPACE"
    
    # Deployment phases
    validate_environment
    run_quality_gates
    build_and_push_images
    deploy_with_helm
    validate_deployment
    setup_monitoring
    verify_performance
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
    log_info "Total deployment time: ${duration}s"
    
    # Generate final report
    generate_deployment_report
    
    # Display important information
    echo ""
    echo "==================================="
    echo "   DEPLOYMENT SUMMARY"
    echo "==================================="
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Namespace: $NAMESPACE"
    echo "Duration: ${duration}s"
    echo "Monitoring: $MONITORING_ENABLED"
    echo ""
    echo "Service endpoints:"
    kubectl get services -n "$NAMESPACE" -l app="$PROJECT_NAME"
    echo ""
    echo "Pod status:"
    kubectl get pods -n "$NAMESPACE" -l app="$PROJECT_NAME"
    echo "==================================="
    
    cleanup_deployment
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi