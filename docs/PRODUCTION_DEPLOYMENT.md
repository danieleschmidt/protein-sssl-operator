# Production Deployment Guide

## Overview

This comprehensive guide provides step-by-step instructions for deploying the protein-sssl-operator in production environments worldwide. The deployment framework supports multi-region setups, compliance with international regulations, and enterprise-grade security.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Security Configuration](#security-configuration)
4. [Environment Setup](#environment-setup)
5. [Production Deployment](#production-deployment)
6. [Multi-Region Deployment](#multi-region-deployment)
7. [Compliance Configuration](#compliance-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Monitoring Setup](#monitoring-setup)
10. [Validation and Testing](#validation-and-testing)
11. [Post-Deployment Checklist](#post-deployment-checklist)

## Prerequisites

### System Requirements

#### Minimum Production Requirements
- **CPU**: 16 cores (Intel Xeon or AMD EPYC recommended)
- **Memory**: 64GB RAM (128GB+ recommended for large-scale deployments)
- **Storage**: 2TB NVMe SSD (with 50,000+ IOPS)
- **GPU**: NVIDIA A100 or V100 (8GB+ VRAM) for GPU-accelerated inference
- **Network**: 10Gbps network interface with low latency

#### Recommended Production Configuration
- **CPU**: 32+ cores with AVX-512 support
- **Memory**: 256GB+ RAM with ECC
- **Storage**: 10TB+ NVMe SSD in RAID 10 configuration
- **GPU**: Multiple NVIDIA H100 or A100 GPUs
- **Network**: 25Gbps+ with redundant connections

### Software Dependencies

#### Container Platform
- **Kubernetes**: 1.25+ (1.28+ recommended)
- **Docker**: 20.10+ or containerd 1.6+
- **Helm**: 3.10+ (3.12+ recommended)

#### Cloud Platform Support
- **AWS**: EKS 1.25+, EC2 instances with SR-IOV
- **Google Cloud**: GKE 1.25+, Compute Engine with gVNIC
- **Azure**: AKS 1.25+, Virtual Machines with accelerated networking
- **On-premises**: OpenShift 4.12+, Rancher 2.7+

#### Networking
- **CNI**: Calico, Cilium, or Weave Net
- **Ingress**: NGINX Ingress Controller 1.5+, Istio, or Traefik
- **Load Balancer**: MetalLB, HAProxy, or cloud provider load balancer
- **Service Mesh**: Istio 1.16+ or Linkerd 2.12+ (optional)

## Infrastructure Requirements

### Kubernetes Cluster Setup

#### Node Configuration
```yaml
# Production node pool specification
apiVersion: v1
kind: Node
metadata:
  labels:
    node-type: protein-sssl-worker
    accelerator: nvidia-a100
    storage: nvme-ssd
spec:
  capacity:
    cpu: "32"
    memory: 256Gi
    nvidia.com/gpu: "4"
    ephemeral-storage: 2Ti
```

#### Storage Classes
```yaml
# Fast SSD storage class for models and checkpoints
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: protein-sssl-fast
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "10000"
  throughput: "500"
  fsType: ext4
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
---
# Shared storage for datasets
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: protein-sssl-shared
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-xxxxx
  directoryPerms: "0755"
volumeBindingMode: Immediate
```

### Network Architecture

#### Production Network Topology
```
Internet
    │
    ▼
[Load Balancer] ── [WAF/DDoS Protection]
    │
    ▼
[Ingress Controller] ── [TLS Termination]
    │
    ▼
[Application Services] ── [Service Mesh (Optional)]
    │
    ▼
[Backend Services] ── [Internal Load Balancing]
    │
    ▼
[Data Storage] ── [Encrypted Communication]
```

#### Security Groups/Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: protein-sssl-production
spec:
  podSelector:
    matchLabels:
      app: protein-sssl
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
```

## Security Configuration

### TLS/SSL Setup

#### Certificate Management
```bash
# Install cert-manager for automated certificate management
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Create production certificate issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

#### TLS Configuration
```yaml
# TLS 1.3 configuration with strong ciphers
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-tls-config
data:
  ssl-protocols: "TLSv1.3"
  ssl-ciphers: "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256"
  ssl-prefer-server-ciphers: "false"
  ssl-session-cache: "shared:SSL:10m"
  ssl-session-timeout: "10m"
```

### Authentication and Authorization

#### RBAC Configuration
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: protein-sssl-operator
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies"]
  verbs: ["get", "list", "watch"]
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: protein-sssl-operator
  namespace: protein-sssl-prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: protein-sssl-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: protein-sssl-operator
subjects:
- kind: ServiceAccount
  name: protein-sssl-operator
  namespace: protein-sssl-prod
```

#### OAuth2/OIDC Integration
```yaml
# Example OAuth2 proxy configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oauth2-proxy
spec:
  template:
    spec:
      containers:
      - name: oauth2-proxy
        image: quay.io/oauth2-proxy/oauth2-proxy:v7.4.0
        args:
        - --provider=oidc
        - --oidc-issuer-url=https://your-oidc-provider.com
        - --client-id=your-client-id
        - --client-secret=your-client-secret
        - --cookie-secret=your-cookie-secret
        - --upstream=http://protein-sssl.protein-sssl-prod.svc.cluster.local
        - --email-domain=your-domain.com
```

### Secrets Management

#### Using Kubernetes Secrets
```bash
# Create secrets for sensitive configuration
kubectl create secret generic protein-sssl-secrets \
  --namespace protein-sssl-prod \
  --from-literal=database-url="postgresql://user:$(openssl rand -base64 32)@postgres:5432/protein_sssl" \
  --from-literal=redis-url="redis://redis:6379/0" \
  --from-literal=jwt-secret="$(openssl rand -base64 64)" \
  --from-literal=encryption-key="$(openssl rand -base64 32)"

# Create image pull secrets
kubectl create secret docker-registry ghcr-secret \
  --namespace protein-sssl-prod \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN
```

#### External Secrets Operator (Recommended)
```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secretsmanager
  namespace: protein-sssl-prod
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        secretRef:
          accessKeyID:
            name: awssm-secret
            key: access-key
          secretAccessKey:
            name: awssm-secret
            key: secret-access-key
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: protein-sssl-external-secrets
  namespace: protein-sssl-prod
spec:
  refreshInterval: 5m
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: protein-sssl-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-url
    remoteRef:
      key: protein-sssl/prod/database-url
  - secretKey: jwt-secret
    remoteRef:
      key: protein-sssl/prod/jwt-secret
```

## Environment Setup

### Production Environment Configuration

#### Environment Variables
```bash
# Core application settings
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export DEBUG=false

# Resource allocation
export MAX_WORKERS=32
export BATCH_SIZE=64
export MEMORY_LIMIT=64Gi
export GPU_MEMORY_FRACTION=0.9

# Performance tuning
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=all
export PYTHONPATH=/app

# Security settings
export SECURE_COOKIES=true
export FORCE_HTTPS=true
export CONTENT_SECURITY_POLICY=true

# Monitoring and observability
export METRICS_ENABLED=true
export TRACING_ENABLED=true
export PROFILING_ENABLED=false
```

#### ConfigMap Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: protein-sssl-config
  namespace: protein-sssl-prod
data:
  application.yaml: |
    environment: production
    logging:
      level: INFO
      format: json
      structured: true
    
    model:
      cache_size: 10000
      batch_size: 64
      max_sequence_length: 2048
      gpu_acceleration: true
      mixed_precision: true
    
    performance:
      max_workers: 32
      connection_pool_size: 20
      keep_alive_timeout: 30
      request_timeout: 300
    
    security:
      force_https: true
      secure_cookies: true
      csrf_protection: true
      rate_limiting:
        enabled: true
        requests_per_minute: 1000
    
    monitoring:
      metrics_enabled: true
      health_check_interval: 30
      prometheus_port: 9090
      jaeger_enabled: true
```

### Database Setup

#### PostgreSQL Configuration (for metadata)
```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: protein-sssl-prod
spec:
  instances: 3
  primaryUpdateStrategy: unsupervised
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "2GB"
      effective_cache_size: "8GB"
      work_mem: "16MB"
      maintenance_work_mem: "512MB"
      checkpoint_timeout: "15min"
      wal_buffers: "16MB"
      max_wal_size: "4GB"
    
  bootstrap:
    initdb:
      database: protein_sssl
      owner: protein_sssl_user
      secret:
        name: postgres-credentials
  
  storage:
    size: 500Gi
    storageClass: protein-sssl-fast
  
  monitoring:
    enabled: true
```

#### Redis Configuration (for caching)
```yaml
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: Redis
metadata:
  name: redis-cluster
  namespace: protein-sssl-prod
spec:
  kubernetesConfig:
    image: redis:7.0.11-alpine
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: 2
        memory: 4Gi
  
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: protein-sssl-fast
        resources:
          requests:
            storage: 100Gi
  
  redisConfig:
    maxmemory: 3gb
    maxmemory-policy: allkeys-lru
    save: "900 1 300 10 60 10000"
    appendonly: yes
    appendfsync: everysec
```

## Production Deployment

### Single-Region Production Deployment

#### Step 1: Prepare the Environment
```bash
# Set deployment variables
export ENVIRONMENT=production
export NAMESPACE=protein-sssl-prod
export VERSION=v1.0.0
export REGISTRY=ghcr.io/terragonlabs
export DOMAIN=protein-sssl.your-domain.com

# Create production namespace
kubectl create namespace $NAMESPACE
kubectl label namespace $NAMESPACE environment=production
kubectl label namespace $NAMESPACE compliance-required=true
```

#### Step 2: Deploy Dependencies
```bash
# Install monitoring stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword="$(openssl rand -base64 32)"

# Install ingress controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.metrics.enabled=true \
  --set controller.podAnnotations."prometheus\.io/scrape"=true
```

#### Step 3: Deploy Application
```bash
# Deploy using Helm
helm upgrade --install protein-sssl ./helm/protein-sssl \
  --namespace $NAMESPACE \
  --values ./helm/protein-sssl/values-production.yaml \
  --set global.environment=production \
  --set image.repository=$REGISTRY/protein-sssl-operator \
  --set image.tag=$VERSION \
  --set ingress.hosts[0].host=$DOMAIN \
  --set resources.inference.requests.memory=16Gi \
  --set resources.inference.requests.cpu=4 \
  --set resources.inference.limits.memory=32Gi \
  --set resources.inference.limits.cpu=8 \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=3 \
  --set autoscaling.maxReplicas=20 \
  --wait \
  --timeout=20m
```

#### Step 4: Configure Production Values
Create `values-production.yaml`:
```yaml
global:
  environment: production
  region: us-east-1
  compliance:
    frameworks: ["soc2", "iso27001", "hipaa"]
    dataClassification: "confidential"
    auditLogging: true

replicaCount:
  inference: 5
  training: 1

image:
  repository: ghcr.io/terragonlabs/protein-sssl-operator
  tag: v1.0.0
  pullPolicy: IfNotPresent

resources:
  inference:
    requests:
      cpu: 4
      memory: 16Gi
      nvidia.com/gpu: 1
    limits:
      cpu: 8
      memory: 32Gi
      nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
  prometheusRule:
    enabled: true

security:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop: ["ALL"]

networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
      ports:
      - protocol: TCP
        port: 8000
  egress:
    - to: []
      ports:
      - protocol: TCP
        port: 443
      - protocol: TCP
        port: 5432
      - protocol: TCP
        port: 6379
```

### GPU-Optimized Deployment

#### GPU Node Configuration
```bash
# Label GPU nodes
kubectl label nodes gpu-node-1 accelerator=nvidia-a100
kubectl label nodes gpu-node-2 accelerator=nvidia-a100

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

#### GPU-Specific Helm Values
```yaml
nodeSelector:
  accelerator: nvidia-a100

tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule

resources:
  inference:
    requests:
      nvidia.com/gpu: 1
    limits:
      nvidia.com/gpu: 1

env:
- name: CUDA_VISIBLE_DEVICES
  value: "all"
- name: NVIDIA_VISIBLE_DEVICES
  value: "all"
- name: NVIDIA_DRIVER_CAPABILITIES
  value: "compute,utility"
```

## Multi-Region Deployment

### Global Multi-Region Setup

#### Step 1: Region Configuration
```bash
# Deploy to primary region (US East)
export PRIMARY_REGION=us-east-1
export PRIMARY_CLUSTER=protein-sssl-us-east-1

# Deploy to secondary regions
export SECONDARY_REGIONS=("eu-west-1" "asia-southeast-1")
export EU_CLUSTER=protein-sssl-eu-west-1
export ASIA_CLUSTER=protein-sssl-asia-southeast-1
```

#### Step 2: Cross-Region Networking
```bash
# Set up cross-region VPC peering (AWS example)
aws ec2 create-vpc-peering-connection \
  --vpc-id vpc-12345678 \
  --peer-vpc-id vpc-87654321 \
  --peer-region eu-west-1

# Configure DNS resolution across regions
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: protein-sssl-global
  annotations:
    external-dns.alpha.kubernetes.io/hostname: protein-sssl.global.your-domain.com
spec:
  type: ExternalName
  externalName: protein-sssl.us-east-1.your-domain.com
EOF
```

#### Step 3: Data Replication Setup
```yaml
# Cross-region database replication
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-replica-eu
  namespace: protein-sssl-prod
spec:
  instances: 3
  
  bootstrap:
    pg_basebackup:
      source: postgres-cluster-us
  
  externalClusters:
  - name: postgres-cluster-us
    connectionParameters:
      host: postgres.us-east-1.your-domain.com
      user: postgres
      dbname: protein_sssl
    password:
      name: postgres-us-credentials
      key: password
```

#### Step 4: Global Load Balancing
```yaml
# Global HTTP(S) Load Balancer configuration
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: protein-sssl-global-ssl
spec:
  domains:
  - protein-sssl.global.your-domain.com
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: protein-sssl-global
  annotations:
    kubernetes.io/ingress.global-static-ip-name: protein-sssl-global-ip
    networking.gke.io/managed-certificates: protein-sssl-global-ssl
    kubernetes.io/ingress.class: gce
spec:
  rules:
  - host: protein-sssl.global.your-domain.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: protein-sssl-backend
            port:
              number: 80
```

## Compliance Configuration

### GDPR Compliance Setup

#### Data Processing Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gdpr-config
data:
  gdpr_settings.yaml: |
    data_protection:
      purpose_limitation: true
      data_minimization: true
      accuracy_requirements: true
      storage_limitation: true
      
    consent_management:
      explicit_consent_required: true
      consent_withdrawal_enabled: true
      consent_audit_trail: true
      
    data_subject_rights:
      access_request_response_time: "72h"
      rectification_enabled: true
      erasure_enabled: true
      portability_enabled: true
      objection_handling: true
      
    data_retention:
      default_retention_period: "365d"
      automatic_deletion: true
      legal_hold_support: true
      
    breach_notification:
      supervisory_authority_notification: "72h"
      data_subject_notification: "without_undue_delay"
      internal_escalation: "immediate"
```

### HIPAA Compliance Setup

#### Security Controls Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hipaa-config
data:
  hipaa_settings.yaml: |
    access_controls:
      unique_user_identification: true
      automatic_logoff: true
      encryption_decryption: true
      
    audit_controls:
      audit_log_generation: true
      audit_log_protection: true
      audit_log_review: true
      
    integrity:
      data_integrity_controls: true
      transmission_integrity: true
      
    person_authentication:
      multi_factor_authentication: true
      password_complexity: true
      account_lockout: true
      
    transmission_security:
      end_to_end_encryption: true
      network_controls: true
      integrity_controls: true
```

### SOC 2 Compliance Setup

#### Control Framework Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: soc2-config
data:
  soc2_settings.yaml: |
    security:
      access_controls: true
      authorization: true
      vulnerability_management: true
      network_security: true
      
    availability:
      system_monitoring: true
      incident_response: true
      backup_recovery: true
      capacity_planning: true
      
    processing_integrity:
      data_validation: true
      error_handling: true
      processing_controls: true
      
    confidentiality:
      data_classification: true
      encryption: true
      secure_disposal: true
      
    privacy:
      notice_choice: true
      collection_use: true
      disclosure_notification: true
      data_quality: true
```

## Performance Optimization

### CPU Optimization

#### CPU Affinity and Isolation
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: protein-sssl
    resources:
      requests:
        cpu: 8
        memory: 32Gi
      limits:
        cpu: 16
        memory: 64Gi
    env:
    - name: OMP_NUM_THREADS
      value: "8"
    - name: MKL_NUM_THREADS
      value: "8"
    - name: NUMEXPR_NUM_THREADS
      value: "8"
  nodeSelector:
    node.kubernetes.io/instance-type: c5.4xlarge
  tolerations:
  - key: dedicated
    value: compute-optimized
    effect: NoSchedule
```

#### NUMA Topology Optimization
```bash
# Configure NUMA awareness
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: numa-config
data:
  numa-policy: |
    topology_manager_policy: single-numa-node
    cpu_manager_policy: static
    memory_manager_policy: Static
EOF
```

### Memory Optimization

#### Memory Allocation Strategy
```yaml
# HugePages configuration
apiVersion: v1
kind: Node
spec:
  allocatable:
    hugepages-2Mi: 4Gi
    hugepages-1Gi: 8Gi
---
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: protein-sssl
    resources:
      requests:
        hugepages-2Mi: 2Gi
        memory: 30Gi
      limits:
        hugepages-2Mi: 2Gi
        memory: 32Gi
    env:
    - name: MALLOC_ARENA_MAX
      value: "4"
    - name: MALLOC_MMAP_THRESHOLD_
      value: "131072"
```

### Storage Optimization

#### High-Performance Storage Configuration
```yaml
# NVMe-optimized storage class
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nvme-high-performance
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "20000"
  throughput: "1000"
  fsType: ext4
  encrypted: "true"
mountOptions:
- noatime
- nodiratime
- nobarrier
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

#### Cache Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cache-config
data:
  cache_settings.yaml: |
    model_cache:
      size: "20Gi"
      eviction_policy: "lru"
      persistence: true
      
    prediction_cache:
      size: "10Gi"
      ttl: "1h"
      compression: true
      
    dataset_cache:
      size: "50Gi"
      preload: true
      memory_mapping: true
```

## Monitoring Setup

### Prometheus Configuration

#### Metrics Collection Setup
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: protein-sssl-metrics
spec:
  selector:
    matchLabels:
      app: protein-sssl
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    scrapeTimeout: 10s
  - port: model-metrics
    interval: 60s
    path: /model/metrics
    scrapeTimeout: 30s
```

#### Custom Metrics Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 30s
      evaluation_interval: 30s
      
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    scrape_configs:
    - job_name: 'protein-sssl'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: protein-sssl
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### Grafana Dashboard Configuration

#### Performance Dashboard
```json
{
  "dashboard": {
    "title": "Protein-SSSL Production Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job='protein-sssl'}[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job='protein-sssl'}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "Model Prediction Accuracy",
        "type": "singlestat",
        "targets": [
          {
            "expr": "avg(model_prediction_accuracy)",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ]
  }
}
```

### Alerting Configuration

#### Critical Alerts
```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: protein-sssl-alerts
spec:
  groups:
  - name: protein-sssl.rules
    rules:
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} requests/second"
        
    - alert: HighMemoryUsage
      expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High memory usage"
        description: "Memory usage is {{ $value | humanizePercentage }}"
        
    - alert: ModelPredictionAccuracyDrop
      expr: model_prediction_accuracy < 0.85
      for: 15m
      labels:
        severity: critical
      annotations:
        summary: "Model prediction accuracy dropped"
        description: "Accuracy is {{ $value | humanizePercentage }}"
        
    - alert: GPUUtilizationHigh
      expr: nvidia_gpu_utilization_gpu > 95
      for: 20m
      labels:
        severity: warning
      annotations:
        summary: "GPU utilization very high"
        description: "GPU utilization is {{ $value }}%"
```

## Validation and Testing

### Deployment Validation

#### Health Check Validation
```bash
# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=protein-sssl \
  -n $NAMESPACE \
  --timeout=600s

# Check service endpoints
kubectl get endpoints protein-sssl -n $NAMESPACE

# Test health endpoint
kubectl run health-test --rm -i --restart=Never \
  --image=curlimages/curl \
  -- curl -f http://protein-sssl.$NAMESPACE.svc.cluster.local:8000/health
```

#### Functional Testing
```bash
# Test prediction endpoint
kubectl run prediction-test --rm -i --restart=Never \
  --image=curlimages/curl \
  -- curl -X POST \
  http://protein-sssl.$NAMESPACE.svc.cluster.local:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV", "return_confidence": true}'

# Test batch prediction endpoint
kubectl run batch-test --rm -i --restart=Never \
  --image=curlimages/curl \
  -- curl -X POST \
  http://protein-sssl.$NAMESPACE.svc.cluster.local:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["MKFL", "ACDE"], "batch_size": 2}'
```

### Performance Testing

#### Load Testing with k6
```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 10 },
    { duration: '5m', target: 50 },
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 0 },
  ],
};

export default function() {
  let payload = {
    sequence: 'MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV',
    return_confidence: true
  };
  
  let response = http.post(
    'https://protein-sssl.your-domain.com/predict',
    JSON.stringify(payload),
    { headers: { 'Content-Type': 'application/json' } }
  );
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 5s': (r) => r.timings.duration < 5000,
    'has prediction': (r) => JSON.parse(r.body).coordinates !== undefined,
  });
  
  sleep(1);
}
```

```bash
# Run load test
k6 run --out prometheus load-test.js
```

### Security Testing

#### Security Scan with Trivy
```bash
# Scan container images
trivy image ghcr.io/terragonlabs/protein-sssl-operator:v1.0.0

# Scan Kubernetes manifests
trivy config ./helm/protein-sssl/

# Scan for secrets
trivy fs --security-checks secret ./
```

#### Network Policy Testing
```bash
# Test network isolation
kubectl run network-test --rm -i --restart=Never \
  --image=busybox \
  -- nc -zv protein-sssl.$NAMESPACE.svc.cluster.local 8000

# Verify external access is blocked
kubectl run external-test --rm -i --restart=Never \
  --image=busybox \
  -- nc -zv google.com 80
```

## Post-Deployment Checklist

### Security Verification
- [ ] TLS certificates properly configured and valid
- [ ] Network policies restricting unnecessary traffic
- [ ] Pod security standards enforced
- [ ] Secrets properly encrypted and rotated
- [ ] RBAC permissions follow principle of least privilege
- [ ] Container images scanned and free of critical vulnerabilities
- [ ] Security monitoring and alerting configured

### Performance Verification
- [ ] Resource requests and limits properly configured
- [ ] Horizontal Pod Autoscaler (HPA) functioning correctly
- [ ] GPU resources properly allocated and utilized
- [ ] Storage performance meets requirements
- [ ] Network latency within acceptable bounds
- [ ] Cache hit rates optimized
- [ ] Memory usage patterns stable

### Monitoring and Observability
- [ ] All critical metrics being collected
- [ ] Dashboards displaying relevant information
- [ ] Alerting rules configured and tested
- [ ] Log aggregation and analysis working
- [ ] Distributed tracing functional
- [ ] Health checks responding correctly
- [ ] Performance baselines established

### Compliance and Legal
- [ ] Required compliance frameworks active
- [ ] Data retention policies implemented
- [ ] Audit logging configured
- [ ] Privacy controls functional
- [ ] Export control screening active
- [ ] Data sovereignty requirements met
- [ ] Breach notification procedures tested

### Operational Readiness
- [ ] Backup and recovery procedures tested
- [ ] Disaster recovery plan documented
- [ ] Runbook documentation complete
- [ ] On-call procedures established
- [ ] Incident response plan tested
- [ ] Scaling procedures documented
- [ ] Update and rollback procedures tested

### Documentation and Training
- [ ] Deployment documentation updated
- [ ] Operations runbooks current
- [ ] Team training completed
- [ ] Emergency contact information current
- [ ] Escalation procedures documented
- [ ] Knowledge base updated
- [ ] User documentation published

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod status and events
kubectl describe pod <pod-name> -n $NAMESPACE

# Check logs for startup errors
kubectl logs <pod-name> -n $NAMESPACE --previous

# Check resource constraints
kubectl top pods -n $NAMESPACE
kubectl describe nodes
```

#### Performance Issues
```bash
# Check resource utilization
kubectl top pods -n $NAMESPACE
kubectl top nodes

# Check HPA status
kubectl describe hpa protein-sssl -n $NAMESPACE

# Check for CPU throttling
kubectl exec <pod-name> -n $NAMESPACE -- cat /sys/fs/cgroup/cpu/cpu.stat
```

#### Network Connectivity Issues
```bash
# Test service DNS resolution
kubectl run dns-test --rm -i --restart=Never \
  --image=busybox \
  -- nslookup protein-sssl.$NAMESPACE.svc.cluster.local

# Check network policies
kubectl describe networkpolicy -n $NAMESPACE

# Test ingress connectivity
curl -v https://$DOMAIN/health
```

### Emergency Procedures

#### Rapid Scaling for High Load
```bash
# Immediate scale up
kubectl scale deployment protein-sssl --replicas=20 -n $NAMESPACE

# Monitor scaling progress
kubectl rollout status deployment protein-sssl -n $NAMESPACE

# Adjust HPA if needed
kubectl patch hpa protein-sssl -n $NAMESPACE \
  --patch '{"spec":{"maxReplicas":50}}'
```

#### Emergency Rollback
```bash
# Check rollout history
kubectl rollout history deployment protein-sssl -n $NAMESPACE

# Rollback to previous version
kubectl rollout undo deployment protein-sssl -n $NAMESPACE

# Monitor rollback progress
kubectl rollout status deployment protein-sssl -n $NAMESPACE
```

#### Service Degradation Response
```bash
# Enable circuit breaker mode
kubectl patch configmap protein-sssl-config -n $NAMESPACE \
  --patch '{"data":{"circuit_breaker_enabled":"true"}}'

# Restart pods to pick up config change
kubectl rollout restart deployment protein-sssl -n $NAMESPACE

# Monitor service recovery
kubectl logs -l app=protein-sssl -n $NAMESPACE -f
```

## Support and Contacts

### Technical Support
- **Primary**: support@terragonlabs.ai
- **Emergency**: +1-800-PROTEIN (24/7)
- **Slack**: #protein-sssl-production
- **Documentation**: https://docs.protein-sssl.terragonlabs.ai

### Escalation Matrix
1. **L1 Support**: Basic issues, configuration help
2. **L2 Support**: Performance issues, complex debugging
3. **L3 Support**: Architecture issues, critical failures
4. **Engineering**: Code issues, security incidents

### Monitoring and Alerting Contacts
- **Operations Team**: ops@terragonlabs.ai
- **Security Team**: security@terragonlabs.ai
- **Compliance Team**: compliance@terragonlabs.ai
- **PagerDuty**: protein-sssl-production service

---

This production deployment guide provides comprehensive instructions for deploying the protein-sssl-operator in enterprise production environments. Follow all sections carefully and customize configurations based on your specific requirements and compliance needs.