# Operations & Maintenance Guide

## Overview

This comprehensive guide provides detailed procedures for operating and maintaining the protein-sssl-operator in production environments. It covers monitoring, alerting, backup and recovery, scaling strategies, troubleshooting, and performance optimization.

## Table of Contents

1. [System Monitoring](#system-monitoring)
2. [Alerting and Notification](#alerting-and-notification)
3. [Backup and Recovery](#backup-and-recovery)
4. [Scaling Strategies](#scaling-strategies)
5. [Performance Optimization](#performance-optimization)
6. [Security Operations](#security-operations)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Incident Response](#incident-response)
10. [Capacity Planning](#capacity-planning)
11. [Automation and Runbooks](#automation-and-runbooks)

## System Monitoring

### Core Metrics Overview

#### Application Metrics
| Metric | Description | Alert Threshold | Critical Threshold |
|--------|-------------|-----------------|-------------------|
| Request Rate | Requests per second | > 1000 req/s | > 2000 req/s |
| Response Time | 95th percentile latency | > 2s | > 5s |
| Error Rate | 5xx errors percentage | > 1% | > 5% |
| Model Accuracy | Prediction accuracy | < 90% | < 85% |
| GPU Utilization | GPU usage percentage | > 85% | > 95% |
| Memory Usage | Memory consumption | > 80% | > 90% |
| CPU Usage | CPU utilization | > 70% | > 85% |
| Queue Depth | Pending requests | > 100 | > 500 |

#### Infrastructure Metrics
| Metric | Description | Alert Threshold | Critical Threshold |
|--------|-------------|-----------------|-------------------|
| Node CPU | Node CPU usage | > 80% | > 90% |
| Node Memory | Node memory usage | > 85% | > 95% |
| Disk Usage | Storage utilization | > 80% | > 90% |
| Network I/O | Network throughput | > 80% capacity | > 95% capacity |
| Pod Restarts | Pod restart count | > 5/hour | > 10/hour |
| API Server | Kubernetes API latency | > 500ms | > 1000ms |

### Monitoring Stack Configuration

#### Prometheus Setup
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 30s
      evaluation_interval: 30s
      external_labels:
        cluster: 'protein-sssl-prod'
        region: 'us-east-1'
    
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    scrape_configs:
    # Application metrics
    - job_name: 'protein-sssl-app'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['protein-sssl-prod']
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
    
    # GPU metrics
    - job_name: 'nvidia-dcgm'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: nvidia-dcgm-exporter
    
    # Node metrics
    - job_name: 'node-exporter'
      kubernetes_sd_configs:
      - role: node
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/${1}/proxy/metrics
    
    # Kubernetes metrics
    - job_name: 'kube-state-metrics'
      static_configs:
      - targets: ['kube-state-metrics:8080']
    
    # Custom business metrics
    - job_name: 'protein-sssl-business'
      static_configs:
      - targets: ['protein-sssl-metrics:9091']
      scrape_interval: 60s
```

#### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "id": null,
    "title": "Protein-SSSL Operations Dashboard",
    "tags": ["protein-sssl", "production", "monitoring"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "Service Overview",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 1},
        "targets": [
          {
            "expr": "rate(http_requests_total{job='protein-sssl-app'}[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "thresholds": [
          {"value": 1000, "colorMode": "critical", "op": "gt"},
          {"value": 2000, "colorMode": "critical", "op": "gt"}
        ]
      },
      {
        "id": 3,
        "title": "Response Time (95th percentile)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 1},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job='protein-sssl-app'}[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job='protein-sssl-app'}[5m]))",
            "legendFormat": "99th percentile"
          }
        ],
        "yAxes": [{"unit": "s"}]
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "singlestat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 9},
        "targets": [
          {
            "expr": "rate(http_requests_total{job='protein-sssl-app', status=~'5..'}[5m]) / rate(http_requests_total{job='protein-sssl-app'}[5m]) * 100"
          }
        ],
        "thresholds": "1,5",
        "colorBackground": true,
        "format": "percent"
      },
      {
        "id": 5,
        "title": "Active Pods",
        "type": "singlestat",
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 9},
        "targets": [
          {
            "expr": "count(up{job='protein-sssl-app'} == 1)"
          }
        ]
      },
      {
        "id": 6,
        "title": "Model Accuracy",
        "type": "gauge",
        "gridPos": {"h": 4, "w": 6, "x": 12, "y": 9},
        "targets": [
          {
            "expr": "avg(model_prediction_accuracy{job='protein-sssl-app'})"
          }
        ],
        "thresholds": {
          "steps": [
            {"color": "red", "value": 0},
            {"color": "yellow", "value": 0.85},
            {"color": "green", "value": 0.90}
          ]
        }
      },
      {
        "id": 7,
        "title": "Resource Usage",
        "type": "row",
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": 13}
      },
      {
        "id": 8,
        "title": "CPU Usage",
        "type": "graph",
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 14},
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{pod=~'protein-sssl-.*'}[5m]) * 100",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "id": 9,
        "title": "Memory Usage",
        "type": "graph",
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 14},
        "targets": [
          {
            "expr": "container_memory_usage_bytes{pod=~'protein-sssl-.*'} / container_spec_memory_limit_bytes * 100",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "id": 10,
        "title": "GPU Utilization",
        "type": "graph",
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 14},
        "targets": [
          {
            "expr": "DCGM_FI_DEV_GPU_UTIL",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      }
    ],
    "refresh": "30s",
    "time": {"from": "now-1h", "to": "now"}
  }
}
```

#### Custom Metrics Collection

##### Application Metrics Instrumentation
```python
# protein_sssl/utils/metrics.py
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from functools import wraps

# Initialize metrics registry
registry = CollectorRegistry()

# Request metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    registry=registry
)

# Model metrics
model_prediction_accuracy = Gauge(
    'model_prediction_accuracy',
    'Model prediction accuracy',
    ['model_version'],
    registry=registry
)

model_inference_time = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_type', 'sequence_length_bucket'],
    registry=registry
)

gpu_memory_usage = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage',
    ['gpu_id'],
    registry=registry
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    registry=registry
)

queue_depth = Gauge(
    'prediction_queue_depth',
    'Number of pending predictions',
    registry=registry
)

def track_request_metrics(func):
    """Decorator to track request metrics"""
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        start_time = time.time()
        method = request.method
        endpoint = request.url.path
        
        try:
            response = func(request, *args, **kwargs)
            status = str(response.status_code)
            return response
        except Exception as e:
            status = "500"
            raise
        finally:
            duration = time.time() - start_time
            request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    return wrapper

def update_system_metrics():
    """Update system-level metrics"""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    
    # GPU metrics (if available)
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_usage.labels(gpu_id=str(i)).set(memory_info.used)
    except ImportError:
        pass

def get_metrics():
    """Get current metrics in Prometheus format"""
    update_system_metrics()
    return generate_latest(registry)
```

### Log Management

#### Structured Logging Configuration
```python
# protein_sssl/utils/logging_config.py
import logging
import json
import os
from datetime import datetime
from pythonjsonlogger import jsonlogger

class ProteinSSLFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['service'] = 'protein-sssl-operator'
        log_record['version'] = os.getenv('APP_VERSION', 'unknown')
        log_record['environment'] = os.getenv('ENVIRONMENT', 'development')
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id

def setup_logging():
    """Configure structured logging"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create formatters
    json_formatter = ProteinSSLFormatter(
        '%(timestamp)s %(level)s %(logger)s %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for local development
    if os.getenv('ENVIRONMENT') == 'development':
        file_handler = logging.FileHandler('app.log')
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Usage in application code
logger = setup_logging()

# Example usage
logger.info(
    "Prediction completed", 
    extra={
        'request_id': '12345',
        'sequence_length': 256,
        'inference_time': 1.5,
        'confidence_score': 0.92
    }
)
```

#### Log Aggregation with Fluentd
```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/protein-sssl-*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_key timestamp
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
    </filter>
    
    <filter kubernetes.**>
      @type parser
      format json
      key_name log
      reserve_data true
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name protein-sssl-logs
      type_name _doc
      include_tag_key true
      tag_key @log_name
      
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        retry_type exponential_backoff
        flush_thread_count 2
        flush_interval 5s
        retry_forever
        retry_max_interval 30
        chunk_limit_size 2M
        queue_limit_length 8
        overflow_action block
      </buffer>
    </match>
```

## Alerting and Notification

### Alert Rules Configuration

#### Critical Alerts
```yaml
# prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: protein-sssl-critical-alerts
  namespace: monitoring
spec:
  groups:
  - name: protein-sssl.critical
    interval: 30s
    rules:
    - alert: ServiceDown
      expr: up{job="protein-sssl-app"} == 0
      for: 1m
      labels:
        severity: critical
        team: sre
        service: protein-sssl
      annotations:
        summary: "Protein-SSSL service is down"
        description: "Service {{ $labels.instance }} has been down for more than 1 minute"
        runbook_url: "https://docs.protein-sssl.com/runbooks/service-down"
        
    - alert: HighErrorRate
      expr: rate(http_requests_total{job="protein-sssl-app", status=~"5.."}[5m]) / rate(http_requests_total{job="protein-sssl-app"}[5m]) * 100 > 5
      for: 5m
      labels:
        severity: critical
        team: sre
        service: protein-sssl
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
        runbook_url: "https://docs.protein-sssl.com/runbooks/high-error-rate"
        
    - alert: ModelAccuracyDegraded
      expr: avg(model_prediction_accuracy{job="protein-sssl-app"}) < 0.85
      for: 15m
      labels:
        severity: critical
        team: ml-ops
        service: protein-sssl
      annotations:
        summary: "Model prediction accuracy severely degraded"
        description: "Model accuracy is {{ $value | humanizePercentage }}, below critical threshold"
        runbook_url: "https://docs.protein-sssl.com/runbooks/model-accuracy-degraded"
        
    - alert: OutOfMemory
      expr: container_memory_usage_bytes{pod=~"protein-sssl-.*"} / container_spec_memory_limit_bytes > 0.95
      for: 2m
      labels:
        severity: critical
        team: sre
        service: protein-sssl
      annotations:
        summary: "Container running out of memory"
        description: "Pod {{ $labels.pod }} memory usage is {{ $value | humanizePercentage }}"
        runbook_url: "https://docs.protein-sssl.com/runbooks/out-of-memory"
```

#### Warning Alerts
```yaml
# prometheus-warning-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: protein-sssl-warning-alerts
  namespace: monitoring
spec:
  groups:
  - name: protein-sssl.warning
    interval: 60s
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="protein-sssl-app"}[5m])) > 2
      for: 10m
      labels:
        severity: warning
        team: sre
        service: protein-sssl
      annotations:
        summary: "High request latency"
        description: "95th percentile latency is {{ $value }}s"
        
    - alert: PodRestartingFrequently
      expr: rate(kube_pod_container_status_restarts_total{pod=~"protein-sssl-.*"}[1h]) > 0.1
      for: 5m
      labels:
        severity: warning
        team: sre
        service: protein-sssl
      annotations:
        summary: "Pod restarting frequently"
        description: "Pod {{ $labels.pod }} is restarting {{ $value }} times per hour"
        
    - alert: HighCPUUsage
      expr: rate(container_cpu_usage_seconds_total{pod=~"protein-sssl-.*"}[5m]) * 100 > 80
      for: 15m
      labels:
        severity: warning
        team: sre
        service: protein-sssl
      annotations:
        summary: "High CPU usage"
        description: "CPU usage is {{ $value }}% for pod {{ $labels.pod }}"
        
    - alert: LowDiskSpace
      expr: (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes * 100) < 20
      for: 5m
      labels:
        severity: warning
        team: sre
        service: protein-sssl
      annotations:
        summary: "Low disk space"
        description: "Disk space is {{ $value }}% on {{ $labels.instance }}"
```

### Notification Channels

#### Slack Integration
```yaml
# alertmanager-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
data:
  alertmanager.yml: |
    global:
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    
    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'default'
      routes:
      - match:
          severity: critical
        receiver: 'critical-alerts'
        continue: true
      - match:
          severity: warning
        receiver: 'warning-alerts'
        continue: true
      - match:
          team: ml-ops
        receiver: 'ml-ops-team'
        continue: true
    
    receivers:
    - name: 'default'
      slack_configs:
      - channel: '#protein-sssl-alerts'
        title: 'Protein-SSSL Alert'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Severity:* {{ .Labels.severity }}
          *Service:* {{ .Labels.service }}
          {{ end }}
    
    - name: 'critical-alerts'
      slack_configs:
      - channel: '#protein-sssl-critical'
        title: '🚨 CRITICAL: Protein-SSSL Alert'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Runbook:* {{ .Annotations.runbook_url }}
          {{ end }}
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .Annotations.summary }}'
    
    - name: 'warning-alerts'
      slack_configs:
      - channel: '#protein-sssl-warnings'
        title: '⚠️ WARNING: Protein-SSSL Alert'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          {{ end }}
    
    - name: 'ml-ops-team'
      email_configs:
      - to: 'ml-ops@your-company.com'
        subject: 'ML Model Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          Time: {{ .StartsAt }}
          {{ end }}
```

#### PagerDuty Integration
```yaml
# pagerduty-service.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pagerduty-config
type: Opaque
stringData:
  service_key: "YOUR_PAGERDUTY_INTEGRATION_KEY"
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: pagerduty-routing
data:
  routing.yml: |
    critical_services:
      - protein-sssl-production
      - protein-sssl-api
    
    escalation_policy:
      - level: 1
        timeout: 5m
        targets:
          - sre-oncall-primary
      - level: 2
        timeout: 10m
        targets:
          - sre-oncall-secondary
          - engineering-manager
      - level: 3
        timeout: 15m
        targets:
          - director-engineering
```

## Backup and Recovery

### Data Backup Strategy

#### Database Backup Configuration
```yaml
# postgres-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: protein-sssl-prod
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:14-alpine
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-credentials
                  key: password
            - name: PGHOST
              value: "postgres.protein-sssl-prod.svc.cluster.local"
            - name: PGUSER
              value: "protein_sssl_user"
            - name: PGDATABASE
              value: "protein_sssl"
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: backup-credentials
                  key: access_key_id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: backup-credentials
                  key: secret_access_key
            command:
            - /bin/sh
            - -c
            - |
              BACKUP_FILE="protein_sssl_$(date +%Y%m%d_%H%M%S).sql"
              echo "Creating backup: $BACKUP_FILE"
              
              # Create database dump
              pg_dump -h $PGHOST -U $PGUSER -d $PGDATABASE > /tmp/$BACKUP_FILE
              
              # Compress backup
              gzip /tmp/$BACKUP_FILE
              
              # Upload to S3
              apk add --no-cache aws-cli
              aws s3 cp /tmp/${BACKUP_FILE}.gz s3://protein-sssl-backups/database/$(date +%Y/%m/%d)/${BACKUP_FILE}.gz
              
              # Verify backup
              if aws s3 ls s3://protein-sssl-backups/database/$(date +%Y/%m/%d)/${BACKUP_FILE}.gz; then
                echo "Backup successful: ${BACKUP_FILE}.gz"
              else
                echo "Backup failed!" >&2
                exit 1
              fi
              
              # Cleanup local files
              rm /tmp/${BACKUP_FILE}.gz
              
              # Delete old backups (keep 30 days)
              aws s3 ls s3://protein-sssl-backups/database/ --recursive | \
                awk '$1 < "'$(date -d '30 days ago' '+%Y-%m-%d')'" {print $4}' | \
                xargs -I {} aws s3 rm s3://protein-sssl-backups/{}
            
            volumeMounts:
            - name: backup-storage
              mountPath: /tmp
          
          volumes:
          - name: backup-storage
            emptyDir: {}
          
          restartPolicy: OnFailure
```

#### Model and Configuration Backup
```yaml
# model-backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-backup
  namespace: protein-sssl-prod
spec:
  schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: model-backup
            image: ghcr.io/terragonlabs/protein-sssl-operator:v1.0.0
            env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: backup-credentials
                  key: access_key_id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: backup-credentials
                  key: secret_access_key
            command:
            - /bin/sh
            - -c
            - |
              BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
              BACKUP_DIR="/tmp/model_backup_$BACKUP_DATE"
              
              echo "Creating model backup: $BACKUP_DIR"
              mkdir -p $BACKUP_DIR
              
              # Backup model files
              cp -r /app/models/* $BACKUP_DIR/
              
              # Backup configuration
              cp -r /app/configs/* $BACKUP_DIR/
              
              # Create metadata file
              cat > $BACKUP_DIR/metadata.json << EOF
              {
                "backup_date": "$BACKUP_DATE",
                "model_version": "$MODEL_VERSION",
                "config_version": "$CONFIG_VERSION",
                "backup_type": "weekly_full"
              }
              EOF
              
              # Create tarball
              cd /tmp
              tar -czf model_backup_$BACKUP_DATE.tar.gz model_backup_$BACKUP_DATE/
              
              # Upload to S3
              aws s3 cp model_backup_$BACKUP_DATE.tar.gz s3://protein-sssl-backups/models/$(date +%Y/%m/)model_backup_$BACKUP_DATE.tar.gz
              
              # Verify upload
              if aws s3 ls s3://protein-sssl-backups/models/$(date +%Y/%m/)model_backup_$BACKUP_DATE.tar.gz; then
                echo "Model backup successful"
              else
                echo "Model backup failed!" >&2
                exit 1
              fi
              
              # Cleanup
              rm -rf /tmp/model_backup_*
            
            volumeMounts:
            - name: model-storage
              mountPath: /app/models
              readOnly: true
            - name: config-storage
              mountPath: /app/configs
              readOnly: true
          
          volumes:
          - name: model-storage
            persistentVolumeClaim:
              claimName: model-storage-pvc
          - name: config-storage
            configMap:
              name: protein-sssl-config
          
          restartPolicy: OnFailure
```

### Disaster Recovery Procedures

#### Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Component | RTO | RPO | Recovery Strategy |
|-----------|-----|-----|------------------|
| API Service | 15 minutes | 5 minutes | Multi-region deployment with automated failover |
| Database | 30 minutes | 1 hour | Point-in-time recovery from automated backups |
| Model Storage | 1 hour | 24 hours | Restore from S3 backup with model re-deployment |
| Configuration | 5 minutes | 1 hour | GitOps-based configuration recovery |
| User Data | 2 hours | 4 hours | Cross-region database replication |

#### Automated Disaster Recovery
```bash
#!/bin/bash
# disaster-recovery.sh

set -euo pipefail

RECOVERY_TYPE="${1:-full}"
BACKUP_DATE="${2:-latest}"
TARGET_REGION="${3:-us-west-2}"

echo "Starting disaster recovery: $RECOVERY_TYPE from $BACKUP_DATE to $TARGET_REGION"

# 1. Setup recovery environment
setup_recovery_environment() {
    echo "Setting up recovery environment in $TARGET_REGION"
    
    # Create namespace
    kubectl create namespace protein-sssl-recovery || true
    
    # Apply RBAC
    kubectl apply -f kubernetes/rbac.yaml -n protein-sssl-recovery
    
    # Create secrets
    kubectl create secret generic recovery-secrets \
        --from-literal=database-url="$RECOVERY_DB_URL" \
        --from-literal=backup-access-key="$BACKUP_ACCESS_KEY" \
        -n protein-sssl-recovery
}

# 2. Restore database
restore_database() {
    echo "Restoring database from backup dated $BACKUP_DATE"
    
    if [ "$BACKUP_DATE" = "latest" ]; then
        BACKUP_FILE=$(aws s3 ls s3://protein-sssl-backups/database/ --recursive | sort | tail -n 1 | awk '{print $4}')
    else
        BACKUP_FILE="database/${BACKUP_DATE}/protein_sssl_${BACKUP_DATE}.sql.gz"
    fi
    
    # Download backup
    aws s3 cp s3://protein-sssl-backups/$BACKUP_FILE /tmp/restore.sql.gz
    
    # Restore database
    gunzip /tmp/restore.sql.gz
    PGPASSWORD="$RECOVERY_DB_PASSWORD" psql -h "$RECOVERY_DB_HOST" -U "$RECOVERY_DB_USER" -d "$RECOVERY_DB_NAME" < /tmp/restore.sql
    
    echo "Database restoration completed"
}

# 3. Restore models
restore_models() {
    echo "Restoring models from backup"
    
    if [ "$BACKUP_DATE" = "latest" ]; then
        BACKUP_FILE=$(aws s3 ls s3://protein-sssl-backups/models/ --recursive | sort | tail -n 1 | awk '{print $4}')
    else
        BACKUP_FILE="models/${BACKUP_DATE}/model_backup_${BACKUP_DATE}.tar.gz"
    fi
    
    # Download and extract models
    aws s3 cp s3://protein-sssl-backups/$BACKUP_FILE /tmp/models.tar.gz
    mkdir -p /tmp/models
    tar -xzf /tmp/models.tar.gz -C /tmp/models
    
    # Create PVC and copy models
    kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: recovered-models-pvc
  namespace: protein-sssl-recovery
spec:
  accessModes: [ReadWriteMany]
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 500Gi
EOF
    
    # Copy models to PVC
    kubectl run model-restore --rm -i --restart=Never \
        --image=busybox \
        --overrides='{"spec":{"containers":[{"name":"model-restore","image":"busybox","command":["sh","-c","sleep 3600"],"volumeMounts":[{"name":"models","mountPath":"/models"}]}],"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"recovered-models-pvc"}}]}}' \
        -n protein-sssl-recovery &
    
    sleep 10
    kubectl cp /tmp/models/ protein-sssl-recovery/model-restore:/models/
    kubectl delete pod model-restore -n protein-sssl-recovery
    
    echo "Model restoration completed"
}

# 4. Deploy application
deploy_application() {
    echo "Deploying application in recovery mode"
    
    helm upgrade --install protein-sssl-recovery ./helm/protein-sssl \
        --namespace protein-sssl-recovery \
        --values ./helm/protein-sssl/values-recovery.yaml \
        --set global.environment=recovery \
        --set global.region="$TARGET_REGION" \
        --set persistence.models.existingClaim=recovered-models-pvc \
        --wait \
        --timeout=20m
}

# 5. Verify recovery
verify_recovery() {
    echo "Verifying recovery"
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=protein-sssl \
        -n protein-sssl-recovery \
        --timeout=600s
    
    # Test health endpoint
    kubectl run recovery-test --rm -i --restart=Never \
        --image=curlimages/curl \
        -n protein-sssl-recovery \
        -- curl -f http://protein-sssl.protein-sssl-recovery.svc.cluster.local:8000/health
    
    # Test prediction endpoint
    kubectl run prediction-test --rm -i --restart=Never \
        --image=curlimages/curl \
        -n protein-sssl-recovery \
        -- curl -X POST \
        http://protein-sssl.protein-sssl-recovery.svc.cluster.local:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"sequence": "MKFLKFSLLTAV", "return_confidence": true}'
    
    echo "Recovery verification completed successfully"
}

# Execute recovery based on type
case $RECOVERY_TYPE in
    "full")
        setup_recovery_environment
        restore_database
        restore_models
        deploy_application
        verify_recovery
        ;;
    "database")
        restore_database
        ;;
    "models")
        restore_models
        ;;
    "application")
        deploy_application
        verify_recovery
        ;;
    *)
        echo "Unknown recovery type: $RECOVERY_TYPE"
        echo "Usage: $0 [full|database|models|application] [backup_date] [target_region]"
        exit 1
        ;;
esac

echo "Disaster recovery completed successfully"
```

## Scaling Strategies

### Horizontal Pod Autoscaling (HPA)

#### HPA Configuration
```yaml
# hpa-config.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: protein-sssl-hpa
  namespace: protein-sssl-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: protein-sssl
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: prediction_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  - type: Object
    object:
      metric:
        name: http_requests_per_second
      describedObject:
        apiVersion: v1
        kind: Service
        name: protein-sssl
      target:
        type: Value
        value: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 5
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min
```

#### Custom Metrics for Scaling
```python
# protein_sssl/utils/custom_metrics.py
import time
from kubernetes import client, config
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

class CustomMetricsExporter:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.queue_depth = Gauge(
            'prediction_queue_depth',
            'Number of pending predictions',
            registry=self.registry
        )
        self.response_time = Gauge(
            'average_response_time',
            'Average response time',
            registry=self.registry
        )
        self.active_connections = Gauge(
            'active_connections_count',
            'Number of active connections',
            registry=self.registry
        )
    
    def export_metrics(self, queue_size, avg_response_time, connections):
        """Export custom metrics for HPA"""
        self.queue_depth.set(queue_size)
        self.response_time.set(avg_response_time)
        self.active_connections.set(connections)
        
        # Push to Prometheus pushgateway
        push_to_gateway(
            'prometheus-pushgateway:9091',
            job='protein-sssl-custom-metrics',
            registry=self.registry
        )

# Usage in application
metrics_exporter = CustomMetricsExporter()

def export_scaling_metrics():
    """Export metrics for autoscaling decisions"""
    queue_size = get_current_queue_size()
    avg_response_time = get_average_response_time()
    connections = get_active_connections()
    
    metrics_exporter.export_metrics(queue_size, avg_response_time, connections)
```

### Vertical Pod Autoscaling (VPA)

#### VPA Configuration
```yaml
# vpa-config.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: protein-sssl-vpa
  namespace: protein-sssl-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: protein-sssl
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: protein-sssl
      minAllowed:
        cpu: 500m
        memory: 2Gi
      maxAllowed:
        cpu: 8
        memory: 32Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
```

### Cluster Autoscaling

#### Cluster Autoscaler Configuration
```yaml
# cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.25.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/protein-sssl-cluster
        - --balance-similar-node-groups
        - --scale-down-enabled=true
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5
        - --max-node-provision-time=15m
        env:
        - name: AWS_REGION
          value: us-east-1
```

### Predictive Scaling

#### Machine Learning-Based Scaling
```python
# protein_sssl/utils/predictive_scaling.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime, timedelta

class PredictiveScaler:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_columns = [
            'hour_of_day', 'day_of_week', 'requests_per_minute',
            'avg_response_time', 'cpu_utilization', 'memory_utilization',
            'queue_depth', 'active_connections'
        ]
    
    def prepare_features(self, timestamp, metrics):
        """Prepare features for prediction"""
        dt = datetime.fromtimestamp(timestamp)
        
        features = {
            'hour_of_day': dt.hour,
            'day_of_week': dt.weekday(),
            'requests_per_minute': metrics.get('requests_per_minute', 0),
            'avg_response_time': metrics.get('avg_response_time', 0),
            'cpu_utilization': metrics.get('cpu_utilization', 0),
            'memory_utilization': metrics.get('memory_utilization', 0),
            'queue_depth': metrics.get('queue_depth', 0),
            'active_connections': metrics.get('active_connections', 0)
        }
        
        return pd.DataFrame([features])
    
    def train_model(self, historical_data):
        """Train the predictive model on historical data"""
        # Prepare training data
        X = historical_data[self.feature_columns]
        y = historical_data['required_replicas']
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate model
        predictions = self.model.predict(X)
        mae = mean_absolute_error(y, predictions)
        
        return {
            'model_accuracy': mae,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict_required_replicas(self, current_metrics, forecast_horizon=30):
        """Predict required replicas for the next forecast_horizon minutes"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_time = datetime.now().timestamp()
        
        for i in range(forecast_horizon):
            future_time = current_time + (i * 60)  # i minutes in the future
            features = self.prepare_features(future_time, current_metrics)
            prediction = self.model.predict(features)[0]
            predictions.append(max(1, int(prediction)))  # Minimum 1 replica
        
        return predictions
    
    def get_scaling_recommendation(self, current_replicas, current_metrics):
        """Get scaling recommendation based on prediction"""
        predictions = self.predict_required_replicas(current_metrics, 15)  # 15-minute forecast
        
        # Calculate target replicas (with safety margin)
        target_replicas = max(predictions) + 2  # Add 2 replica buffer
        
        # Calculate scaling decision
        if target_replicas > current_replicas * 1.2:  # Scale up if 20% increase needed
            return {
                'action': 'scale_up',
                'target_replicas': target_replicas,
                'confidence': self.calculate_confidence(predictions),
                'reason': f"Predicted peak demand of {max(predictions)} replicas"
            }
        elif target_replicas < current_replicas * 0.8:  # Scale down if 20% decrease possible
            return {
                'action': 'scale_down',
                'target_replicas': target_replicas,
                'confidence': self.calculate_confidence(predictions),
                'reason': f"Predicted low demand, max {max(predictions)} replicas needed"
            }
        else:
            return {
                'action': 'no_change',
                'target_replicas': current_replicas,
                'confidence': self.calculate_confidence(predictions),
                'reason': "Current replica count is optimal"
            }
    
    def calculate_confidence(self, predictions):
        """Calculate confidence in prediction based on variance"""
        variance = np.var(predictions)
        # High confidence if low variance
        confidence = max(0, min(1, 1 - (variance / 10)))
        return confidence

# Usage in scaling controller
scaler = PredictiveScaler()

def automated_scaling_loop():
    """Main scaling loop"""
    while True:
        try:
            # Get current metrics
            current_metrics = get_current_metrics()
            current_replicas = get_current_replica_count()
            
            # Get scaling recommendation
            recommendation = scaler.get_scaling_recommendation(
                current_replicas, current_metrics
            )
            
            # Apply scaling if confidence is high enough
            if recommendation['confidence'] > 0.7:
                if recommendation['action'] != 'no_change':
                    apply_scaling(recommendation['target_replicas'])
                    log_scaling_action(recommendation)
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in scaling loop: {e}")
            time.sleep(60)
```

---

*This operations and maintenance guide provides comprehensive procedures for managing the protein-sssl-operator in production. Regular review and updates of these procedures ensure optimal system performance and reliability.*