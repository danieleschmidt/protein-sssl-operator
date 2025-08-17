#!/bin/bash

# Asia Pacific Regional Deployment Template for protein-sssl-operator
# Optimized for PDPA compliance and Asian cultural requirements

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOBAL_SCRIPT="$(dirname "$SCRIPT_DIR")/global-deployment.sh"

# Asia-specific configuration
ASIA_REGIONS="asia-southeast-1,asia-northeast-1,asia-south-1"
ASIA_COMPLIANCE="pdpa,soc2,iso27001"
ASIA_LANGUAGES="en,ja,zh,ko,hi"  # Major Asian languages
ASIA_CULTURAL="east_asia,southeast_asia,south_asia"
ASIA_LOCALE="en-SG"  # Using Singapore locale

# Default environment
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-protein-sssl-asia}"
NAMESPACE="${NAMESPACE:-protein-sssl-asia}"

echo "=== Asia Pacific Regional Deployment Template ==="
echo "Regions: $ASIA_REGIONS"
echo "Compliance: $ASIA_COMPLIANCE"
echo "Languages: $ASIA_LANGUAGES"
echo "Environment: $ENVIRONMENT"
echo ""

# Execute global deployment script with Asia-specific parameters
exec "$GLOBAL_SCRIPT" \
    --name "$DEPLOYMENT_NAME" \
    --namespace "$NAMESPACE" \
    --environment "$ENVIRONMENT" \
    --regions "$ASIA_REGIONS" \
    --compliance "$ASIA_COMPLIANCE" \
    --languages "$ASIA_LANGUAGES" \
    --cultural "$ASIA_CULTURAL" \
    --locale "$ASIA_LOCALE" \
    --multi-region \
    --cross-region-replication \
    --data-sovereignty "strict_local" \
    --export-control \
    --audit-logging \
    --accessibility \
    --security-level "restricted" \
    --data-classification "confidential" \
    "$@"