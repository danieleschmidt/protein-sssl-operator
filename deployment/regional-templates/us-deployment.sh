#!/bin/bash

# US Regional Deployment Template for protein-sssl-operator
# Optimized for US compliance and cultural requirements

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOBAL_SCRIPT="$(dirname "$SCRIPT_DIR")/global-deployment.sh"

# US-specific configuration
US_REGIONS="us-east-1,us-west-2"
US_COMPLIANCE="ccpa,soc2,iso27001,hipaa"
US_LANGUAGES="en,es"  # English and Spanish for US market
US_CULTURAL="north_america"
US_LOCALE="en-US"

# Default environment
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-protein-sssl-us}"
NAMESPACE="${NAMESPACE:-protein-sssl-us}"

echo "=== US Regional Deployment Template ==="
echo "Regions: $US_REGIONS"
echo "Compliance: $US_COMPLIANCE"
echo "Languages: $US_LANGUAGES"
echo "Environment: $ENVIRONMENT"
echo ""

# Execute global deployment script with US-specific parameters
exec "$GLOBAL_SCRIPT" \
    --name "$DEPLOYMENT_NAME" \
    --namespace "$NAMESPACE" \
    --environment "$ENVIRONMENT" \
    --regions "$US_REGIONS" \
    --compliance "$US_COMPLIANCE" \
    --languages "$US_LANGUAGES" \
    --cultural "$US_CULTURAL" \
    --locale "$US_LOCALE" \
    --multi-region \
    --cross-region-replication \
    --data-sovereignty "prefer_local" \
    --export-control \
    --audit-logging \
    --accessibility \
    --security-level "restricted" \
    --data-classification "confidential" \
    "$@"