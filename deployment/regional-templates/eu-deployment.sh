#!/bin/bash

# EU Regional Deployment Template for protein-sssl-operator
# Optimized for GDPR compliance and European cultural requirements

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOBAL_SCRIPT="$(dirname "$SCRIPT_DIR")/global-deployment.sh"

# EU-specific configuration
EU_REGIONS="eu-west-1,eu-central-1,eu-north-1"
EU_COMPLIANCE="gdpr,soc2,iso27001"
EU_LANGUAGES="en,fr,de,es,it,nl"  # Major European languages
EU_CULTURAL="western_europe"
EU_LOCALE="en-GB"  # Using GB locale for European deployment

# Default environment
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-protein-sssl-eu}"
NAMESPACE="${NAMESPACE:-protein-sssl-eu}"

echo "=== EU Regional Deployment Template ==="
echo "Regions: $EU_REGIONS"
echo "Compliance: $EU_COMPLIANCE"
echo "Languages: $EU_LANGUAGES"
echo "Environment: $ENVIRONMENT"
echo ""

# Execute global deployment script with EU-specific parameters
exec "$GLOBAL_SCRIPT" \
    --name "$DEPLOYMENT_NAME" \
    --namespace "$NAMESPACE" \
    --environment "$ENVIRONMENT" \
    --regions "$EU_REGIONS" \
    --compliance "$EU_COMPLIANCE" \
    --languages "$EU_LANGUAGES" \
    --cultural "$EU_CULTURAL" \
    --locale "$EU_LOCALE" \
    --multi-region \
    --cross-region-replication \
    --data-sovereignty "strict_local" \
    --export-control \
    --audit-logging \
    --accessibility \
    --security-level "restricted" \
    --data-classification "confidential" \
    "$@"