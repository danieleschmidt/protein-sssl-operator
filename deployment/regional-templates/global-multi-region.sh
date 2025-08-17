#!/bin/bash

# Global Multi-Region Deployment Template for protein-sssl-operator
# Deploys across all major regions with comprehensive compliance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOBAL_SCRIPT="$(dirname "$SCRIPT_DIR")/global-deployment.sh"

# Global multi-region configuration
GLOBAL_REGIONS="us-east-1,eu-west-1,asia-southeast-1"
GLOBAL_COMPLIANCE="gdpr,ccpa,pdpa,soc2,iso27001"
GLOBAL_LANGUAGES="en,es,fr,de,ja,zh,ar,pt,it,ru"  # Major world languages
GLOBAL_CULTURAL="western_europe,north_america,east_asia,southeast_asia"
GLOBAL_LOCALE="en-US"

# Default environment
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-protein-sssl-global}"
NAMESPACE="${NAMESPACE:-protein-sssl-global}"

echo "=== Global Multi-Region Deployment Template ==="
echo "Regions: $GLOBAL_REGIONS"
echo "Compliance: $GLOBAL_COMPLIANCE"
echo "Languages: $GLOBAL_LANGUAGES"
echo "Environment: $ENVIRONMENT"
echo ""
echo "This deployment will create a globally distributed system with:"
echo "- Full GDPR, CCPA, and PDPA compliance"
echo "- Multi-language support for major world languages"
echo "- Cultural adaptation for key regions"
echo "- Cross-region data replication with strict data sovereignty"
echo "- Comprehensive accessibility features"
echo ""

# Execute global deployment script with comprehensive parameters
exec "$GLOBAL_SCRIPT" \
    --name "$DEPLOYMENT_NAME" \
    --namespace "$NAMESPACE" \
    --environment "$ENVIRONMENT" \
    --regions "$GLOBAL_REGIONS" \
    --compliance "$GLOBAL_COMPLIANCE" \
    --languages "$GLOBAL_LANGUAGES" \
    --cultural "$GLOBAL_CULTURAL" \
    --locale "$GLOBAL_LOCALE" \
    --multi-region \
    --cross-region-replication \
    --data-sovereignty "strict_local" \
    --export-control \
    --audit-logging \
    --accessibility \
    --security-level "restricted" \
    --data-classification "confidential" \
    "$@"