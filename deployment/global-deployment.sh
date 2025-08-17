#!/bin/bash

# Global Deployment Script for protein-sssl-operator
# Supports multi-region deployment with compliance, accessibility, and cultural adaptation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HELM_CHART_PATH="$PROJECT_ROOT/helm/protein-sssl"

# Default values
DEFAULT_DEPLOYMENT_NAME="protein-sssl"
DEFAULT_NAMESPACE="protein-sssl"
DEFAULT_ENVIRONMENT="production"
DEFAULT_REGION="us-east-1"
DEFAULT_COMPLIANCE_FRAMEWORKS="gdpr,ccpa,soc2"
DEFAULT_LANGUAGES="en,es,fr,de"

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

# Usage function
usage() {
    cat << EOF
Global Deployment Script for protein-sssl-operator

Usage: $0 [OPTIONS]

Options:
    -n, --name NAME                 Deployment name (default: $DEFAULT_DEPLOYMENT_NAME)
    -N, --namespace NAMESPACE       Kubernetes namespace (default: $DEFAULT_NAMESPACE)
    -e, --environment ENV           Environment (development|staging|production) (default: $DEFAULT_ENVIRONMENT)
    -r, --region REGION             Primary region (default: $DEFAULT_REGION)
    -R, --regions REGIONS           Comma-separated list of regions (default: primary region only)
    -c, --compliance FRAMEWORKS    Comma-separated compliance frameworks (default: $DEFAULT_COMPLIANCE_FRAMEWORKS)
    -l, --languages LANGUAGES      Comma-separated supported languages (default: $DEFAULT_LANGUAGES)
    -L, --locale LOCALE             Default locale (default: en-US)
    -a, --accessibility             Enable accessibility features (default: enabled)
    -C, --cultural REGIONS          Comma-separated cultural regions (default: auto-detect)
    -s, --security-level LEVEL      Security level (baseline|restricted) (default: restricted)
    -d, --data-classification CLASS Data classification (public|internal|confidential|restricted) (default: confidential)
    -m, --multi-region              Enable multi-region deployment (default: auto-detect)
    -x, --cross-region-replication  Enable cross-region replication (default: false)
    -D, --data-sovereignty LEVEL   Data sovereignty level (none|prefer_local|strict_local) (default: strict_local)
    -E, --export-control            Enable export control screening (default: true)
    -A, --audit-logging             Enable audit logging (default: true)
    -v, --values-file FILE          Additional Helm values file
    -k, --kubeconfig CONFIG         Kubeconfig file path
    -t, --timeout TIMEOUT          Deployment timeout in seconds (default: 600)
    -y, --yes                       Skip confirmation prompts
    -V, --validate-only             Validate configuration without deploying
    -h, --help                      Show this help message

Examples:
    # Basic deployment in US East
    $0 --region us-east-1

    # Multi-region deployment with GDPR compliance
    $0 --regions us-east-1,eu-west-1 --compliance gdpr,ccpa --multi-region

    # High-security deployment for healthcare
    $0 --compliance hipaa,soc2,iso27001 --data-classification restricted --export-control

    # Development deployment with cultural adaptation
    $0 --environment development --cultural western_europe,north_america --languages en,fr,de

    # Accessibility-focused deployment
    $0 --accessibility --languages en,es,fr,de,ar --cultural western_europe,middle_east

    # Validate configuration only
    $0 --validate-only --regions us-east-1,eu-west-1,asia-southeast-1

Regional Templates:
    # North America (US + Canada)
    $0 --regions us-east-1,us-west-2,canada-central --compliance ccpa,pipeda --cultural north_america

    # Europe (GDPR compliance)
    $0 --regions eu-west-1,eu-central-1 --compliance gdpr --cultural western_europe --languages en,fr,de,es,it

    # Asia Pacific
    $0 --regions asia-southeast-1,asia-northeast-1 --compliance pdpa --cultural east_asia,southeast_asia --languages en,ja,zh,ko

    # Global deployment
    $0 --regions us-east-1,eu-west-1,asia-southeast-1 --compliance gdpr,ccpa,pdpa,soc2 --multi-region --cross-region-replication

EOF
}

# Parse command line arguments
parse_args() {
    DEPLOYMENT_NAME="$DEFAULT_DEPLOYMENT_NAME"
    NAMESPACE="$DEFAULT_NAMESPACE"
    ENVIRONMENT="$DEFAULT_ENVIRONMENT"
    PRIMARY_REGION="$DEFAULT_REGION"
    REGIONS="$DEFAULT_REGION"
    COMPLIANCE_FRAMEWORKS="$DEFAULT_COMPLIANCE_FRAMEWORKS"
    SUPPORTED_LANGUAGES="$DEFAULT_LANGUAGES"
    DEFAULT_LOCALE="en-US"
    ACCESSIBILITY_ENABLED="true"
    CULTURAL_REGIONS=""
    SECURITY_LEVEL="restricted"
    DATA_CLASSIFICATION="confidential"
    MULTI_REGION="auto"
    CROSS_REGION_REPLICATION="false"
    DATA_SOVEREIGNTY="strict_local"
    EXPORT_CONTROL="true"
    AUDIT_LOGGING="true"
    VALUES_FILE=""
    KUBECONFIG=""
    TIMEOUT="600"
    SKIP_CONFIRMATION="false"
    VALIDATE_ONLY="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--name)
                DEPLOYMENT_NAME="$2"
                shift 2
                ;;
            -N|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                PRIMARY_REGION="$2"
                REGIONS="$2"
                shift 2
                ;;
            -R|--regions)
                REGIONS="$2"
                shift 2
                ;;
            -c|--compliance)
                COMPLIANCE_FRAMEWORKS="$2"
                shift 2
                ;;
            -l|--languages)
                SUPPORTED_LANGUAGES="$2"
                shift 2
                ;;
            -L|--locale)
                DEFAULT_LOCALE="$2"
                shift 2
                ;;
            -a|--accessibility)
                ACCESSIBILITY_ENABLED="true"
                shift
                ;;
            -C|--cultural)
                CULTURAL_REGIONS="$2"
                shift 2
                ;;
            -s|--security-level)
                SECURITY_LEVEL="$2"
                shift 2
                ;;
            -d|--data-classification)
                DATA_CLASSIFICATION="$2"
                shift 2
                ;;
            -m|--multi-region)
                MULTI_REGION="true"
                shift
                ;;
            -x|--cross-region-replication)
                CROSS_REGION_REPLICATION="true"
                shift
                ;;
            -D|--data-sovereignty)
                DATA_SOVEREIGNTY="$2"
                shift 2
                ;;
            -E|--export-control)
                EXPORT_CONTROL="true"
                shift
                ;;
            -A|--audit-logging)
                AUDIT_LOGGING="true"
                shift
                ;;
            -v|--values-file)
                VALUES_FILE="$2"
                shift 2
                ;;
            -k|--kubeconfig)
                KUBECONFIG="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -y|--yes)
                SKIP_CONFIRMATION="true"
                shift
                ;;
            -V|--validate-only)
                VALIDATE_ONLY="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Extract primary region from regions list
    PRIMARY_REGION=$(echo "$REGIONS" | cut -d',' -f1)

    # Auto-detect multi-region
    if [[ "$MULTI_REGION" == "auto" ]]; then
        if [[ "$REGIONS" =~ "," ]]; then
            MULTI_REGION="true"
        else
            MULTI_REGION="false"
        fi
    fi

    # Auto-detect cultural regions based on deployment regions
    if [[ -z "$CULTURAL_REGIONS" ]]; then
        CULTURAL_REGIONS=$(auto_detect_cultural_regions "$REGIONS")
    fi
}

# Auto-detect cultural regions based on deployment regions
auto_detect_cultural_regions() {
    local regions="$1"
    local cultural_regions=""

    if [[ "$regions" =~ (us-|ca-) ]]; then
        cultural_regions="${cultural_regions},north_america"
    fi

    if [[ "$regions" =~ eu- ]]; then
        cultural_regions="${cultural_regions},western_europe"
    fi

    if [[ "$regions" =~ (asia-northeast|asia-east) ]]; then
        cultural_regions="${cultural_regions},east_asia"
    fi

    if [[ "$regions" =~ asia-southeast ]]; then
        cultural_regions="${cultural_regions},southeast_asia"
    fi

    if [[ "$regions" =~ asia-south ]]; then
        cultural_regions="${cultural_regions},south_asia"
    fi

    if [[ "$regions" =~ middle-east ]]; then
        cultural_regions="${cultural_regions},middle_east"
    fi

    if [[ "$regions" =~ (south-america|latin-america) ]]; then
        cultural_regions="${cultural_regions},latin_america"
    fi

    # Remove leading comma
    echo "${cultural_regions#,}"
}

# Validate configuration
validate_config() {
    log_info "Validating deployment configuration..."

    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be development, staging, or production."
        return 1
    fi

    # Validate security level
    if [[ ! "$SECURITY_LEVEL" =~ ^(baseline|restricted)$ ]]; then
        log_error "Invalid security level: $SECURITY_LEVEL. Must be baseline or restricted."
        return 1
    fi

    # Validate data classification
    if [[ ! "$DATA_CLASSIFICATION" =~ ^(public|internal|confidential|restricted)$ ]]; then
        log_error "Invalid data classification: $DATA_CLASSIFICATION. Must be public, internal, confidential, or restricted."
        return 1
    fi

    # Validate data sovereignty
    if [[ ! "$DATA_SOVEREIGNTY" =~ ^(none|prefer_local|strict_local)$ ]]; then
        log_error "Invalid data sovereignty level: $DATA_SOVEREIGNTY. Must be none, prefer_local, or strict_local."
        return 1
    fi

    # Validate compliance frameworks
    local valid_frameworks="gdpr,ccpa,hipaa,pdpa,soc2,iso27001,lgpd,pipeda"
    IFS=',' read -ra FRAMEWORKS <<< "$COMPLIANCE_FRAMEWORKS"
    for framework in "${FRAMEWORKS[@]}"; do
        if [[ ! ",$valid_frameworks," =~ ,$framework, ]]; then
            log_error "Invalid compliance framework: $framework. Valid frameworks: $valid_frameworks"
            return 1
        fi
    done

    # Validate regions
    local valid_regions="us-east-1,us-west-2,canada-central,eu-west-1,eu-central-1,eu-north-1,uk-south,asia-east-1,asia-southeast-1,asia-northeast-1,asia-south-1,australia-east,south-america-east,africa-south,middle-east-1"
    IFS=',' read -ra REGION_LIST <<< "$REGIONS"
    for region in "${REGION_LIST[@]}"; do
        if [[ ! ",$valid_regions," =~ ,$region, ]]; then
            log_error "Invalid region: $region. Valid regions: $valid_regions"
            return 1
        fi
    done

    # Validate language codes
    local valid_languages="en,es,fr,de,ja,zh,ar,pt,it,ru,ko,hi,nl"
    IFS=',' read -ra LANGUAGE_LIST <<< "$SUPPORTED_LANGUAGES"
    for language in "${LANGUAGE_LIST[@]}"; do
        if [[ ! ",$valid_languages," =~ ,$language, ]]; then
            log_error "Invalid language code: $language. Valid languages: $valid_languages"
            return 1
        fi
    done

    # Check compliance-region alignment
    validate_compliance_region_alignment

    # Check cultural adaptation alignment
    validate_cultural_alignment

    log_success "Configuration validation passed"
    return 0
}

# Validate compliance framework and region alignment
validate_compliance_region_alignment() {
    IFS=',' read -ra REGION_LIST <<< "$REGIONS"
    IFS=',' read -ra FRAMEWORKS <<< "$COMPLIANCE_FRAMEWORKS"

    for region in "${REGION_LIST[@]}"; do
        case "$region" in
            eu-*|uk-*)
                if [[ ! " ${FRAMEWORKS[*]} " =~ " gdpr " ]]; then
                    log_warning "GDPR compliance recommended for EU/UK region: $region"
                fi
                ;;
            us-*|canada-*)
                if [[ ! " ${FRAMEWORKS[*]} " =~ " ccpa " ]] && [[ ! " ${FRAMEWORKS[*]} " =~ " pipeda " ]]; then
                    log_warning "CCPA or PIPEDA compliance recommended for North American region: $region"
                fi
                ;;
            asia-southeast-*)
                if [[ ! " ${FRAMEWORKS[*]} " =~ " pdpa " ]]; then
                    log_warning "PDPA compliance recommended for Southeast Asian region: $region"
                fi
                ;;
            south-america-*)
                if [[ ! " ${FRAMEWORKS[*]} " =~ " lgpd " ]]; then
                    log_warning "LGPD compliance recommended for South American region: $region"
                fi
                ;;
        esac
    done
}

# Validate cultural adaptation alignment
validate_cultural_alignment() {
    if [[ -n "$CULTURAL_REGIONS" ]]; then
        IFS=',' read -ra CULTURAL_LIST <<< "$CULTURAL_REGIONS"
        IFS=',' read -ra REGION_LIST <<< "$REGIONS"

        for cultural_region in "${CULTURAL_LIST[@]}"; do
            case "$cultural_region" in
                western_europe)
                    if [[ ! "$REGIONS" =~ eu- ]]; then
                        log_warning "Western Europe cultural adaptation specified but no EU regions targeted"
                    fi
                    ;;
                north_america)
                    if [[ ! "$REGIONS" =~ (us-|canada-) ]]; then
                        log_warning "North America cultural adaptation specified but no North American regions targeted"
                    fi
                    ;;
                east_asia)
                    if [[ ! "$REGIONS" =~ (asia-northeast|asia-east) ]]; then
                        log_warning "East Asia cultural adaptation specified but no East Asian regions targeted"
                    fi
                    ;;
            esac
        done
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        return 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is required but not installed"
        return 1
    fi

    # Check kubeconfig
    if [[ -n "$KUBECONFIG" ]]; then
        if [[ ! -f "$KUBECONFIG" ]]; then
            log_error "Kubeconfig file not found: $KUBECONFIG"
            return 1
        fi
        export KUBECONFIG="$KUBECONFIG"
    fi

    # Test cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi

    # Check Helm chart
    if [[ ! -f "$HELM_CHART_PATH/Chart.yaml" ]]; then
        log_error "Helm chart not found at: $HELM_CHART_PATH"
        return 1
    fi

    log_success "Prerequisites check passed"
    return 0
}

# Generate Helm values
generate_helm_values() {
    local values_file="$1"

    log_info "Generating Helm values..."

    # Convert regions string to YAML array
    local regions_yaml=""
    IFS=',' read -ra REGION_LIST <<< "$REGIONS"
    for i in "${!REGION_LIST[@]}"; do
        local region="${REGION_LIST[$i]}"
        local zone="secondary"
        if [[ "$region" == "$PRIMARY_REGION" ]]; then
            zone="primary"
        fi
        regions_yaml="${regions_yaml}      - name: \"$region\"
        zone: \"$zone\"
        dataResidency: true
"
    done

    # Convert compliance frameworks to YAML array
    local compliance_yaml=""
    IFS=',' read -ra FRAMEWORKS <<< "$COMPLIANCE_FRAMEWORKS"
    for framework in "${FRAMEWORKS[@]}"; do
        compliance_yaml="${compliance_yaml}      - \"$framework\"
"
    done

    # Convert languages to YAML array
    local languages_yaml=""
    IFS=',' read -ra LANGUAGE_LIST <<< "$SUPPORTED_LANGUAGES"
    for language in "${LANGUAGE_LIST[@]}"; do
        languages_yaml="${languages_yaml}      - \"$language\"
"
    done

    # Convert cultural regions to YAML array
    local cultural_yaml=""
    if [[ -n "$CULTURAL_REGIONS" ]]; then
        IFS=',' read -ra CULTURAL_LIST <<< "$CULTURAL_REGIONS"
        for cultural_region in "${CULTURAL_LIST[@]}"; do
            cultural_yaml="${cultural_yaml}      - \"$cultural_region\"
"
        done
    fi

    cat > "$values_file" << EOF
# Generated Helm values for global deployment
global:
  region: "$PRIMARY_REGION"
  zone: "primary"
  environment: "$ENVIRONMENT"
  deploymentMode: "active"
  
  multiRegion:
    enabled: $MULTI_REGION
    primaryRegion: "$PRIMARY_REGION"
    regions:
$regions_yaml
    crossRegionReplication: $CROSS_REGION_REPLICATION
    dataSovereignty: "$DATA_SOVEREIGNTY"
  
  compliance:
    frameworks:
$compliance_yaml
    dataClassification: "$DATA_CLASSIFICATION"
    auditLogging: $AUDIT_LOGGING
    dataRetention:
      enabled: true
      defaultPeriodDays: 365
      autoDelete: true
    exportControl:
      enabled: $EXPORT_CONTROL
      screening: true
    
  i18n:
    enabled: true
    defaultLanguage: "$(echo "$SUPPORTED_LANGUAGES" | cut -d',' -f1)"
    supportedLanguages:
$languages_yaml
    defaultLocale: "$DEFAULT_LOCALE"
    rtlSupport: true
    
  accessibility:
    enabled: $ACCESSIBILITY_ENABLED
    wcagLevel: "AA"
    screenReaderSupport: true
    keyboardNavigation: true
    highContrast: true
    
  cultural:
    enabled: $(if [[ -n "$CULTURAL_REGIONS" ]]; then echo "true"; else echo "false"; fi)
    regions:
$cultural_yaml
    scientificNotation: "decimal_point"
    adaptColors: true
    adaptCommunication: true
    
  security:
    encryption:
      atRest: true
      inTransit: true
      keyRotation: true
    networkPolicies: true
    podSecurityStandards: "$SECURITY_LEVEL"

# Environment-specific overrides
$(if [[ "$ENVIRONMENT" == "development" ]]; then cat << DEV_EOF
replicaCount:
  training: 1
  inference: 1

resources:
  training:
    limits:
      cpu: "2"
      memory: 8Gi
    requests:
      cpu: "1"
      memory: 4Gi
  inference:
    limits:
      cpu: "1"
      memory: 4Gi
    requests:
      cpu: "500m"
      memory: 2Gi

autoscaling:
  enabled: false
DEV_EOF
elif [[ "$ENVIRONMENT" == "staging" ]]; then cat << STAGING_EOF
replicaCount:
  training: 1
  inference: 2

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
STAGING_EOF
fi)

# Security-level specific settings
$(if [[ "$SECURITY_LEVEL" == "restricted" ]]; then cat << RESTRICTED_EOF
podSecurityContext:
  runAsNonRoot: true
  runAsUser: 65534
  runAsGroup: 65534
  fsGroup: 65534
  seccompProfile:
    type: RuntimeDefault

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 65534
  seccompProfile:
    type: RuntimeDefault
RESTRICTED_EOF
fi)
EOF

    log_success "Generated Helm values at: $values_file"
}

# Display deployment summary
display_summary() {
    cat << EOF

$(echo -e "${BLUE}========================================")
$(echo -e "  GLOBAL DEPLOYMENT SUMMARY")
$(echo -e "========================================${NC}")

Deployment Name:     $DEPLOYMENT_NAME
Namespace:          $NAMESPACE
Environment:        $ENVIRONMENT
Primary Region:     $PRIMARY_REGION
All Regions:        $REGIONS
Multi-Region:       $MULTI_REGION

Compliance:         $COMPLIANCE_FRAMEWORKS
Data Classification: $DATA_CLASSIFICATION
Data Sovereignty:   $DATA_SOVEREIGNTY
Export Control:     $EXPORT_CONTROL
Audit Logging:      $AUDIT_LOGGING

Languages:          $SUPPORTED_LANGUAGES
Default Locale:     $DEFAULT_LOCALE
Accessibility:      $ACCESSIBILITY_ENABLED
Cultural Regions:   ${CULTURAL_REGIONS:-"auto-detected"}

Security Level:     $SECURITY_LEVEL
Cross-Region Sync:  $CROSS_REGION_REPLICATION
Timeout:           ${TIMEOUT}s

EOF
}

# Confirm deployment
confirm_deployment() {
    if [[ "$SKIP_CONFIRMATION" == "true" ]]; then
        return 0
    fi

    echo -n "Proceed with deployment? [y/N]: "
    read -r response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            log_info "Deployment cancelled by user"
            exit 0
            ;;
    esac
}

# Deploy with Helm
deploy_helm() {
    local values_file="$1"

    log_info "Starting Helm deployment..."

    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Label namespace for compliance and regional requirements
    kubectl label namespace "$NAMESPACE" \
        "global.deployment/region=$PRIMARY_REGION" \
        "global.deployment/environment=$ENVIRONMENT" \
        "compliance.frameworks=$COMPLIANCE_FRAMEWORKS" \
        --overwrite

    # Prepare Helm command
    local helm_cmd="helm upgrade --install $DEPLOYMENT_NAME $HELM_CHART_PATH"
    helm_cmd="$helm_cmd --namespace $NAMESPACE"
    helm_cmd="$helm_cmd --values $values_file"
    helm_cmd="$helm_cmd --timeout ${TIMEOUT}s"
    helm_cmd="$helm_cmd --wait"

    # Add additional values file if specified
    if [[ -n "$VALUES_FILE" ]]; then
        if [[ -f "$VALUES_FILE" ]]; then
            helm_cmd="$helm_cmd --values $VALUES_FILE"
        else
            log_error "Additional values file not found: $VALUES_FILE"
            return 1
        fi
    fi

    # Execute Helm deployment
    log_info "Executing: $helm_cmd"
    if eval "$helm_cmd"; then
        log_success "Helm deployment completed successfully"
    else
        log_error "Helm deployment failed"
        return 1
    fi

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    if kubectl wait --for=condition=available deployment \
        --selector="app.kubernetes.io/name=$DEPLOYMENT_NAME" \
        --namespace="$NAMESPACE" \
        --timeout="${TIMEOUT}s"; then
        log_success "Deployment is ready"
    else
        log_error "Deployment failed to become ready within timeout"
        return 1
    fi

    return 0
}

# Post-deployment verification
verify_deployment() {
    log_info "Verifying deployment..."

    # Check pods
    local pods_ready
    pods_ready=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$DEPLOYMENT_NAME" --no-headers | grep -c "Running" || true)
    if [[ "$pods_ready" -gt 0 ]]; then
        log_success "Pods are running: $pods_ready"
    else
        log_warning "No pods are running"
    fi

    # Check services
    local services
    services=$(kubectl get services -n "$NAMESPACE" -l "app.kubernetes.io/name=$DEPLOYMENT_NAME" --no-headers | wc -l || true)
    if [[ "$services" -gt 0 ]]; then
        log_success "Services created: $services"
    else
        log_warning "No services found"
    fi

    # Check compliance resources
    local compliance_configs
    compliance_configs=$(kubectl get configmaps -n "$NAMESPACE" | grep -c "compliance\|data-retention\|audit" || true)
    if [[ "$compliance_configs" -gt 0 ]]; then
        log_success "Compliance configurations deployed: $compliance_configs"
    else
        log_warning "No compliance configurations found"
    fi

    # Check multi-region resources
    if [[ "$MULTI_REGION" == "true" ]]; then
        local region_configs
        region_configs=$(kubectl get configmaps -n "$NAMESPACE" | grep -c "multi-region\|regional\|failover" || true)
        if [[ "$region_configs" -gt 0 ]]; then
            log_success "Multi-region configurations deployed: $region_configs"
        else
            log_warning "No multi-region configurations found"
        fi
    fi

    log_success "Deployment verification completed"
}

# Display post-deployment information
display_post_deployment_info() {
    cat << EOF

$(echo -e "${GREEN}========================================")
$(echo -e "  DEPLOYMENT COMPLETED SUCCESSFULLY")
$(echo -e "========================================${NC}")

Access Information:
  Namespace: $NAMESPACE
  Primary Region: $PRIMARY_REGION
  
To check deployment status:
  kubectl get pods -n $NAMESPACE
  kubectl get services -n $NAMESPACE
  kubectl get configmaps -n $NAMESPACE

To view logs:
  kubectl logs -n $NAMESPACE -l app.kubernetes.io/name=$DEPLOYMENT_NAME

To access compliance dashboard:
  kubectl port-forward -n $NAMESPACE svc/$DEPLOYMENT_NAME-compliance 8080:80

To check multi-region status:
  kubectl get configmap $DEPLOYMENT_NAME-multi-region-config -n $NAMESPACE -o yaml

To uninstall:
  helm uninstall $DEPLOYMENT_NAME -n $NAMESPACE

EOF
}

# Cleanup function
cleanup() {
    local temp_values_file="$1"
    if [[ -f "$temp_values_file" ]]; then
        rm -f "$temp_values_file"
    fi
}

# Main function
main() {
    # Parse arguments
    parse_args "$@"

    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi

    # Validate configuration
    if ! validate_config; then
        exit 1
    fi

    # Display summary
    display_summary

    # Exit if validation only
    if [[ "$VALIDATE_ONLY" == "true" ]]; then
        log_success "Configuration validation completed successfully"
        exit 0
    fi

    # Confirm deployment
    confirm_deployment

    # Generate temporary values file
    local temp_values_file
    temp_values_file=$(mktemp)
    trap "cleanup '$temp_values_file'" EXIT

    generate_helm_values "$temp_values_file"

    # Deploy
    if deploy_helm "$temp_values_file"; then
        verify_deployment
        display_post_deployment_info
        log_success "Global deployment completed successfully!"
    else
        log_error "Deployment failed"
        exit 1
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi