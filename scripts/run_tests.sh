#!/bin/bash

# Comprehensive test runner for protein-sssl-operator
# Terragon Labs - Quality Gates Implementation

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="protein-sssl-operator"
TEST_DIR="tests"
REPORTS_DIR="test_reports"
COVERAGE_MIN=80
PERFORMANCE_TIMEOUT=600  # 10 minutes

# Test categories
UNIT_TESTS=("test_models.py" "test_data.py" "test_config.py" "test_cli.py")
INTEGRATION_TESTS=("test_integration.py")
SECURITY_TESTS=("test_security.py")
PERFORMANCE_TESTS=("test_performance.py" "test_monitoring.py")

# Create reports directory
mkdir -p "$REPORTS_DIR"

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

log_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log_section "CHECKING PREREQUISITES"
    
    # Check if pytest is available
    if ! command -v python3 -m pytest &> /dev/null; then
        log_error "pytest is not available. Installing..."
        pip3 install pytest pytest-cov pytest-xdist pytest-timeout
    fi
    
    # Check if coverage tools are available
    if ! python3 -c "import coverage" 2>/dev/null; then
        log_info "Installing coverage tools..."
        pip3 install coverage pytest-cov
    fi
    
    # Check if security tools are available
    if ! command -v bandit &> /dev/null; then
        log_info "Installing security scanning tools..."
        pip3 install bandit safety
    fi
    
    # Check if performance tools are available  
    if ! python3 -c "import psutil" 2>/dev/null; then
        log_info "Installing performance monitoring tools..."
        pip3 install psutil memory-profiler
    fi
    
    log_success "Prerequisites check completed"
}

# Run unit tests
run_unit_tests() {
    log_section "RUNNING UNIT TESTS"
    
    local failed_tests=()
    local total_tests=${#UNIT_TESTS[@]}
    local passed_tests=0
    
    for test_file in "${UNIT_TESTS[@]}"; do
        log_info "Running unit tests: $test_file"
        
        if python3 -m pytest "$TEST_DIR/$test_file" \
            --verbose \
            --tb=short \
            --junit-xml="$REPORTS_DIR/unit_${test_file%.py}.xml" \
            --cov=protein_sssl \
            --cov-report=html:"$REPORTS_DIR/coverage_${test_file%.py}" \
            --cov-report=term-missing \
            --timeout=120; then
            
            log_success "Unit tests passed: $test_file"
            ((passed_tests++))
        else
            log_error "Unit tests failed: $test_file"
            failed_tests+=("$test_file")
        fi
    done
    
    # Summary
    log_info "Unit tests summary: $passed_tests/$total_tests passed"
    
    if [ ${#failed_tests[@]} -gt 0 ]; then
        log_warning "Failed unit tests: ${failed_tests[*]}"
        return 1
    fi
    
    return 0
}

# Run integration tests
run_integration_tests() {
    log_section "RUNNING INTEGRATION TESTS"
    
    local failed_tests=()
    local total_tests=${#INTEGRATION_TESTS[@]}
    local passed_tests=0
    
    for test_file in "${INTEGRATION_TESTS[@]}"; do
        log_info "Running integration tests: $test_file"
        
        if python3 -m pytest "$TEST_DIR/$test_file" \
            --verbose \
            --tb=long \
            --junit-xml="$REPORTS_DIR/integration_${test_file%.py}.xml" \
            --timeout=300; then
            
            log_success "Integration tests passed: $test_file"
            ((passed_tests++))
        else
            log_error "Integration tests failed: $test_file"
            failed_tests+=("$test_file")
        fi
    done
    
    # Summary
    log_info "Integration tests summary: $passed_tests/$total_tests passed"
    
    if [ ${#failed_tests[@]} -gt 0 ]; then
        log_warning "Failed integration tests: ${failed_tests[*]}"
        return 1
    fi
    
    return 0
}

# Run security tests
run_security_tests() {
    log_section "RUNNING SECURITY TESTS"
    
    local security_passed=true
    
    # Run security-specific pytest tests
    log_info "Running security unit tests..."
    if python3 -m pytest "$TEST_DIR/test_security.py" \
        --verbose \
        --tb=short \
        --junit-xml="$REPORTS_DIR/security_tests.xml" \
        --timeout=180; then
        log_success "Security unit tests passed"
    else
        log_error "Security unit tests failed"
        security_passed=false
    fi
    
    # Run Bandit security linter
    log_info "Running Bandit security analysis..."
    if bandit -r protein_sssl/ -f json -o "$REPORTS_DIR/bandit_report.json" -ll; then
        log_success "Bandit security scan passed"
    else
        log_warning "Bandit found potential security issues (check $REPORTS_DIR/bandit_report.json)"
        # Don't fail on Bandit warnings, just log them
    fi
    
    # Run Safety check for known vulnerabilities
    log_info "Running Safety vulnerability check..."
    if safety check --json --output "$REPORTS_DIR/safety_report.json"; then
        log_success "Safety vulnerability check passed"
    else
        log_warning "Safety found potential vulnerabilities (check $REPORTS_DIR/safety_report.json)"
        # Don't fail on Safety warnings for now
    fi
    
    # Check for hardcoded secrets (simple grep-based check)
    log_info "Checking for potential hardcoded secrets..."
    if grep -r -i -E "(password|secret|key|token).*=.*['\"][^'\"]{8,}" protein_sssl/ > "$REPORTS_DIR/secrets_check.txt" 2>/dev/null; then
        log_warning "Potential hardcoded secrets found (check $REPORTS_DIR/secrets_check.txt)"
    else
        log_success "No obvious hardcoded secrets found"
    fi
    
    if [ "$security_passed" = true ]; then
        log_success "Security tests completed successfully"
        return 0
    else
        log_error "Security tests failed"
        return 1
    fi
}

# Run performance tests
run_performance_tests() {
    log_section "RUNNING PERFORMANCE TESTS"
    
    local failed_tests=()
    local total_tests=${#PERFORMANCE_TESTS[@]}
    local passed_tests=0
    
    for test_file in "${PERFORMANCE_TESTS[@]}"; do
        log_info "Running performance tests: $test_file"
        
        if timeout $PERFORMANCE_TIMEOUT python3 -m pytest "$TEST_DIR/$test_file" \
            --verbose \
            --tb=short \
            --junit-xml="$REPORTS_DIR/performance_${test_file%.py}.xml" \
            --timeout=300 \
            -m "not slow"; then  # Skip slow tests in regular runs
            
            log_success "Performance tests passed: $test_file"
            ((passed_tests++))
        else
            log_error "Performance tests failed: $test_file"
            failed_tests+=("$test_file")
        fi
    done
    
    # Summary
    log_info "Performance tests summary: $passed_tests/$total_tests passed"
    
    if [ ${#failed_tests[@]} -gt 0 ]; then
        log_warning "Failed performance tests: ${failed_tests[*]}"
        return 1
    fi
    
    return 0
}

# Generate coverage report
generate_coverage_report() {
    log_section "GENERATING COVERAGE REPORT"
    
    log_info "Generating comprehensive coverage report..."
    
    # Run all tests with coverage
    python3 -m pytest "$TEST_DIR" \
        --cov=protein_sssl \
        --cov-report=html:"$REPORTS_DIR/coverage_html" \
        --cov-report=xml:"$REPORTS_DIR/coverage.xml" \
        --cov-report=term-missing \
        --cov-fail-under=$COVERAGE_MIN \
        --quiet \
        -x  # Stop on first failure for coverage run
    
    local coverage_result=$?
    
    if [ $coverage_result -eq 0 ]; then
        log_success "Coverage report generated successfully"
        log_info "Coverage report available at: $REPORTS_DIR/coverage_html/index.html"
    else
        log_warning "Coverage below minimum threshold of $COVERAGE_MIN%"
    fi
    
    return $coverage_result
}

# Run code quality checks
run_code_quality_checks() {
    log_section "RUNNING CODE QUALITY CHECKS"
    
    local quality_passed=true
    
    # Python code formatting check (if black is available)
    if command -v black &> /dev/null; then
        log_info "Checking code formatting with Black..."
        if black --check protein_sssl/ tests/ > "$REPORTS_DIR/black_check.txt" 2>&1; then
            log_success "Code formatting check passed"
        else
            log_warning "Code formatting issues found (run 'black protein_sssl/ tests/' to fix)"
            quality_passed=false
        fi
    fi
    
    # Import sorting check (if isort is available)
    if command -v isort &> /dev/null; then
        log_info "Checking import sorting with isort..."
        if isort --check-only protein_sssl/ tests/ > "$REPORTS_DIR/isort_check.txt" 2>&1; then
            log_success "Import sorting check passed"
        else
            log_warning "Import sorting issues found (run 'isort protein_sssl/ tests/' to fix)"
            quality_passed=false
        fi
    fi
    
    # Linting check (if flake8 is available)
    if command -v flake8 &> /dev/null; then
        log_info "Running flake8 linting..."
        if flake8 protein_sssl/ tests/ --output-file="$REPORTS_DIR/flake8_report.txt" --tee; then
            log_success "Linting check passed"
        else
            log_warning "Linting issues found (check $REPORTS_DIR/flake8_report.txt)"
            quality_passed=false
        fi
    fi
    
    # Type checking (if mypy is available)
    if command -v mypy &> /dev/null; then
        log_info "Running type checking with mypy..."
        if mypy protein_sssl/ --ignore-missing-imports > "$REPORTS_DIR/mypy_report.txt" 2>&1; then
            log_success "Type checking passed"
        else
            log_warning "Type checking issues found (check $REPORTS_DIR/mypy_report.txt)"
            # Don't fail on mypy issues for now
        fi
    fi
    
    if [ "$quality_passed" = true ]; then
        log_success "Code quality checks completed successfully"
        return 0
    else
        log_error "Code quality checks found issues"
        return 1
    fi
}

# Run stress tests (optional)
run_stress_tests() {
    log_section "RUNNING STRESS TESTS"
    
    if [ "$RUN_STRESS_TESTS" = "true" ]; then
        log_info "Running stress tests..."
        
        python3 -m pytest "$TEST_DIR" \
            -m "slow" \
            --verbose \
            --tb=short \
            --junit-xml="$REPORTS_DIR/stress_tests.xml" \
            --timeout=1800  # 30 minutes timeout for stress tests
        
        local stress_result=$?
        
        if [ $stress_result -eq 0 ]; then
            log_success "Stress tests completed successfully"
        else
            log_warning "Stress tests failed or timed out"
        fi
        
        return $stress_result
    else
        log_info "Stress tests skipped (set RUN_STRESS_TESTS=true to enable)"
        return 0
    fi
}

# Generate final report
generate_final_report() {
    log_section "GENERATING FINAL TEST REPORT"
    
    local report_file="$REPORTS_DIR/test_summary_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Test Report for $PROJECT_NAME

**Generated**: $(date)  
**Environment**: $(uname -a)  
**Python Version**: $(python3 --version)  

## Test Results Summary

EOF
    
    # Count test results
    local total_xml_files=$(find "$REPORTS_DIR" -name "*.xml" | wc -l)
    local total_tests=$(grep -r "tests=" "$REPORTS_DIR"/*.xml 2>/dev/null | cut -d'"' -f2 | paste -sd+ | bc 2>/dev/null || echo "0")
    local total_failures=$(grep -r "failures=" "$REPORTS_DIR"/*.xml 2>/dev/null | cut -d'"' -f2 | paste -sd+ | bc 2>/dev/null || echo "0")
    local total_errors=$(grep -r "errors=" "$REPORTS_DIR"/*.xml 2>/dev/null | cut -d'"' -f2 | paste -sd+ | bc 2>/dev/null || echo "0")
    
    cat >> "$report_file" << EOF
- **Total Tests**: $total_tests
- **Failures**: $total_failures
- **Errors**: $total_errors
- **Success Rate**: $(( (total_tests - total_failures - total_errors) * 100 / total_tests ))%

## Coverage
$(grep "TOTAL" "$REPORTS_DIR"/coverage_html/index.html 2>/dev/null || echo "Coverage data not available")

## Test Categories

### Unit Tests
$(ls -1 "$REPORTS_DIR"/unit_*.xml 2>/dev/null | wc -l) test suites executed

### Integration Tests
$(ls -1 "$REPORTS_DIR"/integration_*.xml 2>/dev/null | wc -l) test suites executed

### Security Tests
$(ls -1 "$REPORTS_DIR"/security_*.xml 2>/dev/null | wc -l) test suites executed

### Performance Tests
$(ls -1 "$REPORTS_DIR"/performance_*.xml 2>/dev/null | wc -l) test suites executed

## Reports Generated

- HTML Coverage Report: \`$REPORTS_DIR/coverage_html/index.html\`
- Security Analysis: \`$REPORTS_DIR/bandit_report.json\`
- Vulnerability Check: \`$REPORTS_DIR/safety_report.json\`
- Code Quality: \`$REPORTS_DIR/flake8_report.txt\`

## Next Steps

EOF
    
    if [ $total_failures -gt 0 ] || [ $total_errors -gt 0 ]; then
        cat >> "$report_file" << EOF
⚠️ **Action Required**: Tests have failures or errors. Please review the XML reports and fix issues before deployment.
EOF
    else
        cat >> "$report_file" << EOF
✅ **All Tests Passed**: The codebase is ready for deployment.
EOF
    fi
    
    log_success "Final test report generated: $report_file"
    
    # Display summary in terminal
    echo -e "\n${CYAN}=== FINAL TEST SUMMARY ===${NC}"
    echo -e "Total Tests: ${BLUE}$total_tests${NC}"
    echo -e "Failures: ${RED}$total_failures${NC}"
    echo -e "Errors: ${RED}$total_errors${NC}"
    echo -e "Success Rate: ${GREEN}$(( (total_tests - total_failures - total_errors) * 100 / total_tests ))%${NC}"
    echo -e "Report: ${PURPLE}$report_file${NC}"
    echo
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    log_info "Starting comprehensive test suite for $PROJECT_NAME"
    log_info "Test reports will be saved to: $REPORTS_DIR"
    
    # Track overall results
    local overall_success=true
    
    # Run test categories
    check_prerequisites || overall_success=false
    
    run_unit_tests || overall_success=false
    run_integration_tests || overall_success=false
    run_security_tests || overall_success=false
    run_performance_tests || overall_success=false
    
    # Run quality and coverage checks
    run_code_quality_checks || overall_success=false
    generate_coverage_report || overall_success=false
    
    # Optional stress tests
    run_stress_tests || true  # Don't fail overall on stress test issues
    
    # Generate final report
    generate_final_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ "$overall_success" = true ]; then
        log_success "🎉 All test suites completed successfully in ${duration}s!"
        echo -e "${GREEN}Ready for production deployment!${NC}"
        exit 0
    else
        log_error "❌ Some tests failed. Check reports in $REPORTS_DIR"
        echo -e "${RED}Fix issues before deployment!${NC}"
        exit 1
    fi
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stress)
            export RUN_STRESS_TESTS=true
            shift
            ;;
        --coverage-min)
            COVERAGE_MIN="$2"
            shift 2
            ;;
        --reports-dir)
            REPORTS_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --stress         Run stress tests"
            echo "  --coverage-min   Minimum coverage percentage (default: 80)"
            echo "  --reports-dir    Test reports directory (default: test_reports)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"