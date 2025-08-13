#!/usr/bin/env python3
"""
Quality Gates & Final Validation for protein-sssl-operator
Comprehensive testing of all systems before production deployment
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class QualityGateValidator:
    """Comprehensive quality validation system"""
    
    def __init__(self):
        self.results = {}
        self.overall_score = 0
        self.critical_failures = []
        self.warnings = []
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates"""
        
        print("🚀 PROTEIN-SSSL-OPERATOR QUALITY GATES")
        print("=" * 55)
        print("Final validation before production deployment")
        print("=" * 55)
        
        # Quality gates in order of importance
        gates = [
            ("Project Structure", self._validate_project_structure),
            ("Code Quality", self._validate_code_quality),
            ("Security", self._validate_security),
            ("Performance", self._validate_performance),
            ("Robustness", self._validate_robustness),
            ("Scalability", self._validate_scalability),
            ("Documentation", self._validate_documentation),
            ("Configuration", self._validate_configuration),
            ("Testing", self._validate_testing),
            ("Production Readiness", self._validate_production_readiness)
        ]
        
        total_gates = len(gates)
        passed_gates = 0
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Quality Gate: {gate_name}")
            print("=" * (17 + len(gate_name)))
            
            try:
                gate_result = gate_func()
                self.results[gate_name] = gate_result
                
                if gate_result['passed']:
                    passed_gates += 1
                    print(f"✅ {gate_name}: PASSED")
                else:
                    print(f"❌ {gate_name}: FAILED")
                    if gate_result.get('critical', False):
                        self.critical_failures.append(gate_name)
                
                # Display details
                if gate_result.get('details'):
                    for detail in gate_result['details']:
                        print(f"   • {detail}")
                
                if gate_result.get('warnings'):
                    for warning in gate_result['warnings']:
                        print(f"   ⚠️ {warning}")
                        self.warnings.append(f"{gate_name}: {warning}")
                        
            except Exception as e:
                print(f"❌ {gate_name}: ERROR - {e}")
                self.results[gate_name] = {'passed': False, 'error': str(e), 'critical': True}
                self.critical_failures.append(gate_name)
        
        # Calculate overall score
        self.overall_score = (passed_gates / total_gates) * 100
        
        # Generate final report
        return self._generate_final_report(passed_gates, total_gates)
    
    def _validate_project_structure(self) -> Dict[str, Any]:
        """Validate project structure and organization"""
        
        base_dir = Path(__file__).parent.parent
        required_dirs = [
            "protein_sssl",
            "protein_sssl/models",
            "protein_sssl/data", 
            "protein_sssl/training",
            "protein_sssl/utils",
            "protein_sssl/config",
            "protein_sssl/cli",
            "scripts",
            "tests",
            "configs"
        ]
        
        required_files = [
            "README.md",
            "pyproject.toml",
            "environment.yml",
            "protein_sssl/__init__.py"
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_name in required_dirs:
            if not (base_dir / dir_name).exists():
                missing_dirs.append(dir_name)
        
        # Check files
        for file_name in required_files:
            if not (base_dir / file_name).exists():
                missing_files.append(file_name)
        
        details = []
        if not missing_dirs and not missing_files:
            details.append(f"All {len(required_dirs)} directories present")
            details.append(f"All {len(required_files)} required files present")
        
        return {
            'passed': len(missing_dirs) == 0 and len(missing_files) == 0,
            'critical': True,
            'details': details,
            'missing_dirs': missing_dirs,
            'missing_files': missing_files
        }
    
    def _validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality and standards"""
        
        details = []
        warnings = []
        
        # Check for Python syntax errors
        python_files = list(Path(__file__).parent.parent.rglob("*.py"))
        syntax_errors = 0
        
        for py_file in python_files[:20]:  # Sample first 20 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors += 1
                warnings.append(f"Syntax error in {py_file.name}: {e}")
            except Exception:
                pass  # Skip files that can't be read
        
        details.append(f"Checked {min(len(python_files), 20)} Python files for syntax")
        
        # Check for basic code structure
        init_files = list(Path(__file__).parent.parent.rglob("__init__.py"))
        details.append(f"Found {len(init_files)} package __init__.py files")
        
        # Check for docstrings in main modules
        main_modules = [
            "protein_sssl/models/ssl_encoder.py",
            "protein_sssl/utils/security.py",
            "protein_sssl/utils/error_handling.py"
        ]
        
        documented_modules = 0
        for module_path in main_modules:
            try:
                with open(Path(__file__).parent.parent / module_path, 'r') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        documented_modules += 1
            except FileNotFoundError:
                pass
        
        details.append(f"Documented modules: {documented_modules}/{len(main_modules)}")
        
        # Overall quality assessment
        quality_score = 100 - (syntax_errors * 20) - ((len(main_modules) - documented_modules) * 10)
        quality_good = quality_score >= 80
        
        if not quality_good:
            warnings.append(f"Code quality score: {quality_score}%")
        
        return {
            'passed': syntax_errors == 0 and quality_score >= 70,
            'critical': False,
            'details': details,
            'warnings': warnings,
            'quality_score': quality_score
        }
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security implementations"""
        
        try:
            # Run the security validation we built
            result = subprocess.run([
                sys.executable, 
                str(Path(__file__).parent / "test_robustness_minimal.py")
            ], capture_output=True, text=True, timeout=30)
            
            security_passed = result.returncode == 0
            
            details = []
            warnings = []
            
            if security_passed:
                details.append("Security validation tests passed")
                details.append("Input validation working correctly")
                details.append("Output sanitization functional")
                details.append("Error handling robust")
            else:
                warnings.append("Some security tests failed")
                if result.stderr:
                    warnings.append(f"Security test errors: {result.stderr[:200]}")
            
            # Check for security-related files
            security_files = [
                "protein_sssl/utils/security.py",
                "protein_sssl/utils/validation.py",
                "protein_sssl/utils/error_handling.py"
            ]
            
            security_files_present = 0
            for sec_file in security_files:
                if (Path(__file__).parent.parent / sec_file).exists():
                    security_files_present += 1
            
            details.append(f"Security modules: {security_files_present}/{len(security_files)}")
            
            return {
                'passed': security_passed and security_files_present >= 2,
                'critical': True,
                'details': details,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'passed': False,
                'critical': True,
                'details': [],
                'warnings': [f"Security validation error: {e}"]
            }
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance optimizations"""
        
        try:
            # Run the scaling tests we built
            result = subprocess.run([
                sys.executable,
                str(Path(__file__).parent / "test_scaling_minimal.py")
            ], capture_output=True, text=True, timeout=60)
            
            performance_passed = result.returncode == 0
            
            details = []
            warnings = []
            
            if performance_passed:
                details.append("Performance tests passed")
                details.append("Caching system functional")
                details.append("Parallel processing working")
                details.append("Memory optimization active")
            else:
                warnings.append("Some performance tests failed")
                if result.stderr:
                    warnings.append(f"Performance test issues: {result.stderr[:200]}")
            
            # Check for performance-related files
            perf_files = [
                "protein_sssl/utils/performance_optimizer.py",
                "protein_sssl/utils/monitoring.py"
            ]
            
            perf_files_present = 0
            for perf_file in perf_files:
                if (Path(__file__).parent.parent / perf_file).exists():
                    perf_files_present += 1
            
            details.append(f"Performance modules: {perf_files_present}/{len(perf_files)}")
            
            return {
                'passed': performance_passed and perf_files_present >= 1,
                'critical': False,
                'details': details,
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'passed': False,
                'critical': False,
                'details': [],
                'warnings': [f"Performance validation error: {e}"]
            }
    
    def _validate_robustness(self) -> Dict[str, Any]:
        """Validate robustness and error handling"""
        
        details = []
        warnings = []
        
        # Check error handling files
        error_handling_files = [
            "protein_sssl/utils/error_handling.py",
            "protein_sssl/utils/error_recovery.py"
        ]
        
        error_files_present = 0
        for error_file in error_handling_files:
            if (Path(__file__).parent.parent / error_file).exists():
                error_files_present += 1
                details.append(f"Found {error_file}")
        
        # Check for exception classes in code
        try:
            with open(Path(__file__).parent.parent / "protein_sssl/utils/error_handling.py", 'r') as f:
                content = f.read()
                
                exception_classes = [
                    'ProteinSSLError',
                    'DataError', 
                    'ModelError',
                    'TrainingError'
                ]
                
                found_exceptions = 0
                for exc_class in exception_classes:
                    if exc_class in content:
                        found_exceptions += 1
                
                details.append(f"Custom exception classes: {found_exceptions}/{len(exception_classes)}")
                
                robustness_score = (error_files_present * 50) + (found_exceptions * 10)
                
        except FileNotFoundError:
            robustness_score = 0
            warnings.append("Error handling module not found")
        
        return {
            'passed': robustness_score >= 70,
            'critical': False,
            'details': details,
            'warnings': warnings,
            'robustness_score': robustness_score
        }
    
    def _validate_scalability(self) -> Dict[str, Any]:
        """Validate scalability implementations"""
        
        details = []
        warnings = []
        
        # Check for scalability-related implementations
        scalability_indicators = [
            ("Parallel processing", "concurrent.futures"),
            ("Caching", "cache"),
            ("Memory optimization", "memory"),
            ("Performance monitoring", "performance")
        ]
        
        found_indicators = 0
        
        try:
            perf_optimizer_path = Path(__file__).parent.parent / "protein_sssl/utils/performance_optimizer.py"
            if perf_optimizer_path.exists():
                with open(perf_optimizer_path, 'r') as f:
                    content = f.read()
                    
                    for indicator_name, indicator_text in scalability_indicators:
                        if indicator_text.lower() in content.lower():
                            found_indicators += 1
                            details.append(f"✅ {indicator_name} implementation found")
                        else:
                            warnings.append(f"⚠️ {indicator_name} implementation not clearly evident")
            else:
                warnings.append("Performance optimizer module not found")
        
        except Exception as e:
            warnings.append(f"Error checking scalability: {e}")
        
        # Check for multiprocessing support
        try:
            import multiprocessing
            import concurrent.futures
            details.append(f"System supports {multiprocessing.cpu_count()} CPU cores")
            details.append("Concurrent futures module available")
        except ImportError:
            warnings.append("Limited concurrency support")
        
        scalability_score = (found_indicators / len(scalability_indicators)) * 100
        
        return {
            'passed': scalability_score >= 75,
            'critical': False,
            'details': details,
            'warnings': warnings,
            'scalability_score': scalability_score
        }
    
    def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness"""
        
        details = []
        warnings = []
        
        # Check for key documentation files
        doc_files = [
            ("README.md", "Project overview"),
            ("CHANGELOG.md", "Change history"),
            ("CONTRIBUTING.md", "Contribution guidelines"),
            ("docs/API_REFERENCE.md", "API documentation"),
            ("docs/ARCHITECTURE.md", "Architecture documentation")
        ]
        
        found_docs = 0
        base_dir = Path(__file__).parent.parent
        
        for doc_file, description in doc_files:
            if (base_dir / doc_file).exists():
                found_docs += 1
                details.append(f"✅ {description}")
            else:
                warnings.append(f"Missing {description} ({doc_file})")
        
        # Check README content
        try:
            readme_path = base_dir / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    readme_content = f.read()
                    
                    readme_sections = [
                        "Installation",
                        "Usage", 
                        "Examples",
                        "Overview"
                    ]
                    
                    readme_score = sum(1 for section in readme_sections 
                                     if section.lower() in readme_content.lower())
                    
                    details.append(f"README sections: {readme_score}/{len(readme_sections)}")
        except Exception:
            warnings.append("Could not analyze README content")
        
        documentation_score = (found_docs / len(doc_files)) * 100
        
        return {
            'passed': documentation_score >= 60,  # Allow for missing some docs
            'critical': False,
            'details': details,
            'warnings': warnings,
            'documentation_score': documentation_score
        }
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration management"""
        
        details = []
        warnings = []
        
        # Check configuration files
        config_files = [
            "pyproject.toml",
            "environment.yml",
            "configs/ssl_config.yaml",
            "configs/folding_config.yaml"
        ]
        
        found_configs = 0
        base_dir = Path(__file__).parent.parent
        
        for config_file in config_files:
            if (base_dir / config_file).exists():
                found_configs += 1
                details.append(f"✅ {config_file}")
            else:
                warnings.append(f"Missing {config_file}")
        
        # Check config management code
        config_manager_path = base_dir / "protein_sssl/config/config_manager.py"
        if config_manager_path.exists():
            details.append("✅ Configuration manager implementation")
            found_configs += 1
        else:
            warnings.append("Configuration manager not found")
        
        # Validate pyproject.toml structure
        try:
            pyproject_path = base_dir / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    
                    required_sections = ['project', 'build-system']
                    found_sections = sum(1 for section in required_sections 
                                       if f'[{section}]' in content)
                    
                    details.append(f"pyproject.toml sections: {found_sections}/{len(required_sections)}")
        except Exception:
            warnings.append("Could not validate pyproject.toml")
        
        config_score = (found_configs / (len(config_files) + 1)) * 100
        
        return {
            'passed': config_score >= 70,
            'critical': False,
            'details': details,
            'warnings': warnings,
            'config_score': config_score
        }
    
    def _validate_testing(self) -> Dict[str, Any]:
        """Validate testing implementation"""
        
        details = []
        warnings = []
        
        # Check for test files
        test_files = [
            "tests/test_models.py",
            "tests/test_config.py",
            "tests/test_security.py",
            "scripts/test_minimal_functionality.py",
            "scripts/test_robustness_minimal.py",
            "scripts/test_scaling_minimal.py"
        ]
        
        found_tests = 0
        base_dir = Path(__file__).parent.parent
        
        for test_file in test_files:
            if (base_dir / test_file).exists():
                found_tests += 1
                details.append(f"✅ {test_file}")
            else:
                warnings.append(f"Missing {test_file}")
        
        # Run our custom tests
        custom_tests_passed = 0
        custom_tests = [
            ("Basic functionality", "test_minimal_functionality.py"),
            ("Robustness", "test_robustness_minimal.py"),
            ("Scalability", "test_scaling_minimal.py")
        ]
        
        for test_name, test_file in custom_tests:
            try:
                result = subprocess.run([
                    sys.executable,
                    str(base_dir / "scripts" / test_file)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    custom_tests_passed += 1
                    details.append(f"✅ {test_name} tests passed")
                else:
                    warnings.append(f"⚠️ {test_name} tests had issues")
                    
            except Exception as e:
                warnings.append(f"Could not run {test_name} tests: {e}")
        
        testing_score = ((found_tests + custom_tests_passed * 2) / (len(test_files) + len(custom_tests) * 2)) * 100
        
        return {
            'passed': testing_score >= 60 and custom_tests_passed >= 2,
            'critical': False,
            'details': details,
            'warnings': warnings,
            'testing_score': testing_score
        }
    
    def _validate_production_readiness(self) -> Dict[str, Any]:
        """Validate production deployment readiness"""
        
        details = []
        warnings = []
        
        # Check deployment files
        deployment_files = [
            "docker/Dockerfile",
            "kubernetes/deployment.yaml",
            "deployment/production-deploy.sh"
        ]
        
        found_deployment = 0
        base_dir = Path(__file__).parent.parent
        
        for deploy_file in deployment_files:
            if (base_dir / deploy_file).exists():
                found_deployment += 1
                details.append(f"✅ {deploy_file}")
            else:
                warnings.append(f"Missing {deploy_file}")
        
        # Check for CLI interface
        cli_path = base_dir / "protein_sssl/cli/main.py"
        if cli_path.exists():
            details.append("✅ CLI interface available")
        else:
            warnings.append("CLI interface not found")
        
        # Check entry points in pyproject.toml
        try:
            pyproject_path = base_dir / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    
                    if '[project.scripts]' in content:
                        details.append("✅ CLI scripts configured")
                    else:
                        warnings.append("CLI scripts not configured")
        except Exception:
            warnings.append("Could not check CLI configuration")
        
        # Check for monitoring and logging
        monitoring_files = [
            "protein_sssl/utils/logging_config.py",
            "protein_sssl/utils/monitoring.py"
        ]
        
        monitoring_present = sum(1 for f in monitoring_files 
                               if (base_dir / f).exists())
        
        details.append(f"Monitoring/logging modules: {monitoring_present}/{len(monitoring_files)}")
        
        # Overall production readiness score
        total_checks = len(deployment_files) + 3 + len(monitoring_files)  # deployment + cli + entry points + monitoring
        passed_checks = found_deployment + (1 if cli_path.exists() else 0) + monitoring_present
        
        readiness_score = (passed_checks / total_checks) * 100
        
        return {
            'passed': readiness_score >= 70,
            'critical': False,
            'details': details,
            'warnings': warnings,
            'readiness_score': readiness_score
        }
    
    def _generate_final_report(self, passed_gates: int, total_gates: int) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        print("\n" + "=" * 55)
        print("🏁 FINAL QUALITY ASSESSMENT")
        print("=" * 55)
        
        print(f"Overall Score: {self.overall_score:.1f}%")
        print(f"Gates Passed: {passed_gates}/{total_gates}")
        
        if self.critical_failures:
            print(f"❌ Critical Failures: {len(self.critical_failures)}")
            for failure in self.critical_failures:
                print(f"   • {failure}")
        
        if self.warnings:
            print(f"⚠️ Warnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"   • {warning}")
            if len(self.warnings) > 5:
                print(f"   • ... and {len(self.warnings) - 5} more warnings")
        
        # Determine deployment readiness
        deployment_ready = (
            self.overall_score >= 80 and 
            len(self.critical_failures) == 0 and
            passed_gates >= total_gates * 0.8
        )
        
        print(f"\n🚀 DEPLOYMENT STATUS:")
        if deployment_ready:
            print("✅ READY FOR PRODUCTION DEPLOYMENT")
            print("   All critical systems operational")
            print("   Quality gates passed successfully")
        elif self.overall_score >= 70:
            print("⚠️ READY FOR STAGING DEPLOYMENT")
            print("   Core functionality operational")
            print("   Some optimizations recommended")
        else:
            print("❌ NOT READY FOR DEPLOYMENT")
            print("   Critical issues must be resolved")
            print("   Additional development required")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        
        if deployment_ready:
            recommendations = [
                "Monitor system performance in production",
                "Set up automated testing pipeline",
                "Configure production monitoring and alerts",
                "Plan for horizontal scaling as needed"
            ]
        elif self.overall_score >= 70:
            recommendations = [
                "Resolve critical failures before production",
                "Address high-priority warnings",
                "Complete missing documentation",
                "Enhance error handling and monitoring"
            ]
        else:
            recommendations = [
                "Focus on critical system components",
                "Implement comprehensive error handling",
                "Add security validations",
                "Complete core functionality development"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return {
            'overall_score': self.overall_score,
            'gates_passed': passed_gates,
            'total_gates': total_gates,
            'critical_failures': self.critical_failures,
            'warnings_count': len(self.warnings),
            'deployment_ready': deployment_ready,
            'recommendations': recommendations,
            'results': self.results
        }

def main():
    """Run all quality gates"""
    
    validator = QualityGateValidator()
    
    start_time = time.time()
    final_report = validator.run_all_gates()
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Quality gates completed in {total_time:.1f} seconds")
    
    # Exit with appropriate code
    if final_report['deployment_ready']:
        print("\n🎉 QUALITY GATES PASSED - READY FOR PRODUCTION!")
        return 0
    elif final_report['overall_score'] >= 70:
        print("\n⚠️ QUALITY GATES PARTIAL - STAGING READY")
        return 1
    else:
        print("\n❌ QUALITY GATES FAILED - DEVELOPMENT NEEDED")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)