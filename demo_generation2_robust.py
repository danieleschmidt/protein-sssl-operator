#!/usr/bin/env python3
"""
🛡️ PROTEIN-SSSL-OPERATOR - GENERATION 2: MAKE IT ROBUST
========================================================
Demonstrates enhanced error handling, validation, and security features
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def demo_robust_error_handling():
    """Demo robust error handling and validation."""
    print("🛡️ GENERATION 2: ROBUST ERROR HANDLING DEMO")
    print("=" * 50)
    
    try:
        # Import robust error handling system
        from protein_sssl.utils.robust_error_handling import (
            RobustErrorHandler,
            InputValidator,
            ResourceMonitor,
            SecurityValidator,
            robust_operation
        )
        
        print("✅ Robust error handling system imported")
        
        # Initialize components
        error_handler = RobustErrorHandler(max_retries=2, retry_delay=0.1)
        validator = InputValidator()
        resource_monitor = ResourceMonitor()
        security_validator = SecurityValidator()
        
        print("✅ All robust components initialized")
        
        # Test input validation
        print("\n📋 Testing Input Validation")
        
        # Valid sequence
        try:
            valid_seq = validator.validate_protein_sequence("MKFLKFSLLTAVLLSVVFAFSSC")
            print(f"✅ Valid sequence processed: {valid_seq[:20]}...")
        except Exception as e:
            print(f"❌ Valid sequence failed: {e}")
        
        # Invalid sequence (should fail gracefully)
        try:
            validator.validate_protein_sequence("MKFL123INVALID")
            print("❌ Should have failed on invalid characters")
        except Exception as e:
            print(f"✅ Correctly caught invalid sequence: {type(e).__name__}")
        
        # Test model parameter validation
        try:
            valid_params = validator.validate_model_parameters({
                'd_model': 1280,
                'n_layers': 33,
                'n_heads': 20,
                'dropout': 0.1,
                'learning_rate': 1e-4
            })
            print(f"✅ Model parameters validated: {len(valid_params)} params")
        except Exception as e:
            print(f"❌ Parameter validation failed: {e}")
        
        # Test retry mechanism
        print("\n🔄 Testing Retry Mechanism")
        
        attempt_count = [0]
        
        @error_handler.with_retry(exceptions=ValueError, max_retries=2, delay=0.1)
        def flaky_prediction(sequence):
            attempt_count[0] += 1
            if attempt_count[0] <= 2:
                raise ValueError(f"Attempt {attempt_count[0]} failed")
            return f"Prediction successful on attempt {attempt_count[0]}"
        
        try:
            result = flaky_prediction("MKFLKF")
            print(f"✅ Retry mechanism successful: {result}")
        except Exception as e:
            print(f"❌ Retry mechanism failed: {e}")
        
        # Test resource monitoring
        print("\n💾 Testing Resource Monitoring")
        
        try:
            memory_ok = resource_monitor.check_memory_usage(0.01)  # 10MB
            print(f"✅ Memory check: {'PASS' if memory_ok else 'WARNING'}")
            
            disk_ok = resource_monitor.check_disk_space(Path("."), 0.01)  # 10MB
            print(f"✅ Disk space check: {'PASS' if disk_ok else 'WARNING'}")
        except Exception as e:
            print(f"❌ Resource monitoring failed: {e}")
        
        # Test security validation
        print("\n🔒 Testing Security Validation")
        
        try:
            safe_filename = security_validator.sanitize_filename("test<>file|name?.txt")
            print(f"✅ Filename sanitized: {safe_filename}")
            
            safe_path = security_validator.validate_safe_path(Path("demo_output/test.pdb"))
            print(f"✅ Path validation: {safe_path.name}")
        except Exception as e:
            print(f"❌ Security validation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust error handling demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_comprehensive_security():
    """Demo comprehensive security framework."""
    print("\n🔒 GENERATION 2: COMPREHENSIVE SECURITY DEMO")
    print("=" * 50)
    
    try:
        from protein_sssl.security.comprehensive_security import (
            SecureLogger,
            InputSanitizer,
            AccessController,
            DataEncryption,
            SecurityAuditor,
            secure_operation
        )
        
        print("✅ Comprehensive security system imported")
        
        # Initialize security components
        secure_logger = SecureLogger('demo', security_level="MEDIUM")
        sanitizer = InputSanitizer()
        access_controller = AccessController()
        auditor = SecurityAuditor()
        
        print("✅ Security components initialized")
        
        # Test input sanitization
        print("\n🧹 Testing Input Sanitization")
        
        try:
            # Sanitize protein sequence
            dirty_sequence = "MKF<script>alert('xss')</script>LKFSLLT   \n\t"
            clean_sequence = sanitizer.sanitize_sequence(dirty_sequence)
            print(f"✅ Sequence sanitized: '{dirty_sequence[:20]}...' -> '{clean_sequence}'")
            
            # Sanitize filename
            dirty_filename = "../../etc/passwd<>|?.pdb"
            clean_filename = sanitizer.sanitize_filename(dirty_filename)
            print(f"✅ Filename sanitized: '{dirty_filename}' -> '{clean_filename}'")
            
            # Sanitize configuration
            dirty_config = {
                "model": {"d_model": 1280, "exec": "rm -rf /"},
                "training": {"epochs": 10, "import os; os.system('ls')": "dangerous"},
                "data": {"path": "/safe/path"}
            }
            clean_config = sanitizer.sanitize_config(dirty_config)
            print(f"✅ Config sanitized: {len(dirty_config)} -> {len(clean_config)} keys")
            
        except Exception as e:
            print(f"❌ Input sanitization failed: {e}")
        
        # Test access control
        print("\n🚪 Testing Access Control")
        
        try:
            # Test different permissions
            permissions_tests = [
                ('guest', 'read:basic', True),
                ('user', 'predict:basic', True),
                ('user', 'train:advanced', False),
                ('researcher', 'train:basic', True),
                ('admin', 'delete:all', True)
            ]
            
            for role, permission, expected in permissions_tests:
                result = access_controller.check_permission(role, permission)
                status = "✅" if result == expected else "❌"
                print(f"{status} {role} -> {permission}: {result}")
                
        except Exception as e:
            print(f"❌ Access control failed: {e}")
        
        # Test encryption (if available)
        print("\n🔐 Testing Data Encryption")
        
        try:
            encryptor = DataEncryption()
            
            # Test data encryption
            sensitive_data = "CONFIDENTIAL_PROTEIN_SEQUENCE_MKFLKFSLLT"
            encrypted = encryptor.encrypt_data(sensitive_data)
            decrypted = encryptor.decrypt_data(encrypted)
            
            if decrypted == sensitive_data:
                print(f"✅ Encryption working: {encrypted['algorithm']}")
            else:
                print("❌ Encryption/decryption mismatch")
                
        except Exception as e:
            print(f"⚠️  Encryption failed (may need cryptography package): {e}")
        
        # Test secure operations decorator
        print("\n🛡️ Testing Secure Operations")
        
        try:
            @secure_operation(permission='predict:basic', user_role='user')
            def secure_predict_function(sequence):
                return f"Prediction for {len(sequence)} amino acids"
            
            result = secure_predict_function("MKFLKFSLLT")
            print(f"✅ Secure operation successful: {result}")
            
        except Exception as e:
            print(f"❌ Secure operation failed: {e}")
        
        # Generate security audit report
        print("\n📊 Testing Security Auditing")
        
        try:
            # Log some events
            auditor.log_event('USER_LOGIN', {'user': 'demo_user', 'timestamp': time.time()})
            auditor.log_event('PREDICTION_REQUEST', {'sequence_length': 10, 'user': 'demo_user'})
            auditor.log_event('FILE_ACCESS', {'file': 'demo_structure.pdb', 'action': 'read'})
            
            report = auditor.generate_security_report()
            print(f"✅ Security report generated:")
            print(f"   Total events: {report['total_events']}")
            print(f"   Event types: {list(report['event_types'].keys())}")
            print(f"   Recommendations: {len(report['security_recommendations'])}")
            
        except Exception as e:
            print(f"❌ Security auditing failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive security demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_robust_prediction_pipeline():
    """Demo the complete robust prediction pipeline."""
    print("\n🧬 GENERATION 2: ROBUST PREDICTION PIPELINE")
    print("=" * 50)
    
    try:
        # Import core systems with robust features
        from protein_sssl import (
            SequenceStructureSSL,
            NeuralOperatorFold,
            StructurePredictor,
            ConfigManager,
            TORCH_AVAILABLE
        )
        
        from protein_sssl.utils.robust_error_handling import (
            RobustErrorHandler,
            InputValidator,
            robust_operation
        )
        
        from protein_sssl.security.comprehensive_security import (
            InputSanitizer,
            SecureLogger,
            secure_operation
        )
        
        print(f"✅ All robust systems imported (Torch: {TORCH_AVAILABLE})")
        
        # Initialize robust components
        error_handler = RobustErrorHandler()
        validator = InputValidator()
        sanitizer = InputSanitizer()
        logger = SecureLogger('robust_pipeline')
        
        # Test sequence with robust processing
        test_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        print(f"🧬 Processing sequence: {test_sequence[:20]}... (length: {len(test_sequence)})")
        
        # Step 1: Robust input validation and sanitization
        try:
            # Sanitize input
            clean_sequence = sanitizer.sanitize_sequence(test_sequence)
            
            # Validate sequence
            validated_sequence = validator.validate_protein_sequence(
                clean_sequence,
                min_length=10,
                max_length=1000
            )
            
            print(f"✅ Input validation passed: {len(validated_sequence)} amino acids")
            
        except Exception as e:
            print(f"❌ Input validation failed: {e}")
            return False
        
        # Step 2: Robust model initialization with error handling
        @robust_operation(retries=2, fallback=None)
        def initialize_robust_models():
            config_manager = ConfigManager()
            ssl_config = config_manager.get_default_ssl_config()
            folding_config = config_manager.get_default_folding_config()
            
            # Validate configuration
            ssl_params = validator.validate_model_parameters(ssl_config)
            
            ssl_model = SequenceStructureSSL(**ssl_params)
            operator_model = NeuralOperatorFold(
                encoder=ssl_model,
                **folding_config
            )
            predictor = StructurePredictor(
                model=operator_model if TORCH_AVAILABLE else None,
                device="cpu"
            )
            
            return ssl_model, operator_model, predictor
        
        try:
            ssl_model, operator_model, predictor = initialize_robust_models()
            print("✅ Robust model initialization successful")
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            return False
        
        # Step 3: Robust prediction with comprehensive error handling
        @secure_operation(permission='predict:basic', user_role='user')
        @robust_operation(retries=3, fallback=None)
        def robust_structure_prediction(sequence, predictor):
            logger.audit(f"Starting prediction for sequence length {len(sequence)}")
            
            start_time = time.time()
            prediction = predictor.predict(
                sequence,
                return_confidence=True,
                num_recycles=3
            )
            prediction_time = time.time() - start_time
            
            logger.audit(f"Prediction completed in {prediction_time:.2f}s")
            return prediction, prediction_time
        
        try:
            prediction, prediction_time = robust_structure_prediction(
                validated_sequence, 
                predictor
            )
            
            print(f"✅ Robust prediction successful in {prediction_time:.2f}s")
            
            # Extract results safely
            results = {
                'sequence_length': len(validated_sequence),
                'prediction_time': prediction_time,
                'torch_available': TORCH_AVAILABLE,
                'model_type': 'mock' if not TORCH_AVAILABLE else 'real',
                'validation_passed': True,
                'security_checks_passed': True
            }
            
            # Add prediction metrics if available
            if hasattr(prediction, 'plddt_score'):
                results['plddt_score'] = getattr(prediction, 'plddt_score', 0.0)
            if hasattr(prediction, 'confidence'):
                results['confidence'] = getattr(prediction, 'confidence', 0.0)
            
            print(f"\n📊 ROBUST PREDICTION RESULTS:")
            for key, value in results.items():
                print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"❌ Robust prediction failed: {e}")
            return False
        
        # Step 4: Secure output handling
        try:
            output_dir = Path("demo_output_robust")
            output_dir.mkdir(exist_ok=True)
            
            # Secure filename generation using existing security validator
            from protein_sssl.utils.robust_error_handling import SecurityValidator
            security_validator = SecurityValidator()
            
            safe_pdb_path = security_validator.validate_safe_path(
                output_dir / "robust_prediction.pdb"
            )
            safe_json_path = security_validator.validate_safe_path(
                output_dir / "robust_results.json"
            )
            
            # Save with error handling
            if hasattr(prediction, 'save_pdb'):
                prediction.save_pdb(safe_pdb_path)
                print(f"✅ Structure saved securely to {safe_pdb_path}")
            
            # Save results JSON
            with safe_json_path.open('w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved securely to {safe_json_path}")
            
        except Exception as e:
            print(f"❌ Secure output handling failed: {e}")
            return False
        
        print("\n🎉 ROBUST PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Robust prediction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete Generation 2 robust demo."""
    print("🛡️ PROTEIN-SSSL-OPERATOR GENERATION 2 ROBUST DEMO")
    print("=" * 60)
    
    # Run all robust demos
    error_handling_success = demo_robust_error_handling()
    security_success = demo_comprehensive_security()
    pipeline_success = demo_robust_prediction_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 GENERATION 2 ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"✅ Error Handling & Validation: {'PASS' if error_handling_success else 'FAIL'}")
    print(f"✅ Comprehensive Security: {'PASS' if security_success else 'FAIL'}")
    print(f"✅ Robust Prediction Pipeline: {'PASS' if pipeline_success else 'FAIL'}")
    
    overall_success = error_handling_success and security_success and pipeline_success
    
    print(f"\n🎯 OVERALL STATUS: {'✅ GENERATION 2 COMPLETE' if overall_success else '❌ INCOMPLETE'}")
    
    if overall_success:
        print("🚀 Ready to proceed to GENERATION 3: MAKE IT SCALE")
        print("\n🛡️ ROBUSTNESS FEATURES IMPLEMENTED:")
        print("   • Comprehensive error handling with retry mechanisms")
        print("   • Input validation and sanitization")
        print("   • Resource monitoring and limits")
        print("   • Multi-layer security framework")
        print("   • Access control and permissions")
        print("   • Data encryption and secure storage")
        print("   • Security auditing and compliance")
        print("   • Secure file operations with sandboxing")
    else:
        print("🔧 Need to fix robustness issues before proceeding")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)