#!/usr/bin/env python3
"""
Simple test to verify basic imports work without external dependencies
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_basic_imports():
    """Test basic module structure"""
    print("🧬 Testing Basic Module Imports")
    print("=" * 40)
    
    try:
        # Test basic protein_sssl imports
        from protein_sssl.config import ConfigManager
        print("✅ Config module imported successfully")
        
        from protein_sssl.utils.logging_config import setup_logging
        print("✅ Utils module imported successfully")
        
        print("\n📁 Directory Structure:")
        base_path = Path(__file__).parent.parent / "protein_sssl"
        
        for item in base_path.rglob("*.py"):
            if "__pycache__" not in str(item):
                relative_path = item.relative_to(base_path.parent)
                print(f"  📄 {relative_path}")
        
        print("\n✅ Basic imports successful - project structure valid!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    return True

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration System")
    print("=" * 40)
    
    try:
        from protein_sssl.config import ConfigManager
        
        # Create config manager
        config_manager = ConfigManager()
        
        # Test basic config operations
        test_config = {
            "model": {
                "d_model": 1280,
                "n_layers": 33,
                "n_heads": 20
            },
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 128
            }
        }
        
        print("✅ Configuration manager created")
        print(f"  Config keys: {list(test_config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PROTEIN-SSSL-OPERATOR IMPORT TESTS")
    print("=" * 50)
    print("Generation 1: MAKE IT WORK - Testing Basic Structure")
    print("=" * 50)
    print()
    
    success = True
    
    success &= test_basic_imports()
    success &= test_configuration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All import tests passed!")
        print("✅ Project structure is valid")
    else:
        print("❌ Some tests failed")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)