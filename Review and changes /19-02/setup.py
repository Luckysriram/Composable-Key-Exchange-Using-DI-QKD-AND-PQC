#!/usr/bin/env python3
"""
CuKEM Setup Script
Automated installation and verification of dependencies
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required, but you have {sys.version}")
        return False
    print(f"✓ Python {sys.version.split()[0]} OK")
    return True


def install_requirements():
    """Install all required packages"""
    print("\n📦 Installing dependencies from requirements.txt...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade", "pip"
        ])
        print("✓ pip upgraded")
        
        # Try installing all requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", "requirements.txt"
        ])
        print("✓ All requirements installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Installation had issues: {e}")
        print("   Some optional packages may have failed (e.g., liboqs-python)")
        print("   You can continue with pip install issues or try:")
        print("   - For liboqs: pip install --only-binary :all: liboqs-python")
        print("   - For Qiskit: pip install --upgrade qiskit qiskit-aer --no-cache-dir")
        return True  # Continue anyway


def verify_imports():
    """Verify that critical modules can be imported"""
    print("\n🔍 Verifying module imports...")
    
    critical_modules = {
        'yaml': 'PyYAML',
        'qiskit': 'Qiskit',
        'cryptography': 'cryptography',
        'transitions': 'transitions',
        'numpy': 'numpy',
    }
    
    optional_modules = {
        'oqs': 'liboqs-python',
    }
    
    # Check critical modules
    failed_critical = []
    for module, package in critical_modules.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - REQUIRED")
            failed_critical.append(package)
    
    # Check optional modules
    failed_optional = []
    for module, package in optional_modules.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠️  {package} - OPTIONAL (some features will be limited)")
            failed_optional.append(package)
    
    if failed_critical:
        print(f"\n❌ Critical packages missing: {', '.join(failed_critical)}")
        print("Try: pip install " + " ".join(failed_critical))
        return False
    
    if failed_optional:
        print(f"\n⚠️  Optional packages missing: {', '.join(failed_optional)}")
        print("Most demos will still work, but PQC-related features may be limited")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "output",
        "config",
        "tests"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ {directory}/")
        except Exception as e:
            print(f"⚠️  Could not create {directory}: {e}")
    
    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports of our modules
        import utils
        import config
        from cukem import CuKEMMode
        
        print("✓ Core modules can be imported")
        
        # Test utils functions
        from utils import generate_random_bytes, bytes_to_hex
        test_bytes = generate_random_bytes(16)
        test_hex = bytes_to_hex(test_bytes)
        print(f"✓ Utility functions work (generated {len(test_hex)} char hex)")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Run setup procedure"""
    print("\n" + "="*70)
    print(" CuKEM SETUP WIZARD")
    print(" Hybrid Post-Quantum Cryptography + Quantum Key Distribution")
    print("="*70)
    
    # Step 1: Check Python
    if not check_python_version():
        return 1
    
    # Step 2: Install requirements
    if not install_requirements():
        return 1
    
    # Step 3: Verify imports
    if not verify_imports():
        print("\n⚠️  Some dependencies are missing, but you can try to run demos anyway")
    
    # Step 4: Create directories
    if not create_directories():
        print("⚠️  Could not create all directories")
    
    # Step 5: Test functionality
    if not test_basic_functionality():
        return 1
    
    # Success!
    print("\n" + "="*70)
    print(" ✓ SETUP COMPLETE!")
    print("="*70)
    print("\n🚀 Next steps:")
    print("   1. Review config.yml to customize settings (optional)")
    print("   2. Run: python demo.py")
    print("   3. Check README.md for usage examples")
    print("\n📚 For more information:")
    print("   - Architecture: See compass_artifact_wf-*.md")
    print("   - Examples: Check demo.py")
    print("   - Config: Edit config.yml")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
