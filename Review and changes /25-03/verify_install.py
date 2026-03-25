#!/usr/bin/env python3
"""
Installation Verification Script
Checks that all dependencies are properly installed and files are in place
"""

import sys
import os
from pathlib import Path


def check_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    print(f"✅ Python {sys.version.split()[0]} installed")
    return True


def check_files():
    """Check that all required files exist"""
    required_files = [
        # Backend
        'backend/quantum_simulator.py',
        'backend/bb84.py',
        'backend/chsh.py',
        'backend/diqkd_simulator.py',
        'backend/api.py',
        'backend/__init__.py',
        # Frontend
        'frontend/index.html',
        'frontend/app.js',
        # Documentation
        'README.md',
        'QUICKSTART.md',
        'ARCHITECTURE.md',
        # Scripts
        'demo.py',
        'setup.py',
        'test_simulator.py',
        'requirements.txt',
    ]
    
    missing = []
    for filepath in required_files:
        if not os.path.exists(filepath):
            missing.append(filepath)
    
    if missing:
        print(f"❌ {len(missing)} files missing:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print(f"✅ All {len(required_files)} required files found")
    return True


def check_dependencies():
    """Check that all required packages are installed"""
    dependencies = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('numpy', 'NumPy'),
        ('pytest', 'pytest'),
    ]
    
    failed = []
    print("\nChecking dependencies...")
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} - NOT INSTALLED")
            failed.append(display_name)
    
    if failed:
        print(f"\n❌ {len(failed)} dependencies missing:")
        for name in failed:
            print(f"   - {name}")
        print("\nRun: pip install -r requirements.txt")
        return False
    
    return True


def check_imports():
    """Check that backend modules can be imported"""
    print("\nChecking backend imports...")
    
    modules = [
        'backend.quantum_simulator',
        'backend.bb84',
        'backend.chsh',
        'backend.diqkd_simulator',
    ]
    
    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name} - {str(e)}")
            failed.append(module_name)
    
    if failed:
        print(f"\n❌ {len(failed)} import(s) failed")
        return False
    
    return True


def check_api_start():
    """Check that Flask API can start"""
    print("\nChecking Flask API configuration...")
    
    try:
        from backend.api import app
        print(f"  ✅ Flask app created successfully")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create Flask app: {str(e)}")
        return False


def run_quick_test():
    """Run a quick functional test"""
    print("\nRunning quick functionality test...")
    
    try:
        from backend.diqkd_simulator import DIQKDSimulator
        
        # Create and run a minimal simulation
        sim = DIQKDSimulator(key_size=64, num_chsh_rounds=100)
        
        print("  Running BB84...")
        bb84_results = sim.run_bb84_protocol()
        
        print("  Running CHSH...")
        chsh_results = sim.run_chsh_bell_test(state_type='entangled')
        
        if bb84_results and chsh_results:
            print("  ✅ Functional test passed!")
            return True
        else:
            print("  ❌ Test results invalid")
            return False
            
    except Exception as e:
        print(f"  ❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "DI-QKD Simulator - Verification Check" + " "*15 + "║")
    print("╚" + "="*68 + "╝\n")
    
    checks = [
        ("Python Version", check_version),
        ("Required Files", check_files),
        ("Dependencies", check_dependencies),
        ("Backend Imports", check_imports),
        ("Flask API Setup", check_api_start),
        ("Functional Test", run_quick_test),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*70}")
        print(f"Checking: {name}")
        print(f"{'='*70}")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error during check: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:>8} - {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run demo:     python demo.py")
        print("  2. Start server: python -m backend.api")
        print("  3. Open UI:      frontend/index.html")
        print("  4. Run tests:    python -m pytest test_simulator.py -v")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed. Fix these issues before running.")
        if 'Dependencies' in [name for name, r in results if not r]:
            print("\nFix missing dependencies:")
            print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
