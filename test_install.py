#!/usr/bin/env python3
"""
Test script to verify XRD Phase Matcher installation
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'PyQt5',
        'matplotlib',
        'numpy',
        'pandas',
        'requests',
        'scipy',
        'bs4',  # beautifulsoup4
        'lxml'
    ]
    
    optional_packages = [
        'pymatgen',
        'gemmi'
    ]
    
    print("Testing required package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    print("\nTesting optional package imports...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"⚠ {package}: {e} (optional)")
    
    return failed_imports

def test_gui_components():
    """Test if GUI components can be imported"""
    print("\nTesting GUI components...")
    
    try:
        from gui.main_window import XRDMainWindow
        print("✓ Main window")
    except ImportError as e:
        print(f"✗ Main window: {e}")
        return False
        
    try:
        from gui.pattern_tab import PatternTab
        print("✓ Pattern tab")
    except ImportError as e:
        print(f"✗ Pattern tab: {e}")
        return False
        
    try:
        from gui.database_tab import DatabaseTab
        print("✓ Database tab")
    except ImportError as e:
        print(f"✗ Database tab: {e}")
        return False
        
    try:
        from gui.matching_tab import MatchingTab
        print("✓ Matching tab")
    except ImportError as e:
        print(f"✗ Matching tab: {e}")
        return False
        
    try:
        from gui.settings_tab import SettingsTab
        print("✓ Settings tab")
    except ImportError as e:
        print(f"✗ Settings tab: {e}")
        return False
        
    return True

def test_utilities():
    """Test utility modules"""
    print("\nTesting utility modules...")
    
    try:
        from utils.cif_parser import CIFParser
        print("✓ CIF parser")
        
        # Test basic functionality
        parser = CIFParser()
        print("✓ CIF parser instantiation")
        
    except ImportError as e:
        print(f"✗ CIF parser: {e}")
        return False
    except Exception as e:
        print(f"⚠ CIF parser warning: {e}")
        
    return True

def main():
    """Run all tests"""
    print("XRD Phase Matcher Installation Test")
    print("=" * 40)
    
    # Test package imports
    failed_imports = test_imports()
    
    if failed_imports:
        print(f"\n❌ Installation incomplete. Missing packages: {', '.join(failed_imports)}")
        print("Please install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    # Test GUI components
    if not test_gui_components():
        print("\n❌ GUI components failed to import")
        return False
    
    # Test utilities
    if not test_utilities():
        print("\n❌ Utility modules failed to import")
        return False
    
    print("\n✅ All tests passed! Installation appears to be successful.")
    print("\nTo run the application:")
    print("python main.py")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
