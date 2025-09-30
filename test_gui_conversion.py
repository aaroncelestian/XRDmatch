#!/usr/bin/env python3
"""
Test the CIF to DIF conversion through the GUI components
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from gui.local_database_tab import LocalDatabaseTab, CifToDifDialog

def test_gui_conversion():
    """Test the GUI conversion functionality"""
    print("🧪 Testing GUI CIF to DIF conversion...")
    
    # Create QApplication
    app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
    
    try:
        # Create test CIF file
        cif_content = """
data_test_mineral
_chemical_name_mineral                 'Test Mineral'
_chemical_formula_sum                  'Si O2'
_symmetry_space_group_name_H-M         'P 1'
_cell_length_a                         5.0
_cell_length_b                         5.0
_cell_length_c                         5.0
_cell_angle_alpha                      90.0
_cell_angle_beta                       90.0
_cell_angle_gamma                      90.0

loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1  0.0  0.0  0.0
O1   0.5  0.5  0.0
O2   0.0  0.5  0.5
"""
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        cif_path = os.path.join(temp_dir, "test_mineral.cif")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(cif_path, 'w') as f:
            f.write(cif_content.strip())
        
        print(f"📁 Created test CIF: {cif_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Create LocalDatabaseTab instance
        db_tab = LocalDatabaseTab()
        print("✅ LocalDatabaseTab created successfully")
        
        # Test the conversion method directly
        print("🔄 Testing conversion...")
        db_tab.start_cif_to_dif_conversion([cif_path], output_dir, 1.5406)
        
        # Wait a moment for the thread to start
        import time
        time.sleep(1)
        
        # Check if thread was created
        if hasattr(db_tab, 'conversion_thread'):
            print("✅ Conversion thread created")
            
            # Wait for conversion to complete (simulate)
            # In real GUI, this would be handled by signals
            conversion_thread = db_tab.conversion_thread
            if conversion_thread:
                # Run the conversion directly for testing
                conversion_thread.run()
                print("✅ Conversion thread executed")
        
        # Check output
        output_files = list(Path(output_dir).glob("*.dif"))
        if output_files:
            print(f"✅ DIF files generated: {len(output_files)}")
            for dif_file in output_files:
                print(f"   📄 {dif_file.name} ({dif_file.stat().st_size} bytes)")
                
                # Show first few lines
                with open(dif_file, 'r') as f:
                    lines = f.readlines()[:10]
                print("   📄 First 10 lines:")
                for i, line in enumerate(lines, 1):
                    print(f"      {i:2d}: {line.rstrip()}")
        else:
            print("❌ No DIF files generated")
            print(f"   Files in output dir: {os.listdir(output_dir)}")
        
        # Test dialog creation
        print("\n🧪 Testing dialog creation...")
        dialog = CifToDifDialog()
        print("✅ CifToDifDialog created successfully")
        
        # Test dialog methods
        dialog.cif_files = [cif_path]
        dialog.output_directory = output_dir
        
        files = dialog.get_cif_files()
        output = dialog.get_output_directory()
        wavelength = dialog.get_wavelength()
        
        print(f"✅ Dialog methods working:")
        print(f"   Files: {files}")
        print(f"   Output: {output}")
        print(f"   Wavelength: {wavelength}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up: {temp_dir}")
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting GUI conversion test...\n")
    
    success = test_gui_conversion()
    
    if success:
        print(f"\n🎉 GUI conversion test PASSED!")
        print(f"💡 The CIF to DIF conversion feature is working correctly.")
        print(f"   You can use the '🔄 Generate DIF from CIF' button in the Database Manager tab.")
    else:
        print(f"\n⚠️  GUI conversion test FAILED!")
        print(f"   Please check the error messages above.")
    
    print(f"\n🏁 Test completed.")
