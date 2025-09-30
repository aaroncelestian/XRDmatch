#!/usr/bin/env python3
"""
Test script for CIF to DIF conversion functionality
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.local_database_tab import CifToDifConversionThread
from utils.local_database import LocalCIFDatabase

def create_test_cif():
    """Create a simple test CIF file for testing"""
    cif_content = """
data_quartz
_chemical_name_mineral                 'Quartz'
_chemical_formula_sum                  'Si O2'
_symmetry_space_group_name_H-M         'P 31 2 1'
_cell_length_a                         4.9134
_cell_length_b                         4.9134
_cell_length_c                         5.4052
_cell_angle_alpha                      90.0
_cell_angle_beta                       90.0
_cell_angle_gamma                      120.0
_diffrn_radiation_wavelength           1.5406

loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1  0.4697  0.0000  0.0000
O1   0.4135  0.2669  0.1191
O2   0.4135  0.1466  0.8809

_publ_author_name                      'Test Author'
_journal_name_full                     'Test Journal'
"""
    
    # Create temporary CIF file
    temp_dir = tempfile.mkdtemp()
    cif_path = os.path.join(temp_dir, "test_quartz.cif")
    
    with open(cif_path, 'w') as f:
        f.write(cif_content.strip())
    
    return cif_path, temp_dir

def test_cif_to_dif_conversion():
    """Test the CIF to DIF conversion functionality"""
    print("🧪 Testing CIF to DIF conversion...")
    
    try:
        # Create test CIF file
        cif_path, temp_dir = create_test_cif()
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 Created test CIF file: {cif_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Initialize database manager
        db_manager = LocalCIFDatabase()
        
        # Create conversion thread
        conversion_thread = CifToDifConversionThread(
            db_manager=db_manager,
            cif_files=[cif_path],
            output_dir=output_dir,
            wavelength=1.5406
        )
        
        # Test the conversion method directly
        print("🔄 Starting conversion...")
        dif_path = conversion_thread.convert_cif_to_dif(cif_path)
        
        if dif_path and os.path.exists(dif_path):
            print(f"✅ Conversion successful! DIF file created: {dif_path}")
            
            # Read and display first few lines of DIF file
            with open(dif_path, 'r') as f:
                lines = f.readlines()[:20]  # First 20 lines
            
            print("\n📄 DIF file content (first 20 lines):")
            print("=" * 50)
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}: {line.rstrip()}")
            print("=" * 50)
            
            # Check file size
            file_size = os.path.getsize(dif_path)
            print(f"📊 DIF file size: {file_size} bytes")
            
            # Count data lines
            with open(dif_path, 'r') as f:
                all_lines = f.readlines()
            
            data_lines = [line for line in all_lines if not line.strip().startswith('#') 
                         and not line.strip().startswith('_') 
                         and not line.strip().startswith('data_')
                         and not line.strip().startswith('loop_')
                         and line.strip() 
                         and not line.strip().startswith("'")]
            
            print(f"📈 Number of diffraction data points: {len(data_lines)}")
            
            if data_lines:
                print("\n📊 Sample diffraction data:")
                for i, line in enumerate(data_lines[:5]):  # First 5 data points
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        print(f"   2θ={parts[0]:>8s}°  d={parts[1]:>8s}Å  I={parts[2]:>8s}")
            
            return True
            
        else:
            print("❌ Conversion failed - no DIF file created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")
        except:
            pass

def test_pseudo_voigt_profile():
    """Test the pseudo-Voigt profile generation"""
    print("\n🧪 Testing pseudo-Voigt profile generation...")
    
    try:
        import numpy as np
        from gui.local_database_tab import CifToDifConversionThread
        from utils.local_database import LocalCIFDatabase
        
        # Create conversion thread instance
        db_manager = LocalCIFDatabase()
        conversion_thread = CifToDifConversionThread(
            db_manager=db_manager,
            cif_files=[],
            output_dir="",
            wavelength=1.5406
        )
        
        # Test data: simple peaks
        peak_positions = np.array([20.0, 26.6, 50.1])  # 2theta values
        peak_intensities = np.array([100.0, 80.0, 60.0])  # Intensities
        
        print(f"📊 Input peaks: {len(peak_positions)} peaks")
        for pos, intensity in zip(peak_positions, peak_intensities):
            print(f"   2θ={pos:6.2f}°  I={intensity:6.1f}")
        
        # Generate profile
        profile_data = conversion_thread.generate_pseudo_voigt_profile(
            peak_positions, peak_intensities, fwhm_base=0.1
        )
        
        print(f"✅ Generated profile with {len(profile_data['two_theta'])} points")
        print(f"📈 2θ range: {profile_data['two_theta'].min():.2f}° - {profile_data['two_theta'].max():.2f}°")
        print(f"📈 Intensity range: {profile_data['intensity'].min():.2f} - {profile_data['intensity'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Profile test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting CIF to DIF conversion tests...\n")
    
    # Test 1: Basic conversion
    test1_passed = test_cif_to_dif_conversion()
    
    # Test 2: Pseudo-Voigt profile generation
    test2_passed = test_pseudo_voigt_profile()
    
    # Summary
    print(f"\n📋 Test Results:")
    print(f"   CIF to DIF conversion: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Pseudo-Voigt profiles: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! CIF to DIF conversion is working correctly.")
        print(f"\n💡 You can now use the 'Generate DIF from CIF' button in the Database Manager tab.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the error messages above.")
    
    print(f"\n🏁 Test completed.")
