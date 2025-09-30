#!/usr/bin/env python3
"""
Demonstration of CIF to DIF conversion
Creates a real DIF file you can examine
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.local_database_tab import CifToDifConversionThread
from utils.local_database import LocalCIFDatabase

def create_demo_cif():
    """Create a demo CIF file for calcite"""
    cif_content = """
data_calcite
_chemical_name_mineral                 'Calcite'
_chemical_formula_sum                  'Ca C O3'
_symmetry_space_group_name_H-M         'R -3 c'
_cell_length_a                         4.9896
_cell_length_b                         4.9896
_cell_length_c                         17.0610
_cell_angle_alpha                      90.0
_cell_angle_beta                       90.0
_cell_angle_gamma                      120.0
_publ_author_name                      'Demo Author'
_journal_name_full                     'Demo Journal'

loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ca1  0.0000  0.0000  0.0000
C1   0.0000  0.0000  0.2500
O1   0.2569  0.0000  0.2500
"""
    
    # Create demo directory
    demo_dir = Path(__file__).parent / "demo_conversion"
    demo_dir.mkdir(exist_ok=True)
    
    cif_path = demo_dir / "calcite_demo.cif"
    output_dir = demo_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    with open(cif_path, 'w') as f:
        f.write(cif_content.strip())
    
    return str(cif_path), str(output_dir)

def main():
    print("🚀 CIF to DIF Conversion Demonstration")
    print("=" * 50)
    
    # Create demo files
    cif_path, output_dir = create_demo_cif()
    print(f"📁 Created demo CIF: {cif_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize database
    db_manager = LocalCIFDatabase()
    
    # Create conversion thread
    conversion_thread = CifToDifConversionThread(
        db_manager=db_manager,
        cif_files=[cif_path],
        output_dir=output_dir,
        wavelength=1.5406
    )
    
    print("\n🔄 Starting conversion...")
    
    # Perform conversion
    dif_path = conversion_thread.convert_cif_to_dif(cif_path)
    
    if dif_path and os.path.exists(dif_path):
        print(f"✅ SUCCESS! DIF file created: {dif_path}")
        
        # Show file info
        file_size = os.path.getsize(dif_path)
        print(f"📊 File size: {file_size:,} bytes")
        
        # Show content preview
        with open(dif_path, 'r') as f:
            lines = f.readlines()
        
        print(f"📄 Total lines: {len(lines)}")
        
        # Show header
        print(f"\n📋 DIF File Header (first 25 lines):")
        print("-" * 50)
        for i, line in enumerate(lines[:25], 1):
            print(f"{i:2d}: {line.rstrip()}")
        
        # Find data section
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#') and not line.startswith('_') and not line.startswith('data_') and not line.startswith('loop_') and not line.strip().startswith("'"):
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        float(parts[0])  # Try to parse as number
                        data_start = i
                        break
                except:
                    continue
        
        if data_start:
            print(f"\n📊 Diffraction Data (starting at line {data_start + 1}):")
            print("-" * 50)
            print("    2θ (°)     d (Å)    Intensity")
            print("-" * 50)
            
            data_lines = lines[data_start:data_start + 10]  # Show first 10 data points
            for line in data_lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        two_theta = float(parts[0])
                        d_spacing = float(parts[1])
                        intensity = float(parts[2])
                        print(f"  {two_theta:8.4f}  {d_spacing:8.6f}  {intensity:8.2f}")
                    except:
                        continue
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📁 You can examine the full DIF file at: {dif_path}")
        print(f"💡 This file can now be imported into the XRD database for pattern matching.")
        
    else:
        print("❌ Conversion failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Demo completed successfully!")
        print(f"🔍 Check the 'demo_conversion/output/' directory for the generated DIF file.")
    else:
        print(f"\n❌ Demo failed!")
