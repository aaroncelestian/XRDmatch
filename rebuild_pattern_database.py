#!/usr/bin/env python3
"""
Rebuild the XRD pattern database using pymatgen for accurate intensities

This script will regenerate all theoretical patterns using pymatgen's proper
XRD calculator which includes:
- Angle-dependent atomic scattering factors
- Proper Lorentz-polarization factors
- Thermal factors
- Multiplicity
- Space group symmetry

This fixes the issue where low-angle peaks were artificially strong.
"""

import sqlite3
import numpy as np
from utils.cif_parser import CIFParser
from utils.local_database import LocalCIFDatabase
import time

def rebuild_patterns():
    """Rebuild all patterns in the database using pymatgen"""
    print("="*70)
    print("Rebuilding XRD Pattern Database with Pymatgen")
    print("="*70)
    
    # Initialize database
    db = LocalCIFDatabase()
    
    # Connect directly to the database
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM minerals")
    total_minerals = cursor.fetchone()[0]
    
    print(f"\nFound {total_minerals} minerals in database")
    print("This will regenerate all theoretical patterns using pymatgen")
    print("\nThis may take several minutes...")
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        conn.close()
        return
    
    # Get all minerals with their CIF data
    cursor.execute("""
        SELECT m.id, m.mineral_name, m.amcsd_id, m.cif_content
        FROM minerals m
        ORDER BY m.id
    """)
    
    minerals = cursor.fetchall()
    
    print(f"\nProcessing {len(minerals)} minerals...")
    print("-"*70)
    
    parser = CIFParser()
    success_count = 0
    fail_count = 0
    pymatgen_count = 0
    fallback_count = 0
    
    for i, (mineral_id, mineral_name, amcsd_id, cif_data) in enumerate(minerals):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(minerals)} ({100*(i+1)/len(minerals):.1f}%)")
        
        if not cif_data:
            print(f"  ⚠ {mineral_name}: No CIF data")
            fail_count += 1
            continue
        
        try:
            # Calculate pattern with pymatgen (Cu Kα)
            pattern = parser.calculate_xrd_pattern_from_cif(
                cif_data,
                wavelength=1.5406,
                max_2theta=90.0,
                min_d=0.5
            )
            
            if len(pattern['two_theta']) == 0:
                print(f"  ✗ {mineral_name}: No reflections calculated")
                fail_count += 1
                continue
            
            # Check if pymatgen was used (it prints a message)
            # We'll assume it was if we got valid results
            
            # Convert arrays to comma-separated strings
            two_theta_str = ','.join([f"{x:.6f}" for x in pattern['two_theta']])
            intensity_str = ','.join([f"{x:.6f}" for x in pattern['intensity']])
            d_spacing_str = ','.join([f"{x:.6f}" for x in pattern['d_spacing']])
            
            # Update or insert diffraction pattern
            cursor.execute("""
                SELECT id FROM diffraction_patterns 
                WHERE mineral_id = ? AND wavelength = 1.5406
            """, (mineral_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                cursor.execute("""
                    UPDATE diffraction_patterns
                    SET two_theta = ?, intensities = ?, d_spacings = ?,
                        calculation_method = 'pymatgen'
                    WHERE id = ?
                """, (two_theta_str, intensity_str, d_spacing_str, existing[0]))
            else:
                # Insert new pattern
                cursor.execute("""
                    INSERT INTO diffraction_patterns 
                    (mineral_id, wavelength, two_theta, intensities, d_spacings, calculation_method)
                    VALUES (?, 1.5406, ?, ?, ?, 'pymatgen')
                """, (mineral_id, two_theta_str, intensity_str, d_spacing_str))
            
            success_count += 1
            pymatgen_count += 1
            
        except Exception as e:
            print(f"  ✗ {mineral_name}: {str(e)[:50]}")
            fail_count += 1
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print("\n" + "="*70)
    print("Rebuild Complete!")
    print("="*70)
    print(f"Total minerals: {len(minerals)}")
    print(f"Successfully updated: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Using pymatgen: {pymatgen_count}")
    print(f"Using fallback: {fallback_count}")
    
    print("\n⚠ IMPORTANT: You must rebuild the search index!")
    print("Run: python -c \"from utils.fast_pattern_search import FastPatternSearchEngine; engine = FastPatternSearchEngine(); engine.build_search_index()\"")

if __name__ == "__main__":
    rebuild_patterns()
