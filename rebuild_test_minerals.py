#!/usr/bin/env python3
"""
Quick test rebuild for specific minerals to verify the angle-dependent scattering factor fix
"""

import sqlite3
import numpy as np
from utils.cif_parser import CIFParser
from utils.local_database import LocalCIFDatabase

def rebuild_test_minerals():
    """Rebuild patterns for test minerals only"""
    print("="*70)
    print("Test Rebuild: Angle-Dependent Scattering Factors")
    print("="*70)
    
    # Test minerals
    test_minerals = [
        'Epsomite',
        'Hexahydrite', 
        'Calcite',
        'Meridianiite',
        'Quartz'
    ]
    
    print(f"\nRebuilding patterns for: {', '.join(test_minerals)}")
    print("-"*70)
    
    # Initialize database
    db = LocalCIFDatabase()
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    parser = CIFParser()
    success_count = 0
    fail_count = 0
    
    for mineral_name in test_minerals:
        print(f"\n{'='*70}")
        print(f"Processing: {mineral_name}")
        print(f"{'='*70}")
        
        # Find mineral in database (case-insensitive, partial match)
        cursor.execute("""
            SELECT m.id, m.mineral_name, m.amcsd_id, m.cif_content
            FROM minerals m
            WHERE LOWER(m.mineral_name) LIKE LOWER(?)
            ORDER BY m.id
            LIMIT 5
        """, (f'%{mineral_name}%',))
        
        minerals = cursor.fetchall()
        
        if not minerals:
            print(f"  ✗ {mineral_name}: Not found in database")
            fail_count += 1
            continue
        
        print(f"  Found {len(minerals)} match(es)")
        
        for mineral_id, found_name, amcsd_id, cif_data in minerals:
            print(f"\n  → {found_name} (AMCSD: {amcsd_id})")
            
            if not cif_data:
                print(f"    ✗ No CIF data")
                continue
            
            try:
                # Calculate pattern with angle-dependent scattering factors
                print(f"    Calculating pattern...")
                pattern = parser.calculate_xrd_pattern_from_cif(
                    cif_data,
                    wavelength=1.5406,
                    max_2theta=90.0,
                    min_d=0.5
                )
                
                if len(pattern['two_theta']) == 0:
                    print(f"    ✗ No reflections calculated")
                    fail_count += 1
                    continue
                
                # Show first 10 peaks
                print(f"    ✓ Calculated {len(pattern['two_theta'])} reflections")
                print(f"\n    First 10 peaks:")
                print(f"    {'2θ (°)':>8} {'d (Å)':>8} {'I':>8}")
                print(f"    {'-'*30}")
                for i in range(min(10, len(pattern['two_theta']))):
                    tt = pattern['two_theta'][i]
                    d = pattern['d_spacing'][i]
                    intensity = pattern['intensity'][i]
                    print(f"    {tt:8.3f} {d:8.4f} {intensity:8.1f}")
                
                # Find strongest peak
                max_idx = np.argmax(pattern['intensity'])
                strongest_2theta = pattern['two_theta'][max_idx]
                strongest_d = pattern['d_spacing'][max_idx]
                
                print(f"\n    Strongest peak:")
                print(f"      2θ = {strongest_2theta:.3f}°")
                print(f"      d = {strongest_d:.4f} Å")
                print(f"      I = {pattern['intensity'][max_idx]:.1f}")
                
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
                            calculation_method = 'angle_dependent_fallback'
                        WHERE id = ?
                    """, (two_theta_str, intensity_str, d_spacing_str, existing[0]))
                    print(f"    ✓ Updated pattern in database")
                else:
                    # Insert new pattern
                    cursor.execute("""
                        INSERT INTO diffraction_patterns 
                        (mineral_id, wavelength, two_theta, intensities, d_spacings, calculation_method)
                        VALUES (?, 1.5406, ?, ?, ?, 'angle_dependent_fallback')
                    """, (mineral_id, two_theta_str, intensity_str, d_spacing_str))
                    print(f"    ✓ Inserted new pattern in database")
                
                success_count += 1
                
            except Exception as e:
                print(f"    ✗ Error: {str(e)[:100]}")
                fail_count += 1
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print("\n" + "="*70)
    print("Test Rebuild Complete!")
    print("="*70)
    print(f"Successfully updated: {success_count}")
    print(f"Failed: {fail_count}")
    
    print("\n" + "="*70)
    print("Expected Results for Common Minerals:")
    print("="*70)
    print("Calcite:")
    print("  Strongest peak: ~29.4° (d = 3.04 Å) - (104) reflection")
    print("  NOT at ~23° (which was the bug)")
    print("\nQuartz:")
    print("  Strongest peak: ~26.6° (d = 3.34 Å) - (101) reflection")
    print("\nEpsomite:")
    print("  Strongest peak: ~7.4° (d = 11.86 Å) for Cu Kα")
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Test in the app - load Calcite data")
    print("2. Check if strongest peak is now at ~29.4° (not ~23°)")
    print("3. If correct, run full rebuild: python rebuild_pattern_database.py")
    print("4. After full rebuild, rebuild search index")

if __name__ == "__main__":
    rebuild_test_minerals()
