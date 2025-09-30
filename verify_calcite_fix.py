#!/usr/bin/env python3
"""
Verify that Calcite pattern is now correct after the fix
"""

import sqlite3
import numpy as np
from utils.local_database import LocalCIFDatabase

def verify_calcite():
    """Check Calcite pattern in database"""
    print("="*70)
    print("Verifying Calcite Pattern Fix")
    print("="*70)
    
    db = LocalCIFDatabase()
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Get Calcite pattern
    cursor.execute("""
        SELECT m.mineral_name, m.amcsd_id, dp.two_theta, dp.intensities, dp.d_spacings
        FROM minerals m
        JOIN diffraction_patterns dp ON m.id = dp.mineral_id
        WHERE m.mineral_name LIKE '%Calcite%'
        AND dp.wavelength = 1.5406
        ORDER BY m.id
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    
    if not result:
        print("✗ Calcite not found!")
        return
    
    mineral_name, amcsd_id, two_theta_str, intensities_str, d_spacings_str = result
    
    # Parse arrays
    two_theta = np.array([float(x) for x in two_theta_str.split(',')])
    intensities = np.array([float(x) for x in intensities_str.split(',')])
    d_spacings = np.array([float(x) for x in d_spacings_str.split(',')])
    
    print(f"\nMineral: {mineral_name}")
    print(f"AMCSD ID: {amcsd_id}")
    print(f"Number of peaks: {len(two_theta)}")
    
    # Find strongest peak
    max_idx = np.argmax(intensities)
    strongest_2theta = two_theta[max_idx]
    strongest_d = d_spacings[max_idx]
    
    print(f"\n{'='*70}")
    print("STRONGEST PEAK:")
    print(f"{'='*70}")
    print(f"2θ = {strongest_2theta:.3f}°")
    print(f"d = {strongest_d:.4f} Å")
    print(f"I = {intensities[max_idx]:.1f}")
    
    # Check if it's correct
    print(f"\n{'='*70}")
    print("VERIFICATION:")
    print(f"{'='*70}")
    
    if 29.0 <= strongest_2theta <= 30.0:
        print("✅ CORRECT! Strongest peak is at ~29.4° (104 reflection)")
        print("   This is the expected position for Calcite's main peak")
        print("   The angle-dependent scattering factor fix is WORKING!")
    elif 22.5 <= strongest_2theta <= 23.5:
        print("✗ WRONG! Strongest peak is still at ~23°")
        print("   This indicates the old bug is still present")
        print("   The pattern needs to be regenerated")
    else:
        print(f"⚠ UNEXPECTED! Strongest peak at {strongest_2theta:.1f}°")
        print("   This doesn't match either expected position")
    
    # Show top 5 peaks
    print(f"\n{'='*70}")
    print("TOP 5 PEAKS:")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'2θ (°)':>8} {'d (Å)':>8} {'I':>8}")
    print("-"*35)
    
    # Sort by intensity
    sorted_indices = np.argsort(intensities)[::-1]
    for rank, idx in enumerate(sorted_indices[:5], 1):
        print(f"{rank:<6} {two_theta[idx]:8.3f} {d_spacings[idx]:8.4f} {intensities[idx]:8.1f}")
    
    # Expected Calcite peaks (for reference)
    print(f"\n{'='*70}")
    print("EXPECTED CALCITE PEAKS (for reference):")
    print(f"{'='*70}")
    print("1. 2θ ≈ 29.4° (d = 3.04 Å) - (104) - Strongest")
    print("2. 2θ ≈ 23.0° (d = 3.86 Å) - (012) - Second")
    print("3. 2θ ≈ 39.4° (d = 2.28 Å) - (113) - Third")
    print("4. 2θ ≈ 43.2° (d = 2.09 Å) - (202)")
    print("5. 2θ ≈ 47.5° (d = 1.91 Å) - (018)")
    
    conn.close()

if __name__ == "__main__":
    verify_calcite()
