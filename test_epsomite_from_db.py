#!/usr/bin/env python3
"""
Check what Epsomite pattern is actually in the database
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def test_epsomite_from_database():
    """Check Epsomite pattern in the local database"""
    print("="*70)
    print("Epsomite Pattern from Local Database")
    print("="*70)
    
    # Connect to database
    db_path = "data/minerals.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Search for Epsomite
        cursor.execute("""
            SELECT m.id, m.mineral_name, m.chemical_formula, m.amcsd_id,
                   dp.wavelength, dp.two_theta, dp.intensities, dp.d_spacings
            FROM minerals m
            JOIN diffraction_patterns dp ON m.id = dp.mineral_id
            WHERE m.mineral_name LIKE '%Epsomite%'
            OR m.chemical_formula LIKE '%MgSO4%'
            ORDER BY m.mineral_name
        """)
        
        results = cursor.fetchall()
        
        if not results:
            print("✗ No Epsomite found in database!")
            return
        
        print(f"\n✓ Found {len(results)} Epsomite pattern(s)")
        print("="*70)
        
        for row in results:
            mineral_id, mineral_name, formula, amcsd_id, wavelength, two_theta_str, intensities_str, d_spacings_str = row
            
            print(f"\nMineral: {mineral_name}")
            print(f"Formula: {formula}")
            print(f"AMCSD ID: {amcsd_id}")
            print(f"Wavelength: {wavelength} Å")
            
            # Parse arrays
            try:
                two_theta = np.array([float(x) for x in two_theta_str.split(',')])
                intensities = np.array([float(x) for x in intensities_str.split(',')])
                d_spacings = np.array([float(x) for x in d_spacings_str.split(',')])
                
                print(f"Number of peaks: {len(two_theta)}")
                print(f"2θ range: {np.min(two_theta):.2f}° to {np.max(two_theta):.2f}°")
                print(f"d-spacing range: {np.min(d_spacings):.3f} to {np.max(d_spacings):.3f} Å")
                
                # Find strongest peak
                max_idx = np.argmax(intensities)
                print(f"\nStrongest peak (in database):")
                print(f"  2θ = {two_theta[max_idx]:.3f}° (at λ={wavelength}Å)")
                print(f"  d = {d_spacings[max_idx]:.4f} Å")
                print(f"  I = {intensities[max_idx]:.1f}")
                
                # Show first 10 peaks
                print(f"\nFirst 10 peaks:")
                print(f"{'2θ (°)':>8} {'d (Å)':>8} {'I':>8}")
                print("-"*30)
                for i in range(min(10, len(two_theta))):
                    print(f"{two_theta[i]:8.3f} {d_spacings[i]:8.4f} {intensities[i]:8.1f}")
                
                # Convert to synchrotron wavelength
                print(f"\n" + "="*70)
                print("Converting to Synchrotron (0.2401 Å):")
                print("="*70)
                
                target_wavelength = 0.2401
                
                # Calculate new 2θ from d-spacings
                new_two_theta = []
                valid_intensities = []
                valid_d = []
                
                for d, intensity in zip(d_spacings, intensities):
                    sin_theta = target_wavelength / (2 * d)
                    if sin_theta <= 1.0:
                        theta = np.arcsin(sin_theta)
                        tt = 2 * np.degrees(theta)
                        if 1.0 <= tt <= 20.0:
                            new_two_theta.append(tt)
                            valid_intensities.append(intensity)
                            valid_d.append(d)
                
                new_two_theta = np.array(new_two_theta)
                valid_intensities = np.array(valid_intensities)
                valid_d = np.array(valid_d)
                
                print(f"\nAfter conversion:")
                print(f"Number of peaks in range: {len(new_two_theta)}")
                print(f"2θ range: {np.min(new_two_theta):.2f}° to {np.max(new_two_theta):.2f}°")
                
                # Find strongest peak after conversion
                max_idx = np.argmax(valid_intensities)
                print(f"\nStrongest peak (after conversion, NO LP correction):")
                print(f"  2θ = {new_two_theta[max_idx]:.3f}°")
                print(f"  d = {valid_d[max_idx]:.4f} Å")
                print(f"  I = {valid_intensities[max_idx]:.1f}")
                
                # Show first 10 peaks
                print(f"\nFirst 10 peaks after conversion:")
                print(f"{'2θ (°)':>8} {'d (Å)':>8} {'I':>8}")
                print("-"*30)
                for i in range(min(10, len(new_two_theta))):
                    print(f"{new_two_theta[i]:8.3f} {valid_d[i]:8.4f} {valid_intensities[i]:8.1f}")
                
                # Plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Original Cu Kα
                for tt, intensity in zip(two_theta, intensities):
                    ax1.plot([tt, tt], [0, intensity], 'b-', linewidth=1.5)
                ax1.set_xlim(0, 90)
                ax1.set_ylim(0, 110)
                ax1.set_xlabel('2θ (degrees)')
                ax1.set_ylabel('Intensity')
                ax1.set_title(f'Epsomite from Database - Cu Kα ({wavelength} Å)')
                ax1.grid(True, alpha=0.3)
                
                # Converted to synchrotron
                for tt, intensity in zip(new_two_theta, valid_intensities):
                    ax2.plot([tt, tt], [0, intensity], 'r-', linewidth=1.5)
                ax2.set_xlim(0, 20)
                ax2.set_ylim(0, 110)
                ax2.set_xlabel('2θ (degrees)')
                ax2.set_ylabel('Intensity')
                ax2.set_title('Epsomite Converted to Synchrotron (0.2401 Å) - NO LP correction')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('epsomite_from_database.png', dpi=150, bbox_inches='tight')
                print("\n✓ Saved plot: epsomite_from_database.png")
                
                plt.show()
                
            except Exception as e:
                print(f"✗ Error parsing pattern data: {e}")
                import traceback
                traceback.print_exc()
        
        conn.close()
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_epsomite_from_database()
