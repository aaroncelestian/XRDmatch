#!/usr/bin/env python3
"""
Detailed test of Epsomite CIF parsing and pattern calculation
Compare with CrystalDiffract to find the discrepancy
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.cif_parser import CIFParser
import requests

def test_epsomite_detailed():
    """Detailed analysis of Epsomite pattern calculation"""
    print("="*70)
    print("Epsomite CIF Parsing and Pattern Calculation Test")
    print("="*70)
    
    # Try multiple Epsomite structures from AMCSD
    epsomite_ids = ["0015636", "0001391", "0011138"]
    
    for amcsd_id in epsomite_ids:
        print(f"\n{'='*70}")
        print(f"Testing AMCSD ID: {amcsd_id}")
        print(f"{'='*70}")
        
        # Fetch CIF
        cif_url = f"https://rruff.geo.arizona.edu/AMS/xtal_data/CIFfiles/{amcsd_id}.cif"
        
        try:
            response = requests.get(cif_url, timeout=10)
            if response.status_code != 200:
                print(f"✗ Failed to fetch CIF: Status {response.status_code}")
                continue
            
            cif_content = response.text
            print(f"✓ Successfully fetched CIF data ({len(cif_content)} bytes)")
            
        except Exception as e:
            print(f"✗ Error fetching CIF: {e}")
            continue
        
        # Parse CIF
        parser = CIFParser()
        cif_data = parser.parse_content(cif_content)
        
        if not cif_data:
            print("✗ Failed to parse CIF")
            continue
        
        block_name = list(cif_data.keys())[0]
        data = cif_data[block_name]
        
        # Extract crystal info
        print("\n1. Crystal Structure Information:")
        print("-"*70)
        
        mineral = data.get('_chemical_name_mineral', 'Unknown')
        formula = data.get('_chemical_formula_sum', 'Unknown')
        space_group = (data.get('_space_group_name_H-M_alt') or 
                      data.get('_symmetry_space_group_name_H-M', 'Unknown'))
        
        print(f"Mineral: {mineral}")
        print(f"Formula: {formula}")
        print(f"Space group: {space_group}")
        
        # Unit cell
        try:
            a = float(data.get('_cell_length_a', '0').split('(')[0])
            b = float(data.get('_cell_length_b', '0').split('(')[0])
            c = float(data.get('_cell_length_c', '0').split('(')[0])
            alpha = float(data.get('_cell_angle_alpha', '90').split('(')[0])
            beta = float(data.get('_cell_angle_beta', '90').split('(')[0])
            gamma = float(data.get('_cell_angle_gamma', '90').split('(')[0])
            
            print(f"\nUnit cell:")
            print(f"  a = {a:.4f} Å")
            print(f"  b = {b:.4f} Å")
            print(f"  c = {c:.4f} Å")
            print(f"  α = {alpha:.2f}°")
            print(f"  β = {beta:.2f}°")
            print(f"  γ = {gamma:.2f}°")
            
        except Exception as e:
            print(f"✗ Error parsing unit cell: {e}")
            continue
        
        # Atomic positions
        atoms = parser.extract_atomic_positions(data)
        print(f"\n2. Atomic Positions: {len(atoms)} atoms found")
        print("-"*70)
        
        if atoms:
            print(f"{'Label':<10} {'Symbol':<8} {'x':>8} {'y':>8} {'z':>8} {'Occ':>6}")
            print("-"*70)
            for atom in atoms[:10]:  # Show first 10
                label = atom.get('label', 'N/A')
                symbol = atom.get('symbol', 'N/A')
                x = atom.get('x', 0.0)
                y = atom.get('y', 0.0)
                z = atom.get('z', 0.0)
                occ = atom.get('occupancy', 1.0)
                print(f"{label:<10} {symbol:<8} {x:8.5f} {y:8.5f} {z:8.5f} {occ:6.3f}")
            
            if len(atoms) > 10:
                print(f"... and {len(atoms)-10} more atoms")
        else:
            print("⚠ WARNING: No atomic positions found!")
            print("This will result in incorrect intensities!")
        
        # Calculate pattern with pymatgen
        print("\n3. Calculating XRD Pattern with pymatgen:")
        print("-"*70)
        
        try:
            pattern_pymatgen = parser.calculate_xrd_pattern_from_cif(
                cif_content,
                wavelength=0.2401,
                max_2theta=20.0,
                min_d=0.5
            )
            
            if len(pattern_pymatgen['two_theta']) > 0:
                print(f"✓ Pymatgen: {len(pattern_pymatgen['two_theta'])} reflections")
                
                # Show first 10 peaks
                print(f"\n{'2θ (°)':>8} {'d (Å)':>8} {'I':>8}")
                print("-"*30)
                for i in range(min(10, len(pattern_pymatgen['two_theta']))):
                    tt = pattern_pymatgen['two_theta'][i]
                    d = pattern_pymatgen['d_spacing'][i]
                    intensity = pattern_pymatgen['intensity'][i]
                    print(f"{tt:8.3f} {d:8.4f} {intensity:8.1f}")
                
                # Check if strongest peak is at low angle
                max_idx = np.argmax(pattern_pymatgen['intensity'])
                strongest_2theta = pattern_pymatgen['two_theta'][max_idx]
                strongest_d = pattern_pymatgen['d_spacing'][max_idx]
                
                print(f"\nStrongest peak:")
                print(f"  2θ = {strongest_2theta:.3f}°")
                print(f"  d = {strongest_d:.4f} Å")
                print(f"  I = {pattern_pymatgen['intensity'][max_idx]:.1f}")
                
                # For Epsomite, strongest should be around 1.16° (d~11.86Å) for synchrotron
                if strongest_2theta < 2.0 and strongest_d > 10.0:
                    print("  ✓ Looks correct for Epsomite!")
                else:
                    print("  ⚠ This doesn't look like Epsomite's strongest peak")
                
            else:
                print("✗ Pymatgen produced no reflections")
                
        except Exception as e:
            print(f"✗ Pymatgen calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Try fallback method
        print("\n4. Calculating XRD Pattern with fallback method:")
        print("-"*70)
        
        try:
            pattern_fallback = parser._calculate_xrd_pattern_improved_fallback(
                cif_content,
                wavelength=0.2401,
                max_2theta=20.0,
                min_d=0.5
            )
            
            if len(pattern_fallback['two_theta']) > 0:
                print(f"✓ Fallback: {len(pattern_fallback['two_theta'])} reflections")
                
                # Show first 10 peaks
                print(f"\n{'2θ (°)':>8} {'d (Å)':>8} {'I':>8}")
                print("-"*30)
                for i in range(min(10, len(pattern_fallback['two_theta']))):
                    tt = pattern_fallback['two_theta'][i]
                    d = pattern_fallback['d_spacing'][i]
                    intensity = pattern_fallback['intensity'][i]
                    print(f"{tt:8.3f} {d:8.4f} {intensity:8.1f}")
                
                max_idx = np.argmax(pattern_fallback['intensity'])
                strongest_2theta = pattern_fallback['two_theta'][max_idx]
                strongest_d = pattern_fallback['d_spacing'][max_idx]
                
                print(f"\nStrongest peak:")
                print(f"  2θ = {strongest_2theta:.3f}°")
                print(f"  d = {strongest_d:.4f} Å")
                print(f"  I = {pattern_fallback['intensity'][max_idx]:.1f}")
                
            else:
                print("✗ Fallback produced no reflections")
                
        except Exception as e:
            print(f"✗ Fallback calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Check if DIF file exists
        print("\n5. Checking for DIF file:")
        print("-"*70)
        
        dif_data = parser.fetch_dif_from_amcsd(amcsd_id)
        if dif_data and len(dif_data['two_theta']) > 0:
            print(f"✓ DIF file found: {len(dif_data['two_theta'])} peaks")
            
            # Show first 10 peaks
            print(f"\n{'2θ (°)':>8} {'d (Å)':>8} {'I':>8}")
            print("-"*30)
            for i in range(min(10, len(dif_data['two_theta']))):
                tt = dif_data['two_theta'][i]
                d = dif_data['d_spacing'][i]
                intensity = dif_data['intensity'][i]
                print(f"{tt:8.3f} {d:8.4f} {intensity:8.1f}")
            
            max_idx = np.argmax(dif_data['intensity'])
            print(f"\nStrongest peak in DIF:")
            print(f"  2θ = {dif_data['two_theta'][max_idx]:.3f}° (Cu Kα)")
            print(f"  d = {dif_data['d_spacing'][max_idx]:.4f} Å")
            print(f"  I = {dif_data['intensity'][max_idx]:.1f}")
            
        else:
            print("✗ No DIF file found")
        
        print("\n" + "="*70)
        break  # Only test first successful one

if __name__ == "__main__":
    test_epsomite_detailed()
