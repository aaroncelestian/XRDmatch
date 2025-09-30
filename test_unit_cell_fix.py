#!/usr/bin/env python3
"""
Test that unit cell parameters are properly retrieved and passed to Le Bail refinement
"""

import sqlite3
import numpy as np

def test_unit_cell_retrieval():
    """Test that unit cell parameters are in the database and can be retrieved"""
    print("="*70)
    print("Testing Unit Cell Parameter Retrieval")
    print("="*70)
    
    # Connect to database
    db_path = "data/local_cif_database.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test query similar to what fast_pattern_search uses
        cursor.execute('''
            SELECT m.id, m.mineral_name, m.chemical_formula, m.space_group,
                   m.cell_a, m.cell_b, m.cell_c, m.cell_alpha, m.cell_beta, m.cell_gamma,
                   dp.wavelength
            FROM minerals m
            JOIN diffraction_patterns dp ON m.id = dp.mineral_id
            WHERE m.mineral_name LIKE '%Epsomite%'
            OR m.mineral_name LIKE '%Hexahydrite%'
            OR m.mineral_name LIKE '%Meridianiite%'
            LIMIT 5
        ''')
        
        results = cursor.fetchall()
        
        if not results:
            print("\n✗ No test minerals found in database!")
            return False
        
        print(f"\n✓ Found {len(results)} test mineral(s)\n")
        print("="*70)
        
        all_have_cell_params = True
        
        for row in results:
            mineral_id, mineral_name, formula, space_group, cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, wavelength = row
            
            print(f"\nMineral: {mineral_name}")
            print(f"Formula: {formula}")
            print(f"Space Group: {space_group}")
            print(f"Wavelength: {wavelength} Å")
            print(f"\nUnit Cell Parameters:")
            print(f"  a = {cell_a} Å")
            print(f"  b = {cell_b} Å")
            print(f"  c = {cell_c} Å")
            print(f"  α = {cell_alpha}°")
            print(f"  β = {cell_beta}°")
            print(f"  γ = {cell_gamma}°")
            
            # Check if parameters are valid (not None and not default 10.0)
            if cell_a is None or cell_b is None or cell_c is None:
                print(f"  ✗ WARNING: Missing unit cell parameters!")
                all_have_cell_params = False
            elif (cell_a == 10.0 and cell_b == 10.0 and cell_c == 10.0 and 
                  cell_alpha == 90.0 and cell_beta == 90.0 and cell_gamma == 90.0):
                print(f"  ⚠️  WARNING: Unit cell has default values (10.0, 10.0, 10.0, 90, 90, 90)")
                all_have_cell_params = False
            else:
                print(f"  ✓ Unit cell parameters look valid")
            
            print("-"*70)
        
        conn.close()
        
        print("\n" + "="*70)
        if all_have_cell_params:
            print("✓ All test minerals have valid unit cell parameters")
            print("="*70)
            return True
        else:
            print("⚠️  Some minerals have missing or default unit cell parameters")
            print("   This may indicate the database needs to be rebuilt or updated")
            print("="*70)
            return False
            
    except Exception as e:
        print(f"\n✗ Database error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unit_cell_retrieval()
    exit(0 if success else 1)
