#!/usr/bin/env python3
"""
Import DIF data from AMCSD difdata.txt file and populate database with correct patterns
This uses pre-calculated patterns with proper space group symmetry
"""

import sqlite3
import numpy as np
import re
from utils.local_database import LocalCIFDatabase
from utils.ima_mineral_database import get_ima_database

def parse_dif_file(filepath):
    """Parse the AMCSD difdata.txt file"""
    print("="*70)
    print("Parsing AMCSD DIF Data File with IMA Validation")
    print("="*70)
    
    # Load IMA database for mineral name validation
    print("\nLoading IMA database for name validation...")
    ima_db = get_ima_database()
    
    print(f"\nReading file: {filepath}...")
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        content = f.read()
    
    print(f"✓ File read: {len(content):,} characters")
    
    # Split into individual mineral sections
    # Each section ends with "===" or "_END_"
    print("\nSplitting into mineral sections...")
    sections = re.split(r'_END_|={50,}', content)
    print(f"✓ Found {len(sections)} sections")
    
    minerals = []
    parsed_count = 0
    skipped_count = 0
    ima_corrected_count = 0
    ima_verified_count = 0
    
    print("\nParsing mineral data...")
    for i, section in enumerate(sections):
        if (i + 1) % 500 == 0:
            print(f"  Processing section {i+1}/{len(sections)} ({100*(i+1)/len(sections):.1f}%) - Found {parsed_count} minerals so far...")
        if not section.strip():
            continue
        
        # Look for AMCSD code
        amcsd_match = re.search(r'_database_code_amcsd\s+(\d+)', section)
        if not amcsd_match:
            continue
        
        amcsd_id = amcsd_match.group(1).zfill(7)  # Pad to 7 digits
        
        # Extract mineral name - NEW STRATEGY based on DIF format
        # The mineral name comes on the first non-empty line of the section
        # (which is right after _END_ from the previous section)
        lines = section.split('\n')
        mineral_name = None
        possible_names = []
        
        # Strategy 1: First non-empty line is the mineral name
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and not line.startswith('='):
                possible_names.append(line)
                break  # Take the FIRST non-empty line
        
        # Strategy 2: Look for _chemical_name_mineral tag (backup)
        mineral_tag_match = re.search(r'_chemical_name_mineral\s+["\']?([^"\'\n]+)["\']?', section)
        if mineral_tag_match:
            possible_names.insert(0, mineral_tag_match.group(1).strip())  # Prioritize this
        
        # Strategy 3: Validate against IMA database
        original_name = None
        for candidate in possible_names:
            # Try exact match first
            ima_info = ima_db.get_mineral_info(candidate)
            if ima_info:
                original_name = candidate
                mineral_name = ima_info['name']  # Use official IMA name
                if original_name.lower() == mineral_name.lower():
                    ima_verified_count += 1
                else:
                    ima_corrected_count += 1
                break
            
            # Try fuzzy match
            fuzzy_result = ima_db.fuzzy_match_mineral(candidate, threshold=0.85)
            if fuzzy_result:
                matched_name, score, ima_info = fuzzy_result
                original_name = candidate
                mineral_name = matched_name  # Use matched IMA name
                ima_corrected_count += 1
                break
        
        # Fallback: use first candidate if no IMA match found
        if not mineral_name and possible_names:
            mineral_name = possible_names[0]
        
        if not mineral_name:
            continue
        
        # Extract wavelength
        wavelength = 1.5406  # Default Cu Kα
        wavelength_match = re.search(r'X-RAY WAVELENGTH:\s+([\d.]+)', section)
        if wavelength_match:
            wavelength = float(wavelength_match.group(1))
        
        # Extract peak data
        # Format: 2-THETA  INTENSITY  D-SPACING  H  K  L
        peak_pattern = r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)'
        
        # Find where peak data starts (after "2-THETA" header)
        peak_section_start = section.find('2-THETA')
        if peak_section_start == -1:
            continue
        
        peak_section = section[peak_section_start:]
        peaks = re.findall(peak_pattern, peak_section)
        
        if not peaks:
            continue
        
        two_theta = []
        intensities = []
        d_spacings = []
        hkl = []
        
        for peak in peaks:
            tt, intensity, d, h, k, l = peak
            two_theta.append(float(tt))
            intensities.append(float(intensity))
            d_spacings.append(float(d))
            hkl.append((int(h), int(k), int(l)))
        
        if len(two_theta) == 0:
            skipped_count += 1
            continue
        
        minerals.append({
            'mineral_name': mineral_name,
            'amcsd_id': amcsd_id,
            'wavelength': wavelength,
            'two_theta': np.array(two_theta),
            'intensities': np.array(intensities),
            'd_spacings': np.array(d_spacings),
            'hkl': hkl
        })
        parsed_count += 1
    
    print(f"\n{'='*70}")
    print(f"✓ Parsing complete!")
    print(f"  Successfully parsed: {parsed_count} minerals")
    print(f"  IMA verified (exact match): {ima_verified_count}")
    print(f"  IMA corrected (fuzzy match): {ima_corrected_count}")
    print(f"  Not in IMA database: {parsed_count - ima_verified_count - ima_corrected_count}")
    print(f"  Skipped (no data): {skipped_count} sections")
    print(f"{'='*70}")
    return minerals

def import_to_database(minerals):
    """Import DIF patterns into the database"""
    print("\n" + "="*70)
    print("Importing DIF Patterns to Database")
    print("="*70)
    
    db = LocalCIFDatabase()
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    success_count = 0
    not_found_count = 0
    updated_count = 0
    
    print(f"\nProcessing {len(minerals)} minerals...")
    print("(Progress updates every 100 minerals)\n")
    
    for i, mineral_data in enumerate(minerals):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(minerals)} ({100*(i+1)/len(minerals):.1f}%) - "
                  f"Added: {success_count}, Updated: {updated_count}, Not found: {not_found_count}")
        
        mineral_name = mineral_data['mineral_name']
        amcsd_id = mineral_data['amcsd_id']
        wavelength = mineral_data['wavelength']
        
        # Find mineral in database by AMCSD ID
        cursor.execute("""
            SELECT id, mineral_name FROM minerals 
            WHERE amcsd_id = ?
        """, (amcsd_id,))
        
        result = cursor.fetchone()
        
        if not result:
            not_found_count += 1
            continue
        
        mineral_id, db_mineral_name = result
        
        # Convert arrays to comma-separated strings
        two_theta_str = ','.join([f"{x:.6f}" for x in mineral_data['two_theta']])
        intensity_str = ','.join([f"{x:.6f}" for x in mineral_data['intensities']])
        d_spacing_str = ','.join([f"{x:.6f}" for x in mineral_data['d_spacings']])
        
        # Check if pattern already exists
        cursor.execute("""
            SELECT id FROM diffraction_patterns 
            WHERE mineral_id = ? AND ABS(wavelength - ?) < 0.0001
        """, (mineral_id, wavelength))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing pattern
            cursor.execute("""
                UPDATE diffraction_patterns
                SET two_theta = ?, intensities = ?, d_spacings = ?,
                    calculation_method = 'AMCSD_DIF'
                WHERE id = ?
            """, (two_theta_str, intensity_str, d_spacing_str, existing[0]))
            updated_count += 1
        else:
            # Insert new pattern
            cursor.execute("""
                INSERT INTO diffraction_patterns 
                (mineral_id, wavelength, two_theta, intensities, d_spacings, calculation_method)
                VALUES (?, ?, ?, ?, ?, 'AMCSD_DIF')
            """, (mineral_id, wavelength, two_theta_str, intensity_str, d_spacing_str))
            success_count += 1
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*70)
    print("Import Complete!")
    print("="*70)
    print(f"New patterns added: {success_count}")
    print(f"Existing patterns updated: {updated_count}")
    print(f"Minerals not found in database: {not_found_count}")
    print(f"Total processed: {len(minerals)}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Rebuild search index:")
    print("   python -c \"from utils.fast_pattern_search import FastPatternSearchEngine;")
    print("   engine = FastPatternSearchEngine(); engine.build_search_index()\"")
    print("\n2. Test with Calcite - should now show correct intensities!")
    print("   Strongest peak should be at ~29.4° (not ~5° or ~23°)")

def main():
    """Main import function"""
    # Try both possible filenames
    import os
    if os.path.exists('data/difdata.dif'):
        dif_file = 'data/difdata.dif'
    elif os.path.exists('data/difdata.txt'):
        dif_file = 'data/difdata.txt'
    else:
        print("❌ Error: Could not find DIF file!")
        print("   Looking for: data/difdata.dif or data/difdata.txt")
        return
    
    print("Starting DIF data import...")
    print(f"Reading from: {dif_file}\n")
    
    # Parse DIF file
    minerals = parse_dif_file(dif_file)
    
    if not minerals:
        print("✗ No minerals parsed from DIF file!")
        return
    
    # Show some examples
    print("\nExample minerals found:")
    print("-"*70)
    for mineral in minerals[:5]:
        print(f"{mineral['mineral_name']:<30} AMCSD: {mineral['amcsd_id']}  "
              f"Peaks: {len(mineral['two_theta']):<4}  λ: {mineral['wavelength']:.4f}Å")
    
    if len(minerals) > 5:
        print(f"... and {len(minerals)-5} more")
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Import these patterns to database? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Cancelled")
        return
    
    # Import to database
    import_to_database(minerals)

if __name__ == "__main__":
    main()
