#!/usr/bin/env python3
"""
Rebuild database from scratch using only DIF data from AMCSD
This creates a clean database with correct patterns
"""

import os
import sqlite3
import numpy as np
import re
from pathlib import Path

def delete_old_database():
    """Delete old database and cache files"""
    print("="*70)
    print("Step 1: Cleaning up old files")
    print("="*70)
    
    files_to_delete = [
        'data/local_cif_database.db',
        'data/xrd_search_index_0.020_5_90.pkl'
    ]
    
    for filepath in files_to_delete:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✓ Deleted: {filepath}")
        else:
            print(f"  (not found: {filepath})")
    
    print()

def create_new_database():
    """Create fresh database with simplified schema"""
    print("="*70)
    print("Step 2: Creating new database")
    print("="*70)
    
    db_path = 'data/local_cif_database.db'
    os.makedirs('data', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create minerals table
    cursor.execute('''
        CREATE TABLE minerals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mineral_name TEXT NOT NULL,
            chemical_formula TEXT,
            space_group TEXT,
            amcsd_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create diffraction patterns table
    cursor.execute('''
        CREATE TABLE diffraction_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mineral_id INTEGER,
            wavelength REAL NOT NULL,
            two_theta TEXT NOT NULL,
            intensities TEXT NOT NULL,
            d_spacings TEXT NOT NULL,
            calculation_method TEXT DEFAULT 'AMCSD_DIF',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mineral_id) REFERENCES minerals (id),
            UNIQUE (mineral_id, wavelength)
        )
    ''')
    
    # Create indices
    cursor.execute('CREATE INDEX idx_mineral_name ON minerals (mineral_name)')
    cursor.execute('CREATE INDEX idx_amcsd_id ON minerals (amcsd_id)')
    cursor.execute('CREATE INDEX idx_formula ON minerals (chemical_formula)')
    
    conn.commit()
    conn.close()
    
    print(f"✓ Created new database: {db_path}")
    print()

def parse_dif_file(filepath):
    """Parse AMCSD DIF file"""
    print("="*70)
    print("Step 3: Parsing DIF data")
    print("="*70)
    
    print(f"Reading: {filepath}...")
    with open(filepath, 'r', encoding='latin-1', errors='ignore') as f:
        content = f.read()
    
    print(f"✓ File read: {len(content):,} characters")
    
    sections = re.split(r'_END_|={50,}', content)
    print(f"✓ Found {len(sections)} sections")
    
    minerals = []
    print("\nParsing minerals...")
    
    for i, section in enumerate(sections):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(sections)} ({100*(i+1)/len(sections):.1f}%)")
        
        if not section.strip():
            continue
        
        # Extract AMCSD ID
        amcsd_match = re.search(r'_database_code_amcsd\s+(\d+)', section)
        if not amcsd_match:
            continue
        
        amcsd_id = amcsd_match.group(1).zfill(7)
        
        # Extract mineral name
        lines = section.split('\n')
        mineral_name = None
        for line in lines:
            line = line.strip()
            if line and not line.startswith('=') and len(line) < 100:
                if not any(x in line.lower() for x in ['copyright', 'reference', 'locality', 'american']):
                    mineral_name = line
                    break
        
        if not mineral_name:
            continue
        
        # Extract space group
        space_group = None
        sg_match = re.search(r'SPACE GROUP:\s+(.+)', section)
        if sg_match:
            space_group = sg_match.group(1).strip()
        
        # Extract wavelength
        wavelength = 1.5406
        wl_match = re.search(r'X-RAY WAVELENGTH:\s+([\d.]+)', section)
        if wl_match:
            wavelength = float(wl_match.group(1))
        
        # Extract peak data
        peak_pattern = r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)'
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
        
        for peak in peaks:
            tt, intensity, d, h, k, l = peak
            two_theta.append(float(tt))
            intensities.append(float(intensity))
            d_spacings.append(float(d))
        
        if len(two_theta) == 0:
            continue
        
        minerals.append({
            'mineral_name': mineral_name,
            'amcsd_id': amcsd_id,
            'space_group': space_group,
            'wavelength': wavelength,
            'two_theta': np.array(two_theta),
            'intensities': np.array(intensities),
            'd_spacings': np.array(d_spacings)
        })
    
    print(f"\n✓ Parsed {len(minerals)} minerals with DIF data")
    print()
    return minerals

def import_to_database(minerals):
    """Import minerals and patterns to database"""
    print("="*70)
    print("Step 4: Importing to database")
    print("="*70)
    
    conn = sqlite3.connect('data/local_cif_database.db')
    cursor = conn.cursor()
    
    print(f"Importing {len(minerals)} minerals...")
    
    for i, mineral_data in enumerate(minerals):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(minerals)} ({100*(i+1)/len(minerals):.1f}%)")
        
        # Check if mineral already exists
        cursor.execute('SELECT id FROM minerals WHERE amcsd_id = ?', (mineral_data['amcsd_id'],))
        existing = cursor.fetchone()
        
        if existing:
            mineral_id = existing[0]
        else:
            # Insert mineral
            cursor.execute('''
                INSERT INTO minerals (mineral_name, chemical_formula, space_group, amcsd_id)
                VALUES (?, NULL, ?, ?)
            ''', (mineral_data['mineral_name'], mineral_data['space_group'], mineral_data['amcsd_id']))
            
            mineral_id = cursor.lastrowid
        
        # Insert pattern (skip if already exists)
        cursor.execute('SELECT id FROM diffraction_patterns WHERE mineral_id = ? AND wavelength = ?', 
                      (mineral_id, mineral_data['wavelength']))
        if not cursor.fetchone():
            two_theta_str = ','.join([f"{x:.6f}" for x in mineral_data['two_theta']])
            intensity_str = ','.join([f"{x:.6f}" for x in mineral_data['intensities']])
            d_spacing_str = ','.join([f"{x:.6f}" for x in mineral_data['d_spacings']])
            
            cursor.execute('''
                INSERT INTO diffraction_patterns 
                (mineral_id, wavelength, two_theta, intensities, d_spacings, calculation_method)
                VALUES (?, ?, ?, ?, ?, 'AMCSD_DIF')
            ''', (mineral_id, mineral_data['wavelength'], two_theta_str, intensity_str, d_spacing_str))
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Imported {len(minerals)} minerals and patterns")
    print()

def build_search_index():
    """Build search index"""
    print("="*70)
    print("Step 5: Building search index")
    print("="*70)
    
    from utils.fast_pattern_search import FastPatternSearchEngine
    
    engine = FastPatternSearchEngine()
    result = engine.build_search_index(force_rebuild=True)
    
    if result:
        print(f"\n✓ Search index built successfully")
    else:
        print(f"\n✗ Search index build failed")
    
    print()

def main():
    """Main rebuild function"""
    print("\n" + "="*70)
    print("REBUILD DATABASE FROM SCRATCH")
    print("="*70)
    print("\nThis will:")
    print("1. Delete old database and cache files")
    print("2. Create new clean database")
    print("3. Import DIF data from AMCSD")
    print("4. Build search index")
    print("\n" + "="*70)
    
    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        return
    
    # Step 1: Delete old files
    delete_old_database()
    
    # Step 2: Create new database
    create_new_database()
    
    # Step 3: Parse DIF file
    minerals = parse_dif_file('data/difdata.txt')
    
    if not minerals:
        print("✗ No minerals parsed!")
        return
    
    # Step 4: Import to database
    import_to_database(minerals)
    
    # Step 5: Build search index
    build_search_index()
    
    print("="*70)
    print("✅ DATABASE REBUILD COMPLETE!")
    print("="*70)
    print(f"\nDatabase contains {len(minerals)} minerals with correct DIF patterns")
    print("\nNext: Update GUI to remove AMCSD tab and simplify Local Database tab")

if __name__ == "__main__":
    main()
