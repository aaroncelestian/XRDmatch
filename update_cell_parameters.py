#!/usr/bin/env python3
"""
Quick script to update existing database with unit cell parameters from DIF file
"""

import sqlite3
import re

def update_cell_parameters():
    """Update minerals table with cell parameters from DIF file"""
    
    print("Reading DIF file...")
    with open('data/difdata.dif', 'r', encoding='latin-1', errors='ignore') as f:
        content = f.read()
    
    sections = re.split(r'_END_|={50,}', content)
    print(f"Found {len(sections)} sections")
    
    # Connect to database
    conn = sqlite3.connect('data/local_cif_database.db')
    cursor = conn.cursor()
    
    updated = 0
    not_found = 0
    
    for i, section in enumerate(sections):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(sections)} ({100*(i+1)/len(sections):.1f}%) - Updated: {updated}, Not found: {not_found}")
        
        if not section.strip():
            continue
        
        # Extract AMCSD ID
        amcsd_match = re.search(r'_database_code_amcsd\s+(\d+)', section)
        if not amcsd_match:
            continue
        
        amcsd_id = amcsd_match.group(1).zfill(7)
        
        # Extract space group
        space_group = None
        sg_match = re.search(r'SPACE GROUP:\s+(.+)', section)
        if sg_match:
            space_group = sg_match.group(1).strip().split()[0]
        
        # Extract unit cell parameters
        cell_match = re.search(r'CELL PARAMETERS:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', section)
        if not cell_match:
            continue
        
        cell_a = float(cell_match.group(1))
        cell_b = float(cell_match.group(2))
        cell_c = float(cell_match.group(3))
        cell_alpha = float(cell_match.group(4))
        cell_beta = float(cell_match.group(5))
        cell_gamma = float(cell_match.group(6))
        
        # Update database
        cursor.execute('''
            UPDATE minerals 
            SET cell_a = ?, cell_b = ?, cell_c = ?, 
                cell_alpha = ?, cell_beta = ?, cell_gamma = ?
            WHERE amcsd_id = ?
        ''', (cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, amcsd_id))
        
        if cursor.rowcount > 0:
            updated += 1
        else:
            not_found += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Updated {updated} minerals with cell parameters")
    print(f"  {not_found} minerals not found in database")

if __name__ == "__main__":
    update_cell_parameters()
