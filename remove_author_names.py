#!/usr/bin/env python3
"""
Simple script: Remove obvious author names from database
This is faster than trying to correct them
"""

import sqlite3
from utils.local_database import LocalCIFDatabase

def remove_author_names():
    """Remove entries that are obviously author names"""
    print("="*70)
    print("Removing Author Names from Database")
    print("="*70)
    
    db = LocalCIFDatabase()
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Get all minerals
    print("\nFetching minerals from database...")
    cursor.execute("SELECT id, mineral_name FROM minerals ORDER BY mineral_name")
    minerals = cursor.fetchall()
    
    print(f"Found {len(minerals)} minerals to check\n")
    
    # Identify author names
    author_patterns = [
        ' A,', ' B,', ' C,', ' D,', ' E,', ' F,', ' G,', ' H,', 
        ' I,', ' J,', ' K,', ' L,', ' M,', ' N,', ' O,', ' P,',
        ' Q,', ' R,', ' S,', ' T,', ' U,', ' V,', ' W,', ' X,',
        ' Y,', ' Z,',
        ' A ', ' B ', ' C ', ' D ', ' E ', ' F ', ' G ', ' H ',
        ' I ', ' J ', ' K ', ' L ', ' M ', ' N ', ' O ', ' P ',
        ' Q ', ' R ', ' S ', ' T ', ' U ', ' V ', ' W ', ' X ',
        ' Y ', ' Z '
    ]
    
    author_entries = []
    
    print("Identifying author names...")
    for mineral_id, mineral_name in minerals:
        # Check for author name patterns
        if any(pattern in mineral_name for pattern in author_patterns):
            author_entries.append((mineral_id, mineral_name))
    
    print(f"\n✓ Found {len(author_entries)} entries that appear to be author names")
    
    if len(author_entries) == 0:
        print("\n✅ No author names found - database is clean!")
        conn.close()
        return
    
    # Show examples
    print("\nExamples of entries to be removed:")
    print("-"*70)
    for mineral_id, name in author_entries[:20]:
        print(f"  - {name}")
    
    if len(author_entries) > 20:
        print(f"  ... and {len(author_entries) - 20} more")
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input(f"\nDelete {len(author_entries)} author name entries? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Cancelled - no changes made")
        conn.close()
        return
    
    # Delete entries
    print("\nDeleting author name entries...")
    for mineral_id, name in author_entries:
        # Delete diffraction patterns first (foreign key)
        cursor.execute("DELETE FROM diffraction_patterns WHERE mineral_id = ?", (mineral_id,))
        # Delete mineral elements
        cursor.execute("DELETE FROM mineral_elements WHERE mineral_id = ?", (mineral_id,))
        # Delete mineral
        cursor.execute("DELETE FROM minerals WHERE id = ?", (mineral_id,))
    
    conn.commit()
    
    print(f"✅ Deleted {len(author_entries)} author name entries")
    
    # Show remaining count
    cursor.execute("SELECT COUNT(*) FROM minerals")
    remaining = cursor.fetchone()[0]
    print(f"✅ Database now has {remaining} minerals")
    
    conn.close()
    
    # Recommend next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Rebuild search index:")
    print("   python -c \"from utils.fast_pattern_search import FastPatternSearchEngine;")
    print("   engine = FastPatternSearchEngine(); engine.build_search_index(force_rebuild=True)\"")
    print("\n2. Optionally re-import DIF file to restore correct minerals:")
    print("   python import_dif_data.py")

if __name__ == "__main__":
    remove_author_names()
