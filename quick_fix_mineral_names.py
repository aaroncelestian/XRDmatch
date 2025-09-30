#!/usr/bin/env python3
"""
Quick fix: Update mineral names in database using IMA validation
This is faster than re-importing but less thorough
"""

import sqlite3
from utils.local_database import LocalCIFDatabase
from utils.ima_mineral_database import get_ima_database

def quick_fix_names():
    """Update mineral names in database using IMA database"""
    print("="*70)
    print("Quick Fix: Correcting Mineral Names with IMA Database")
    print("="*70)
    
    # Load databases
    print("\nLoading databases...")
    db = LocalCIFDatabase()
    ima_db = get_ima_database()
    
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Get all minerals
    print("Fetching minerals from database...")
    cursor.execute("SELECT id, mineral_name FROM minerals ORDER BY mineral_name")
    minerals = cursor.fetchall()
    
    print(f"Found {len(minerals)} minerals to check\n")
    
    # Track corrections
    exact_matches = 0
    fuzzy_corrections = 0
    not_found = 0
    corrections_made = []
    
    print("Validating and correcting names...")
    print("-"*70)
    
    for mineral_id, mineral_name in minerals:
        # Skip empty names
        if not mineral_name or len(mineral_name) < 2:
            continue
        
        # Check if it's already correct (exact match)
        ima_info = ima_db.get_mineral_info(mineral_name)
        if ima_info:
            # Already correct
            exact_matches += 1
            continue
        
        # Try fuzzy match to find correct name
        fuzzy_result = ima_db.fuzzy_match_mineral(mineral_name, threshold=0.80)
        if fuzzy_result:
            correct_name, score, ima_info = fuzzy_result
            
            # Update database
            cursor.execute("UPDATE minerals SET mineral_name = ? WHERE id = ?", 
                          (correct_name, mineral_id))
            
            fuzzy_corrections += 1
            corrections_made.append({
                'id': mineral_id,
                'old_name': mineral_name,
                'new_name': correct_name,
                'score': score,
                'chemistry': ima_info.get('chemistry', 'N/A')
            })
            
            print(f"✓ Corrected: '{mineral_name}' → '{correct_name}' (score: {score:.2f})")
        else:
            not_found += 1
            if not_found <= 10:  # Show first 10
                print(f"⚠️  Not found: '{mineral_name}'")
    
    # Show summary
    print("\n" + "="*70)
    print("CORRECTION SUMMARY")
    print("="*70)
    print(f"Total minerals checked: {len(minerals)}")
    print(f"  ✅ Already correct (exact match): {exact_matches}")
    print(f"  🔧 Corrected (fuzzy match): {fuzzy_corrections}")
    print(f"  ❌ Not found in IMA: {not_found}")
    
    if fuzzy_corrections > 0:
        print("\n" + "="*70)
        print("CORRECTIONS MADE")
        print("="*70)
        for correction in corrections_made[:20]:  # Show first 20
            print(f"\n{correction['old_name']} → {correction['new_name']}")
            print(f"  Chemistry: {correction['chemistry']}")
            print(f"  Match score: {correction['score']:.2f}")
        
        if len(corrections_made) > 20:
            print(f"\n... and {len(corrections_made) - 20} more corrections")
    
    # Ask for confirmation
    print("\n" + "="*70)
    if fuzzy_corrections > 0:
        response = input(f"\nCommit {fuzzy_corrections} corrections to database? (yes/no): ")
        if response.lower() == 'yes':
            conn.commit()
            print(f"✅ Committed {fuzzy_corrections} corrections to database")
            
            # Recommend rebuilding search index
            print("\n" + "="*70)
            print("NEXT STEPS")
            print("="*70)
            print("Rebuild the search index to use corrected names:")
            print("\npython -c \"from utils.fast_pattern_search import FastPatternSearchEngine;")
            print("engine = FastPatternSearchEngine(); engine.build_search_index(force_rebuild=True)\"")
        else:
            print("❌ Changes NOT committed - database unchanged")
    else:
        print("\n✅ No corrections needed - all mineral names are already correct!")
    
    conn.close()

if __name__ == "__main__":
    quick_fix_names()
