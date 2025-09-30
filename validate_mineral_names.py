#!/usr/bin/env python3
"""
Validate mineral names in the database against IMA records
Identifies potential author names or incorrect entries
"""

import sqlite3
from utils.local_database import LocalCIFDatabase
from utils.ima_mineral_database import get_ima_database

def validate_database_minerals():
    """Check all minerals in database against IMA records"""
    print("="*70)
    print("Validating Mineral Names Against IMA Database")
    print("="*70)
    
    # Load databases
    local_db = LocalCIFDatabase()
    ima_db = get_ima_database()
    
    conn = sqlite3.connect(local_db.db_path)
    cursor = conn.cursor()
    
    # Get all unique mineral names
    cursor.execute('''
        SELECT DISTINCT mineral_name, COUNT(*) as count
        FROM minerals
        GROUP BY mineral_name
        ORDER BY mineral_name
    ''')
    
    minerals = cursor.fetchall()
    total_minerals = len(minerals)
    
    print(f"\n📊 Database contains {total_minerals} unique mineral names")
    print(f"   Checking against {len(ima_db.minerals)} IMA records...\n")
    
    # Categories
    verified = []
    fuzzy_matches = []
    not_found = []
    suspicious = []
    
    for mineral_name, count in minerals:
        # Skip empty or very short names
        if not mineral_name or len(mineral_name) < 3:
            suspicious.append((mineral_name, count, "Too short"))
            continue
        
        # Check for exact match
        ima_info = ima_db.get_mineral_info(mineral_name)
        if ima_info:
            verified.append((mineral_name, count, ima_info))
            continue
        
        # Try fuzzy match
        fuzzy_result = ima_db.fuzzy_match_mineral(mineral_name, threshold=0.85)
        if fuzzy_result:
            matched_name, score, ima_info = fuzzy_result
            fuzzy_matches.append((mineral_name, count, matched_name, score, ima_info))
            continue
        
        # Check if it looks like an author name (contains initials or common patterns)
        if any(pattern in mineral_name for pattern in [' P.', ' R.', ' J.', ' A.', ' et al']):
            suspicious.append((mineral_name, count, "Possible author name"))
        else:
            not_found.append((mineral_name, count))
    
    # Print results
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    print(f"\n✅ Verified (exact IMA match): {len(verified)}")
    if verified and len(verified) <= 20:
        for name, count, info in verified[:20]:
            print(f"   - {name} ({count} entries): {info['chemistry']}")
    
    print(f"\n🔍 Fuzzy Matches (similar IMA names): {len(fuzzy_matches)}")
    if fuzzy_matches:
        print("\n   Suggested corrections:")
        for db_name, count, ima_name, score, info in fuzzy_matches[:20]:
            print(f"   - '{db_name}' → '{ima_name}' (score: {score:.2f}, {count} entries)")
            print(f"     Chemistry: {info['chemistry']}")
    
    print(f"\n⚠️  Suspicious Entries: {len(suspicious)}")
    if suspicious:
        for name, count, reason in suspicious[:20]:
            print(f"   - '{name}' ({count} entries): {reason}")
    
    print(f"\n❌ Not Found in IMA: {len(not_found)}")
    if not_found:
        print("\n   These may be:")
        print("   - Synthetic materials")
        print("   - Author names")
        print("   - Obsolete mineral names")
        print("   - Typos or data entry errors")
        print("\n   First 20 entries:")
        for name, count in not_found[:20]:
            print(f"   - '{name}' ({count} entries)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total unique minerals: {total_minerals}")
    print(f"  ✅ Verified: {len(verified)} ({100*len(verified)/total_minerals:.1f}%)")
    print(f"  🔍 Fuzzy matches: {len(fuzzy_matches)} ({100*len(fuzzy_matches)/total_minerals:.1f}%)")
    print(f"  ⚠️  Suspicious: {len(suspicious)} ({100*len(suspicious)/total_minerals:.1f}%)")
    print(f"  ❌ Not found: {len(not_found)} ({100*len(not_found)/total_minerals:.1f}%)")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if fuzzy_matches:
        print("\n1. Review fuzzy matches and consider updating mineral names")
        print("   Example SQL to update:")
        print("   UPDATE minerals SET mineral_name = 'Correct Name'")
        print("   WHERE mineral_name = 'Incorrect Name';")
    
    if suspicious:
        print("\n2. Review suspicious entries - these may be author names")
        print("   Check the CIF files to determine correct mineral names")
    
    if not_found:
        print("\n3. Investigate entries not found in IMA database")
        print("   - Verify these are actual minerals")
        print("   - Check for typos in mineral names")
        print("   - Consider if these are synthetic or non-mineral phases")
    
    print("\n" + "="*70)
    
    conn.close()
    
    return {
        'verified': verified,
        'fuzzy_matches': fuzzy_matches,
        'suspicious': suspicious,
        'not_found': not_found
    }

def export_validation_report(results, output_file='mineral_validation_report.txt'):
    """Export validation results to a text file"""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Mineral Name Validation Report\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Verified Minerals: {len(results['verified'])}\n")
        f.write("-"*70 + "\n")
        for name, count, info in results['verified']:
            f.write(f"{name} ({count} entries): {info['chemistry']}\n")
        
        f.write(f"\n\nFuzzy Matches: {len(results['fuzzy_matches'])}\n")
        f.write("-"*70 + "\n")
        for db_name, count, ima_name, score, info in results['fuzzy_matches']:
            f.write(f"'{db_name}' → '{ima_name}' (score: {score:.2f}, {count} entries)\n")
            f.write(f"  Chemistry: {info['chemistry']}\n")
        
        f.write(f"\n\nSuspicious Entries: {len(results['suspicious'])}\n")
        f.write("-"*70 + "\n")
        for name, count, reason in results['suspicious']:
            f.write(f"'{name}' ({count} entries): {reason}\n")
        
        f.write(f"\n\nNot Found in IMA: {len(results['not_found'])}\n")
        f.write("-"*70 + "\n")
        for name, count in results['not_found']:
            f.write(f"'{name}' ({count} entries)\n")
    
    print(f"\n📄 Validation report exported to: {output_file}")

if __name__ == "__main__":
    results = validate_database_minerals()
    
    # Ask if user wants to export report
    response = input("\nExport detailed report to file? (yes/no): ")
    if response.lower() == 'yes':
        export_validation_report(results)
