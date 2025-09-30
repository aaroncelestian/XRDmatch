#!/usr/bin/env python3
"""
Test script for IMA mineral database integration
"""

from utils.ima_mineral_database import IMAMineralDatabase, get_ima_database

def test_ima_database():
    """Test IMA database functionality"""
    print("="*70)
    print("Testing IMA Mineral Database Integration")
    print("="*70)
    
    # Load database
    ima_db = get_ima_database()
    
    # Get statistics
    stats = ima_db.get_statistics()
    print(f"\n📊 Database Statistics:")
    print(f"   Total minerals: {stats.get('total_minerals', 0)}")
    print(f"   Database path: {stats.get('database_path', 'Unknown')}")
    
    # Test exact lookups
    print(f"\n🔍 Testing Exact Mineral Lookups:")
    print("-"*70)
    
    test_minerals = ['Calcite', 'Quartz', 'Halite', 'Gypsum', 'Epsomite']
    
    for mineral_name in test_minerals:
        info = ima_db.get_mineral_info(mineral_name)
        if info:
            print(f"\n✅ {mineral_name}:")
            print(f"   Chemistry: {info.get('chemistry', 'N/A')}")
            print(f"   Space Group: {info.get('space_group', 'N/A')}")
            print(f"   Crystal System: {info.get('crystal_system', 'N/A')}")
            print(f"   IMA Status: {info.get('ima_status', 'N/A')}")
        else:
            print(f"\n❌ {mineral_name}: Not found")
    
    # Test fuzzy matching
    print(f"\n🔍 Testing Fuzzy Matching:")
    print("-"*70)
    
    test_fuzzy = ['Calcit', 'Quarz', 'Gypsm', 'Epsomit']
    
    for fuzzy_name in test_fuzzy:
        result = ima_db.fuzzy_match_mineral(fuzzy_name, threshold=0.7)
        if result:
            matched_name, score, info = result
            print(f"\n'{fuzzy_name}' → '{matched_name}' (score: {score:.2f})")
            print(f"   Chemistry: {info.get('chemistry', 'N/A')}")
        else:
            print(f"\n'{fuzzy_name}' → No match found")
    
    # Test element search
    print(f"\n🔍 Testing Element Search:")
    print("-"*70)
    
    elements = ['Ca', 'C', 'O']
    results = ima_db.search_by_chemistry(elements)
    print(f"\nMinerals containing {elements}: {len(results)} found")
    if results:
        print("\nFirst 5 matches:")
        for mineral in results[:5]:
            print(f"   - {mineral['name']}: {mineral['chemistry']}")
    
    # Test space group search
    print(f"\n🔍 Testing Space Group Search:")
    print("-"*70)
    
    space_group = 'R-3c'
    results = ima_db.search_by_space_group(space_group)
    print(f"\nMinerals with space group {space_group}: {len(results)} found")
    if results:
        print("\nFirst 5 matches:")
        for mineral in results[:5]:
            print(f"   - {mineral['name']}: {mineral['chemistry']}")
    
    # Test author name correction
    print(f"\n🔍 Testing Author Name Correction:")
    print("-"*70)
    
    # Simulate cases where author names might be confused with mineral names
    test_cases = [
        ('Spinat P. Pruchart R', None),  # Author name that might appear
        ('Calcite', 'Smith J.'),
        ('Unknown Mineral', 'Johnson A.')
    ]
    
    for possible_name, possible_author in test_cases:
        result = ima_db.correct_mineral_name(possible_name, possible_author)
        if result:
            print(f"\n'{possible_name}' → Corrected to: {result['name']}")
            print(f"   Chemistry: {result.get('chemistry', 'N/A')}")
        else:
            print(f"\n'{possible_name}' → Could not correct")
    
    print("\n" + "="*70)
    print("IMA Database Test Complete!")
    print("="*70)

if __name__ == "__main__":
    test_ima_database()
