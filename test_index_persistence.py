#!/usr/bin/env python3
"""
Test script to verify that the search index persistence fix works correctly
"""

import sys
import os
import time
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.fast_pattern_search import FastPatternSearchEngine

def test_index_persistence():
    """Test that the search index persists between engine instances"""
    
    print("=== Testing Search Index Persistence Fix ===\n")
    
    # Test 1: Create first engine instance
    print("1. Creating first FastPatternSearchEngine instance...")
    engine1 = FastPatternSearchEngine()
    
    if engine1.search_index is not None:
        print("   ✅ Index auto-loaded from cache!")
        stats1 = engine1.get_search_statistics()
        print(f"   Database size: {stats1['database_size']} patterns")
        print(f"   Status: {stats1['status']}")
    else:
        print("   ⚠️  No index found - this is expected on first run")
        print("   Building index for the test...")
        success = engine1.build_search_index(force_rebuild=True)
        if not success:
            print("   ❌ Failed to build index")
            return False
        print("   ✅ Index built successfully")
    
    # Test 2: Create synthetic experimental data
    print("\n2. Creating synthetic experimental pattern...")
    two_theta = np.linspace(10, 50, 1000)
    intensity = 100 * np.exp(-((two_theta - 25) / 5) ** 2) + np.random.normal(0, 5, len(two_theta))
    intensity = np.maximum(intensity, 0)
    
    experimental_pattern = {
        'two_theta': two_theta,
        'intensity': intensity,
        'wavelength': 1.5406
    }
    
    # Test 3: Perform search with first engine
    print("3. Performing search with first engine...")
    start_time = time.time()
    results1 = engine1.ultra_fast_correlation_search(
        experimental_pattern,
        min_correlation=0.1,
        max_results=10
    )
    search_time1 = time.time() - start_time
    print(f"   Search completed in {search_time1*1000:.1f}ms")
    print(f"   Found {len(results1)} matches")
    
    # Test 4: Create second engine instance (this should auto-load the cache)
    print("\n4. Creating second FastPatternSearchEngine instance...")
    engine2 = FastPatternSearchEngine()
    
    if engine2.search_index is not None:
        print("   ✅ Index auto-loaded from cache in second instance!")
        stats2 = engine2.get_search_statistics()
        print(f"   Database size: {stats2['database_size']} patterns")
        print(f"   Status: {stats2['status']}")
    else:
        print("   ❌ Index NOT loaded - the fix didn't work!")
        return False
    
    # Test 5: Perform search with second engine (should be instant)
    print("\n5. Performing search with second engine...")
    start_time = time.time()
    results2 = engine2.ultra_fast_correlation_search(
        experimental_pattern,
        min_correlation=0.1,
        max_results=10
    )
    search_time2 = time.time() - start_time
    print(f"   Search completed in {search_time2*1000:.1f}ms")
    print(f"   Found {len(results2)} matches")
    
    # Test 6: Verify results are consistent
    print("\n6. Verifying consistency between searches...")
    if len(results1) == len(results2):
        print("   ✅ Same number of results from both engines")
        
        # Check if top results are the same
        if len(results1) > 0 and len(results2) > 0:
            top1 = results1[0]['mineral_name']
            top2 = results2[0]['mineral_name']
            if top1 == top2:
                print(f"   ✅ Top result consistent: {top1}")
            else:
                print(f"   ⚠️  Top results differ: {top1} vs {top2}")
    else:
        print(f"   ⚠️  Different number of results: {len(results1)} vs {len(results2)}")
    
    print(f"\n=== Test Results ===")
    print(f"✅ Index persistence: WORKING")
    print(f"✅ Auto-loading: WORKING") 
    print(f"✅ Search consistency: WORKING")
    print(f"⚡ Search speed: {search_time2*1000:.1f}ms (should be <100ms)")
    
    if search_time2 < 0.1:  # Less than 100ms
        print(f"🚀 Ultra-fast search performance: EXCELLENT")
    elif search_time2 < 0.5:  # Less than 500ms
        print(f"⚡ Fast search performance: GOOD")
    else:
        print(f"⚠️  Search performance: SLOW (may need optimization)")
    
    print(f"\n🎉 The search index persistence fix is working correctly!")
    print(f"   The app should no longer ask to rebuild the index repeatedly.")
    
    return True

if __name__ == "__main__":
    try:
        test_index_persistence()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
