#!/usr/bin/env python3
"""
Test script to demonstrate ultra-fast pattern search performance
Compares traditional vs optimized correlation search speeds
"""

import numpy as np
import time
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.pattern_search import PatternSearchEngine
from utils.fast_pattern_search import FastPatternSearchEngine

def generate_synthetic_xrd_pattern(num_peaks=20, noise_level=0.1):
    """Generate a synthetic XRD pattern for testing"""
    
    # Create 2θ range
    two_theta = np.linspace(5, 90, 4250)  # 0.02° resolution
    
    # Generate random peak positions
    peak_positions = np.random.uniform(10, 80, num_peaks)
    peak_intensities = np.random.uniform(0.1, 1.0, num_peaks)
    
    # Create pattern with Gaussian peaks
    pattern = np.zeros_like(two_theta)
    for pos, intensity in zip(peak_positions, peak_intensities):
        fwhm = np.random.uniform(0.1, 0.3)  # Random peak width
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        peak = intensity * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
        pattern += peak
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.max(pattern), len(pattern))
        pattern += noise
        pattern = np.maximum(pattern, 0)  # No negative intensities
    
    return {
        'two_theta': two_theta,
        'intensity': pattern,
        'wavelength': 1.5406
    }

def test_search_performance():
    """Test and compare search performance"""
    
    print("🧪 XRD Pattern Search Performance Test")
    print("=" * 50)
    
    # Generate test pattern
    print("📊 Generating synthetic XRD pattern...")
    test_pattern = generate_synthetic_xrd_pattern(num_peaks=25, noise_level=0.05)
    print(f"   Pattern: {len(test_pattern['two_theta'])} data points")
    print(f"   2θ range: {test_pattern['two_theta'][0]:.1f}° - {test_pattern['two_theta'][-1]:.1f}°")
    
    # Initialize search engines
    print("\n🔧 Initializing search engines...")
    traditional_engine = PatternSearchEngine()
    fast_engine = FastPatternSearchEngine()
    
    # Check database availability
    try:
        from utils.local_database import LocalCIFDatabase
        db = LocalCIFDatabase()
        
        # Get database statistics
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM minerals")
        mineral_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM diffraction_patterns WHERE wavelength = 1.5406")
        pattern_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"   Database: {mineral_count} minerals, {pattern_count} diffraction patterns")
        
        if pattern_count == 0:
            print("❌ No pre-calculated diffraction patterns found!")
            print("   Run 'Calculate All Diffraction Patterns' in the Local Database tab first.")
            return
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return
    
    # Test 1: Traditional correlation search
    print(f"\n🐌 Testing Traditional Correlation Search...")
    try:
        start_time = time.time()
        traditional_results = traditional_engine.search_by_correlation(
            test_pattern,
            min_correlation=0.1,  # Low threshold to test full database
            max_results=50
        )
        traditional_time = time.time() - start_time
        
        print(f"   ✅ Traditional search: {traditional_time:.2f}s")
        print(f"   Found {len(traditional_results)} matches")
        
    except Exception as e:
        print(f"   ❌ Traditional search failed: {e}")
        traditional_time = None
        traditional_results = []
    
    # Test 2: Build fast search index
    print(f"\n🔨 Building Ultra-Fast Search Index...")
    try:
        start_time = time.time()
        success = fast_engine.build_search_index(
            grid_resolution=0.02,
            two_theta_range=(5.0, 90.0)
        )
        index_build_time = time.time() - start_time
        
        if success:
            stats = fast_engine.get_search_statistics()
            print(f"   ✅ Index built in {index_build_time:.2f}s")
            print(f"   Matrix size: {stats['matrix_size_mb']:.1f} MB")
            print(f"   Grid points: {stats['grid_points']}")
        else:
            print("   ❌ Index build failed")
            return
            
    except Exception as e:
        print(f"   ❌ Index build error: {e}")
        return
    
    # Test 3: Ultra-fast correlation search
    print(f"\n🚀 Testing Ultra-Fast Correlation Search...")
    try:
        # Warm-up run
        fast_engine.ultra_fast_correlation_search(test_pattern, min_correlation=0.1, max_results=50)
        
        # Timed runs
        times = []
        for i in range(5):
            start_time = time.time()
            fast_results = fast_engine.ultra_fast_correlation_search(
                test_pattern,
                min_correlation=0.1,
                max_results=50
            )
            times.append(time.time() - start_time)
        
        avg_fast_time = np.mean(times)
        min_fast_time = np.min(times)
        
        print(f"   ✅ Ultra-fast search: {avg_fast_time*1000:.1f}ms (avg), {min_fast_time*1000:.1f}ms (best)")
        print(f"   Found {len(fast_results)} matches")
        
    except Exception as e:
        print(f"   ❌ Ultra-fast search failed: {e}")
        avg_fast_time = None
        fast_results = []
    
    # Performance comparison
    print(f"\n📈 Performance Comparison")
    print("=" * 30)
    
    if traditional_time and avg_fast_time:
        speedup = traditional_time / avg_fast_time
        print(f"Traditional search:  {traditional_time:.2f}s")
        print(f"Ultra-fast search:   {avg_fast_time*1000:.1f}ms")
        print(f"Speedup factor:      {speedup:.0f}x faster! 🚀")
        
        # Patterns per second
        patterns_per_sec_traditional = pattern_count / traditional_time
        patterns_per_sec_fast = pattern_count / avg_fast_time
        
        print(f"\nSearch Rate:")
        print(f"Traditional:         {patterns_per_sec_traditional:.0f} patterns/second")
        print(f"Ultra-fast:          {patterns_per_sec_fast:.0f} patterns/second")
        
    else:
        print("❌ Could not complete performance comparison")
    
    # Memory and setup costs
    print(f"\n💾 Resource Usage")
    print("=" * 20)
    print(f"Index build time:    {index_build_time:.2f}s (one-time setup)")
    print(f"Memory usage:        {stats['matrix_size_mb']:.1f} MB")
    print(f"Database coverage:   {pattern_count} patterns indexed")
    
    print(f"\n🎯 Key Optimizations Used:")
    print("• Pre-computed pattern matrix on common 2θ grid")
    print("• Single matrix multiplication for all correlations")
    print("• Fast binning instead of individual peak profiles")
    print("• L2-normalized patterns for better correlation")
    print("• NumPy vectorized operations throughout")
    
    print(f"\n✨ This is similar to how Raman spectroscopy databases achieve instant searching!")

if __name__ == "__main__":
    test_search_performance()
