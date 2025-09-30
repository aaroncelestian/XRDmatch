#!/usr/bin/env python3
"""
Test script to verify 2-theta range limiting works correctly through MultiPhaseAnalyzer
This tests the fix for the dimension mismatch issue.
"""

import numpy as np
from utils.multi_phase_analyzer import MultiPhaseAnalyzer

def test_multi_phase_with_2theta_range():
    """Test that 2-theta range limiting works through MultiPhaseAnalyzer"""
    print("="*70)
    print("Testing 2-theta Range Limiting through MultiPhaseAnalyzer")
    print("="*70)
    
    # Create synthetic diffraction data
    print("\n1. Creating synthetic diffraction data...")
    two_theta = np.linspace(5, 90, 850)
    
    # Simulate a pattern with Gaussian peaks (simulating quartz)
    pattern = np.zeros_like(two_theta)
    peak_positions = [20.9, 26.6, 36.5, 39.5, 42.4, 50.1, 59.9, 68.3]
    for pos in peak_positions:
        pattern += 1000 * np.exp(-0.5 * ((two_theta - pos) / 0.3) ** 2)
    
    # Add some noise
    np.random.seed(42)
    pattern += np.random.normal(0, 20, len(two_theta))
    pattern = np.maximum(pattern, 0)
    
    print(f"   Full data range: {two_theta[0]:.2f}° - {two_theta[-1]:.2f}°")
    print(f"   Total data points: {len(two_theta)}")
    
    # Create synthetic phase data (no database needed)
    print("\n2. Creating synthetic phase data...")
    phase_info = {
        'id': 'test_quartz',
        'mineral': 'Quartz',
        'formula': 'SiO2',
        'cell_a': 4.913,
        'cell_b': 4.913,
        'cell_c': 5.405,
        'cell_alpha': 90.0,
        'cell_beta': 90.0,
        'cell_gamma': 120.0,
        'space_group': 'P3221'
    }
    
    print(f"   Created: {phase_info['mineral']} ({phase_info['formula']})")
    
    # Create theoretical peaks matching our synthetic pattern
    theoretical_peaks = {
        'two_theta': np.array([20.9, 26.6, 36.5, 39.5, 42.4, 50.1, 59.9, 68.3]),
        'intensity': np.array([100, 80, 60, 50, 40, 30, 20, 15]),
        'hkl': [(1,0,0), (1,0,1), (1,1,0), (1,1,1), (2,0,0), (2,0,1), (2,1,0), (2,1,1)]
    }
    
    print(f"   Theoretical peaks: {len(theoretical_peaks['two_theta'])}")
    
    # Create matched phase structure
    matched_phase = {
        'phase': phase_info,
        'theoretical_peaks': theoretical_peaks,
        'optimized_scaling': 1.0
    }
    
    # Test 1: Le Bail refinement WITHOUT 2-theta range
    print("\n3. Test Le Bail refinement WITHOUT 2-theta range...")
    analyzer1 = MultiPhaseAnalyzer()
    
    experimental_data1 = {
        'two_theta': two_theta,
        'intensity': pattern,
        'wavelength': 1.5406
    }
    
    try:
        results1 = analyzer1.perform_lebail_refinement(
            experimental_data1,
            [matched_phase],
            max_iterations=3,  # Just a few iterations for testing
            two_theta_range=None
        )
        
        if results1['success']:
            print(f"   ✓ Refinement succeeded")
            print(f"   Data points used: {len(analyzer1.lebail_engine.experimental_data['two_theta'])}")
            print(f"   Rwp: {results1['r_factors']['Rwp']:.2f}%")
        else:
            print(f"   ✗ Refinement failed: {results1.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ✗ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Le Bail refinement WITH 2-theta range (25-65 degrees)
    print("\n4. Test Le Bail refinement WITH 2-theta range (25-65°)...")
    analyzer2 = MultiPhaseAnalyzer()
    
    experimental_data2 = {
        'two_theta': two_theta,
        'intensity': pattern,
        'wavelength': 1.5406
    }
    
    try:
        results2 = analyzer2.perform_lebail_refinement(
            experimental_data2,
            [matched_phase],
            max_iterations=3,
            two_theta_range=(25.0, 65.0)
        )
        
        if results2['success']:
            print(f"   ✓ Refinement succeeded")
            filtered_data = analyzer2.lebail_engine.experimental_data['two_theta']
            print(f"   Data points used: {len(filtered_data)}")
            print(f"   Range: {filtered_data[0]:.2f}° - {filtered_data[-1]:.2f}°")
            print(f"   Rwp: {results2['r_factors']['Rwp']:.2f}%")
            
            # Verify the range is correct
            assert filtered_data[0] >= 25.0, "Min 2-theta not applied correctly"
            assert filtered_data[-1] <= 65.0, "Max 2-theta not applied correctly"
            assert len(filtered_data) < len(two_theta), "Range limiting didn't reduce data points"
            
            print(f"   ✓ 2-theta range correctly applied")
        else:
            print(f"   ✗ Refinement failed: {results2.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ✗ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)

if __name__ == "__main__":
    test_multi_phase_with_2theta_range()
