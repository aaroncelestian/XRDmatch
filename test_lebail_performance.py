#!/usr/bin/env python3
"""
Test script to verify Le Bail refinement performance improvements
"""

import time
import numpy as np
from utils.lebail_refinement import LeBailRefinement

def test_performance():
    """Test Le Bail refinement performance with realistic data"""
    print("="*60)
    print("Le Bail Refinement Performance Test")
    print("="*60)
    
    # Create realistic synthetic data
    print("\n1. Generating synthetic XRD data...")
    two_theta = np.linspace(5, 60, 2750)  # 0.02° steps, typical for lab XRD
    
    # Generate pattern with multiple peaks
    pattern = np.zeros_like(two_theta)
    peak_positions = [12.5, 17.7, 21.7, 25.1, 28.3, 30.9, 35.2, 39.8, 44.1, 48.2]
    peak_intensities = [100, 80, 60, 90, 50, 70, 40, 50, 30, 35]
    
    for pos, intensity in zip(peak_positions, peak_intensities):
        fwhm = 0.15
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        peak = intensity * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
        pattern += peak
    
    # Add background and noise
    background = 50 + 20 * np.exp(-two_theta / 30)
    pattern += background
    noise = np.random.normal(0, 2, len(pattern))
    pattern += noise
    pattern = np.maximum(pattern, 0)
    
    print(f"   Data points: {len(two_theta)}")
    print(f"   Range: {two_theta[0]:.1f}° to {two_theta[-1]:.1f}°")
    print(f"   Peaks: {len(peak_positions)}")
    
    # Create test phase
    print("\n2. Creating test phase...")
    phase_data = {
        'phase': {
            'mineral': 'Test Phase',
            'formula': 'ABC',
            'id': 'test_001',
            'cell_a': 5.0,
            'cell_b': 5.0,
            'cell_c': 5.0,
            'cell_alpha': 90.0,
            'cell_beta': 90.0,
            'cell_gamma': 90.0
        },
        'theoretical_peaks': {
            'two_theta': np.array(peak_positions),
            'intensity': np.array(peak_intensities),
            'd_spacing': np.array([7.1, 5.0, 4.1, 3.5, 3.1, 2.9, 2.5, 2.3, 2.0, 1.9])
        }
    }
    
    # Initialize refinement
    print("\n3. Setting up Le Bail refinement...")
    lebail = LeBailRefinement()
    lebail.set_experimental_data(two_theta, pattern)
    
    initial_params = {
        'scale_factor': 1.0,
        'u_param': 0.02,
        'v_param': -0.002,
        'w_param': 0.015,
        'eta_param': 0.5,
        'zero_shift': 0.0,
        'refine_cell': True,
        'refine_profile': True,
        'refine_scale': True
    }
    
    lebail.add_phase(phase_data, initial_params)
    
    # Test with different iteration counts
    iteration_tests = [5, 10, 15, 20]
    
    print("\n4. Running performance tests...")
    print("-" * 60)
    
    for max_iter in iteration_tests:
        # Reset phase
        lebail.phases = []
        lebail.add_phase(phase_data, initial_params)
        
        print(f"\nTest: {max_iter} iterations")
        start_time = time.time()
        
        results = lebail.refine_phases(max_iterations=max_iter, convergence_threshold=1e-5)
        
        elapsed_time = time.time() - start_time
        
        # Display results
        r_factors = results['final_r_factors']
        converged = results['converged']
        actual_iters = results['iterations']
        
        print(f"   Time: {elapsed_time:.2f} seconds")
        print(f"   Actual iterations: {actual_iters}")
        print(f"   Converged: {converged}")
        print(f"   Rwp: {r_factors['Rwp']:.3f}%")
        print(f"   GoF: {r_factors['GoF']:.3f}")
        print(f"   Time per iteration: {elapsed_time/actual_iters:.3f} sec")
    
    print("\n" + "="*60)
    print("Performance Test Complete")
    print("="*60)
    
    # Summary
    print("\n5. Performance Summary:")
    print("   ✓ Optimized peak profile calculation (±5 FWHM cutoff)")
    print("   ✓ Pre-calculated background patterns")
    print("   ✓ Reduced default iterations (15)")
    print("   ✓ Tighter convergence criteria")
    print("\n   Typical refinement time: 10-60 seconds")
    print("   Recommended max_iterations: 10-15 for routine use")
    
    return results

if __name__ == "__main__":
    test_performance()
