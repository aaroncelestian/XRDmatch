#!/usr/bin/env python3
"""
Test script to demonstrate the importance of Lorentz-polarization correction
when converting between wavelengths
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_lp_factor(two_theta_deg):
    """
    Calculate Lorentz-polarization factor
    LP(θ) = (1 + cos²(2θ)) / (sin²(θ) × cos(θ))
    """
    theta_rad = np.radians(two_theta_deg / 2)
    
    if theta_rad > 0 and np.sin(theta_rad) > 0:
        lp = ((1 + np.cos(2 * theta_rad)**2) / 
              (np.sin(theta_rad)**2 * np.cos(theta_rad)))
    else:
        lp = 1.0
    
    return lp

def convert_wavelength_with_lp(d_spacing, intensity_cu, wavelength_cu=1.5406, wavelength_target=0.2401):
    """
    Convert diffraction pattern from one wavelength to another with LP correction
    """
    # Calculate 2θ for Cu Kα
    sin_theta_cu = wavelength_cu / (2 * d_spacing)
    if sin_theta_cu > 1.0:
        return None, None, None
    
    theta_cu = np.arcsin(sin_theta_cu)
    two_theta_cu = 2 * np.degrees(theta_cu)
    
    # Calculate 2θ for target wavelength
    sin_theta_target = wavelength_target / (2 * d_spacing)
    if sin_theta_target > 1.0:
        return None, None, None
    
    theta_target = np.arcsin(sin_theta_target)
    two_theta_target = 2 * np.degrees(theta_target)
    
    # Calculate LP factors
    lp_cu = calculate_lp_factor(two_theta_cu)
    lp_target = calculate_lp_factor(two_theta_target)
    
    # Correct intensity
    intensity_target = intensity_cu * (lp_target / lp_cu)
    
    return two_theta_target, intensity_target, lp_target / lp_cu

def test_lp_correction():
    """Test LP correction for wavelength conversion"""
    print("="*70)
    print("Lorentz-Polarization Correction Test")
    print("Converting Cu Kα (1.5406 Å) → Synchrotron (0.2401 Å)")
    print("="*70)
    
    # Example d-spacings for Epsomite (first few peaks)
    d_spacings = np.array([11.86, 5.93, 4.69, 4.29, 3.95, 3.56, 3.39, 3.05, 2.87, 2.68])
    intensities_cu = np.array([100, 15, 30, 12, 18, 8, 25, 10, 15, 12])  # Example intensities
    
    print("\nOriginal Cu Kα pattern:")
    print("-"*70)
    print(f"{'d (Å)':>8} {'2θ (Cu)':>10} {'I (Cu)':>10} {'LP (Cu)':>10}")
    print("-"*70)
    
    results_cu = []
    results_syn = []
    lp_corrections = []
    
    for d, i_cu in zip(d_spacings, intensities_cu):
        # Cu Kα
        sin_theta = 1.5406 / (2 * d)
        if sin_theta <= 1.0:
            two_theta_cu = 2 * np.degrees(np.arcsin(sin_theta))
            lp_cu = calculate_lp_factor(two_theta_cu)
            results_cu.append((d, two_theta_cu, i_cu, lp_cu))
            print(f"{d:8.3f} {two_theta_cu:10.3f} {i_cu:10.1f} {lp_cu:10.2f}")
    
    print("\nConverted Synchrotron pattern (WITH LP correction):")
    print("-"*70)
    print(f"{'d (Å)':>8} {'2θ (Syn)':>10} {'I (old)':>10} {'I (new)':>10} {'LP ratio':>10}")
    print("-"*70)
    
    for d, i_cu in zip(d_spacings, intensities_cu):
        two_theta_syn, i_syn, lp_ratio = convert_wavelength_with_lp(d, i_cu)
        
        if two_theta_syn is not None:
            results_syn.append((d, two_theta_syn, i_syn))
            lp_corrections.append(lp_ratio)
            print(f"{d:8.3f} {two_theta_syn:10.3f} {i_cu:10.1f} {i_syn:10.1f} {lp_ratio:10.3f}")
    
    # Renormalize synchrotron intensities
    if results_syn:
        intensities_syn = np.array([r[2] for r in results_syn])
        intensities_syn_normalized = 100.0 * intensities_syn / np.max(intensities_syn)
        
        print("\nAfter renormalization:")
        print("-"*70)
        print(f"{'d (Å)':>8} {'2θ (Syn)':>10} {'I (norm)':>10}")
        print("-"*70)
        
        for (d, two_theta, _), i_norm in zip(results_syn, intensities_syn_normalized):
            print(f"{d:8.3f} {two_theta:10.3f} {i_norm:10.1f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Cu Kα pattern
    ax = axes[0]
    for d, two_theta, intensity, _ in results_cu:
        ax.plot([two_theta, two_theta], [0, intensity], 'b-', linewidth=2)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 110)
    ax.set_xlabel('2θ (degrees)')
    ax.set_ylabel('Intensity')
    ax.set_title('Original: Cu Kα (1.5406 Å)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Synchrotron WITHOUT LP correction
    ax = axes[1]
    for d, i_cu in zip(d_spacings, intensities_cu):
        sin_theta = 0.2401 / (2 * d)
        if sin_theta <= 1.0:
            two_theta = 2 * np.degrees(np.arcsin(sin_theta))
            ax.plot([two_theta, two_theta], [0, i_cu], 'r-', linewidth=2)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 110)
    ax.set_xlabel('2θ (degrees)')
    ax.set_ylabel('Intensity')
    ax.set_title('Synchrotron (0.2401 Å) - WITHOUT LP correction (WRONG)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Synchrotron WITH LP correction
    ax = axes[2]
    for (d, two_theta, _), i_norm in zip(results_syn, intensities_syn_normalized):
        ax.plot([two_theta, two_theta], [0, i_norm], 'g-', linewidth=2)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 110)
    ax.set_xlabel('2θ (degrees)')
    ax.set_ylabel('Intensity')
    ax.set_title('Synchrotron (0.2401 Å) - WITH LP correction (CORRECT)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wavelength_lp_correction_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved comparison plot: wavelength_lp_correction_comparison.png")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"\nLP correction factors range: {np.min(lp_corrections):.3f} to {np.max(lp_corrections):.3f}")
    print(f"Average LP correction: {np.mean(lp_corrections):.3f}")
    print(f"\nThis shows that intensity ratios change by up to {np.max(lp_corrections):.1f}x")
    print("when converting from Cu Kα to synchrotron wavelengths!")
    print("\n⚠ Without LP correction, the relative intensities will be WRONG,")
    print("  causing poor phase matching results.")
    
    plt.show()

if __name__ == "__main__":
    test_lp_correction()
