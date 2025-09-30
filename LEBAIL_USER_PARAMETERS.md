# Le Bail Refinement - User Parameters Guide

## Overview
The Le Bail refinement now includes user-configurable parameters to help you get better fits. These controls are in the **Visualization tab** under "Le Bail Refinement".

## Parameters

### 1. **Initial FWHM** (Full Width at Half Maximum)
- **Range:** 0.01° to 1.0°
- **Default:** 0.1°
- **What it does:** Sets the starting peak width
- **When to adjust:**
  - **Decrease (0.03-0.05°)** if your experimental peaks are sharp (high-resolution data, synchrotron)
  - **Increase (0.15-0.3°)** if your peaks are broad (low-resolution, nanocrystalline samples)
  
**Tip:** Look at your experimental peaks - measure the width at half the peak height. Start with that value.

### 2. **Max Scale Factor**
- **Range:** 1.0 to 1000.0
- **Default:** 100.0
- **What it does:** Controls how tall peaks can grow during refinement
- **When to adjust:**
  - **Increase (200-500)** if calculated peaks are too short compared to experimental
  - **Decrease (10-50)** if you want to constrain peak heights more tightly
  
**Tip:** If you see the message "Scale factor at boundary (10.0)", increase this value significantly.

### 3. **Peak Shape (η - eta)**
- **Range:** 0.0 to 1.0
- **Default:** 0.5
- **What it does:** Controls the Gaussian vs Lorentzian character of peaks
  - **η = 0.0:** Pure Gaussian (symmetric, bell-shaped)
  - **η = 0.5:** 50/50 mix (pseudo-Voigt)
  - **η = 1.0:** Pure Lorentzian (broader tails)
- **When to adjust:**
  - **Decrease (0.2-0.3)** for sharper, more symmetric peaks
  - **Increase (0.6-0.8)** if peaks have broader tails
  
**Tip:** Most XRD data works well with 0.4-0.6. Synchrotron data often prefers lower values.

### 4. **Refine Unit Cell** (checkbox)
- **Default:** Checked (enabled)
- **What it does:** Allows unit cell parameters (a, b, c) to refine within ±5%
- **When to disable:**
  - If you have high-quality unit cell parameters from single crystal data
  - If refinement is unstable (unit cell wandering)
  - For quick tests where you just want to fit peak shapes

### 5. **Refine Peak Profile** (checkbox)
- **Default:** Checked (enabled)
- **What it does:** Allows U, V, W parameters to refine (controls peak width variation with angle)
- **When to disable:**
  - If you want to keep the initial FWHM fixed
  - If profile refinement causes instability
  - For initial tests to see if the starting FWHM is reasonable

### 6. **Max Iterations**
- **Range:** 5 to 50
- **Default:** 15
- **What it does:** Number of refinement cycles
- **When to adjust:**
  - **Increase (20-30)** if refinement hasn't converged (Rwp still decreasing)
  - **Decrease (5-10)** for quick tests

### 7. **2θ Range** (optional)
- **What it does:** Limits refinement to a specific angular range
- **When to use:**
  - Focus on a specific region with good peaks
  - Exclude noisy low-angle or high-angle regions
  - Speed up refinement by using fewer data points

## Recommended Workflow

### For Sharp Peaks (High-Resolution Data):
```
Initial FWHM: 0.03-0.05°
Max Scale: 100-200
Peak Shape (η): 0.3-0.4
Refine Unit Cell: ✓
Refine Peak Profile: ✓
```

### For Broad Peaks (Nanocrystalline, Low-Resolution):
```
Initial FWHM: 0.2-0.4°
Max Scale: 50-100
Peak Shape (η): 0.5-0.7
Refine Unit Cell: ✓
Refine Peak Profile: ✓
```

### For Quick Tests:
```
Initial FWHM: 0.1°
Max Scale: 100
Peak Shape (η): 0.5
Refine Unit Cell: ✗ (disabled)
Refine Peak Profile: ✗ (disabled)
Max Iterations: 5-10
```

## Interpreting Results

### Good Fit Indicators:
- **Rwp < 10%** - Excellent fit
- **Rwp 10-20%** - Good fit
- **GoF ≈ 1-2** - Reasonable fit
- Calculated peaks match experimental peak positions and heights
- Difference curve is relatively flat (near zero)

### Problem Indicators:
- **Peaks too broad:** Decrease Initial FWHM, or disable "Refine Peak Profile"
- **Peaks too narrow:** Increase Initial FWHM
- **Peaks too short:** Increase Max Scale Factor
- **Large zero shift (>0.05°):** Check instrument calibration
- **Rwp not improving:** Try different initial FWHM, check phase identification

## Real-Time Monitoring

Watch the plot during refinement:
- **Blue line:** Your experimental data
- **Red line:** Calculated pattern from refinement
- **Green line:** Difference (experimental - calculated)

**Goal:** Make the red line match the blue line as closely as possible, with the green line near zero.

## Tips for Best Results

1. **Start with good phase identification** - Le Bail can't fix wrong phases
2. **Use 2θ range** to focus on the best data region
3. **Adjust Initial FWHM first** - This has the biggest impact on fit quality
4. **Increase Max Scale if needed** - Don't artificially limit peak heights
5. **Watch the real-time plot** - You'll see immediately if parameters are wrong
6. **Iterate** - Try different FWHM values if first attempt doesn't work well

## Common Issues

### Issue: "Peaks are still too broad"
**Solution:** 
- Decrease Initial FWHM to 0.03-0.05°
- Make sure "Refine Peak Profile" is enabled
- Check that U, V, W bounds aren't too loose

### Issue: "Calculated peaks too short"
**Solution:**
- Increase Max Scale Factor to 200-500
- Check that phases are correctly identified

### Issue: "Refinement is unstable"
**Solution:**
- Disable "Refine Unit Cell" temporarily
- Use a smaller 2θ range with good peaks only
- Reduce Max Iterations to 5-10
- Try a different Initial FWHM

### Issue: "Zero shift is large (>0.1°)"
**Solution:**
- This suggests instrument misalignment
- Check your 2θ zero calibration
- Consider recalibrating with a standard (Si, LaB6)
