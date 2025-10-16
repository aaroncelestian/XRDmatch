# Le Bail Refinement Guide

## Critical Fixes Applied (2024)

### Issue: Poor Refinement Quality (Rwp > 200%)

**Root Causes:**
1. **Broken Le Bail intensity extraction** - Was using simple peak height instead of proper partitioning
2. **Wrong FWHM for synchrotron data** - Default 0.1° is 5-10x too large
3. **Profile parameters too broad** - Bounds allowed excessive peak broadening

### Fixes Implemented

#### 1. Le Bail Intensity Extraction (CRITICAL)
- **Before:** Simple peak height extraction (incorrect)
- **After:** Proper iterative partitioning algorithm
- **Location:** `utils/lebail_refinement.py::_extract_lebail_intensities()`
- **Method:** 
  - Initialize with peak heights as first guess
  - Iterate 5 times to partition overlapping peaks
  - Each peak gets fraction of observed intensity based on profile shape

#### 2. Profile Parameter Bounds (UPDATED)
- **U:** 0.0 - 0.05 (widened to allow flexibility)
- **V:** -0.01 - 0.01 (widened to allow flexibility)
- **W:** 0.00001 - 0.05 (widened to allow flexibility)
- **Reason:** Need enough range to fit actual peak shapes, not just theoretical minimum

#### 3. Initial Parameter Calculation
- **U/W ratio:** Changed from 0.1 to 0.05 for synchrotron
- **FWHM format:** Now shows 4 decimal places for precision

## Recommended Settings

### For Synchrotron Data (λ < 0.5 Å)

**Initial FWHM:** 0.015° (range: 0.010 - 0.020°)
- 11-BM typical: 0.012 - 0.015°
- Very high resolution: 0.008 - 0.010°

**Profile Parameters:**
- Max Scale: 10.0 (default is fine)
- η (eta): 0.5 (default is fine)
- Refine Cell: ✓ (enabled)
- Refine Profile: ✓ (enabled)
- **Refine Peak Intensities (Pawley): ✗ (DISABLED)** - Use Le Bail mode

**2θ Range:**
- Start with limited range (e.g., 2-8°) for initial refinement
- Expand once good fit is achieved
- Helps with speed and stability

### For Lab X-ray Data (λ ≈ 1.54 Å)

**Initial FWHM:** 0.08 - 0.12°

**Profile Parameters:**
- Same as above, but FWHM will be larger

## Troubleshooting

### Rwp > 50%
1. **Check FWHM** - Most common issue
   - Too large: Calculated peaks too broad
   - Too small: Calculated peaks too narrow
   - Adjust by ±50% and retry

2. **Check wavelength** - Must match experimental data
   - Synchrotron: typically 0.2-0.8 Å
   - Lab Cu Kα: 1.5406 Å

3. **Check phase identity** - Wrong phase = bad fit
   - Review pattern search results
   - Try next best match

4. **Check background subtraction** - Must be done BEFORE refinement
   - Refinement uses background-subtracted data
   - Poor background = poor fit

### Scale Factor Collapses to Zero
- Wrong phase identified
- FWHM mismatch (usually too large)
- Peak positions don't match

### Refinement is Slow
- Disable "Refine Peak Intensities (Pawley)"
- Use 2θ range to limit data points
- Reduce max iterations to 10

## Expected Results

### Excellent Fit
- Rwp: < 5%
- GoF: 1.0 - 2.0

### Good Fit  
- Rwp: 5 - 15%
- GoF: 2.0 - 3.0

### Acceptable Fit
- Rwp: 15 - 25%
- GoF: 3.0 - 5.0

### Poor Fit (needs investigation)
- Rwp: > 25%
- GoF: > 5.0

## Staged Refinement Process

The refinement uses a **two-stage approach** for better convergence:

### Stage 1: Unit Cell & Zero Shift
- **Duration:** First ~1/3 of iterations
- **Parameters refined:** Unit cell (a, b, c), zero shift
- **Parameters fixed:** Profile (U, V, W, η)
- **Purpose:** Get peak positions correct first
- **Output:** You'll see "STAGE 1: Unit Cell & Zero Shift Refinement"

### Stage 2: Profile Parameters
- **Duration:** Remaining ~2/3 of iterations  
- **Parameters refined:** Unit cell, zero shift, **AND** profile (U, V, W, η)
- **Purpose:** Optimize peak shapes with correct positions
- **Output:** You'll see "STAGE 2: Profile Parameter Refinement"
- **What to watch:** 
  - "Profile params: U=..., V=..., W=..., η=..."
  - "Profile refined: U=..., V=..., W=..., η=..."
  - "→ FWHM at 2θ=5°: ..."

**If you don't see profile parameters changing:**
1. Check that "Refine Peak Profile" checkbox is enabled
2. Look for Stage 2 output in terminal
3. Verify you have enough iterations (>6 recommended)

## Algorithm Details

### Le Bail Method
1. Start with theoretical peak positions from crystal structure
2. Extract observed intensities by partitioning experimental pattern
3. Refine profile parameters (U, V, W, η) and unit cell
4. Re-extract intensities with new parameters
5. Iterate until convergence

**Key Difference from Pawley:**
- Le Bail: Intensities extracted from observed data (faster, more stable)
- Pawley: Intensities are free parameters (slower, can be unstable)

### Caglioti Function
Peak width varies with angle:
```
FWHM² = U·tan²θ + V·tanθ + W
```
- **W:** Dominates at low angles (instrumental resolution)
- **U:** Dominates at high angles (sample effects)
- **V:** Usually small or negative

## Files Modified

1. `utils/lebail_refinement.py`
   - `_extract_lebail_intensities()` - Complete rewrite
   - `_create_parameter_vector()` - Tighter bounds
   - `_calculate_phase_pattern()` - Added diagnostics

2. `gui/visualization_tab.py`
   - `run_lebail_refinement()` - Better U/W ratio for synchrotron
