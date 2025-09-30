# Le Bail Refinement Fixes Summary

## Issues Fixed

### 1. Dimension Mismatch Error (CRITICAL)
**Problem:** When using 2θ range filtering, the calculated pattern had different dimensions than the experimental data, causing plotting errors.

**Root Cause:** The `two_theta_range` parameter was being passed to BOTH `set_experimental_data()` and `refine_phases()` in `multi_phase_analyzer.py`, causing potential double-filtering.

**Solution:**
- Modified `utils/multi_phase_analyzer.py` (lines 399-405): Removed `two_theta_range` from `set_experimental_data()` call
- The range is now only applied in `refine_phases()` method
- Added `two_theta` and `experimental_intensity` arrays to refinement results for proper plotting

**Files Modified:**
- `utils/multi_phase_analyzer.py`
- `utils/lebail_refinement.py` (lines 269-278)
- `gui/visualization_tab.py` (lines 508-533)

### 2. Peak Shape Issues
**Problem:** Calculated peaks were too broad and didn't match experimental peak widths.

**Solution:**
- Tightened parameter bounds in `utils/lebail_refinement.py`:
  - U parameter: 0.0 to 0.05 (was 0.0 to 0.1)
  - V parameter: -0.005 to 0.005 (was -0.01 to 0.01)
  - W parameter: 0.001 to 0.05 (was 0.001 to 0.1)
- Reduced initial parameter values in `utils/multi_phase_analyzer.py`:
  - u_param: 0.005 (was 0.01)
  - w_param: 0.005 (was 0.01)

**Files Modified:**
- `utils/lebail_refinement.py` (lines 356-361)
- `utils/multi_phase_analyzer.py` (lines 418-420)

### 3. Zero Shift Constraints
**Problem:** Zero shift parameter could vary too much (±0.5°), causing unrealistic peak position shifts.

**Solution:**
- Tightened zero shift bounds from ±0.5° to ±0.1° in `utils/lebail_refinement.py` (line 366)

**Files Modified:**
- `utils/lebail_refinement.py` (line 366)

### 4. Real-Time Plotting During Refinement
**Problem:** No visual feedback during refinement, making it impossible to see what's happening.

**Solution:**
- Added `plot_callback` class variable to `LeBailRefinement` class
- Implemented `_realtime_plot_callback()` method in `visualization_tab.py`
- Callback updates plot after each iteration showing:
  - Experimental data (blue)
  - Calculated pattern (red)
  - Difference curve (green)
  - Current iteration, stage, Rwp, and GoF

**Files Modified:**
- `utils/lebail_refinement.py` (lines 17-18, 253-258)
- `gui/visualization_tab.py` (lines 416-430, 638-677)

### 5. Minor Bug Fix
**Problem:** `rwp_change` variable referenced before assignment when refinement loop doesn't execute.

**Solution:**
- Initialize `rwp_change = float('inf')` at the start of refinement

**Files Modified:**
- `utils/lebail_refinement.py` (line 167)

## Testing

Created comprehensive test: `test_multi_phase_2theta_range.py`
- Tests Le Bail refinement without 2θ range (uses all 850 points)
- Tests Le Bail refinement with 2θ range 25-65° (correctly filters to 400 points)
- Both tests pass successfully

## Usage

### Running Le Bail Refinement with 2θ Range:
1. Load experimental data in the Phase Matching tab
2. Identify phases
3. Go to Visualization tab and click "Import from Phase Matching"
4. Check "Limit range" and set min/max 2θ values
5. Click "Run Le Bail Refinement"
6. Watch real-time plot updates during refinement

### Recommended Settings:
- **Max Iterations:** 10-15 (default: 15)
- **2θ Range:** Use if you want to focus on a specific region
- **Staged Refinement:** Enabled by default (refines unit cell first, then profile)

## Parameter Constraints

### Profile Parameters (Caglioti function: FWHM² = U·tan²θ + V·tanθ + W):
- **U:** 0.0 to 0.05 (controls high-angle broadening)
- **V:** -0.005 to 0.005 (controls intermediate broadening)
- **W:** 0.001 to 0.05 (controls low-angle broadening, minimum width)
- **η (eta):** 0.0 to 1.0 (Gaussian/Lorentzian mixing in pseudo-Voigt)

### Other Parameters:
- **Zero Shift:** -0.1° to 0.1° (2θ offset correction)
- **Scale Factor:** 0.01 to 10.0
- **Unit Cell:** ±5% of initial values

## Notes

- Smaller U, V, W values = narrower peaks
- If peaks are still too broad, check experimental data quality
- Zero shift should be small (<0.05°) for well-calibrated instruments
- Real-time plotting adds ~10% overhead but provides valuable feedback
