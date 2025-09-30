# Le Bail Refinement 2-Theta Range Limiting

## Overview

The Le Bail refinement engine now supports limiting the diffraction pattern calculation to a specific 2-theta angular range. This feature allows users to:

- **Focus on specific regions** of interest in the diffraction pattern
- **Improve performance** by reducing the number of data points to process
- **Exclude problematic regions** (e.g., areas with artifacts or background issues)
- **Target specific peaks** for more accurate refinement

## Implementation Details

### Core Changes

#### 1. `utils/lebail_refinement.py`

**Added Features:**
- `two_theta_range` attribute to store the optional range limit
- `two_theta_range` parameter in `set_experimental_data()` method
- `two_theta_range` parameter in `refine_phases()` method
- `_apply_two_theta_filter()` method to filter experimental data

**Usage Example:**
```python
from utils.lebail_refinement import LeBailRefinement

# Initialize refinement engine
lebail = LeBailRefinement()

# Set experimental data with 2-theta range (25° to 65°)
lebail.set_experimental_data(
    two_theta=two_theta_array,
    intensity=intensity_array,
    errors=error_array,
    two_theta_range=(25.0, 65.0)  # Optional range
)

# Or specify range during refinement
results = lebail.refine_phases(
    max_iterations=15,
    convergence_threshold=1e-5,
    two_theta_range=(25.0, 65.0)  # Optional range
)
```

#### 2. `utils/multi_phase_analyzer.py`

**Added Features:**
- `two_theta_range` parameter in `perform_lebail_refinement()` method
- Passes the range to both `set_experimental_data()` and `refine_phases()`

**Usage Example:**
```python
results = analyzer.perform_lebail_refinement(
    experimental_data=exp_data,
    identified_phases=phases,
    max_iterations=15,
    two_theta_range=(20.0, 70.0)  # Optional range
)
```

#### 3. `gui/visualization_tab.py`

**Added UI Controls:**
- **Checkbox**: "Limit range" to enable/disable the feature
- **Min 2θ spin box**: Set minimum 2-theta value (default: 10°)
- **Max 2θ spin box**: Set maximum 2-theta value (default: 90°)
- **Validation**: Ensures min < max before running refinement

**UI Location:**
The controls are located in the "Le Bail Refinement" group box in the Visualization tab, below the "Max Iterations" setting.

## User Guide

### Using the Feature in the GUI

1. **Load your experimental pattern** in the Phase Matching tab
2. **Navigate to the Visualization tab**
3. **Import data** using "Import from Phase Matching" button
4. **Enable range limiting**:
   - Check the "Limit range" checkbox
   - Set the minimum 2θ value (e.g., 25°)
   - Set the maximum 2θ value (e.g., 65°)
5. **Run refinement** by clicking "Run Le Bail Refinement"

### Benefits by Use Case

#### 1. **Performance Optimization**
- **Scenario**: Large datasets with many data points
- **Action**: Limit to the most informative region (e.g., 10-80°)
- **Result**: Faster refinement with minimal quality loss

#### 2. **Artifact Exclusion**
- **Scenario**: Low-angle artifacts or high-angle noise
- **Action**: Exclude problematic regions (e.g., use 15-75° instead of 5-90°)
- **Result**: More accurate R-factors and better fit quality

#### 3. **Focused Analysis**
- **Scenario**: Specific peaks of interest in a narrow range
- **Action**: Limit to the region containing those peaks (e.g., 25-35°)
- **Result**: Refined parameters optimized for that specific region

#### 4. **Sequential Refinement**
- **Scenario**: Different phases dominate different angular regions
- **Action**: Refine each phase in its dominant region
- **Result**: Better phase separation and parameter determination

## Technical Details

### Data Filtering

When a 2-theta range is specified:

1. A boolean mask is created: `(two_theta >= min) & (two_theta <= max)`
2. All experimental data arrays are filtered:
   - `two_theta` values
   - `intensity` values
   - `errors` values
3. The filtered data is used for all subsequent calculations
4. Console output shows the data reduction:
   ```
   Applied 2-theta range filter: 25.00° - 65.00°
   Data points: 850 → 400
   ```

### Performance Impact

**Test Results** (from `test_2theta_range.py`):
- Full range (5-90°): 850 data points
- Limited range (25-65°): 400 data points
- **Data reduction: 52.9%**
- **Expected speedup: ~2-3x** (depends on peak density)

### Backward Compatibility

The feature is **fully backward compatible**:
- The `two_theta_range` parameter is optional (defaults to `None`)
- Existing code without the parameter will work unchanged
- Test scripts (`test_lebail_performance.py`, `test_lebail_refinement.py`) continue to work

## Testing

A comprehensive test script is provided: `test_2theta_range.py`

**Run the test:**
```bash
python test_2theta_range.py
```

**Test Coverage:**
- ✓ Initialization without range limiting
- ✓ Initialization with range limiting
- ✓ Data point reduction verification
- ✓ Range boundary validation
- ✓ Method signature compatibility

## Best Practices

### Recommended Ranges

| Use Case | Recommended Range | Notes |
|----------|------------------|-------|
| General refinement | 10-80° | Excludes low/high angle artifacts |
| High-resolution data | 5-120° | Use full range if data quality is good |
| Quick test | 15-60° | Fast refinement for initial testing |
| Specific phase | Variable | Focus on dominant peaks of the phase |

### Tips

1. **Start broad**: Use a wide range initially, then narrow if needed
2. **Check residuals**: Examine the difference plot to identify problematic regions
3. **Iterative approach**: Run refinement multiple times with different ranges
4. **Document choices**: Note the range used in your analysis for reproducibility

## Future Enhancements

Potential improvements for future versions:

- [ ] Multiple non-contiguous ranges (e.g., exclude specific regions)
- [ ] Automatic range suggestion based on peak density
- [ ] Per-phase range specification for multi-phase refinement
- [ ] Visual range selector on the plot
- [ ] Range presets for common scenarios

## References

- Main implementation: `utils/lebail_refinement.py`
- GUI implementation: `gui/visualization_tab.py`
- Integration: `utils/multi_phase_analyzer.py`
- Test script: `test_2theta_range.py`

---

**Version**: 1.0  
**Date**: 2025-09-30  
**Author**: XRDmatch Development Team
