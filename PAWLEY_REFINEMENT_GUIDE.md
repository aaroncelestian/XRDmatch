# Pawley Refinement for Preferred Orientation

## Overview

The software now supports **Pawley-style refinement** where individual peak intensities can be refined independently. This is essential for samples with **preferred orientation** (texture) where the relative peak intensities deviate from theoretical structure factors.

## When to Use Pawley Refinement

### Use Pawley Mode When:
- ✅ Sample has **preferred orientation** (texture) along unknown direction
- ✅ Certain peaks are much stronger/weaker than expected
- ✅ Sample preparation causes alignment (pressed pellets, thin films, fibers)
- ✅ Le Bail refinement gives poor fit despite correct phase identification
- ✅ You see systematic intensity mismatches (some peaks too tall, others too short)

### Use Le Bail Mode (Default) When:
- ✅ Sample is well-randomized powder
- ✅ No systematic intensity deviations
- ✅ You want more stable refinement
- ✅ You trust the structure factors

## How It Works

### Le Bail Method (Default):
```
Intensity of peak i = Scale_Factor × Theoretical_Intensity_i
```
- One scale factor for all peaks
- Maintains relative intensity ratios from structure factors
- More stable, fewer parameters

### Pawley Method (New Feature):
```
Intensity of peak i = Scale_Factor × Theoretical_Intensity_i × Multiplier_i
```
- Individual multiplier for each peak (0.1x to 10x)
- Breaks relative intensity constraints
- Handles preferred orientation
- More parameters, potentially less stable

## How to Enable

1. Go to **Visualization** tab
2. Under "Le Bail Refinement" section
3. Check **"Refine Peak Intensities (Pawley)"**
4. Run refinement

## Parameter Details

### Intensity Multiplier Bounds:
- **Range:** 0.1 to 10.0
- **Meaning:** Each peak can be 0.1x to 10x its theoretical intensity
- **Initial value:** 1.0 (all peaks start at theoretical ratios)

### Number of Parameters:
- **Le Bail:** ~10-15 parameters (scale, profile, unit cell, zero shift)
- **Pawley:** 10-15 + N_peaks parameters
  - Example: 20 peaks → 30-35 total parameters
  - Example: 50 peaks → 60-65 total parameters
  - **⚠️ WARNING:** >100 peaks → >110 parameters (may be slow/unstable)

### Performance Considerations:
- **< 50 peaks:** Fast, stable
- **50-100 peaks:** Moderate speed, generally stable
- **> 100 peaks:** Slow, may be unstable - **USE 2θ RANGE TO REDUCE PEAKS!**

## Recommended Workflow

### Step 1: Try Le Bail First
```
Refine Peak Intensities: ✗ (unchecked)
Max Iterations: 15
```
- If fit is good → Done!
- If systematic intensity mismatches → Go to Step 2

### Step 2: Enable Pawley
```
Refine Peak Intensities: ✓ (checked)
Max Iterations: 20-30 (needs more iterations)
Initial FWHM: 0.05-0.1° (start with good peak widths)
2θ Range: IMPORTANT - Limit to reduce number of peaks if >100 total
```

**CRITICAL:** If you have >100 peaks total across all phases:
- Use 2θ range to focus on a smaller region (e.g., 5-30°)
- This reduces the number of intensity parameters
- Makes refinement faster and more stable

### Step 3: Monitor Refinement
Watch the real-time plot:
- Individual peaks should adjust to match experimental heights
- Difference curve should improve
- Rwp should decrease

### Step 4: Check Results
Look for:
- **Intensity multipliers:** Should vary systematically if preferred orientation exists
- **Rwp improvement:** Should be significantly better than Le Bail
- **Physical sense:** Multipliers shouldn't be at bounds (0.1 or 10.0)

## Example: Preferred Orientation Patterns

### Fiber Texture (e.g., [001] preferred):
- (00l) peaks: Multipliers >> 1.0 (much stronger)
- (hk0) peaks: Multipliers << 1.0 (much weaker)
- Other peaks: Multipliers ≈ 1.0

### Plate-like Crystals (e.g., [hk0] preferred):
- (hk0) peaks: Multipliers >> 1.0
- (00l) peaks: Multipliers << 1.0
- Other peaks: Multipliers ≈ 1.0

### Random Powder (No Texture):
- All multipliers ≈ 1.0 ± 0.2
- If this is the case, Le Bail is sufficient!

## Tips for Success

### 1. Start with Good Initial Parameters
- Get peak widths right first (Initial FWHM)
- Use correct unit cell (enable "Refine Unit Cell")
- Make sure phases are correctly identified

### 2. Use More Iterations
- Pawley needs 20-30 iterations (vs 10-15 for Le Bail)
- Watch convergence in real-time plot

### 3. Limit 2θ Range if Needed
- Focus on region with best peaks
- Exclude noisy regions
- Reduces number of parameters

### 4. Check for Instability
If refinement is unstable:
- Reduce number of peaks (use 2θ range)
- Disable "Refine Unit Cell" temporarily
- Increase Initial FWHM slightly
- Try fewer iterations first (10-15)

### 5. Interpret Results
After refinement, check which peaks have:
- **Multiplier > 2.0:** Strongly enhanced (preferred orientation favors this reflection)
- **Multiplier < 0.5:** Strongly suppressed (preferred orientation disfavors this reflection)
- **Multiplier ≈ 1.0:** Normal intensity (no texture effect)

## Comparison: Le Bail vs Pawley

| Feature | Le Bail | Pawley |
|---------|---------|--------|
| **Intensity constraints** | Fixed ratios | Free |
| **Parameters** | ~10-15 | 10-15 + N_peaks |
| **Stability** | More stable | Less stable |
| **Preferred orientation** | Cannot handle | Handles well |
| **Computation time** | Faster | Slower |
| **Use case** | Random powder | Textured samples |

## Troubleshooting

### Issue: Refinement doesn't converge
**Solution:**
- Increase Max Iterations to 30-40
- Disable "Refine Unit Cell" temporarily
- Use smaller 2θ range
- Check that Initial FWHM is reasonable

### Issue: Intensity multipliers hit bounds (0.1 or 10.0)
**Solution:**
- This suggests extreme preferred orientation
- Check phase identification (might be wrong phase)
- Consider if sample preparation caused severe texture
- May need March-Dollase or spherical harmonics (future feature)

### Issue: Pawley gives worse fit than Le Bail
**Solution:**
- Sample probably doesn't have preferred orientation
- Use Le Bail mode instead
- Check that you have enough data points
- Verify phase identification is correct

### Issue: Too many parameters, refinement is slow or hangs
**Solution:**
- **MOST IMPORTANT:** Use 2θ range to limit number of peaks to <50 per phase
- Focus on the region with best signal-to-noise
- Example: If you have 200 peaks from 5-90°, limit to 10-40° to get ~60 peaks
- Disable "Refine Unit Cell" if unit cell is known accurately
- Consider using Le Bail mode instead if texture isn't severe

### Issue: Program hangs during refinement
**Cause:** Too many Pawley intensity parameters (>100)
**Solution:**
- **IMMEDIATELY:** Use 2θ range to reduce peaks
- The program will warn you: "⚠️ WARNING: XXX intensity parameters may cause slow/unstable refinement"
- Staged refinement automatically disables Pawley in Stage 1 to prevent hanging

## Future Enhancements

Planned features:
- **March-Dollase correction:** Model fiber texture with known axis
- **Spherical harmonics:** Model complex texture automatically
- **Texture visualization:** Plot pole figures from refined intensities
- **Constraints:** Link similar reflections (e.g., all (h00) peaks)

## References

- **Le Bail method:** Le Bail, A. (1988). Powder Diffr. 3, 47-51.
- **Pawley method:** Pawley, G.S. (1981). J. Appl. Cryst. 14, 357-361.
- **Preferred orientation:** Dollase, W.A. (1986). J. Appl. Cryst. 19, 267-272.
