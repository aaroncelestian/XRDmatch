# Visualization & Export Tab - User Guide

## Overview
The new **Visualization & Export** tab provides advanced plotting capabilities, customization options, and Le Bail refinement visualization for your XRD phase matching results.

## Features

### 1. Data Import
- **Import from Phase Matching**: Click this button to import your experimental pattern and matched phases from the Phase Matching tab
- The tab will automatically load selected phases (checked in the results table) or all phases if none are selected

### 2. Le Bail Refinement
- **Run Le Bail Refinement**: Perform crystallographic refinement on your matched phases
- **Max Iterations**: Control the number of refinement cycles (5-50, default: 15)
- **Progress Tracking**: Visual progress bar and status updates during refinement
- **Results Display**: View detailed refinement report including R-factors, GOF, and refined parameters

### 3. Plot Customization

#### Line Styles
- **Experimental Data**: Customize color and line width
- **Calculated Pattern**: Customize color and line width (shown after Le Bail refinement)
- **Difference Pattern**: Customize color and line width (shown after Le Bail refinement)

#### Plot Options
- **Waterfall Offset**: Create waterfall plots by setting an offset value (0 = standard overlay plot)
- **Show Legend**: Toggle legend display
- **Show Grid**: Toggle grid display
- **Update Plot**: Apply all customization changes

### 4. Export Options

#### Image Export
- **PNG**: High-quality raster image (customizable DPI)
- **PDF**: Vector format for publications
- **SVG**: Scalable vector graphics for editing in Illustrator/Inkscape

#### Data Export
- **CSV**: Export experimental data, calculated pattern, and difference curve
- Includes all data points for further analysis in other software

#### DPI Settings
- Adjust resolution from 72 to 600 DPI
- Default: 300 DPI (publication quality)

### 5. Multiple Views

#### Main Plot Tab
- Interactive matplotlib plot with zoom, pan, and save tools
- Shows experimental data, calculated pattern (after refinement), and difference curve

#### Refinement Results Tab
- Detailed text report of Le Bail refinement
- R-factors (Rp, Rwp, Rexp)
- Goodness-of-fit (GoF)
- Refined parameters for each phase

#### Phase Details Tab
- Table showing all phases with:
  - Phase name and formula
  - Scale factors
  - Rwp values (after refinement)
  - Color indicators
- Customize individual phase colors

## Workflow

### Basic Workflow
1. Load pattern in **Pattern Data** tab
2. Process and find peaks in **Data Processing** tab
3. Search for phases in **AMCSD Search**, **Local Database**, or **Pattern Search** tabs
4. Match phases in **Phase Matching** tab
5. Click **Import from Phase Matching** in **Visualization & Export** tab
6. Customize plot appearance
7. Export as desired format

### Advanced Workflow (with Le Bail Refinement)
1. Follow steps 1-5 above
2. Click **Run Le Bail Refinement**
3. Wait for refinement to complete (progress bar shows status)
4. Review refinement results in the **Refinement Results** tab
5. Customize plot with refined data
6. Export high-quality figures for publication

## Tips

### Performance Optimization
- Le Bail refinement has been optimized for speed:
  - Peak profiles only calculated within ±5 FWHM of peak centers
  - Pre-calculation of unchanging patterns during optimization
  - Reduced default iterations (15) for faster convergence
  - Adjustable convergence thresholds

### Waterfall Plots
- Set **Waterfall Offset** to a positive value (e.g., 100-500 depending on your intensity scale)
- Each phase will be displayed with vertical offset
- Useful for visualizing individual phase contributions
- Experimental pattern shown at top

### Color Customization
- Click color buttons to open color picker
- Colors are preserved when updating plots
- Individual phase colors can be customized in the Phase Details tab

### Export Quality
- For presentations: 150-200 DPI PNG
- For publications: 300-600 DPI PNG or PDF/SVG
- CSV export includes all numerical data for custom plotting in Origin, Igor, etc.

## Troubleshooting

### "No Data" Error
- Make sure you've loaded a pattern and performed phase matching first
- Check that phases are selected (or present) in the Phase Matching tab

### Le Bail Refinement Slow/Hanging
- Reduce max iterations (try 10 or fewer for quick tests)
- The optimization has been significantly improved in this version
- Typical refinement time: 10-60 seconds depending on number of phases and data points

### Export Failed
- Check file permissions in the target directory
- Ensure you have write access to the selected location
- Try a different file format if one fails

## Technical Details

### Le Bail Refinement Improvements
1. **Optimized peak profile calculation**: Only computes profiles within relevant range
2. **Pre-calculated background patterns**: Reduces redundant calculations
3. **Reduced iteration counts**: Default 15 iterations (was 30)
4. **Tighter convergence criteria**: ftol=1e-6, gtol=1e-5
5. **Better parameter bounds**: Prevents optimization from exploring unrealistic parameter space

### Supported Formats
- **Input**: Experimental patterns from Phase Matching tab
- **Output Images**: PNG, PDF, SVG
- **Output Data**: CSV (comma-separated values)

## Future Enhancements
Potential features for future versions:
- Individual phase pattern extraction and export
- Multiple pattern overlay (compare different samples)
- Custom color schemes and themes
- Annotation tools for peak labeling
- Batch export for multiple refinements
