# Quick Start: Visualization & Export Tab

## 🚀 5-Minute Quick Start

### Step 1: Import Your Data
```
1. Complete phase matching in "Phase Matching" tab
2. Switch to "Visualization & Export" tab
3. Click "Import from Phase Matching" button
```

### Step 2: Run Le Bail Refinement (Optional)
```
1. Set Max Iterations: 10-15 (recommended)
2. Click "Run Le Bail Refinement"
3. Wait 1-10 seconds
4. View results in "Refinement Results" tab
```

### Step 3: Customize Plot
```
Experimental Data:
  - Click "Color" button → Choose color
  - Adjust line width (0.1-10.0)

Calculated Pattern (after refinement):
  - Click "Color" button → Choose color
  - Adjust line width

Waterfall Plot:
  - Set offset value (e.g., 200)
  - Click "Update Plot"
```

### Step 4: Export
```
For presentations:
  - DPI: 150-200
  - Format: PNG
  
For publications:
  - DPI: 300-600
  - Format: PDF or SVG
  
For data analysis:
  - Format: CSV
```

---

## 🎨 Common Customizations

### Publication-Ready Plot
```
1. Set DPI to 300
2. Enable legend and grid
3. Adjust line widths: Exp=1.5, Calc=1.5, Diff=1.0
4. Export as PDF
```

### Waterfall Plot for Presentation
```
1. Set Waterfall Offset: 200-500
2. Disable legend (cleaner look)
3. Enable grid
4. Export as PNG (200 DPI)
```

### Data Export for Origin/Igor
```
1. Run Le Bail refinement (optional)
2. Click "Export Data (CSV)"
3. Import CSV into your analysis software
```

---

## ⚡ Performance Tips

### Le Bail Refinement
- **Fast test**: 5-10 iterations (~1-2 seconds)
- **Standard**: 15 iterations (~2-5 seconds)
- **High quality**: 20-30 iterations (~5-15 seconds)

### Export Quality
- **Screen/Web**: 72-150 DPI
- **Presentation**: 150-200 DPI
- **Publication**: 300-600 DPI

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No Data" error | Load pattern and match phases first |
| Refinement slow | Reduce max iterations to 10 |
| Export failed | Check file permissions |
| Colors not updating | Click "Update Plot" button |

---

## 📊 Example Workflows

### Workflow 1: Quick Export
```
Import → Customize colors → Export PNG
Time: 30 seconds
```

### Workflow 2: Publication Figure
```
Import → Le Bail refinement → Customize → Export PDF
Time: 2-3 minutes
```

### Workflow 3: Waterfall Comparison
```
Import → Set waterfall offset → Customize colors → Export
Time: 1-2 minutes
```

---

## 💡 Pro Tips

1. **Save time**: Run Le Bail refinement in Phase Matching tab first, then import to Visualization tab with results already available

2. **Color schemes**: Use contrasting colors for experimental (blue) and calculated (red/orange) patterns

3. **Waterfall offset**: Start with 100 and increase until phases are clearly separated

4. **DPI settings**: Higher DPI = larger file size. Use 300 DPI as default for most purposes

5. **CSV export**: Includes all data points - perfect for custom plotting in other software

---

## 📖 More Information

- Full documentation: `VISUALIZATION_TAB_README.md`
- Technical details: `CHANGES_SUMMARY.md`
- Performance tests: `test_lebail_performance.py`

---

## 🎯 Key Features at a Glance

✅ One-click import from Phase Matching  
✅ Fast Le Bail refinement (1-10 seconds)  
✅ Customizable colors and line widths  
✅ Waterfall plots  
✅ Multiple export formats (PNG, PDF, SVG, CSV)  
✅ High-resolution output (up to 600 DPI)  
✅ Interactive matplotlib tools  
✅ Detailed refinement reports  

---

**Need Help?** Check the full documentation or run test scripts to verify installation.
