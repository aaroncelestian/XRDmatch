# DIF Format Support

## Overview
The application now supports two DIF file formats for importing diffraction patterns.

## Supported Formats

### 1. Standard DIF Format (CIF-style)
Standard crystallographic DIF format with CIF-style headers:

```
_pd_phase_name 'Quartz'
_chemical_formula_sum 'Si O2'
_diffrn_radiation_wavelength 1.5406
_cell_length_a 4.9134
_cell_length_b 4.9134
_cell_length_c 5.4052
_symmetry_space_group_name_H-M 'P 32 2 1'

loop_
_pd_proc_2theta
_pd_proc_d_spacing
_pd_proc_intensity
20.85  4.2567  100.0
26.64  3.3430  50.5
...
```

**Features:**
- CIF-style metadata headers
- Explicit 2θ, d-spacing, and intensity columns
- Standard loop_ structure

---

### 2. AMCSD Bulk DIF Format ✨ NEW
AMCSD database format with multiple minerals in a single file:

```
    3   3   1        24
                45.00         11.10        2.0147    4   2   2        24
                47.89         23.32        1.8995    3   3   3         8
================================================================================
       XPOW Copyright 1993 Bob Downs, Ranjini Swaminathan and Kurt Bartelmehs
_END_
      Abernathyite
      Ross M, Evans H T
      American Mineralogist 49 (1964) 1578-1602
      _database_code_amcsd 0000130

      CELL PARAMETERS:    7.1760   7.1760  18.1260   90.000   90.000   90.000
```

**Format Details:**
- **Data columns:** `d-spacing  intensity  h  k  l  multiplicity`
- **Separator:** `================================================================================`
- **End marker:** `_END_`
- **Mineral name:** Appears after `_END_` and separator
- **Cell parameters:** Line starting with `CELL PARAMETERS:`
- **AMCSD ID:** Line with `_database_code_amcsd`

**Automatic Conversion:**
- 2θ values are calculated from d-spacings using Bragg's law: `λ = 2d sin(θ)`
- Default wavelength: Cu Kα (1.5406 Å)

---

## Import Methods

### Single DIF File Import
Use when you have:
- Individual DIF files
- Standard CIF-style DIF format
- AMCSD bulk file (imports first mineral only)

**Steps:**
1. Go to **Database Manager** tab
2. Click **Import Single DIF File**
3. Select your DIF file
4. Application will auto-detect format and parse accordingly

### Directory Import
Use when you have:
- Multiple individual DIF files in a folder
- Batch import needed

**Steps:**
1. Go to **Database Manager** tab
2. Click **Import DIF Directory**
3. Select directory containing DIF files
4. All `.dif` files will be imported

---

## AMCSD Bulk File Handling

### Current Behavior
When importing an AMCSD bulk DIF file (like `difdata.dif`):
- ⚠️ **Only the first mineral is imported**
- File size: Can be very large (100+ MB)
- Contains: Thousands of minerals

### Recommended Approach

**Option 1: Split the File** (Recommended)
Use external tools to split the bulk file into individual mineral DIF files:

```bash
# Example: Split AMCSD bulk file by separator
csplit -f mineral_ -b "%04d.dif" difdata.dif '/^==========/' '{*}'
```

Then use **Import DIF Directory** to batch import all files.

**Option 2: Use First Mineral Only**
If you just need to test the import:
- Use **Import Single DIF File**
- First mineral will be imported
- Check console for mineral name

**Option 3: Bulk Import Tool** (Future Enhancement)
A dedicated bulk import tool for AMCSD format is planned for a future update.

---

## Parser Features

### AMCSD Format Parser
The `_parse_amcsd_dif_format()` method:

1. **Detects format** by looking for `================` separator
2. **Extracts metadata:**
   - Mineral name (first capitalized line after separator)
   - AMCSD ID from `_database_code_amcsd`
   - Unit cell parameters from `CELL PARAMETERS:` line

3. **Parses diffraction data:**
   - Reads: d-spacing, intensity, h, k, l, multiplicity
   - Calculates 2θ from d-spacing using Bragg's law
   - Validates reflections (sin(θ) ≤ 1.0)

4. **Stops at next separator** (only imports first mineral)

### Format Detection
Automatic detection based on file content:
- If `================` found → AMCSD bulk format
- Otherwise → Standard CIF-style format

---

## Troubleshooting

### "No diffraction data found in DIF file"
**Causes:**
- File format not recognized
- Data section not properly formatted
- Missing required columns

**Solutions:**
1. Check file format matches one of the supported formats
2. Verify data columns are properly separated (whitespace)
3. Check console output for specific parsing errors

### "Failed to parse DIF file"
**Causes:**
- File encoding issues
- Malformed data
- Missing required fields

**Solutions:**
1. Check file encoding (should be UTF-8 or ASCII)
2. Verify file is not corrupted
3. Check console for detailed error traceback

### Large File Import is Slow
**Cause:**
- AMCSD bulk files are very large (100+ MB)
- Only first mineral is imported

**Solution:**
- Split file into individual minerals first
- Use directory import for batch processing

---

## Data Storage

### Database Schema
Imported DIF patterns are stored with:
- `calculation_method = 'DIF_import'`
- Full 2θ, d-spacing, and intensity arrays
- Linked to mineral metadata
- Wavelength information preserved

### Pattern Retrieval
When searching or matching:
- DIF-imported patterns are prioritized
- Wavelength conversion handled automatically
- Bragg's law used for wavelength conversions

---

## Future Enhancements

### Planned Features:
1. **Bulk AMCSD Import Tool**
   - Parse entire AMCSD bulk file
   - Import all minerals with progress tracking
   - Parallel processing for speed

2. **Format Validation**
   - Pre-import format checking
   - Better error messages
   - Format conversion tools

3. **Enhanced Metadata Extraction**
   - Author information
   - Publication details
   - Additional crystallographic data

---

## Testing

### Test Files Needed:
- ✅ AMCSD bulk format (difdata.dif) - Supported
- ⏳ Individual AMCSD DIF files - Needs testing
- ⏳ Standard CIF-style DIF files - Needs testing

### Validation:
After import, verify:
1. Mineral name extracted correctly
2. Peak positions (2θ) are reasonable
3. Intensities are normalized
4. Pattern displays correctly in search results

---

## References

- **AMCSD:** https://rruff.geo.arizona.edu/AMS/amcsd.php
- **XPOW:** Downs et al. (1993) American Mineralogist 78, 1104-1107
- **Bragg's Law:** λ = 2d sin(θ)

---

**Last Updated:** 2025-09-29
**Version:** 1.0
