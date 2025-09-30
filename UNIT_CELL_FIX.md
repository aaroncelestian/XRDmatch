# Unit Cell Parameters Fix for Le Bail Refinement

## Problem

Le Bail refinement was showing default unit cell values (10.0 Å, 90°) instead of actual crystallographic data from the database:

```
Unit cell:
a = 10.0000 Å  ← Should be 11.8971 Å for Epsomite
b = 10.0000 Å  ← Should be 11.9085 Å
c = 10.0000 Å  ← Should be 6.7873 Å
α = 90.000°
β = 90.000°
γ = 90.000°
```

## Root Cause

The SQL queries in the pattern search engines were **not selecting the unit cell parameters** from the database. The queries only retrieved:
- `mineral_id`, `mineral_name`, `chemical_formula`, `space_group`
- Diffraction pattern data (`two_theta`, `intensities`, `d_spacings`)

But **NOT** the unit cell parameters:
- `cell_a`, `cell_b`, `cell_c`, `cell_alpha`, `cell_beta`, `cell_gamma`

## Solution

Updated three files to include unit cell parameters in the data pipeline:

### 1. `utils/fast_pattern_search.py`

**Changed SQL query** (line 105-113):
```python
# BEFORE
SELECT m.id, m.mineral_name, m.chemical_formula, m.space_group,
       dp.two_theta, dp.intensities, dp.d_spacings

# AFTER
SELECT m.id, m.mineral_name, m.chemical_formula, m.space_group,
       m.cell_a, m.cell_b, m.cell_c, m.cell_alpha, m.cell_beta, m.cell_gamma,
       dp.two_theta, dp.intensities, dp.d_spacings
```

**Updated row unpacking** (line 131):
```python
# BEFORE
mineral_id, mineral_name, formula, space_group, two_theta_json, ...

# AFTER
mineral_id, mineral_name, formula, space_group, cell_a, cell_b, cell_c, 
cell_alpha, cell_beta, cell_gamma, two_theta_json, ...
```

**Updated metadata storage** (lines 157-169):
```python
self.mineral_metadata.append({
    'id': mineral_id,
    'name': mineral_name,
    'formula': formula,
    'space_group': space_group,
    'cell_a': cell_a,           # ← Added
    'cell_b': cell_b,           # ← Added
    'cell_c': cell_c,           # ← Added
    'cell_alpha': cell_alpha,   # ← Added
    'cell_beta': cell_beta,     # ← Added
    'cell_gamma': cell_gamma,   # ← Added
    'pattern_norm': pattern_norm
})
```

**Updated search results** (lines 289-303):
```python
result = {
    'mineral_id': metadata['id'],
    'mineral_name': metadata['name'],
    'chemical_formula': metadata['formula'],
    'space_group': metadata['space_group'],
    'cell_a': metadata.get('cell_a'),           # ← Added
    'cell_b': metadata.get('cell_b'),           # ← Added
    'cell_c': metadata.get('cell_c'),           # ← Added
    'cell_alpha': metadata.get('cell_alpha'),   # ← Added
    'cell_beta': metadata.get('cell_beta'),     # ← Added
    'cell_gamma': metadata.get('cell_gamma'),   # ← Added
    'correlation': float(correlation),
    'r_squared': float(correlation ** 2),
    'search_method': 'ultra_fast_correlation'
}
```

### 2. `utils/pattern_search.py`

Applied the same changes to **two SQL queries**:
- Peak-based search (lines 65-71)
- Correlation-based search (lines 190-197)

Updated row unpacking and result dictionaries in both methods.

### 3. `utils/lebail_refinement.py`

The `_extract_unit_cell()` method was already correct - it looks for `cell_a`, `cell_b`, etc. in the phase data. The issue was that these values weren't being provided by the search engines.

## Data Flow

```
Database (local_cif_database.db)
  └─ minerals table (has cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)
      ↓
  Pattern Search (fast_pattern_search.py or pattern_search.py)
      ↓ [NOW INCLUDES UNIT CELL IN SQL QUERY]
  Search Results (dict with cell_a, cell_b, etc.)
      ↓
  Multi-Phase Analyzer (multi_phase_analyzer.py)
      ↓ [Passes phase data to Le Bail]
  Le Bail Refinement (lebail_refinement.py)
      ↓ [_extract_unit_cell() extracts from phase data]
  Refinement Report (shows actual unit cell values)
```

## Verification

Created `test_unit_cell_fix.py` to verify:
1. ✓ Database contains unit cell parameters
2. ✓ SQL queries can retrieve them
3. ✓ Values are valid (not default 10.0, 10.0, 10.0)

**Test Results:**
```
Mineral: Epsomite
Unit Cell Parameters:
  a = 11.8971 Å  ✓
  b = 11.9085 Å  ✓
  c = 6.7873 Å   ✓
  α = 90.0°
  β = 90.0°
  γ = 90.0°

Mineral: Hexahydrite
Unit Cell Parameters:
  a = 10.11 Å    ✓
  b = 7.212 Å    ✓
  c = 24.41 Å    ✓
  α = 90.0°
  β = 98.3°      ✓
  γ = 90.0°

Mineral: Meridianiite
Unit Cell Parameters:
  a = 6.7508 Å   ✓
  b = 6.8146 Å   ✓
  c = 17.2924 Å  ✓
  α = 88.118°    ✓
  β = 89.481°    ✓
  γ = 62.689°    ✓
```

## Expected Outcome

After these changes, Le Bail refinement reports should show **actual crystallographic unit cell parameters** instead of defaults:

```
Phase 1: Epsomite
Unit cell:
a = 11.8971 Å  ← Actual value from database
b = 11.9085 Å  ← Actual value from database
c = 6.7873 Å   ← Actual value from database
α = 90.000°
β = 90.000°
γ = 90.000°
Space group: P2_12_12_1
```

## Files Modified

1. ✅ `utils/fast_pattern_search.py` - Ultra-fast correlation search
2. ✅ `utils/pattern_search.py` - Peak-based and correlation search
3. ✅ `utils/lebail_refinement.py` - Already correct, no changes needed

## Files Created

1. ✅ `test_unit_cell_fix.py` - Verification test
2. ✅ `UNIT_CELL_FIX.md` - This documentation

## Testing

To verify the fix works:

1. **Run the test script:**
   ```bash
   python test_unit_cell_fix.py
   ```

2. **Run a full analysis with Le Bail refinement:**
   - Load an experimental pattern
   - Run phase matching
   - Navigate to Visualization tab
   - Run Le Bail refinement
   - Check the refinement report for actual unit cell values

## Notes

- The database (`data/local_cif_database.db`) already contains correct unit cell parameters
- The `_extract_unit_cell()` method in `lebail_refinement.py` was already looking for the right keys
- The only issue was that the search engines weren't retrieving these values from the database
- This fix ensures unit cell parameters flow through the entire pipeline from database to refinement report

---

**Status:** ✅ **FIXED**  
**Date:** 2025-09-30  
**Impact:** Le Bail refinement now uses actual crystallographic unit cell parameters for all phases
