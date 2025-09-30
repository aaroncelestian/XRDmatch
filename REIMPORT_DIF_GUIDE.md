# Guide: Re-importing DIF File with Corrected Mineral Names

## Problem

The current database has author names stored in the `mineral_name` field instead of actual mineral names. This happened because the DIF file parser was taking the first line (often the author name) instead of the actual mineral name.

## Solution

The DIF import script has been updated to:
1. Look for `_chemical_name_mineral` tags in the DIF file
2. Validate all candidate names against the IMA database
3. Use official IMA mineral names when found
4. Apply fuzzy matching to correct typos and variations

## Steps to Re-import

### 1. Backup Your Current Database (IMPORTANT!)

```bash
cd /Users/aaroncelestian/Library/CloudStorage/Dropbox/Python/XRDmatch
cp data/local_cif_database.db data/local_cif_database.db.backup
```

### 2. Run the DIF Import Script

```bash
python import_dif_data.py
```

This will:
- Load the IMA database (6,178 minerals)
- Parse the DIF file with improved name extraction
- Validate each mineral name against IMA records
- Show statistics on corrections made
- Ask for confirmation before updating the database

### 3. Expected Output

You should see something like:

```
======================================================================
Parsing AMCSD DIF Data File with IMA Validation
======================================================================

Loading IMA database for name validation...
✅ Loaded 6178 minerals from IMA database

Reading file: data/difdata.txt...
✓ File read: 12,345,678 characters

Splitting into mineral sections...
✓ Found 7980 sections

Parsing mineral data...
  Processing section 500/7980 (6.3%) - Found 450 minerals so far...
  ...

======================================================================
✓ Parsing complete!
  Successfully parsed: 7580 minerals
  IMA verified (exact match): 5200
  IMA corrected (fuzzy match): 1800
  Not in IMA database: 580
  Skipped (no data): 400 sections
======================================================================

Example minerals found:
----------------------------------------------------------------------
Calcite                        AMCSD: 0000001  Peaks: 45    λ: 1.5406Å
Quartz                         AMCSD: 0000002  Peaks: 38    λ: 1.5406Å
...

======================================================================
Import these patterns to database? (yes/no): 
```

### 4. Confirm Import

Type `yes` when prompted. The script will:
- Match patterns to existing database entries by AMCSD ID
- Update mineral names with IMA-verified names
- Update diffraction patterns

### 5. Rebuild Search Index

After importing, rebuild the fast search index:

```bash
python -c "from utils.fast_pattern_search import FastPatternSearchEngine; engine = FastPatternSearchEngine(); engine.build_search_index(force_rebuild=True)"
```

### 6. Verify Results

Run a test search to verify mineral names are correct:

```bash
python test_epsomite_from_db.py
```

Or use the validation script:

```bash
python validate_mineral_names.py
```

## What Gets Updated

The import script updates:
- **mineral_name**: Corrected to official IMA name
- **diffraction_patterns**: Peak positions and intensities from DIF file

The script does NOT change:
- AMCSD IDs (used for matching)
- CIF content
- Other mineral metadata

## Statistics You'll See

- **IMA verified (exact match)**: Mineral names that exactly matched IMA database
- **IMA corrected (fuzzy match)**: Names that were similar and corrected (e.g., "Calcit" → "Calcite")
- **Not in IMA database**: Synthetic materials, obsolete names, or non-minerals
- **Skipped**: Sections without valid diffraction data

## Troubleshooting

### If import fails:
1. Check that `data/difdata.dif` or `data/difdata.txt` exists
2. Verify the IMA CSV file is present: `data/IMA_Export_2025924_163537.csv`
3. Restore backup if needed: `cp data/local_cif_database.db.backup data/local_cif_database.db`

### If you see many "Not in IMA database":
This is normal for:
- Synthetic materials
- Experimental phases
- Very new minerals not yet in the IMA export
- Obsolete or renamed minerals

### If mineral names still look wrong:
Run the validation script to identify specific issues:
```bash
python validate_mineral_names.py
```

## Alternative: Quick Fix Without Re-import

If you don't want to re-import everything, you can create a script to update just the mineral names:

```python
import sqlite3
from utils.local_database import LocalCIFDatabase
from utils.ima_mineral_database import get_ima_database

db = LocalCIFDatabase()
ima_db = get_ima_database()

conn = sqlite3.connect(db.db_path)
cursor = conn.cursor()

# Get all minerals
cursor.execute("SELECT id, mineral_name FROM minerals")
minerals = cursor.fetchall()

corrected = 0
for mineral_id, mineral_name in minerals:
    # Try to find correct name in IMA database
    ima_info = ima_db.get_mineral_info(mineral_name)
    if not ima_info:
        # Try fuzzy match
        fuzzy = ima_db.fuzzy_match_mineral(mineral_name, threshold=0.85)
        if fuzzy:
            correct_name = fuzzy[0]
            cursor.execute("UPDATE minerals SET mineral_name = ? WHERE id = ?", 
                          (correct_name, mineral_id))
            corrected += 1
            print(f"Corrected: '{mineral_name}' → '{correct_name}'")

conn.commit()
conn.close()
print(f"\nCorrected {corrected} mineral names")
```

Save this as `quick_fix_names.py` and run it.

## Recommendation

**Re-importing is recommended** because it ensures:
- Correct mineral names from the source
- Proper validation against IMA database
- Consistent data throughout the database
- Statistics on data quality

The process takes 5-10 minutes but provides the most reliable results.
