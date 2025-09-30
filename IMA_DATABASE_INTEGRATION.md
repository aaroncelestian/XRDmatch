# IMA Mineral Database Integration

## Overview

The XRDmatch application now integrates with the IMA (International Mineralogical Association) mineral database to provide authoritative mineral names, chemistry, and space group information during pattern searches. This resolves the issue where author names from DIF files were sometimes displayed instead of mineral names.

## Problem Statement

In the DIF file format used by AMCSD (American Mineralogist Crystal Structure Database), mineral names appear directly before author names in the file structure. During parsing and database import, these could sometimes be confused, leading to:

- Author names appearing in the "Mineral" column of search results
- Incorrect or missing chemistry information
- Inconsistent space group data

## Solution

### 1. IMA Database Module (`utils/ima_mineral_database.py`)

A new utility module that:
- Loads the IMA mineral database from CSV (`data/IMA_Export_2025924_163537.csv`)
- Provides 6,178+ authoritative mineral records
- Supports exact and fuzzy name matching
- Enables search by chemistry elements and space groups
- Cross-references mineral data to correct naming issues

**Key Features:**
- **Exact Lookup**: Case-insensitive mineral name matching
- **Fuzzy Matching**: Finds closest matches with similarity scoring (threshold 0.8)
- **Element Search**: Find minerals containing specific elements
- **Space Group Search**: Find minerals with specific crystallographic space groups
- **Name Correction**: Attempts to distinguish between mineral and author names

### 2. Enhanced Pattern Search Engines

Both search engines now cross-reference results with the IMA database:

#### `utils/pattern_search.py` (PatternSearchEngine)
- Added IMA database integration to `search_by_peaks()` method
- Added IMA database integration to `search_by_correlation()` method
- Each result now includes:
  - `ima_chemistry`: Authoritative chemical formula from IMA
  - `ima_space_group`: Official space group from IMA
  - `ima_verified`: Boolean indicating if mineral was found in IMA database

#### `utils/fast_pattern_search.py` (FastPatternSearchEngine)
- Added IMA database integration to `ultra_fast_correlation_search()` method
- Same enhanced result fields as above

### 3. Data Structure

The IMA CSV contains the following key fields:
- **Mineral Name**: Official IMA-approved name
- **IMA Chemistry**: Chemical formula with valence states
- **Space Groups**: Crystallographic space group(s)
- **Crystal Systems**: Crystal system classification
- **Chemistry Elements**: List of constituent elements
- **IMA Number**: Official IMA registration number
- **IMA Status**: Approval status (Approved, Grandfathered, etc.)

## Usage

### Basic Usage (Automatic)

The IMA database is automatically loaded when you perform a pattern search. No code changes are needed in your existing workflows.

```python
from utils.fast_pattern_search import FastPatternSearchEngine

engine = FastPatternSearchEngine()
results = engine.ultra_fast_correlation_search(experimental_pattern)

# Results now include IMA-verified information
for result in results:
    print(f"Mineral: {result['mineral_name']}")
    print(f"IMA Chemistry: {result.get('ima_chemistry', 'N/A')}")
    print(f"IMA Space Group: {result.get('ima_space_group', 'N/A')}")
    print(f"IMA Verified: {result.get('ima_verified', False)}")
```

### Direct IMA Database Access

You can also directly query the IMA database:

```python
from utils.ima_mineral_database import get_ima_database

ima_db = get_ima_database()

# Exact lookup
info = ima_db.get_mineral_info('Calcite')
print(info['chemistry'])  # CaCO3
print(info['space_group'])  # R-3c|R-32/c, R3c

# Fuzzy matching
result = ima_db.fuzzy_match_mineral('Calcit', threshold=0.8)
if result:
    matched_name, score, info = result
    print(f"Matched: {matched_name} (score: {score:.2f})")

# Search by elements
minerals = ima_db.search_by_chemistry(['Ca', 'C', 'O'])
print(f"Found {len(minerals)} calcium carbonates")

# Search by space group
minerals = ima_db.search_by_space_group('R-3c')
print(f"Found {len(minerals)} minerals with R-3c symmetry")
```

## Benefits

1. **Accurate Mineral Identification**: Authoritative names from IMA database
2. **Correct Chemistry**: Official chemical formulas with proper valence states
3. **Reliable Space Groups**: Crystallographic data from primary source
4. **Quality Assurance**: `ima_verified` flag indicates data reliability
5. **Fuzzy Matching**: Handles spelling variations and typos
6. **Cross-Reference**: Validates database entries against official records

## Testing

Run the test script to verify IMA database functionality:

```bash
python test_ima_database.py
```

This will test:
- Database loading (6,178 minerals)
- Exact mineral lookups
- Fuzzy name matching
- Element-based searches
- Space group searches
- Author name correction

## Files Modified

1. **New Files:**
   - `utils/ima_mineral_database.py` - IMA database interface
   - `test_ima_database.py` - Test script
   - `IMA_DATABASE_INTEGRATION.md` - This documentation

2. **Modified Files:**
   - `utils/pattern_search.py` - Added IMA cross-referencing
   - `utils/fast_pattern_search.py` - Added IMA cross-referencing

3. **Data Files:**
   - `data/IMA_Export_2025924_163537.csv` - IMA mineral database (6,178 entries)

## Future Enhancements

Potential improvements for future versions:

1. **GUI Integration**: Display IMA verification status in search results table
2. **Name Correction Tool**: Batch correct mineral names in existing database
3. **Chemistry Validation**: Flag entries with chemistry mismatches
4. **Space Group Validation**: Highlight space group inconsistencies
5. **IMA Status Filter**: Filter by approval status (Approved, Questionable, etc.)
6. **Periodic Updates**: Mechanism to update IMA database from online source

## Notes

- The IMA database is loaded once per session and cached in memory
- Fuzzy matching uses SequenceMatcher for similarity scoring
- Multiple space groups (separated by `|`) are supported
- The database includes minerals from all geological eras
- IMA verification is non-blocking - searches work even if IMA data is unavailable

## Synchrotron Wavelength Update

As part of this update, the synchrotron wavelength was also corrected:
- **Old value**: 0.2401 Å
- **New value**: 0.24105 Å
- **Location**: `gui/pattern_tab.py` line 88

This provides more accurate wavelength conversion for synchrotron XRD data.
