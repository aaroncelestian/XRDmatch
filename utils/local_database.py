"""
Local CIF database manager for XRD phase matching
Handles parsing, storing, and searching local CIF files
"""

import os
import sqlite3
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
import hashlib
from pathlib import Path
import re
import signal
import time
from utils.cif_parser import CIFParser

class LocalCIFDatabase:
    """Local database for CIF files with SQLite backend"""
    
    def __init__(self, db_path: str = None):
        """Initialize the local CIF database"""
        if db_path is None:
            # Create database in the same directory as the script
            script_dir = Path(__file__).parent.parent
            db_path = script_dir / "data" / "local_cif_database.db"
            
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = str(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create minerals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS minerals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mineral_name TEXT NOT NULL,
                chemical_formula TEXT,
                space_group TEXT,
                crystal_system TEXT,
                cell_a REAL,
                cell_b REAL,
                cell_c REAL,
                cell_alpha REAL,
                cell_beta REAL,
                cell_gamma REAL,
                cell_volume REAL,
                density REAL,
                amcsd_id TEXT,
                authors TEXT,
                journal TEXT,
                year INTEGER,
                doi TEXT,
                cif_content TEXT NOT NULL,
                cif_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create elements table for chemical search
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mineral_elements (
                mineral_id INTEGER,
                element TEXT,
                count INTEGER,
                FOREIGN KEY (mineral_id) REFERENCES minerals (id),
                PRIMARY KEY (mineral_id, element)
            )
        ''')
        
        # Create diffraction patterns table for pre-calculated XRD data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diffraction_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mineral_id INTEGER,
                wavelength REAL NOT NULL,
                two_theta TEXT NOT NULL,  -- JSON array of 2θ values
                intensities TEXT NOT NULL,  -- JSON array of intensity values
                d_spacings TEXT NOT NULL,  -- JSON array of d-spacing values
                max_two_theta REAL DEFAULT 90.0,
                min_d_spacing REAL DEFAULT 0.5,
                calculation_method TEXT DEFAULT 'pymatgen',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (mineral_id) REFERENCES minerals (id),
                UNIQUE (mineral_id, wavelength, max_two_theta, min_d_spacing)
            )
        ''')
        
        # Create search indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mineral_name ON minerals (mineral_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_formula ON minerals (chemical_formula)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_space_group ON minerals (space_group)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_amcsd_id ON minerals (amcsd_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_elements ON mineral_elements (element)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diffraction_mineral ON diffraction_patterns (mineral_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_diffraction_wavelength ON diffraction_patterns (wavelength)')
        
        conn.commit()
        conn.close()
        print(f"Local CIF database initialized at: {self.db_path}")
    
    def parse_cif_with_pymatgen(self, cif_content: str, debug=False) -> List[Dict]:
        """Parse CIF content using pymatgen library with enhanced error handling"""
        try:
            # Parse with pymatgen - try multiple approaches
            try:
                if debug:
                    print("      🔄 Trying primary pymatgen parsing...")
                parser = CifParser.from_string(cif_content)
                cif_dict = parser.as_dict()
                if debug:
                    print(f"      ✅ Primary parsing succeeded, found {len(cif_dict)} blocks")
            except Exception as e:
                if debug:
                    print(f"      ❌ Primary pymatgen parsing failed: {e}")
                # Try with more lenient parsing
                try:
                    if debug:
                        print("      🔄 Trying lenient parsing with higher occupancy tolerance...")
                    parser = CifParser.from_string(cif_content, occupancy_tolerance=10.0)
                    cif_dict = parser.as_dict()
                    if debug:
                        print(f"      ✅ Lenient parsing succeeded, found {len(cif_dict)} blocks")
                except Exception as e2:
                    if debug:
                        print(f"      ❌ Lenient parsing also failed: {e2}")
                        print("      🔄 Falling back to basic CIF parsing...")
                    return self.parse_cif_basic_fallback(cif_content, debug=debug)
            
            results = []
            for key, structure in cif_dict.items():
                try:
                    data = {}
                    
                    # Basic information
                    data['block_name'] = key
                    
                    # Get mineral name from various possible fields
                    mineral_name = (structure.get('_chemical_name_mineral') or 
                                  structure.get('_chemical_name_common') or 
                                  key or 'Unknown')
                    data['mineral_name'] = mineral_name[0] if isinstance(mineral_name, list) else mineral_name
                    
                    # Chemical formula
                    formula = (structure.get('_chemical_formula_sum') or 
                             structure.get('_chemical_formula_analytical') or 
                             structure.get('_chemical_formula_structural') or 'Unknown')
                    data['chemical_formula'] = formula[0] if isinstance(formula, list) else formula
                    
                    # Space group
                    space_group = (structure.get('_space_group_name_H-M_alt') or 
                                 structure.get('_symmetry_space_group_name_H-M') or 
                                 structure.get('_space_group_name_H-M') or 'Unknown')
                    data['space_group'] = space_group[0] if isinstance(space_group, list) else space_group
                    
                    # Crystal system
                    crystal_system = (structure.get('_symmetry_cell_setting') or 
                                    structure.get('_space_group_crystal_system') or 'Unknown')
                    data['crystal_system'] = crystal_system[0] if isinstance(crystal_system, list) else crystal_system
                    
                    # Unit cell parameters
                    try:
                        def extract_float(value):
                            if isinstance(value, list):
                                value = value[0]
                            if isinstance(value, str):
                                # Remove uncertainty in parentheses
                                value = value.split('(')[0]
                            return float(value)
                        
                        data['cell_a'] = extract_float(structure.get('_cell_length_a', 0))
                        data['cell_b'] = extract_float(structure.get('_cell_length_b', 0))
                        data['cell_c'] = extract_float(structure.get('_cell_length_c', 0))
                        data['cell_alpha'] = extract_float(structure.get('_cell_angle_alpha', 90))
                        data['cell_beta'] = extract_float(structure.get('_cell_angle_beta', 90))
                        data['cell_gamma'] = extract_float(structure.get('_cell_angle_gamma', 90))
                        
                        # Calculate volume if not provided
                        cell_volume = structure.get('_cell_volume')
                        if cell_volume:
                            data['cell_volume'] = extract_float(cell_volume)
                        else:
                            # Calculate volume from unit cell parameters
                            a, b, c = data['cell_a'], data['cell_b'], data['cell_c']
                            alpha, beta, gamma = np.radians([data['cell_alpha'], data['cell_beta'], data['cell_gamma']])
                            
                            if all(x > 0 for x in [a, b, c]):
                                volume = a * b * c * np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) - 
                                                           np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2)
                                data['cell_volume'] = volume
                            else:
                                data['cell_volume'] = None
                                
                    except (ValueError, TypeError, KeyError):
                        data.update({
                            'cell_a': None, 'cell_b': None, 'cell_c': None,
                            'cell_alpha': None, 'cell_beta': None, 'cell_gamma': None,
                            'cell_volume': None
                        })
                    
                    # Physical properties
                    try:
                        density = structure.get('_exptl_crystal_density_diffrn')
                        if density:
                            if isinstance(density, list):
                                density = density[0]
                            data['density'] = float(str(density).split('(')[0])
                        else:
                            data['density'] = None
                    except (ValueError, TypeError):
                        data['density'] = None
                    
                    # Publication information
                    authors = structure.get('_publ_author_name', 'Unknown')
                    data['authors'] = authors[0] if isinstance(authors, list) else authors
                    
                    journal = structure.get('_journal_name_full', 'Unknown')
                    data['journal'] = journal[0] if isinstance(journal, list) else journal
                    
                    doi = structure.get('_journal_paper_doi')
                    data['doi'] = doi[0] if isinstance(doi, list) else doi
                    
                    # Extract year
                    try:
                        year = structure.get('_journal_year')
                        if year:
                            if isinstance(year, list):
                                year = year[0]
                            data['year'] = int(year)
                        else:
                            data['year'] = None
                    except (ValueError, TypeError):
                        data['year'] = None
                    
                    # AMCSD ID
                    amcsd_id = structure.get('_database_code_amcsd')
                    data['amcsd_id'] = amcsd_id[0] if isinstance(amcsd_id, list) else amcsd_id
                    
                    # Extract elements from formula
                    data['elements'] = self.extract_elements_from_formula(data['chemical_formula'])
                    
                    results.append(data)
                    
                except Exception as e:
                    print(f"Error parsing CIF block {key}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            if debug:
                print(f"      ❌ Error parsing CIF with pymatgen: {e}")
            # Fallback to basic CIF parsing
            return self.parse_cif_basic_fallback(cif_content, debug=debug)
    
    def parse_cif_basic_fallback(self, cif_content: str, debug=False) -> List[Dict]:
        """Basic CIF parsing fallback when pymatgen fails"""
        try:
            if debug:
                print("      🔄 Attempting basic CIF parsing fallback...")
            
            results = []
            lines = cif_content.split('\n')
            current_block = None
            current_data = {}
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('data_'):
                    # Save previous block if exists
                    if current_block and current_data:
                        parsed_data = self.extract_basic_cif_data(current_data, current_block, debug=debug)
                        if parsed_data:
                            results.append(parsed_data)
                    
                    # Start new block
                    current_block = line[5:]
                    current_data = {}
                    continue
                
                # Extract key-value pairs
                if line.startswith('_') and ' ' in line:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key, value = parts
                        # Clean up value
                        value = value.strip("'\"")
                        current_data[key] = value
            
            # Process last block
            if current_block and current_data:
                parsed_data = self.extract_basic_cif_data(current_data, current_block, debug=debug)
                if parsed_data:
                    results.append(parsed_data)
            
            if debug:
                print(f"      📊 Basic parsing extracted {len(results)} entries")
            return results
            
        except Exception as e:
            if debug:
                print(f"      ❌ Basic CIF parsing also failed: {e}")
            return []
    
    def extract_basic_cif_data(self, cif_data: dict, block_name: str, debug=False) -> dict:
        """Extract basic data from CIF dictionary"""
        try:
            data = {}
            data['block_name'] = block_name
            
            # Extract mineral name
            data['mineral_name'] = (cif_data.get('_chemical_name_mineral') or 
                                  cif_data.get('_chemical_name_common') or 
                                  block_name or 'Unknown')
            
            # Extract formula
            data['chemical_formula'] = (cif_data.get('_chemical_formula_sum') or 
                                      cif_data.get('_chemical_formula_analytical') or 'Unknown')
            
            # Extract space group
            data['space_group'] = (cif_data.get('_space_group_name_H-M_alt') or 
                                 cif_data.get('_symmetry_space_group_name_H-M') or 'Unknown')
            
            # Extract unit cell parameters
            try:
                def safe_float(value, default=None):
                    if not value:
                        return default
                    try:
                        return float(str(value).split('(')[0])
                    except:
                        return default
                
                data['cell_a'] = safe_float(cif_data.get('_cell_length_a'))
                data['cell_b'] = safe_float(cif_data.get('_cell_length_b'))
                data['cell_c'] = safe_float(cif_data.get('_cell_length_c'))
                data['cell_alpha'] = safe_float(cif_data.get('_cell_angle_alpha'), 90.0)
                data['cell_beta'] = safe_float(cif_data.get('_cell_angle_beta'), 90.0)
                data['cell_gamma'] = safe_float(cif_data.get('_cell_angle_gamma'), 90.0)
                data['cell_volume'] = safe_float(cif_data.get('_cell_volume'))
                
            except Exception:
                data.update({
                    'cell_a': None, 'cell_b': None, 'cell_c': None,
                    'cell_alpha': None, 'cell_beta': None, 'cell_gamma': None,
                    'cell_volume': None
                })
            
            # Extract other metadata
            data['crystal_system'] = cif_data.get('_symmetry_cell_setting', 'Unknown')
            data['density'] = None
            data['authors'] = cif_data.get('_publ_author_name', 'Unknown')
            data['journal'] = cif_data.get('_journal_name_full', 'Unknown')
            data['doi'] = cif_data.get('_journal_paper_doi')
            data['amcsd_id'] = cif_data.get('_database_code_amcsd')
            
            # Extract year
            try:
                year_str = cif_data.get('_journal_year')
                data['year'] = int(year_str) if year_str else None
            except:
                data['year'] = None
            
            # Extract elements
            data['elements'] = self.extract_elements_from_formula(data['chemical_formula'])
            
            # Only return if we have essential data
            if data['mineral_name'] != 'Unknown' or data['chemical_formula'] != 'Unknown':
                return data
            
            return None
            
        except Exception as e:
            print(f"Error extracting basic CIF data: {e}")
            return None
    
    def extract_elements_from_formula(self, formula: str) -> Dict[str, int]:
        """Extract elements and their counts from chemical formula"""
        if not formula or formula == 'Unknown':
            return {}
        
        elements = {}
        try:
            # Simple regex-based extraction (could be improved)
            import re
            # Match element symbols (capital letter followed by optional lowercase)
            # followed by optional numbers
            pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
            matches = re.findall(pattern, formula)
            
            for element, count_str in matches:
                if count_str == '':
                    count = 1
                else:
                    try:
                        count = float(count_str)
                        if count.is_integer():
                            count = int(count)
                    except ValueError:
                        count = 1
                elements[element] = count
                
        except Exception as e:
            print(f"Error extracting elements from formula '{formula}': {e}")
        
        return elements
    
    def add_cif_file(self, cif_file_path: str) -> int:
        """Add a single CIF file to the database"""
        try:
            with open(cif_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                cif_content = f.read()
            
            return self.add_cif_content(cif_content)
            
        except Exception as e:
            print(f"Error reading CIF file {cif_file_path}: {e}")
            return 0
    
    def add_cif_content(self, cif_content: str, debug_target_minerals=None) -> int:
        """Add CIF content to the database with enhanced debugging"""
        if debug_target_minerals is None:
            debug_target_minerals = ['epsomite', 'hexahydrite', 'magnesium', 'sulfate']
        
        # Calculate hash to avoid duplicates
        cif_hash = hashlib.md5(cif_content.encode()).hexdigest()
        
        # Check if this CIF contains target minerals (for debugging)
        content_lower = cif_content.lower()
        contains_target = any(target.lower() in content_lower for target in debug_target_minerals)
        
        if contains_target:
            print(f"\n🎯 FOUND TARGET MINERAL CIF! Hash: {cif_hash[:8]}")
            for target in debug_target_minerals:
                if target.lower() in content_lower:
                    print(f"   - Contains '{target}'")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if already exists
        cursor.execute('SELECT id FROM minerals WHERE cif_hash = ?', (cif_hash,))
        if cursor.fetchone():
            conn.close()
            if contains_target:
                print(f"   ❌ Already exists in database")
            return 0  # Already exists
        
        # Parse CIF content with debugging
        if contains_target:
            print(f"   🔍 Attempting to parse CIF content...")
        
        parsed_data = self.parse_cif_with_pymatgen(cif_content, debug=contains_target)
        
        if contains_target:
            print(f"   📊 Parsing result: {len(parsed_data)} structures found")
            for i, data in enumerate(parsed_data):
                print(f"      Structure {i+1}: {data.get('mineral_name', 'Unknown')} - {data.get('chemical_formula', 'Unknown')}")
        
        added_count = 0
        for data in parsed_data:
            try:
                # Debug target minerals
                mineral_name = data.get('mineral_name', 'Unknown')
                formula = data.get('chemical_formula', 'Unknown')
                
                is_target = any(target.lower() in mineral_name.lower() or 
                              target.lower() in formula.lower() 
                              for target in debug_target_minerals)
                
                if is_target:
                    print(f"\n   ✅ PROCESSING TARGET MINERAL:")
                    print(f"      Name: {mineral_name}")
                    print(f"      Formula: {formula}")
                    print(f"      Space Group: {data.get('space_group', 'Unknown')}")
                    print(f"      AMCSD ID: {data.get('amcsd_id', 'Unknown')}")
                    print(f"      Unit Cell: a={data.get('cell_a')}, b={data.get('cell_b')}, c={data.get('cell_c')}")
                
                # Insert mineral data
                cursor.execute('''
                    INSERT INTO minerals (
                        mineral_name, chemical_formula, space_group, crystal_system,
                        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma,
                        cell_volume, density, amcsd_id, authors, journal, year, doi,
                        cif_content, cif_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['mineral_name'], data['chemical_formula'], data['space_group'],
                    data['crystal_system'], data['cell_a'], data['cell_b'], data['cell_c'],
                    data['cell_alpha'], data['cell_beta'], data['cell_gamma'],
                    data['cell_volume'], data['density'], data['amcsd_id'],
                    data['authors'], data['journal'], data['year'], data['doi'],
                    cif_content, cif_hash
                ))
                
                mineral_id = cursor.lastrowid
                
                # Insert element data
                for element, count in data['elements'].items():
                    cursor.execute('''
                        INSERT INTO mineral_elements (mineral_id, element, count)
                        VALUES (?, ?, ?)
                    ''', (mineral_id, element, count))
                
                added_count += 1
                
                if is_target:
                    print(f"      ✅ Successfully added to database with ID: {mineral_id}")
                
            except sqlite3.IntegrityError as e:
                if contains_target:
                    print(f"      ❌ Duplicate entry for mineral {data['mineral_name']}: {e}")
                continue
            except Exception as e:
                if contains_target:
                    print(f"      ❌ Error adding mineral {data['mineral_name']}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        if contains_target:
            print(f"   📈 Final result: {added_count} minerals added from this CIF block")
        
        return added_count
    
    def search_by_mineral_name(self, mineral_name: str, limit: int = 100) -> List[Dict]:
        """Search minerals by name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM minerals 
            WHERE mineral_name LIKE ? 
            ORDER BY mineral_name 
            LIMIT ?
        ''', (f'%{mineral_name}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append(self._row_to_dict(cursor, row))
        
        conn.close()
        return results
    
    def search_by_elements(self, elements: List[str], exact_match: bool = False, limit: int = 100) -> List[Dict]:
        """Search minerals by chemical elements"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if exact_match:
            # Find minerals that contain exactly these elements
            placeholders = ','.join(['?' for _ in elements])
            cursor.execute(f'''
                SELECT m.* FROM minerals m
                WHERE m.id IN (
                    SELECT mineral_id FROM mineral_elements 
                    WHERE element IN ({placeholders})
                    GROUP BY mineral_id 
                    HAVING COUNT(DISTINCT element) = ?
                    AND mineral_id NOT IN (
                        SELECT mineral_id FROM mineral_elements 
                        WHERE element NOT IN ({placeholders})
                    )
                )
                ORDER BY m.mineral_name
                LIMIT ?
            ''', elements + [len(elements)] + elements + [limit])
        else:
            # Find minerals that contain any of these elements
            placeholders = ','.join(['?' for _ in elements])
            cursor.execute(f'''
                SELECT DISTINCT m.* FROM minerals m
                JOIN mineral_elements me ON m.id = me.mineral_id
                WHERE me.element IN ({placeholders})
                ORDER BY m.mineral_name
                LIMIT ?
            ''', elements + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append(self._row_to_dict(cursor, row))
        
        conn.close()
        return results
    
    def search_by_formula(self, formula: str, limit: int = 100) -> List[Dict]:
        """Search minerals by chemical formula"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM minerals 
            WHERE chemical_formula LIKE ? 
            ORDER BY mineral_name 
            LIMIT ?
        ''', (f'%{formula}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append(self._row_to_dict(cursor, row))
        
        conn.close()
        return results
    
    def get_mineral_by_id(self, mineral_id: int) -> Optional[Dict]:
        """Get mineral by database ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM minerals WHERE id = ?', (mineral_id,))
        row = cursor.fetchone()
        
        if row:
            result = self._row_to_dict(cursor, row)
            conn.close()
            return result
        
        conn.close()
        return None
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM minerals')
        total_minerals = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT element) FROM mineral_elements')
        unique_elements = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT space_group) FROM minerals WHERE space_group != "Unknown"')
        unique_space_groups = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_minerals': total_minerals,
            'unique_elements': unique_elements,
            'unique_space_groups': unique_space_groups,
            'database_path': self.db_path
        }
    
    def _row_to_dict(self, cursor, row) -> Dict:
        """Convert SQLite row to dictionary"""
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))
    
    def bulk_import_amcsd_cif(self, cif_file_path: str, progress_callback=None) -> int:
        """Import the complete AMCSD CIF file (contains multiple structures)"""
        print(f"Starting bulk import of AMCSD CIF file: {cif_file_path}")
        
        try:
            with open(cif_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Pre-scan for target minerals
            target_minerals = ['epsomite', 'hexahydrite', 'magnesium', 'sulfate']
            print(f"\n🔍 Pre-scanning for target minerals: {', '.join(target_minerals)}")
            
            content_lower = content.lower()
            found_targets = []
            for target in target_minerals:
                if target.lower() in content_lower:
                    count = content_lower.count(target.lower())
                    found_targets.append(f"{target} ({count} occurrences)")
            
            if found_targets:
                print(f"✅ Found target minerals in file: {', '.join(found_targets)}")
            else:
                print("❌ No target minerals found in file content")
            
            # Split the file into individual CIF blocks
            cif_blocks = self._split_cif_blocks(content)
            print(f"Found {len(cif_blocks)} CIF blocks to process")
            
            total_added = 0
            pymatgen_success = 0
            fallback_success = 0
            total_failures = 0
            
            for i, cif_block in enumerate(cif_blocks):
                try:
                    # Track which parsing method succeeded
                    initial_count = self.get_database_stats()['total_minerals']
                    added = self.add_cif_content(cif_block)
                    
                    if added > 0:
                        total_added += added
                        # Check if it was pymatgen or fallback (rough heuristic)
                        if "Attempting basic CIF parsing fallback" in cif_block:
                            fallback_success += 1
                        else:
                            pymatgen_success += 1
                    else:
                        total_failures += 1
                    
                    if progress_callback and i % 100 == 0:
                        progress = int((i / len(cif_blocks)) * 100)
                        progress_callback(progress)
                        
                        # Print progress stats
                        if i > 0:
                            success_rate = (total_added / (i + 1)) * 100
                            print(f"Progress: {i+1}/{len(cif_blocks)} blocks processed, "
                                  f"{total_added} minerals added ({success_rate:.1f}% success rate)")
                        
                except Exception as e:
                    print(f"Error processing CIF block {i}: {e}")
                    total_failures += 1
                    continue
            
            # Final statistics
            success_rate = (total_added / len(cif_blocks)) * 100
            print(f"\n=== Bulk Import Statistics ===")
            print(f"Total CIF blocks processed: {len(cif_blocks)}")
            print(f"Successfully added minerals: {total_added}")
            print(f"Failed to process: {total_failures}")
            print(f"Overall success rate: {success_rate:.1f}%")
            print(f"Pymatgen successes: {pymatgen_success}")
            print(f"Fallback parser successes: {fallback_success}")
            print(f"===============================")
            
            return total_added
            
        except Exception as e:
            print(f"Error in bulk import: {e}")
            return 0
    
    def _split_cif_blocks(self, content: str) -> List[str]:
        """Split multi-block CIF file into individual CIF blocks"""
        blocks = []
        current_block = []
        
        lines = content.split('\n')
        in_block = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('data_'):
                # Start of new block
                if in_block and current_block:
                    # Save previous block
                    blocks.append('\n'.join(current_block))
                
                # Start new block
                current_block = [line]
                in_block = True
            elif in_block:
                current_block.append(line)
        
        # Add the last block
        if in_block and current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    def calculate_and_store_diffraction_pattern(self, mineral_id: int, wavelength: float = 1.5406, 
                                              max_two_theta: float = 90.0, min_d_spacing: float = 0.5) -> bool:
        """
        Calculate and store diffraction pattern for a mineral
        
        Args:
            mineral_id: Database ID of the mineral
            wavelength: X-ray wavelength in Angstroms
            max_two_theta: Maximum 2θ angle to calculate
            min_d_spacing: Minimum d-spacing to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get mineral data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT mineral_name, cif_content FROM minerals WHERE id = ?
            ''', (mineral_id,))
            
            result = cursor.fetchone()
            if not result:
                print(f"Mineral with ID {mineral_id} not found")
                conn.close()
                return False
            
            mineral_name, cif_content = result
            
            # Check if pattern already exists
            cursor.execute('''
                SELECT id FROM diffraction_patterns 
                WHERE mineral_id = ? AND wavelength = ? AND max_two_theta = ? AND min_d_spacing = ?
            ''', (mineral_id, wavelength, max_two_theta, min_d_spacing))
            
            if cursor.fetchone():
                print(f"Diffraction pattern already exists for {mineral_name} (λ={wavelength}Å)")
                conn.close()
                return True
            
            # Calculate diffraction pattern with appropriate limits for wavelength
            cif_parser = CIFParser()
            
            # Adjust limits based on wavelength to prevent excessive calculations
            if wavelength < 0.5:  # Very short wavelengths like 0.2401Å
                adjusted_max_2theta = min(max_two_theta, 30.0)  # Limit to 30° for short wavelengths
                adjusted_min_d = max(min_d_spacing, 1.0)  # Increase minimum d-spacing
            elif wavelength < 1.0:  # Short wavelengths like 0.7107Å
                adjusted_max_2theta = min(max_two_theta, 60.0)
                adjusted_min_d = max(min_d_spacing, 0.8)
            else:  # Normal wavelengths
                adjusted_max_2theta = max_two_theta
                adjusted_min_d = min_d_spacing
            
            print(f"Calculating with limits: max_2θ={adjusted_max_2theta}°, min_d={adjusted_min_d}Å")
            
            # Add timeout for calculation to prevent hanging
            start_time = time.time()
            try:
                pattern = cif_parser.calculate_xrd_pattern_from_cif(
                    cif_content, wavelength, max_2theta=adjusted_max_2theta, min_d=adjusted_min_d
                )
                calc_time = time.time() - start_time
                
                # Skip if calculation took too long or generated too many peaks
                if calc_time > 30.0:  # 30 second timeout
                    print(f"⚠️ Calculation took too long ({calc_time:.1f}s), skipping {mineral_name}")
                    conn.close()
                    return False
                
                if len(pattern.get('two_theta', [])) > 10000:  # Limit peak count
                    print(f"⚠️ Too many peaks ({len(pattern['two_theta'])}), skipping {mineral_name}")
                    conn.close()
                    return False
                    
            except Exception as calc_error:
                calc_time = time.time() - start_time
                print(f"❌ Calculation failed for {mineral_name} after {calc_time:.1f}s: {calc_error}")
                conn.close()
                return False
            
            if len(pattern.get('two_theta', [])) == 0:
                print(f"Failed to calculate diffraction pattern for {mineral_name}")
                conn.close()
                return False
            
            # Store pattern in database
            cursor.execute('''
                INSERT INTO diffraction_patterns 
                (mineral_id, wavelength, two_theta, intensities, d_spacings, 
                 max_two_theta, min_d_spacing, calculation_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mineral_id,
                wavelength,
                json.dumps(pattern['two_theta'].tolist()),
                json.dumps(pattern['intensity'].tolist()),
                json.dumps(pattern['d_spacing'].tolist()),
                max_two_theta,
                min_d_spacing,
                'pymatgen'
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Calculated and stored diffraction pattern for {mineral_name} ({len(pattern['two_theta'])} peaks)")
            return True
            
        except Exception as e:
            print(f"❌ Error calculating diffraction pattern for mineral ID {mineral_id}: {e}")
            if 'conn' in locals():
                conn.close()
            return False
    
    def get_diffraction_pattern(self, mineral_id: int, wavelength: float = 1.5406, 
                              max_two_theta: float = 90.0, min_d_spacing: float = 0.5) -> Optional[Dict]:
        """
        Retrieve pre-calculated diffraction pattern from database and convert to target wavelength
        
        Args:
            mineral_id: Database ID of the mineral
            wavelength: Target X-ray wavelength in Angstroms
            max_two_theta: Maximum 2θ angle (not used for retrieval, kept for compatibility)
            min_d_spacing: Minimum d-spacing (not used for retrieval, kept for compatibility)
            
        Returns:
            Dictionary with 'two_theta', 'intensity', 'd_spacing' arrays converted to target wavelength or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Always look for Cu Kα pattern (our reference wavelength)
            reference_wavelength = 1.5406
            cursor.execute('''
                SELECT two_theta, intensities, d_spacings FROM diffraction_patterns
                WHERE mineral_id = ? AND wavelength = ?
                ORDER BY max_two_theta DESC, min_d_spacing ASC
                LIMIT 1
            ''', (mineral_id, reference_wavelength))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                two_theta_json, intensities_json, d_spacings_json = result
                
                # Load the reference data
                ref_two_theta = np.array(json.loads(two_theta_json))
                intensities = np.array(json.loads(intensities_json))
                d_spacings = np.array(json.loads(d_spacings_json))
                
                # Convert to target wavelength using Bragg's law if needed
                if abs(wavelength - reference_wavelength) > 1e-6:  # Different wavelength
                    # Calculate new 2θ values: λ = 2d sin(θ) → θ = arcsin(λ / 2d)
                    new_two_theta = []
                    valid_intensities = []
                    valid_d_spacings = []
                    
                    for d, intensity in zip(d_spacings, intensities):
                        if d > 0:
                            sin_theta = wavelength / (2 * d)
                            if sin_theta <= 1.0:  # Valid reflection
                                theta_rad = np.arcsin(sin_theta)
                                two_theta_deg = 2 * np.degrees(theta_rad)
                                
                                # Apply reasonable 2θ limits based on wavelength
                                min_2theta = 1.0 if wavelength < 1.0 else 5.0
                                max_2theta_limit = 90.0 if wavelength > 1.0 else 60.0
                                
                                if min_2theta <= two_theta_deg <= max_2theta_limit:
                                    new_two_theta.append(two_theta_deg)
                                    valid_intensities.append(intensity)
                                    valid_d_spacings.append(d)
                    
                    print(f"Converted pattern from λ={reference_wavelength:.4f}Å to λ={wavelength:.4f}Å: {len(valid_d_spacings)} peaks")
                    
                    return {
                        'two_theta': np.array(new_two_theta),
                        'intensity': np.array(valid_intensities),
                        'd_spacing': np.array(valid_d_spacings)
                    }
                else:
                    # Same wavelength, return as-is
                    return {
                        'two_theta': ref_two_theta,
                        'intensity': intensities,
                        'd_spacing': d_spacings
                    }
            
            return None
            
        except Exception as e:
            print(f"Error retrieving diffraction pattern: {e}")
            return None
    
    def bulk_calculate_diffraction_patterns(self, wavelengths: List[float] = None, 
                                          progress_callback=None) -> int:
        """
        Calculate diffraction patterns for all minerals in the database
        
        Args:
            wavelengths: List of wavelengths to calculate (default: common XRD wavelengths)
            progress_callback: Function to call with progress updates
            
        Returns:
            Number of patterns successfully calculated
        """
        if wavelengths is None:
            # Only calculate for Cu Kα - d-spacings are wavelength independent
            # We'll convert to other wavelengths using Bragg's law during matching
            wavelengths = [
                1.5406,  # Cu Kα1 - standard reference wavelength
            ]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all mineral IDs and names
            cursor.execute('SELECT id, mineral_name FROM minerals ORDER BY id')
            minerals = cursor.fetchall()
            conn.close()
            
            total_calculations = len(minerals) * len(wavelengths)
            completed = 0
            successful = 0
            
            print(f"Starting bulk diffraction pattern calculation for {len(minerals)} minerals...")
            print(f"Using Cu Kα reference wavelength (λ=1.5406Å) - other wavelengths calculated on-demand using Bragg's law")
            print(f"Total calculations: {total_calculations}")
            print(f"Tip: Use 'Calculate Common Minerals Only' for much faster setup (~1000 minerals vs {len(minerals)})")
            
            for mineral_id, mineral_name in minerals:
                print(f"\nProcessing mineral: {mineral_name} (ID: {mineral_id})")
                
                for i, wavelength in enumerate(wavelengths):
                    print(f"  Wavelength {i+1}/{len(wavelengths)}: {wavelength}Å")
                    
                    try:
                        if self.calculate_and_store_diffraction_pattern(mineral_id, wavelength):
                            successful += 1
                        else:
                            print(f"  ❌ Failed for {mineral_name} at λ={wavelength}Å")
                    except KeyboardInterrupt:
                        print(f"\n⚠️ Calculation interrupted by user at mineral {mineral_name}")
                        print(f"Progress: {completed}/{total_calculations} ({completed/total_calculations*100:.1f}%) - {successful} successful")
                        return successful
                    except Exception as e:
                        print(f"  ❌ Error for {mineral_name} at λ={wavelength}Å: {e}")
                    
                    completed += 1
                    
                    if progress_callback:
                        progress = int((completed / total_calculations) * 100)
                        progress_callback(progress)
                    
                    # Progress reporting every 25 calculations for better feedback
                    if completed % 25 == 0:
                        success_rate = (successful / completed) * 100
                        print(f"\nProgress: {completed}/{total_calculations} ({completed/total_calculations*100:.1f}%) - {successful} successful ({success_rate:.1f}% success rate)")
            
            print(f"\n=== Diffraction Pattern Calculation Complete ===")
            print(f"Total calculations attempted: {total_calculations}")
            print(f"Successful calculations: {successful}")
            print(f"Success rate: {successful/total_calculations*100:.1f}%")
            print(f"===============================================")
            
            return successful
            
        except Exception as e:
            print(f"Error in bulk diffraction pattern calculation: {e}")
            return 0
    
    def get_diffraction_statistics(self) -> Dict:
        """Get statistics about stored diffraction patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total patterns
            cursor.execute('SELECT COUNT(*) FROM diffraction_patterns')
            total_patterns = cursor.fetchone()[0]
            
            # Patterns by wavelength
            cursor.execute('''
                SELECT wavelength, COUNT(*) FROM diffraction_patterns 
                GROUP BY wavelength ORDER BY wavelength
            ''')
            by_wavelength = dict(cursor.fetchall())
            
            # Minerals with patterns
            cursor.execute('SELECT COUNT(DISTINCT mineral_id) FROM diffraction_patterns')
            minerals_with_patterns = cursor.fetchone()[0]
            
            # Total minerals
            cursor.execute('SELECT COUNT(*) FROM minerals')
            total_minerals = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_patterns': total_patterns,
                'patterns_by_wavelength': by_wavelength,
                'minerals_with_patterns': minerals_with_patterns,
                'total_minerals': total_minerals,
                'coverage_percentage': (minerals_with_patterns / total_minerals * 100) if total_minerals > 0 else 0
            }
            
        except Exception as e:
            print(f"Error getting diffraction statistics: {e}")
            return {}
    
    def recalculate_all_diffraction_patterns(self, progress_callback=None) -> int:
        """
        Recalculate all existing diffraction patterns with improved intensity calculations
        
        Args:
            progress_callback: Function to call with progress updates
            
        Returns:
            Number of patterns successfully recalculated
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all existing patterns
            cursor.execute('''
                SELECT dp.id, dp.mineral_id, dp.wavelength, dp.max_two_theta, dp.min_d_spacing, 
                       m.mineral_name, m.cif_content
                FROM diffraction_patterns dp
                JOIN minerals m ON dp.mineral_id = m.id
                ORDER BY dp.mineral_id
            ''')
            
            existing_patterns = cursor.fetchall()
            conn.close()
            
            if not existing_patterns:
                print("No existing diffraction patterns found to recalculate")
                return 0
            
            print(f"🔄 Recalculating {len(existing_patterns)} diffraction patterns with improved intensity calculations...")
            print("This will replace existing patterns with more accurate intensity values.")
            
            successful = 0
            failed = 0
            
            for i, (pattern_id, mineral_id, wavelength, max_2theta, min_d, mineral_name, cif_content) in enumerate(existing_patterns):
                try:
                    print(f"\n[{i+1}/{len(existing_patterns)}] Recalculating: {mineral_name} (ID: {mineral_id})")
                    
                    # Calculate new pattern with improved method
                    cif_parser = CIFParser()
                    new_pattern = cif_parser.calculate_xrd_pattern_from_cif(
                        cif_content, wavelength, max_2theta=max_2theta, min_d=min_d
                    )
                    
                    if len(new_pattern.get('two_theta', [])) == 0:
                        print(f"  ❌ Failed to calculate new pattern for {mineral_name}")
                        failed += 1
                        continue
                    
                    # Update the existing pattern in database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        UPDATE diffraction_patterns 
                        SET two_theta = ?, intensities = ?, d_spacings = ?, 
                            calculation_method = 'pymatgen_improved'
                        WHERE id = ?
                    ''', (
                        json.dumps(new_pattern['two_theta'].tolist()),
                        json.dumps(new_pattern['intensity'].tolist()),
                        json.dumps(new_pattern['d_spacing'].tolist()),
                        pattern_id
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                    print(f"  ✅ Updated pattern: {len(new_pattern['two_theta'])} peaks")
                    successful += 1
                    
                except Exception as e:
                    print(f"  ❌ Error recalculating pattern for {mineral_name}: {e}")
                    failed += 1
                    continue
                
                # Progress callback
                if progress_callback:
                    progress = int(((i + 1) / len(existing_patterns)) * 100)
                    progress_callback(progress)
                
                # Progress reporting every 10 patterns
                if (i + 1) % 10 == 0:
                    success_rate = (successful / (i + 1)) * 100
                    print(f"\n📊 Progress: {i+1}/{len(existing_patterns)} ({success_rate:.1f}% success rate)")
            
            print(f"\n=== Diffraction Pattern Recalculation Complete ===")
            print(f"Total patterns processed: {len(existing_patterns)}")
            print(f"Successfully recalculated: {successful}")
            print(f"Failed: {failed}")
            print(f"Success rate: {successful/len(existing_patterns)*100:.1f}%")
            print(f"================================================")
            
            return successful
            
        except Exception as e:
            print(f"Error in bulk diffraction pattern recalculation: {e}")
            return 0
    
    def validate_pattern_intensities(self, sample_size: int = 10) -> Dict:
        """
        Validate intensity calculations by comparing a sample of patterns
        
        Args:
            sample_size: Number of random patterns to validate
            
        Returns:
            Dictionary with validation statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get random sample of patterns
            cursor.execute('''
                SELECT dp.mineral_id, dp.wavelength, m.mineral_name, m.cif_content,
                       dp.two_theta, dp.intensities, dp.calculation_method
                FROM diffraction_patterns dp
                JOIN minerals m ON dp.mineral_id = m.id
                ORDER BY RANDOM()
                LIMIT ?
            ''', (sample_size,))
            
            patterns = cursor.fetchall()
            conn.close()
            
            if not patterns:
                return {'error': 'No patterns found for validation'}
            
            print(f"🔍 Validating intensity calculations for {len(patterns)} random patterns...")
            
            validation_results = {
                'total_validated': 0,
                'improved_patterns': 0,
                'similar_patterns': 0,
                'failed_validations': 0,
                'average_improvement': 0.0,
                'details': []
            }
            
            total_improvement = 0.0
            
            for mineral_id, wavelength, mineral_name, cif_content, old_2theta_json, old_intensities_json, calc_method in patterns:
                try:
                    print(f"\n📋 Validating: {mineral_name}")
                    
                    # Parse old pattern
                    old_intensities = np.array(json.loads(old_intensities_json))
                    
                    # Calculate new pattern
                    cif_parser = CIFParser()
                    new_pattern = cif_parser.calculate_xrd_pattern_from_cif(cif_content, wavelength)
                    
                    if len(new_pattern.get('intensity', [])) == 0:
                        print(f"  ❌ Failed to calculate new pattern")
                        validation_results['failed_validations'] += 1
                        continue
                    
                    new_intensities = new_pattern['intensity']
                    
                    # Compare intensity distributions
                    old_max = np.max(old_intensities) if len(old_intensities) > 0 else 1.0
                    new_max = np.max(new_intensities) if len(new_intensities) > 0 else 1.0
                    
                    # Normalize both to 100 for comparison
                    old_norm = (old_intensities / old_max) * 100 if old_max > 0 else old_intensities
                    new_norm = (new_intensities / new_max) * 100 if new_max > 0 else new_intensities
                    
                    # Calculate improvement metrics
                    intensity_range_old = np.max(old_norm) - np.min(old_norm) if len(old_norm) > 0 else 0
                    intensity_range_new = np.max(new_norm) - np.min(new_norm) if len(new_norm) > 0 else 0
                    
                    improvement = (intensity_range_new - intensity_range_old) / max(intensity_range_old, 1.0) * 100
                    total_improvement += improvement
                    
                    detail = {
                        'mineral_name': mineral_name,
                        'old_method': calc_method,
                        'old_peaks': len(old_intensities),
                        'new_peaks': len(new_intensities),
                        'intensity_improvement': improvement,
                        'old_range': intensity_range_old,
                        'new_range': intensity_range_new
                    }
                    
                    validation_results['details'].append(detail)
                    
                    if abs(improvement) > 10:  # Significant improvement
                        validation_results['improved_patterns'] += 1
                        print(f"  ✅ Significant improvement: {improvement:.1f}% better intensity range")
                    else:
                        validation_results['similar_patterns'] += 1
                        print(f"  ➖ Similar quality: {improvement:.1f}% change")
                    
                    validation_results['total_validated'] += 1
                    
                except Exception as e:
                    print(f"  ❌ Validation error: {e}")
                    validation_results['failed_validations'] += 1
                    continue
            
            if validation_results['total_validated'] > 0:
                validation_results['average_improvement'] = total_improvement / validation_results['total_validated']
            
            print(f"\n=== Intensity Validation Results ===")
            print(f"Patterns validated: {validation_results['total_validated']}")
            print(f"Significantly improved: {validation_results['improved_patterns']}")
            print(f"Similar quality: {validation_results['similar_patterns']}")
            print(f"Failed validations: {validation_results['failed_validations']}")
            print(f"Average improvement: {validation_results['average_improvement']:.1f}%")
            print(f"===================================")
            
            return validation_results
            
        except Exception as e:
            print(f"Error in pattern validation: {e}")
            return {'error': str(e)}
    
    def cleanup_non_cu_patterns(self) -> int:
        """
        Remove all diffraction patterns that are not Cu Kα (1.5406Å) wavelength
        This optimizes the database to use only the reference wavelength
        
        Returns:
            Number of patterns removed
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First, get count of non-Cu patterns
            cu_wavelength = 1.5406
            cursor.execute('''
                SELECT COUNT(*) FROM diffraction_patterns 
                WHERE ABS(wavelength - ?) > 0.0001
            ''', (cu_wavelength,))
            
            non_cu_count = cursor.fetchone()[0]
            
            if non_cu_count == 0:
                print("✅ No non-Cu Kα patterns found. Database is already optimized.")
                conn.close()
                return 0
            
            print(f"🧹 Found {non_cu_count} non-Cu Kα patterns to remove...")
            
            # Get details of patterns to be removed
            cursor.execute('''
                SELECT wavelength, COUNT(*) as count 
                FROM diffraction_patterns 
                WHERE ABS(wavelength - ?) > 0.0001
                GROUP BY wavelength
                ORDER BY wavelength
            ''', (cu_wavelength,))
            
            wavelength_counts = cursor.fetchall()
            
            print("Patterns to be removed:")
            for wavelength, count in wavelength_counts:
                print(f"  λ = {wavelength:.4f} Å: {count} patterns")
            
            # Remove non-Cu patterns
            cursor.execute('''
                DELETE FROM diffraction_patterns 
                WHERE ABS(wavelength - ?) > 0.0001
            ''', (cu_wavelength,))
            
            removed_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"✅ Successfully removed {removed_count} non-Cu Kα patterns")
            print(f"💡 Database now optimized for Cu Kα reference wavelength only")
            print(f"   Other wavelengths will be calculated on-demand using Bragg's law")
            
            return removed_count
            
        except Exception as e:
            print(f"❌ Error cleaning up non-Cu patterns: {e}")
            return 0
    
    def get_wavelength_distribution(self) -> Dict:
        """
        Get distribution of wavelengths in the diffraction patterns table
        
        Returns:
            Dictionary with wavelength statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT wavelength, COUNT(*) as count 
                FROM diffraction_patterns 
                GROUP BY wavelength
                ORDER BY wavelength
            ''')
            
            wavelength_data = cursor.fetchall()
            conn.close()
            
            distribution = {}
            total_patterns = 0
            
            for wavelength, count in wavelength_data:
                distribution[wavelength] = count
                total_patterns += count
            
            return {
                'wavelength_distribution': distribution,
                'total_patterns': total_patterns,
                'unique_wavelengths': len(distribution)
            }
            
        except Exception as e:
            print(f"Error getting wavelength distribution: {e}")
            return {}
