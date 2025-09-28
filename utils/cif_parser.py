"""
CIF file parser for crystal structure data
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import requests
from scipy.signal import find_peaks

class CIFParser:
    """Parser for Crystallographic Information Files (CIF)"""
    
    def __init__(self):
        self.data = {}
        
    def parse_file(self, file_path: str) -> Dict:
        """Parse a CIF file and extract crystal structure information"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        return self.parse_content(content)
        
    def parse_content(self, content: str) -> Dict:
        """Parse CIF content string"""
        self.data = {}
        lines = content.split('\n')
        
        # Parse data blocks
        current_block = None
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
                
            # Data block header
            if line.startswith('data_'):
                current_block = line[5:]
                self.data[current_block] = {}
                i += 1
                continue
                
            # Parse key-value pairs and loops
            if current_block:
                i = self.parse_block_content(lines, i, current_block)
            else:
                i += 1
                
        return self.data
        
    def parse_block_content(self, lines: List[str], start_idx: int, block_name: str) -> int:
        """Parse content within a data block"""
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
                
            # New data block
            if line.startswith('data_'):
                return i
                
            # Loop structure
            if line.startswith('loop_'):
                i = self.parse_loop(lines, i + 1, block_name)
                continue
                
            # Key-value pair
            if line.startswith('_'):
                i = self.parse_key_value(lines, i, block_name)
                continue
                
            i += 1
            
        return i
        
    def parse_key_value(self, lines: List[str], start_idx: int, block_name: str) -> int:
        """Parse a key-value pair"""
        line = lines[start_idx].strip()
        
        # Split key and value
        parts = line.split(None, 1)
        if len(parts) < 2:
            return start_idx + 1
            
        key = parts[0]
        value = parts[1]
        
        # Handle multiline values
        if value.startswith(';'):
            # Multiline text value
            value_lines = [value[1:]]
            i = start_idx + 1
            
            while i < len(lines):
                line = lines[i]
                if line.strip() == ';':
                    break
                value_lines.append(line)
                i += 1
                
            value = '\n'.join(value_lines)
            return i + 1
        else:
            # Single line value
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
                
            self.data[block_name][key] = value
            return start_idx + 1
            
    def parse_loop(self, lines: List[str], start_idx: int, block_name: str) -> int:
        """Parse a loop structure"""
        i = start_idx
        
        # Parse column headers
        headers = []
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            if line.startswith('_'):
                headers.append(line)
                i += 1
            else:
                break
                
        if not headers:
            return i
            
        # Parse data rows
        data_rows = []
        while i < len(lines):
            line = lines[i].strip()
            
            # End of loop
            if not line or line.startswith('#') or line.startswith('_') or line.startswith('loop_') or line.startswith('data_'):
                break
                
            # Parse row data
            row_data = self.parse_data_row(line)
            if row_data:
                data_rows.append(row_data)
                
            i += 1
            
        # Store loop data
        loop_name = f"loop_{len([k for k in self.data[block_name].keys() if k.startswith('loop_')])}"
        self.data[block_name][loop_name] = {
            'headers': headers,
            'data': data_rows
        }
        
        return i
        
    def parse_data_row(self, line: str) -> List[str]:
        """Parse a data row, handling quoted values"""
        values = []
        current_value = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_quotes:
                if char in ['"', "'"]:
                    in_quotes = True
                    quote_char = char
                elif char.isspace():
                    if current_value:
                        values.append(current_value)
                        current_value = ""
                else:
                    current_value += char
            else:
                if char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current_value += char
                    
            i += 1
            
        if current_value:
            values.append(current_value)
            
        return values
        
    def extract_crystal_info(self, block_name: str = None) -> Dict:
        """Extract crystal structure information"""
        if not self.data:
            return {}
            
        # Use first block if none specified
        if block_name is None:
            block_name = list(self.data.keys())[0]
            
        if block_name not in self.data:
            return {}
            
        block_data = self.data[block_name]
        
        crystal_info = {
            'mineral_name': self.get_value(block_data, '_chemical_name_mineral'),
            'chemical_formula': self.get_value(block_data, '_chemical_formula_sum'),
            'space_group': self.get_value(block_data, '_space_group_name_H-M_alt') or 
                          self.get_value(block_data, '_symmetry_space_group_name_H-M'),
            'cell_parameters': {
                'a': self.get_float_value(block_data, '_cell_length_a'),
                'b': self.get_float_value(block_data, '_cell_length_b'),
                'c': self.get_float_value(block_data, '_cell_length_c'),
                'alpha': self.get_float_value(block_data, '_cell_angle_alpha'),
                'beta': self.get_float_value(block_data, '_cell_angle_beta'),
                'gamma': self.get_float_value(block_data, '_cell_angle_gamma'),
                'volume': self.get_float_value(block_data, '_cell_volume')
            },
            'atoms': self.extract_atomic_positions(block_data)
        }
        
        return crystal_info
        
    def get_value(self, data: Dict, key: str) -> Optional[str]:
        """Get a value from CIF data, handling uncertainties"""
        value = data.get(key)
        if value is None:
            return None
            
        # Remove uncertainty notation (e.g., "1.234(5)" -> "1.234")
        value = re.sub(r'\([^)]*\)', '', str(value))
        return value.strip()
        
    def get_float_value(self, data: Dict, key: str) -> Optional[float]:
        """Get a float value from CIF data"""
        value = self.get_value(data, key)
        if value is None:
            return None
            
        try:
            return float(value)
        except ValueError:
            return None
            
    def extract_atomic_positions(self, data: Dict) -> List[Dict]:
        """Extract atomic positions from loop data"""
        atoms = []
        
        # Find atom site loop
        for key, value in data.items():
            if key.startswith('loop_') and isinstance(value, dict):
                headers = value.get('headers', [])
                
                # Check if this is an atom site loop
                if any('_atom_site' in header for header in headers):
                    atoms = self.parse_atom_site_loop(value)
                    break
                    
        return atoms
        
    def parse_atom_site_loop(self, loop_data: Dict) -> List[Dict]:
        """Parse atom site loop data"""
        headers = loop_data['headers']
        data_rows = loop_data['data']
        
        # Map headers to indices
        header_map = {}
        for i, header in enumerate(headers):
            if '_atom_site_label' in header:
                header_map['label'] = i
            elif '_atom_site_type_symbol' in header:
                header_map['symbol'] = i
            elif '_atom_site_fract_x' in header:
                header_map['x'] = i
            elif '_atom_site_fract_y' in header:
                header_map['y'] = i
            elif '_atom_site_fract_z' in header:
                header_map['z'] = i
            elif '_atom_site_occupancy' in header:
                header_map['occupancy'] = i
                
        atoms = []
        for row in data_rows:
            if len(row) >= len(headers):
                atom = {}
                
                if 'label' in header_map:
                    atom['label'] = row[header_map['label']]
                if 'symbol' in header_map:
                    atom['symbol'] = row[header_map['symbol']]
                    
                # Fractional coordinates
                for coord in ['x', 'y', 'z']:
                    if coord in header_map:
                        try:
                            value = re.sub(r'\([^)]*\)', '', row[header_map[coord]])
                            atom[coord] = float(value)
                        except (ValueError, IndexError):
                            atom[coord] = 0.0
                            
                if 'occupancy' in header_map:
                    try:
                        value = re.sub(r'\([^)]*\)', '', row[header_map['occupancy']])
                        atom['occupancy'] = float(value)
                    except (ValueError, IndexError):
                        atom['occupancy'] = 1.0
                        
                atoms.append(atom)
                
        return atoms
        
    def calculate_d_spacings(self, crystal_info: Dict, max_hkl: int = 5) -> List[Tuple[float, Tuple[int, int, int]]]:
        """Calculate d-spacings for given crystal structure"""
        cell_params = crystal_info.get('cell_parameters', {})
        
        a = cell_params.get('a')
        b = cell_params.get('b')
        c = cell_params.get('c')
        alpha = cell_params.get('alpha')
        beta = cell_params.get('beta')
        gamma = cell_params.get('gamma')
        
        if None in [a, b, c, alpha, beta, gamma]:
            return []
            
        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        # Calculate d-spacings for different hkl values
        d_spacings = []
        
        for h in range(-max_hkl, max_hkl + 1):
            for k in range(-max_hkl, max_hkl + 1):
                for l in range(-max_hkl, max_hkl + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                        
                    d = self.calculate_d_spacing_hkl(h, k, l, a, b, c, alpha_rad, beta_rad, gamma_rad)
                    if d > 0.5:  # Only include reasonable d-spacings
                        d_spacings.append((d, (h, k, l)))
                        
        # Sort by d-spacing (descending)
        d_spacings.sort(reverse=True)
        
        return d_spacings
        
    def calculate_d_spacing_hkl(self, h: int, k: int, l: int, a: float, b: float, c: float,
                               alpha: float, beta: float, gamma: float) -> float:
        """Calculate d-spacing for specific hkl indices"""
        # General formula for d-spacing in triclinic system
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_alpha = np.sin(alpha)
        sin_beta = np.sin(beta)
        sin_gamma = np.sin(gamma)
        
        # Volume of unit cell
        V = a * b * c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2*cos_alpha*cos_beta*cos_gamma)
        
        # Reciprocal lattice parameters
        a_star = b * c * sin_alpha / V
        b_star = a * c * sin_beta / V
        c_star = a * b * sin_gamma / V
        
        cos_alpha_star = (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma)
        cos_beta_star = (cos_alpha * cos_gamma - cos_beta) / (sin_alpha * sin_gamma)
        cos_gamma_star = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
        
        # Calculate d-spacing
        d_inv_sq = (h * a_star)**2 + (k * b_star)**2 + (l * c_star)**2 + \
                   2 * h * k * a_star * b_star * cos_gamma_star + \
                   2 * h * l * a_star * c_star * cos_beta_star + \
                   2 * k * l * b_star * c_star * cos_alpha_star
                   
        if d_inv_sq <= 0:
            return 0
            
        return 1.0 / np.sqrt(d_inv_sq)
    
    def calculate_structure_factors(self, crystal_info: Dict, hkl_list: List[Tuple[int, int, int]]) -> List[float]:
        """Calculate structure factors for given hkl reflections"""
        atoms = crystal_info.get('atoms', [])
        if not atoms:
            return [1.0] * len(hkl_list)  # Default intensity if no atomic data
            
        # Atomic scattering factors (simplified - using constant values)
        scattering_factors = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30
        }
        
        structure_factors = []
        
        for h, k, l in hkl_list:
            F_real = 0.0
            F_imag = 0.0
            
            for atom in atoms:
                x = atom.get('x', 0.0)
                y = atom.get('y', 0.0)
                z = atom.get('z', 0.0)
                occupancy = atom.get('occupancy', 1.0)
                symbol = atom.get('symbol', 'C')
                
                # Get atomic scattering factor
                f = scattering_factors.get(symbol, 6)  # Default to carbon
                
                # Calculate phase
                phase = 2 * np.pi * (h * x + k * y + l * z)
                
                # Add to structure factor
                F_real += f * occupancy * np.cos(phase)
                F_imag += f * occupancy * np.sin(phase)
                
            # Calculate intensity (|F|^2)
            intensity = F_real**2 + F_imag**2
            structure_factors.append(intensity)
            
        return structure_factors
    
    def generate_theoretical_pattern(self, crystal_info: Dict, wavelength: float = 1.5406, 
                                   two_theta_range: Tuple[float, float] = (5, 90), 
                                   min_intensity: float = 0.01) -> Dict:
        """Generate theoretical XRD pattern from crystal structure"""
        # Calculate d-spacings
        d_spacings_hkl = self.calculate_d_spacings(crystal_info, max_hkl=8)
        
        if not d_spacings_hkl:
            return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
            
        # Filter by 2theta range
        valid_reflections = []
        valid_hkl = []
        
        for d, hkl in d_spacings_hkl:
            # Calculate 2theta using Bragg's law
            sin_theta = wavelength / (2 * d)
            if sin_theta <= 1.0:  # Valid reflection
                two_theta = 2 * np.degrees(np.arcsin(sin_theta))
                if two_theta_range[0] <= two_theta <= two_theta_range[1]:
                    valid_reflections.append((d, two_theta))
                    valid_hkl.append(hkl)
                    
        if not valid_reflections:
            return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
            
        # Calculate structure factors (intensities)
        structure_factors = self.calculate_structure_factors(crystal_info, valid_hkl)
        
        # Normalize intensities
        if structure_factors:
            max_intensity = max(structure_factors)
            if max_intensity > 0:
                normalized_intensities = [100 * sf / max_intensity for sf in structure_factors]
            else:
                normalized_intensities = structure_factors
        else:
            normalized_intensities = [1.0] * len(valid_reflections)
            
        # Filter by minimum intensity
        filtered_data = [(d, tt, intensity) for (d, tt), intensity in 
                        zip(valid_reflections, normalized_intensities) 
                        if intensity >= min_intensity]
                        
        if not filtered_data:
            return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
            
        # Sort by 2theta
        filtered_data.sort(key=lambda x: x[1])
        
        d_spacings = np.array([item[0] for item in filtered_data])
        two_thetas = np.array([item[1] for item in filtered_data])
        intensities = np.array([item[2] for item in filtered_data])
        
        return {
            'two_theta': two_thetas,
            'intensity': intensities,
            'd_spacing': d_spacings
        }
    
    def fetch_dif_from_amcsd(self, amcsd_id: str) -> Optional[Dict]:
        """Fetch DIF (diffraction) data from AMCSD if available"""
        try:
            # Try the correct DIF file URL format (they use .txt extension)
            # Try both with and without leading zeros
            amcsd_id_no_zeros = amcsd_id.lstrip('0') or '0'  # Remove leading zeros, but keep at least one digit
            
            dif_urls = [
                f"https://rruff.geo.arizona.edu/AMS/xtal_data/DIFfiles/{amcsd_id_no_zeros}.txt",
                f"https://rruff.geo.arizona.edu/AMS/xtal_data/DIFfiles/{amcsd_id}.txt",
                f"http://rruff.geo.arizona.edu/AMS/xtal_data/DIFfiles/{amcsd_id_no_zeros}.txt",
                f"http://rruff.geo.arizona.edu/AMS/xtal_data/DIFfiles/{amcsd_id}.txt"
            ]
            
            for dif_url in dif_urls:
                try:
                    print(f"Trying DIF URL: {dif_url}")
                    response = requests.get(dif_url, timeout=15)
                    
                    if response.status_code == 200 and response.text.strip():
                        # Check if it's actually DIF content (not an error page)
                        content = response.text.strip()
                        # DIF files should contain crystallographic data, not HTML
                        if (not content.lower().startswith('<html') and 
                            not content.lower().startswith('<!doctype') and
                            'error' not in content.lower()[:100] and
                            len(content) > 50):  # DIF files should have substantial content
                            print(f"Successfully fetched DIF data from: {dif_url}")
                            return self.parse_dif_content(content)
                        else:
                            print(f"URL returned HTML/error page: {dif_url}")
                            print(f"Content preview: {content[:100]}...")
                    else:
                        print(f"Failed to fetch from {dif_url}: Status {response.status_code}")
                        
                except requests.RequestException as e:
                    print(f"Request failed for {dif_url}: {e}")
                    continue
            
            print(f"No DIF data found for AMCSD ID: {amcsd_id}")
            return None
                
        except Exception as e:
            print(f"Error fetching DIF data: {e}")
            return None
    
    def parse_dif_content(self, dif_content: str) -> Dict:
        """Parse DIF file content to extract peak positions and intensities"""
        lines = dif_content.strip().split('\n')
        
        two_theta = []
        intensities = []
        d_spacings = []
        
        print(f"Parsing DIF content with {len(lines)} lines")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            
            # Skip header lines that might contain text
            if any(word in line.lower() for word in ['title', 'sample', 'wavelength', 'radiation']):
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Try different DIF formats:
                    # Format 1: 2theta intensity [d-spacing]
                    # Format 2: d-spacing intensity
                    # Format 3: h k l d-spacing intensity
                    
                    # Standard DIF format: 2theta intensity d-spacing [h k l multiplicity]
                    tt = float(parts[0])
                    intensity = float(parts[1])
                    
                    # Get d-spacing from third column if available
                    if len(parts) >= 3 and self._is_number(parts[2]):
                        d = float(parts[2])
                    else:
                        # Calculate d-spacing using Bragg's law with Cu Ka wavelength
                        wavelength = 1.5406
                        if tt > 0.1:  # Avoid very small angles that cause division issues
                            try:
                                d = wavelength / (2 * np.sin(np.radians(tt / 2)))
                            except (ZeroDivisionError, ValueError):
                                continue
                        else:
                            continue
                    
                    # Validate reasonable values
                    if 0 < tt < 180 and intensity > 0 and d > 0:
                        two_theta.append(tt)
                        intensities.append(intensity)
                        d_spacings.append(d)
                    
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Skipping line {i+1}: {line} (Error: {e})")
                    continue
        
        print(f"Successfully parsed {len(two_theta)} peaks from DIF data")
        
        if len(two_theta) == 0:
            print("Warning: No valid peaks found in DIF data")
            
        return {
            'two_theta': np.array(two_theta),
            'intensity': np.array(intensities),
            'd_spacing': np.array(d_spacings)
        }
    
    def _is_number(self, s: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def calculate_xrd_pattern_from_cif(self, cif_content: str, wavelength: float = 1.5406, 
                                     max_2theta: float = 90.0, min_d: float = 0.5) -> Dict:
        """
        Calculate XRD pattern from CIF structure data using pymatgen
        
        Args:
            cif_content: CIF file content as string
            wavelength: X-ray wavelength in Angstroms
            max_2theta: Maximum 2theta angle to calculate
            min_d: Minimum d-spacing to include
            
        Returns:
            Dictionary with 'two_theta', 'intensity', and 'd_spacing' arrays
        """
        try:
            print(f"Calculating XRD pattern from CIF data using pymatgen (λ={wavelength:.4f}Å)")
            
            # Import pymatgen modules
            from pymatgen.io.cif import CifParser
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            # Parse CIF content with pymatgen
            parser = CifParser.from_string(cif_content)
            structures = parser.get_structures()
            
            if not structures:
                print("No structures found in CIF")
                return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
            
            # Use the first structure
            structure = structures[0]
            print(f"Structure: {structure.composition}")
            print(f"Space group: {structure.get_space_group_info()}")
            
            # Create XRD calculator
            calculator = XRDCalculator(wavelength=wavelength)
            
            # Calculate XRD pattern
            pattern = calculator.get_pattern(structure, two_theta_range=(5, max_2theta))
            
            # Extract data
            two_theta_array = pattern.x
            intensity_array = pattern.y
            
            # Calculate d-spacings from 2theta using Bragg's law
            d_spacing_array = wavelength / (2 * np.sin(np.radians(two_theta_array / 2)))
            
            # Filter by minimum d-spacing
            valid_indices = d_spacing_array >= min_d
            two_theta_array = two_theta_array[valid_indices]
            intensity_array = intensity_array[valid_indices]
            d_spacing_array = d_spacing_array[valid_indices]
            
            print(f"Calculated {len(two_theta_array)} reflections using pymatgen XRD calculator")
            
            return {
                'two_theta': two_theta_array,
                'intensity': intensity_array,
                'd_spacing': d_spacing_array
            }
            
        except Exception as e:
            print(f"Error calculating XRD pattern from CIF with pymatgen: {e}")
            # Fallback to simple calculation if pymatgen fails
            return self._calculate_xrd_pattern_simple(cif_content, wavelength, max_2theta, min_d)
    
    def _calculate_xrd_pattern_simple(self, cif_content: str, wavelength: float = 1.5406, 
                                    max_2theta: float = 90.0, min_d: float = 0.5) -> Dict:
        """
        Simple fallback XRD calculation using unit cell parameters only
        """
        try:
            print(f"Using simple XRD calculation fallback (λ={wavelength:.4f}Å)")
            
            # Parse CIF content with our basic parser
            cif_data = self.parse_content(cif_content)
            if not cif_data:
                print("No CIF data found")
                return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
            
            # Get the first data block
            block_name = list(cif_data.keys())[0]
            data = cif_data[block_name]
            
            # Extract unit cell parameters
            try:
                a = float(data.get('_cell_length_a', '0').split('(')[0])
                b = float(data.get('_cell_length_b', '0').split('(')[0])
                c = float(data.get('_cell_length_c', '0').split('(')[0])
                alpha = float(data.get('_cell_angle_alpha', '90').split('(')[0])
                beta = float(data.get('_cell_angle_beta', '90').split('(')[0])
                gamma = float(data.get('_cell_angle_gamma', '90').split('(')[0])
                
                if a <= 0 or b <= 0 or c <= 0:
                    print("Invalid unit cell parameters")
                    return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
                    
                print(f"Unit cell: a={a:.3f}, b={b:.3f}, c={c:.3f}, α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°")
                
            except (ValueError, KeyError) as e:
                print(f"Error parsing unit cell parameters: {e}")
                return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
            
            # Generate h,k,l indices for reflections
            max_h = min(int(wavelength / (2 * min_d * np.sin(np.radians(max_2theta/2))) * a) + 1, 15)
            max_k = min(int(wavelength / (2 * min_d * np.sin(np.radians(max_2theta/2))) * b) + 1, 15)
            max_l = min(int(wavelength / (2 * min_d * np.sin(np.radians(max_2theta/2))) * c) + 1, 15)
            
            print(f"Generating reflections up to h={max_h}, k={max_k}, l={max_l}")
            
            two_theta_list = []
            intensity_list = []
            d_spacing_list = []
            
            # Calculate d-spacings and intensities for each reflection
            for h in range(-max_h, max_h + 1):
                for k in range(-max_k, max_k + 1):
                    for l in range(-max_l, max_l + 1):
                        if h == 0 and k == 0 and l == 0:
                            continue
                        
                        # Calculate d-spacing using the general formula
                        d = self._calculate_d_spacing(h, k, l, a, b, c, alpha, beta, gamma)
                        
                        if d < min_d:
                            continue
                        
                        # Calculate 2theta using Bragg's law
                        sin_theta = wavelength / (2 * d)
                        if sin_theta > 1.0:
                            continue
                        
                        theta = np.arcsin(sin_theta)
                        two_theta = 2 * np.degrees(theta)
                        
                        if two_theta > max_2theta:
                            continue
                        
                        # Simple structure factor approximation
                        intensity = 100.0 * np.exp(-0.5 * (two_theta / 30.0) ** 2)  # Thermal factor
                        intensity *= (1.0 + 0.5 * np.sin(h + k + l))  # Simple modulation
                        
                        if intensity > 1.0:
                            two_theta_list.append(two_theta)
                            intensity_list.append(intensity)
                            d_spacing_list.append(d)
            
            if len(two_theta_list) == 0:
                print("No valid reflections calculated")
                return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
            
            # Sort by 2theta
            sorted_indices = np.argsort(two_theta_list)
            two_theta_array = np.array(two_theta_list)[sorted_indices]
            intensity_array = np.array(intensity_list)[sorted_indices]
            d_spacing_array = np.array(d_spacing_list)[sorted_indices]
            
            print(f"Calculated {len(two_theta_array)} reflections using simple method")
            
            return {
                'two_theta': two_theta_array,
                'intensity': intensity_array,
                'd_spacing': d_spacing_array
            }
            
        except Exception as e:
            print(f"Error in simple XRD calculation: {e}")
            return {'two_theta': np.array([]), 'intensity': np.array([]), 'd_spacing': np.array([])}
    
    def _calculate_d_spacing(self, h: int, k: int, l: int, a: float, b: float, c: float, 
                           alpha: float, beta: float, gamma: float) -> float:
        """
        Calculate d-spacing for given Miller indices and unit cell parameters
        Uses the general formula for any crystal system
        """
        # Convert angles to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        # Calculate volume of unit cell
        V = a * b * c * np.sqrt(1 + 2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad) - 
                               np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2)
        
        # Calculate reciprocal lattice parameters
        a_star = b * c * np.sin(alpha_rad) / V
        b_star = a * c * np.sin(beta_rad) / V
        c_star = a * b * np.sin(gamma_rad) / V
        
        cos_alpha_star = (np.cos(beta_rad)*np.cos(gamma_rad) - np.cos(alpha_rad)) / (np.sin(beta_rad)*np.sin(gamma_rad))
        cos_beta_star = (np.cos(alpha_rad)*np.cos(gamma_rad) - np.cos(beta_rad)) / (np.sin(alpha_rad)*np.sin(gamma_rad))
        cos_gamma_star = (np.cos(alpha_rad)*np.cos(beta_rad) - np.cos(gamma_rad)) / (np.sin(alpha_rad)*np.sin(beta_rad))
        
        # Calculate d-spacing
        d_inv_sq = (h**2 * a_star**2 + k**2 * b_star**2 + l**2 * c_star**2 + 
                   2*h*k*a_star*b_star*cos_gamma_star + 
                   2*h*l*a_star*c_star*cos_beta_star + 
                   2*k*l*b_star*c_star*cos_alpha_star)
        
        if d_inv_sq <= 0:
            return 0.0
        
        return 1.0 / np.sqrt(d_inv_sq)
