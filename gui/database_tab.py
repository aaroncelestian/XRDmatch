"""
Database search tab for AMCSD crystal structure database
"""

import requests
import re
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
                             QGroupBox, QMessageBox, QProgressBar, QTextEdit,
                             QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from bs4 import BeautifulSoup
import pandas as pd

class AMCSDSearchThread(QThread):
    """Thread for searching AMCSD database"""
    
    results_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, search_params):
        super().__init__()
        self.search_params = search_params
        
        # Fallback data for common minerals when AMCSD is unavailable
        self.fallback_minerals = {
            'epsomite': {
                'mineral': 'Epsomite',
                'formula': 'MgSO4·7H2O',
                'space_group': 'P212121',
                'cell_params': {'a': 11.86, 'b': 11.99, 'c': 6.858, 'alpha': 90, 'beta': 90, 'gamma': 90},
                'amcsd_id': '0008280',
                'cif_url': 'https://rruff.geo.arizona.edu/AMS/download.php?id=8280&down=cif',
                'reference': 'AMCSD Database',
                'authors': 'Fortes et al.',
                'journal': 'European Journal of Mineralogy'
            },
            'hexahydrite': {
                'mineral': 'Hexahydrite',
                'formula': 'MgSO4·6H2O',
                'space_group': 'C2/c',
                'cell_params': {'a': 10.1, 'b': 7.21, 'c': 24.41, 'alpha': 90, 'beta': 98.3, 'gamma': 90},
                'amcsd_id': '0010729',
                'cif_url': 'https://rruff.geo.arizona.edu/AMS/download.php?id=10729&down=cif',
                'reference': 'AMCSD Database',
                'authors': 'Zalkin et al.',
                'journal': 'Acta Crystallographica'
            },
            'quartz': {
                'mineral': 'Quartz',
                'formula': 'SiO2',
                'space_group': 'P3221',
                'cell_params': {'a': 4.913, 'b': 4.913, 'c': 5.405, 'alpha': 90, 'beta': 90, 'gamma': 120},
                'amcsd_id': '0011644',
                'cif_url': 'https://rruff.geo.arizona.edu/AMS/download.php?id=11644&down=cif',
                'reference': 'AMCSD Database',
                'authors': 'Antao et al.',
                'journal': 'Physics and Chemistry of Minerals'
            }
        }
        
    def run(self):
        """Run the search in a separate thread"""
        try:
            results = self.search_amcsd()
            self.results_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def search_amcsd(self):
        """Search the AMCSD database with rate limiting and server status check"""
        base_url = "https://rruff.geo.arizona.edu/AMS"
        
        try:
            # Check if AMCSD server is responding
            print("Checking AMCSD server status...")
            try:
                status_response = requests.get(base_url, timeout=10)
                if status_response.status_code != 200:
                    print(f"AMCSD server returned status {status_response.status_code}")
                    return []
                print("AMCSD server is responding")
            except Exception as e:
                print(f"AMCSD server appears to be down or unreachable: {e}")
                print("Attempting to use fallback mineral data...")
                
                # Try fallback search for common minerals
                fallback_results = self.search_fallback_minerals()
                if fallback_results:
                    print(f"Using fallback data for {len(fallback_results)} minerals")
                    return fallback_results
                
                # Emit error signal to show user-friendly message
                self.error_occurred.emit(f"AMCSD server is currently unavailable and no fallback data found. This might be temporary - please try again in a few minutes. Error: {str(e)[:100]}")
                return []
            
            search_type = self.search_params.get('search_type', '').lower()
            if search_type in ['mineral', 'mineral_name']:
                results = self.search_by_mineral(base_url)
            elif search_type in ['chemistry', 'chemical_elements']:
                results = self.search_by_chemistry(base_url)
            elif search_type in ['diffraction', 'diffraction_peaks']:
                results = self.search_by_diffraction(base_url)
            else:
                results = []
            
            # Ensure we always return a list, never None
            return results if results is not None else []
        except Exception as e:
            print(f"Error in search_amcsd: {e}")
            return []
    
    def search_fallback_minerals(self):
        """Search fallback mineral data when AMCSD is unavailable"""
        try:
            search_type = self.search_params.get('search_type', '').lower()
            if search_type not in ['mineral', 'mineral_name']:
                return []
            
            mineral_names = self.search_params.get('mineral_names', [])
            if not mineral_names:
                return []
            
            results = []
            for mineral_name in mineral_names:
                mineral_key = mineral_name.lower().strip()
                if mineral_key in self.fallback_minerals:
                    mineral_data = self.fallback_minerals[mineral_key].copy()
                    mineral_data['fallback'] = True  # Mark as fallback data
                    results.append(mineral_data)
                    print(f"Found fallback data for {mineral_name}")
            
            return results
            
        except Exception as e:
            print(f"Error in fallback search: {e}")
            return []
            
    def search_by_mineral(self, base_url):
        """Search by mineral name(s) using AMCSD search form"""
        try:
            mineral_names = self.search_params.get('mineral_names', [])
            if not mineral_names:
                return []
            
            search_simultaneously = self.search_params.get('search_all_simultaneously', True)
            all_results = []
            
            if search_simultaneously and len(mineral_names) > 1:
                # Search all minerals in one query using space-separated format
                combined_name = ' '.join(mineral_names)
                print(f"Searching for multiple minerals simultaneously: {combined_name}")
                
                search_data = {
                    'Mineral': combined_name,
                    'Author': '',
                    'Periodic': '',
                    'CellParam': '',
                    'diff': '',
                    'Key': '',
                    'logic': 'OR',  # Use OR logic for multiple minerals
                    'Viewing': 'text',
                    'Download': 'no',
                    'hid1': ''
                }
                
                # Try the search with increased timeout and retry logic
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        response = requests.post(f"{base_url}/result.php", 
                                               data=search_data, 
                                               timeout=30)  # Increased timeout
                        break
                    except requests.exceptions.Timeout:
                        print(f"Timeout on attempt {attempt + 1}/{max_retries} for combined search")
                        if attempt == max_retries - 1:
                            raise
                        continue
                response.raise_for_status()
                
                results = self.parse_search_results(response.content, combined_name)
                all_results.extend(results if results is not None else [])
                
            else:
                # Search each mineral individually (slower but more thorough)
                for i, mineral_name in enumerate(mineral_names):
                    print(f"Searching for mineral {i+1}/{len(mineral_names)}: {mineral_name}")
                    
                    # Add rate limiting - wait between searches to be respectful to the server
                    if i > 0:
                        print("Waiting 2 seconds between searches to avoid overwhelming server...")
                        time.sleep(2)
                    
                    search_data = {
                        'Mineral': mineral_name,
                        'Author': '',
                        'Periodic': '',
                        'CellParam': '',
                        'diff': '',
                        'Key': '',
                        'logic': 'AND',
                        'Viewing': 'text',
                        'Download': 'no',
                        'hid1': ''
                    }
                    
                    # Try the search with increased timeout and retry logic
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            response = requests.post(f"{base_url}/result.php", 
                                                   data=search_data, 
                                                   timeout=30)  # Increased timeout
                            break
                        except requests.exceptions.Timeout:
                            print(f"Timeout on attempt {attempt + 1}/{max_retries} for {mineral_name}")
                            if attempt == max_retries - 1:
                                raise
                            continue
                        except requests.exceptions.ConnectionError as e:
                            print(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(5)  # Wait longer on connection errors
                            continue
                    response.raise_for_status()
                    
                    print(f"Search response status: {response.status_code}, content length: {len(response.content)}")
                    
                    results = self.parse_search_results(response.content, mineral_name)
                    print(f"Parsed {len(results) if results else 0} results for {mineral_name}")
                    all_results.extend(results if results is not None else [])
                    
                    # Update progress for individual searches
                    progress = int((i + 1) / len(mineral_names) * 100)
                    self.progress_updated.emit(progress)
            
            # Remove duplicates based on AMCSD ID
            seen_ids = set()
            unique_results = []
            for result in all_results:
                amcsd_id = result.get('amcsd_id')
                if amcsd_id and amcsd_id not in seen_ids:
                    seen_ids.add(amcsd_id)
                    unique_results.append(result)
            
            print(f"Found {len(unique_results)} unique results for {len(mineral_names)} mineral(s)")
            return unique_results
            
        except Exception as e:
            print(f"Error in search_by_mineral: {e}")
            return []
    
    def parse_search_results(self, html_content, mineral_name):
        """Parse AMCSD search results from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # AMCSD results are structured with checkboxes for each unique database entry
            # Look for input checkboxes which represent individual database records
            checkboxes = soup.find_all('input', {'type': 'checkbox', 'name': 'check[]'})
            
            print(f"Found {len(checkboxes)} checkboxes in search results for '{mineral_name}'")
            
            if len(checkboxes) == 0:
                # Debug: check if there are any other forms or indicators of results
                all_inputs = soup.find_all('input')
                print(f"Total input elements found: {len(all_inputs)}")
                if len(all_inputs) > 0:
                    print(f"Input types found: {[inp.get('type', 'no-type') for inp in all_inputs[:5]]}")
                
                # Check for error messages or "no results" indicators
                text_content = soup.get_text().lower()
                if 'no matches' in text_content or 'no results' in text_content:
                    print("Search returned 'no matches' message")
                elif 'error' in text_content:
                    print("Search returned error message")
                else:
                    print("Unknown search result format")
                
                return []
            
            for checkbox in checkboxes:
                try:
                    # Get the AMCSD ID from the checkbox value (format: "12062.amc")
                    checkbox_value = checkbox.get('value')
                    if not checkbox_value:
                        continue
                    
                    # Extract AMCSD ID by removing .amc extension
                    amcsd_id = checkbox_value.replace('.amc', '')
                    if not amcsd_id:
                        continue
                    
                    # Find the table row containing this checkbox
                    parent_row = checkbox.find_parent('tr')
                    if not parent_row:
                        continue
                    
                    # Extract data from the row
                    result_data = self.extract_result_data_from_row(parent_row, amcsd_id)
                    
                    # Look for CIF download link in this row or nearby
                    cif_link = parent_row.find('a', href=re.compile(r'\.cif'))
                    cif_url = None
                    if cif_link:
                        href = cif_link.get('href')
                        if href.startswith('/'):
                            cif_url = f"https://rruff.geo.arizona.edu{href}"
                        else:
                            cif_url = href
                    
                    result = {
                        'mineral': result_data.get('mineral', mineral_name),
                        'formula': result_data.get('formula', 'Unknown'),
                        'space_group': result_data.get('space_group', 'Unknown'),
                        'cell_params': result_data.get('cell_params', {}),
                        'cif_url': cif_url,
                        'reference': result_data.get('reference', 'AMCSD Database'),
                        'authors': result_data.get('authors', 'Unknown'),
                        'journal': result_data.get('journal', 'Unknown'),
                        'amcsd_id': amcsd_id
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error parsing result for checkbox {checkbox}: {e}")
                    continue
            
            # Remove duplicates based on AMCSD ID
            unique_results = []
            seen_ids = set()
            for result in results:
                amcsd_id = result.get('amcsd_id')
                if amcsd_id and amcsd_id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(amcsd_id)
            
            return unique_results
            
        except Exception as e:
            print(f"Error parsing search results: {e}")
            return []
    
    def extract_result_data_from_row(self, row, amcsd_id):
        """Extract result data from a table row containing a database entry"""
        data = {}
        
        try:
            # Get all cells in this row
            cells = row.find_all(['td', 'th'])
            
            # AMCSD results typically have columns like:
            # [checkbox] [mineral] [authors] [journal] [space group] [cell params] [download links]
            
            row_text = row.get_text(separator=' ', strip=True)
            
            # Extract authors (usually contains names with initials)
            author_patterns = [
                r'([A-Z][a-z]+\s+[A-Z]\s*[A-Z]?[,\s]+[A-Z][a-z]+\s+[A-Z]\s*[A-Z]?)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]*\s+[A-Z][a-z]*)',
                r'([A-Z][a-z]+\s*,\s*[A-Z]\s*[A-Z]?)'
            ]
            
            for pattern in author_patterns:
                match = re.search(pattern, row_text)
                if match:
                    data['authors'] = match.group(1).strip()
                    break
            
            # Extract journal reference (contains year in parentheses)
            journal_match = re.search(r'([A-Za-z\s]+\s+\d+\s+\(\d{4}\)\s+[\d\-]+)', row_text)
            if journal_match:
                data['journal'] = journal_match.group(1).strip()
            
            # Extract space group (crystallographic notation)
            space_group_match = re.search(r'([A-Z]\d+[a-z]*|P\d+[a-z]*|I\d+[a-z]*|F\d+[a-z]*|C\d+[a-z]*)', row_text)
            if space_group_match:
                data['space_group'] = space_group_match.group(1)
            
            # Extract cell parameters (look for 6 decimal numbers)
            numbers = re.findall(r'\d+\.\d+', row_text)
            if len(numbers) >= 6:
                try:
                    # Take the first 6 numbers that could be cell parameters
                    params = [float(n) for n in numbers[:6]]
                    if all(0.1 < p < 1000 for p in params):  # Reasonable cell parameter range
                        data['cell_params'] = {
                            'a': str(params[0]),
                            'b': str(params[1]),
                            'c': str(params[2]),
                            'alpha': str(params[3]),
                            'beta': str(params[4]),
                            'gamma': str(params[5])
                        }
                except:
                    pass
            
            # Extract chemical formula (look for element symbols)
            formula_match = re.search(r'([A-Z][a-z]?\d*\s*)+', row_text)
            if formula_match and len(formula_match.group(0)) < 30:
                # Clean up the formula
                formula = re.sub(r'\s+', '', formula_match.group(0))
                if re.match(r'^[A-Z][a-z]?(\d*[A-Z][a-z]?\d*)*$', formula):
                    data['formula'] = formula
            
        except Exception as e:
            print(f"Error extracting data from row: {e}")
        
        return data
    
    def extract_result_data_from_context(self, cif_link, parent_table):
        """Extract detailed data from the context around a CIF link"""
        data = {}
        
        try:
            # Get all text content around the CIF link
            rows = parent_table.find_all('tr')
            
            # Look for patterns in the text
            for row in rows:
                text = row.get_text(strip=True)
                
                # Extract authors (usually first line after mineral name)
                if any(name in text for name in ['L', 'C', 'T', 'D', 'J']) and len(text.split()) > 2:
                    if 'authors' not in data and not any(char.isdigit() for char in text[:20]):
                        data['authors'] = text
                
                # Extract journal reference (contains year and page numbers)
                if re.search(r'\(\d{4}\)', text) and any(word in text.lower() for word in ['american', 'journal', 'mineralogist']):
                    data['journal'] = text
                
                # Extract space group (contains crystallographic notation)
                if re.search(r'\*[A-Z]\d+', text) or re.search(r'P\d+', text):
                    parts = text.split()
                    for part in parts:
                        if '*' in part or (part.startswith('P') and any(c.isdigit() for c in part)):
                            data['space_group'] = part.replace('*', '')
                            break
                
                # Extract cell parameters (6 numbers in a row)
                numbers = re.findall(r'\d+\.?\d*', text)
                if len(numbers) >= 6 and len(text.split()) <= 10:
                    try:
                        if all(float(n) > 0 for n in numbers[:6]):
                            data['cell_params'] = {
                                'a': numbers[0],
                                'b': numbers[1], 
                                'c': numbers[2],
                                'alpha': numbers[3],
                                'beta': numbers[4],
                                'gamma': numbers[5]
                            }
                    except:
                        pass
                
                # Extract chemical formula (look for element symbols)
                if re.search(r'[A-Z][a-z]?\s+\.\d+', text):
                    # This looks like atomic coordinates, skip
                    continue
                elif re.search(r'[A-Z][a-z]?(\d+)?', text) and len(text) < 50:
                    # Could be a formula
                    elements = re.findall(r'[A-Z][a-z]?\d*', text)
                    if elements and len(elements) <= 10:
                        data['formula'] = ''.join(elements)
        
        except Exception as e:
            print(f"Error extracting context data: {e}")
        
        return data
    
    def extract_text_from_cells(self, cells, data_type):
        """Extract specific data from table cells"""
        try:
            # This is a simplified extraction - would need to be refined
            # based on actual AMCSD table structure
            if len(cells) > 1:
                return cells[1].get_text(strip=True)
            return 'Unknown'
        except:
            return 'Unknown'
    
    def extract_cell_params_from_row(self, cells):
        """Extract unit cell parameters from table row"""
        try:
            # Look for cell parameter data in the cells
            # This would need to be customized based on AMCSD format
            params = {}
            for cell in cells:
                text = cell.get_text(strip=True)
                # Look for patterns like "a=5.43" etc.
                if '=' in text and any(param in text.lower() for param in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
                    parts = text.split('=')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        params[key] = value
            return params
        except:
            return {}
        
    def search_by_diffraction(self, base_url):
        """Search by diffraction peaks"""
        peaks = self.search_params['peaks']
        tolerance = self.search_params.get('tolerance', 0.1)
        
        # Use the diffraction search page
        search_url = f"{base_url}/diffpatt.php"
        
        # This would require form submission - simplified for now
        results = []
        return results
        
    def get_mineral_data(self, url):
        """Get detailed data for a specific mineral"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse mineral data from the page
            results = []
            
            # Look for CIF file links
            cif_links = soup.find_all('a', href=re.compile(r'\.cif$'))
            
            for link in cif_links:
                try:
                    result = {
                        'mineral': self.extract_mineral_name(soup),
                        'formula': self.extract_formula(soup),
                        'space_group': self.extract_space_group(soup),
                        'cell_params': self.extract_cell_params(soup),
                        'cif_url': f"https://rruff.geo.arizona.edu/AMS/{link['href']}",
                        'reference': self.extract_reference(soup)
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error parsing mineral result: {e}")
                    continue
                
            return results
            
        except Exception as e:
            print(f"Error getting mineral data from {url}: {e}")
            return []
            
    def extract_mineral_name(self, soup):
        """Extract mineral name from page"""
        # Look for mineral name in title or headers
        title = soup.find('title')
        if title:
            return title.text.strip()
        return "Unknown"
        
    def extract_formula(self, soup):
        """Extract chemical formula"""
        # Look for formula in the page content
        return "Unknown"
        
    def extract_space_group(self, soup):
        """Extract space group"""
        return "Unknown"
        
    def extract_cell_params(self, soup):
        """Extract unit cell parameters"""
        return {}
        
    def extract_reference(self, soup):
        """Extract literature reference"""
        return "Unknown"

class DatabaseTab(QWidget):
    """Tab for searching crystal structure databases"""
    
    phases_selected = pyqtSignal(list)  # Signal emitted when phases are selected
    
    def __init__(self):
        super().__init__()
        self.search_results = []
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Search controls
        search_panel = self.create_search_panel()
        layout.addWidget(search_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results table
        results_panel = self.create_results_panel()
        layout.addWidget(results_panel)
        
    def create_search_panel(self):
        """Create the search control panel"""
        group = QGroupBox("Database Search")
        layout = QVBoxLayout(group)
        
        # Search type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Search Type:"))
        
        self.search_type = QComboBox()
        self.search_type.addItems(["Mineral Name", "Chemical Elements", "Diffraction Peaks"])
        self.search_type.currentTextChanged.connect(self.search_type_changed)
        type_layout.addWidget(self.search_type)
        
        type_layout.addStretch()
        layout.addLayout(type_layout)
        
        # Search parameters (will change based on search type)
        self.search_params_widget = QWidget()
        self.search_params_layout = QVBoxLayout(self.search_params_widget)
        layout.addWidget(self.search_params_widget)
        
        # Initialize with mineral search
        self.setup_mineral_search()
        
        # Search button
        button_layout = QHBoxLayout()
        self.search_btn = QPushButton("Search AMCSD")
        self.search_btn.clicked.connect(self.start_search)
        button_layout.addWidget(self.search_btn)
        
        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return group
        
    def create_results_panel(self):
        """Create the results panel"""
        group = QGroupBox("Search Results")
        layout = QVBoxLayout(group)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            'Select', 'Mineral', 'Formula', 'Space Group', 'Cell Parameters', 'Authors', 'Journal'
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.results_table)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("Download Selected CIF Files")
        self.download_btn.clicked.connect(self.download_selected)
        self.download_btn.setEnabled(False)
        button_layout.addWidget(self.download_btn)
        
        self.add_to_matching_btn = QPushButton("Add to Phase Matching")
        self.add_to_matching_btn.clicked.connect(self.add_to_matching)
        self.add_to_matching_btn.setEnabled(False)
        button_layout.addWidget(self.add_to_matching_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return group
        
    def search_type_changed(self, search_type):
        """Handle search type change"""
        # Clear existing widgets safely
        for i in reversed(range(self.search_params_layout.count())):
            item = self.search_params_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    # Handle spacers and other non-widget items
                    self.search_params_layout.removeItem(item)
            
        if search_type == "Mineral Name":
            self.setup_mineral_search()
        elif search_type == "Chemical Elements":
            self.setup_chemistry_search()
        elif search_type == "Diffraction Peaks":
            self.setup_diffraction_search()
            
    def setup_mineral_search(self):
        """Setup mineral name search interface"""
        layout = QVBoxLayout()
        
        # Single mineral input
        single_layout = QHBoxLayout()
        single_layout.addWidget(QLabel("Mineral Name(s):"))
        
        self.mineral_name_input = QLineEdit()
        self.mineral_name_input.setPlaceholderText("Enter mineral names separated by commas (e.g., quartz, calcite, feldspar)")
        single_layout.addWidget(self.mineral_name_input)
        
        layout.addLayout(single_layout)
        
        # Multi-mineral options
        options_layout = QHBoxLayout()
        self.search_all_minerals = QCheckBox("Search all minerals simultaneously")
        self.search_all_minerals.setChecked(False)  # Default to individual searches (more reliable)
        self.search_all_minerals.setToolTip("Check for faster search, uncheck for more thorough individual searches")
        options_layout.addWidget(self.search_all_minerals)
        options_layout.addStretch()
        
        layout.addLayout(options_layout)
        self.search_params_layout.addLayout(layout)
        
    def setup_chemistry_search(self):
        """Setup chemical element search interface"""
        layout = QVBoxLayout()
        
        # Element selection
        element_layout = QHBoxLayout()
        element_layout.addWidget(QLabel("Elements:"))
        
        self.elements_input = QLineEdit()
        self.elements_input.setPlaceholderText("Enter elements (e.g., Si, O, Al)")
        element_layout.addWidget(self.elements_input)
        
        layout.addLayout(element_layout)
        
        # Search options
        options_layout = QHBoxLayout()
        self.exact_match = QCheckBox("Exact composition match")
        options_layout.addWidget(self.exact_match)
        options_layout.addStretch()
        
        layout.addLayout(options_layout)
        self.search_params_layout.addLayout(layout)
        
    def setup_diffraction_search(self):
        """Setup diffraction peak search interface"""
        layout = QVBoxLayout()
        
        # Peak input
        peak_layout = QHBoxLayout()
        peak_layout.addWidget(QLabel("d-spacings (Å):"))
        
        self.dspacing_input = QLineEdit()
        self.dspacing_input.setPlaceholderText("Enter d-spacings separated by commas (e.g., 3.34, 4.26, 2.45)")
        peak_layout.addWidget(self.dspacing_input)
        
        layout.addLayout(peak_layout)
        
        # Tolerance
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Tolerance (Å):"))
        
        self.tolerance_input = QLineEdit("0.02")
        tolerance_layout.addWidget(self.tolerance_input)
        tolerance_layout.addStretch()
        
        layout.addLayout(tolerance_layout)
        self.search_params_layout.addLayout(layout)
        
    def start_search(self):
        """Start the database search"""
        search_params = self.get_search_parameters()
        
        if not self.validate_search_params(search_params):
            return
            
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.search_btn.setEnabled(False)
        
        # Start search thread
        self.search_thread = AMCSDSearchThread(search_params)
        self.search_thread.results_ready.connect(self.display_results)
        self.search_thread.error_occurred.connect(self.search_error)
        self.search_thread.start()
        
    def get_search_parameters(self):
        """Get search parameters from UI"""
        search_type = self.search_type.currentText()
        
        params = {'search_type': search_type.lower().replace(' ', '_')}
        
        if search_type == "Mineral Name":
            mineral_text = self.mineral_name_input.text().strip()
            # Split by commas and clean up mineral names
            mineral_names = [name.strip() for name in mineral_text.split(',') if name.strip()]
            params['mineral_names'] = mineral_names
            params['search_all_simultaneously'] = self.search_all_minerals.isChecked()
        elif search_type == "Chemical Elements":
            params['elements'] = [e.strip() for e in self.elements_input.text().split(',')]
            params['exact_match'] = self.exact_match.isChecked()
        elif search_type == "Diffraction Peaks":
            try:
                params['peaks'] = [float(d.strip()) for d in self.dspacing_input.text().split(',')]
                params['tolerance'] = float(self.tolerance_input.text())
            except ValueError:
                params['peaks'] = []
                params['tolerance'] = 0.02
                
        return params
        
    def validate_search_params(self, params):
        """Validate search parameters"""
        search_type = params['search_type']
        
        if search_type == 'mineral_name':
            if not params.get('mineral_names') or len(params['mineral_names']) == 0:
                QMessageBox.warning(self, "Warning", "Please enter at least one mineral name")
                return False
        elif search_type == 'chemical_elements':
            if not params.get('elements') or not params['elements'][0]:
                QMessageBox.warning(self, "Warning", "Please enter at least one element")
                return False
        elif search_type == 'diffraction_peaks':
            if not params.get('peaks'):
                QMessageBox.warning(self, "Warning", "Please enter valid d-spacing values")
                return False
                
        return True
        
    def display_results(self, results):
        """Display search results in the table"""
        self.search_results = results
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        
        # Update table
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            # Checkbox for selection
            checkbox = QCheckBox()
            self.results_table.setCellWidget(i, 0, checkbox)
            
            # Result data
            self.results_table.setItem(i, 1, QTableWidgetItem(result.get('mineral', 'Unknown')))
            self.results_table.setItem(i, 2, QTableWidgetItem(result.get('formula', 'Unknown')))
            self.results_table.setItem(i, 3, QTableWidgetItem(result.get('space_group', 'Unknown')))
            
            # Cell parameters as string
            cell_params = result.get('cell_params', {})
            cell_str = ', '.join([f"{k}={v}" for k, v in cell_params.items()]) if cell_params else 'Unknown'
            self.results_table.setItem(i, 4, QTableWidgetItem(cell_str))
            
            self.results_table.setItem(i, 5, QTableWidgetItem(result.get('authors', 'Unknown')))
            self.results_table.setItem(i, 6, QTableWidgetItem(result.get('journal', 'Unknown')))
            
        # Enable action buttons if results exist
        if results:
            self.download_btn.setEnabled(True)
            self.add_to_matching_btn.setEnabled(True)
            
        QMessageBox.information(self, "Search Complete", f"Found {len(results)} results")
        
    def search_error(self, error_message):
        """Handle search error"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        QMessageBox.critical(self, "Search Error", f"Search failed:\n{error_message}")
        
    def clear_results(self):
        """Clear search results"""
        self.search_results = []
        self.results_table.setRowCount(0)
        self.download_btn.setEnabled(False)
        self.add_to_matching_btn.setEnabled(False)
        
    def get_selected_results(self):
        """Get selected results from the table"""
        selected = []
        for i in range(self.results_table.rowCount()):
            checkbox = self.results_table.cellWidget(i, 0)
            if checkbox.isChecked():
                selected.append(self.search_results[i])
        return selected
        
    def download_selected(self):
        """Download selected CIF files"""
        selected = self.get_selected_results()
        
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one result")
            return
            
        # TODO: Implement CIF file download
        QMessageBox.information(self, "Info", f"Would download {len(selected)} CIF files")
        
    def add_to_matching(self):
        """Add selected phases to matching tab"""
        selected = self.get_selected_results()
        
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one result")
            return
        
        # Download CIF content for each selected phase
        enhanced_phases = []
        for phase in selected:
            enhanced_phase = phase.copy()
            
            # Try to download CIF content
            cif_url = phase.get('cif_url')
            if cif_url:
                try:
                    response = requests.get(cif_url, timeout=10)
                    if response.status_code == 200:
                        enhanced_phase['cif_content'] = response.text
                        print(f"Downloaded CIF for {phase.get('mineral', 'Unknown')}")
                    else:
                        print(f"Failed to download CIF from {cif_url}")
                except Exception as e:
                    print(f"Error downloading CIF: {e}")
            
            # Extract AMCSD ID from URL if available
            if cif_url and 'id=' in cif_url:
                try:
                    amcsd_id = cif_url.split('id=')[1].split('&')[0]
                    enhanced_phase['amcsd_id'] = amcsd_id
                    print(f"Extracted AMCSD ID: {amcsd_id} for {phase.get('mineral', 'Unknown')}")
                except Exception as e:
                    print(f"Failed to extract AMCSD ID from {cif_url}: {e}")
            else:
                # Try to extract from other fields if available
                for field in ['amcsd_id', 'id', 'record_id']:
                    if field in phase and phase[field]:
                        enhanced_phase['amcsd_id'] = str(phase[field])
                        print(f"Using {field} as AMCSD ID: {phase[field]}")
                        break
                else:
                    print(f"No AMCSD ID found for {phase.get('mineral', 'Unknown')}")
            
            enhanced_phases.append(enhanced_phase)
            
        self.phases_selected.emit(enhanced_phases)
        QMessageBox.information(self, "Success", f"Added {len(enhanced_phases)} phases to matching")
