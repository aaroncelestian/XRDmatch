"""
Local database management tab for CIF files
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QGroupBox,
                             QProgressBar, QTextEdit, QFileDialog, QMessageBox,
                             QSplitter, QLineEdit, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from utils.local_database import LocalCIFDatabase

class DatabaseImportThread(QThread):
    """Thread for importing CIF files to local database"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    import_complete = pyqtSignal(int)
    
    def __init__(self, db_manager, cif_file_path, import_type='single'):
        super().__init__()
        self.db_manager = db_manager
        self.cif_file_path = cif_file_path
        self.import_type = import_type
    
    def run(self):
        """Run the import in a separate thread"""
        try:
            if self.import_type == 'bulk':
                self.status_updated.emit("Starting bulk AMCSD import...")
                added_count = self.db_manager.bulk_import_amcsd_cif(
                    self.cif_file_path, 
                    progress_callback=self.progress_updated.emit
                )
            else:
                self.status_updated.emit("Importing single CIF file...")
                added_count = self.db_manager.add_cif_file(self.cif_file_path)
                self.progress_updated.emit(100)
            
            self.import_complete.emit(added_count)
            
        except Exception as e:
            self.status_updated.emit(f"Import error: {str(e)}")
            self.import_complete.emit(0)

class DiffractionCalculationThread(QThread):
    """Thread for calculating diffraction patterns"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    calculation_complete = pyqtSignal(int)
    
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
    
    def run(self):
        """Run the diffraction pattern calculation in a separate thread"""
        try:
            self.status_updated.emit("Calculating diffraction patterns for all minerals...")
            successful_count = self.db_manager.bulk_calculate_diffraction_patterns(
                progress_callback=self.progress_updated.emit
            )
            self.calculation_complete.emit(successful_count)
            
        except Exception as e:
            self.status_updated.emit(f"Calculation error: {str(e)}")
            self.calculation_complete.emit(0)

class CommonMineralsCalculationThread(QThread):
    """Thread for calculating diffraction patterns for common minerals only"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    calculation_complete = pyqtSignal(int)
    
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
    
    def run(self):
        """Run the diffraction pattern calculation for common minerals only"""
        try:
            # List of common mineral names to prioritize
            common_minerals = [
                'quartz', 'calcite', 'feldspar', 'albite', 'orthoclase', 'microcline',
                'plagioclase', 'muscovite', 'biotite', 'chlorite', 'kaolinite', 'illite',
                'montmorillonite', 'pyrite', 'hematite', 'magnetite', 'goethite',
                'gypsum', 'anhydrite', 'halite', 'fluorite', 'apatite', 'zircon',
                'rutile', 'anatase', 'brookite', 'corundum', 'spinel', 'olivine',
                'pyroxene', 'amphibole', 'epidote', 'garnet', 'staurolite', 'andalusite',
                'sillimanite', 'kyanite', 'cordierite', 'tourmaline', 'topaz',
                'beryl', 'chrysoberyl', 'dolomite', 'siderite', 'magnesite',
                'rhodochrosite', 'smithsonite', 'cerussite', 'malachite', 'azurite',
                'epsomite', 'hexahydrite', 'meridianiite', 'pentahydrite', 'starkeyite',
                'kieserite', 'sanderite', 'anhydrous magnesium sulfate', 'magnesium sulfate',
                'mirabilite', 'thenardite', 'halotrichite'
            ]
            
            self.status_updated.emit("Finding common minerals in database...")
            
            # Get mineral IDs for common minerals
            conn = self.db_manager.db_path
            import sqlite3
            conn = sqlite3.connect(conn)
            cursor = conn.cursor()
            
            mineral_ids = []
            for mineral_name in common_minerals:
                cursor.execute('''
                    SELECT id, mineral_name FROM minerals 
                    WHERE mineral_name LIKE ? 
                    ORDER BY mineral_name
                    LIMIT 20
                ''', (f'%{mineral_name}%',))
                
                results = cursor.fetchall()
                mineral_ids.extend(results)
            
            # Remove duplicates
            unique_minerals = list(dict.fromkeys(mineral_ids))
            conn.close()
            
            if not unique_minerals:
                self.status_updated.emit("No common minerals found in database")
                self.calculation_complete.emit(0)
                return
            
            self.status_updated.emit(f"Calculating patterns for {len(unique_minerals)} common minerals...")
            
            successful_count = 0
            for i, (mineral_id, mineral_name) in enumerate(unique_minerals):
                try:
                    if self.db_manager.calculate_and_store_diffraction_pattern(mineral_id, 1.5406):
                        successful_count += 1
                    
                    progress = int((i + 1) / len(unique_minerals) * 100)
                    self.progress_updated.emit(progress)
                    
                except Exception as e:
                    print(f"Error calculating pattern for {mineral_name}: {e}")
            
            self.calculation_complete.emit(successful_count)
            
        except Exception as e:
            self.status_updated.emit(f"Calculation error: {str(e)}")
            self.calculation_complete.emit(0)

class LocalDatabaseTab(QWidget):
    """Tab for managing local CIF database"""
    
    # Signal to send selected phases to matching tab
    phases_selected = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.db_manager = LocalCIFDatabase()
        self.search_results = []
        self.init_ui()
        self.update_database_stats()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Database management section
        db_management = self.create_database_management_section()
        layout.addWidget(db_management)
        
        # Search section
        search_section = self.create_search_section()
        layout.addWidget(search_section)
        
        # Results section
        results_section = self.create_results_section()
        layout.addWidget(results_section)
    
    def create_database_management_section(self):
        """Create database management controls"""
        group = QGroupBox("Database Management")
        layout = QVBoxLayout(group)
        
        # Database stats
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Database: Loading...")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        
        refresh_btn = QPushButton("Refresh Stats")
        refresh_btn.clicked.connect(self.update_database_stats)
        stats_layout.addWidget(refresh_btn)
        
        layout.addLayout(stats_layout)
        
        # Import controls
        import_layout = QHBoxLayout()
        
        # Single CIF import
        single_cif_btn = QPushButton("Import Single CIF File")
        single_cif_btn.clicked.connect(self.import_single_cif)
        import_layout.addWidget(single_cif_btn)
        
        # Bulk AMCSD import
        bulk_import_btn = QPushButton("Import AMCSD CIF File")
        bulk_import_btn.clicked.connect(self.import_amcsd_bulk)
        bulk_import_btn.setToolTip("Import the complete AMCSD database CIF file")
        import_layout.addWidget(bulk_import_btn)
        
        # Import from directory
        dir_import_btn = QPushButton("Import CIF Directory")
        dir_import_btn.clicked.connect(self.import_cif_directory)
        import_layout.addWidget(dir_import_btn)
        
        layout.addLayout(import_layout)
        
        # Diffraction pattern calculation controls
        diffraction_layout = QHBoxLayout()
        
        calc_patterns_btn = QPushButton("Calculate All Diffraction Patterns")
        calc_patterns_btn.clicked.connect(self.calculate_all_diffraction_patterns)
        calc_patterns_btn.setToolTip("Pre-calculate Cu Kα diffraction patterns for all minerals (other wavelengths converted on-demand)")
        diffraction_layout.addWidget(calc_patterns_btn)
        
        # Add selective calculation option
        calc_subset_btn = QPushButton("Calculate Common Minerals Only")
        calc_subset_btn.clicked.connect(self.calculate_common_minerals)
        calc_subset_btn.setToolTip("Calculate patterns for ~1000 most common minerals (much faster)")
        diffraction_layout.addWidget(calc_subset_btn)
        
        diffraction_stats_btn = QPushButton("Show Pattern Statistics")
        diffraction_stats_btn.clicked.connect(self.show_diffraction_statistics)
        diffraction_layout.addWidget(diffraction_stats_btn)
        
        layout.addLayout(diffraction_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        return group
    
    def create_search_section(self):
        """Create search controls"""
        group = QGroupBox("Search Local Database")
        layout = QVBoxLayout(group)
        
        # Search type selection
        search_type_layout = QHBoxLayout()
        search_type_layout.addWidget(QLabel("Search by:"))
        
        self.search_type = QComboBox()
        self.search_type.addItems(["Mineral Name", "Chemical Formula", "Elements", "Space Group"])
        self.search_type.currentTextChanged.connect(self.search_type_changed)
        search_type_layout.addWidget(self.search_type)
        
        search_type_layout.addStretch()
        layout.addLayout(search_type_layout)
        
        # Search input
        search_input_layout = QHBoxLayout()
        search_input_layout.addWidget(QLabel("Search term:"))
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search term...")
        self.search_input.returnPressed.connect(self.perform_search)
        search_input_layout.addWidget(self.search_input)
        
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.perform_search)
        search_input_layout.addWidget(search_btn)
        
        layout.addLayout(search_input_layout)
        
        # Search options
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Max results:"))
        self.max_results = QSpinBox()
        self.max_results.setRange(10, 1000)
        self.max_results.setValue(100)
        options_layout.addWidget(self.max_results)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        return group
    
    def create_results_section(self):
        """Create results display"""
        group = QGroupBox("Search Results")
        layout = QVBoxLayout(group)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Mineral', 'Formula', 'Space Group', 'a (Å)', 'b (Å)', 'c (Å)', 'Authors', 'AMCSD ID'
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.itemSelectionChanged.connect(self.show_mineral_details)
        layout.addWidget(self.results_table)
        
        # Details section
        details_layout = QHBoxLayout()
        
        # Mineral details
        details_group = QGroupBox("Mineral Details")
        details_group_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        details_group_layout.addWidget(self.details_text)
        
        details_layout.addWidget(details_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.use_for_matching_btn = QPushButton("Use for Phase Matching")
        self.use_for_matching_btn.clicked.connect(self.use_for_matching)
        self.use_for_matching_btn.setEnabled(False)
        actions_layout.addWidget(self.use_for_matching_btn)
        
        self.view_cif_btn = QPushButton("View CIF Content")
        self.view_cif_btn.clicked.connect(self.view_cif_content)
        self.view_cif_btn.setEnabled(False)
        actions_layout.addWidget(self.view_cif_btn)
        
        details_layout.addWidget(actions_group)
        layout.addLayout(details_layout)
        
        return group
    
    def update_database_stats(self):
        """Update database statistics display"""
        try:
            stats = self.db_manager.get_database_stats()
            diffraction_stats = self.db_manager.get_diffraction_statistics()
            
            # Format the statistics display
            base_stats = (f"Database: {stats['total_minerals']} minerals, "
                         f"{stats['unique_elements']} elements, "
                         f"{stats['unique_space_groups']} space groups")
            
            if diffraction_stats.get('total_patterns', 0) > 0:
                pattern_info = (f" | Patterns: {diffraction_stats['total_patterns']} "
                              f"({diffraction_stats['coverage_percentage']:.1f}% coverage)")
                self.stats_label.setText(base_stats + pattern_info)
            else:
                self.stats_label.setText(base_stats + " | No diffraction patterns calculated")
                
        except Exception as e:
            self.stats_label.setText(f"Database error: {e}")
    
    def import_single_cif(self):
        """Import a single CIF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CIF File", "", "CIF Files (*.cif);;All Files (*)"
        )
        
        if file_path:
            self.start_import(file_path, 'single')
    
    def import_amcsd_bulk(self):
        """Import the complete AMCSD CIF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select AMCSD CIF File", "", "CIF Files (*.cif);;All Files (*)"
        )
        
        if file_path:
            reply = QMessageBox.question(
                self, "Bulk Import", 
                "This will import the entire AMCSD database. This may take several minutes. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.start_import(file_path, 'bulk')
    
    def import_cif_directory(self):
        """Import all CIF files from a directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory with CIF Files")
        
        if dir_path:
            cif_files = list(Path(dir_path).glob("*.cif"))
            if not cif_files:
                QMessageBox.warning(self, "No CIF Files", "No CIF files found in the selected directory.")
                return
            
            reply = QMessageBox.question(
                self, "Directory Import", 
                f"Found {len(cif_files)} CIF files. Import all?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.import_directory_files(cif_files)
    
    def import_directory_files(self, cif_files):
        """Import multiple CIF files from directory"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(cif_files))
        
        total_added = 0
        for i, cif_file in enumerate(cif_files):
            try:
                added = self.db_manager.add_cif_file(str(cif_file))
                total_added += added
                self.progress_bar.setValue(i + 1)
                self.status_label.setText(f"Processing {cif_file.name}... ({i+1}/{len(cif_files)})")
            except Exception as e:
                print(f"Error importing {cif_file}: {e}")
        
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Directory import complete. Added {total_added} new minerals.")
        self.update_database_stats()
    
    def start_import(self, file_path, import_type):
        """Start import in background thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        
        self.import_thread = DatabaseImportThread(self.db_manager, file_path, import_type)
        self.import_thread.progress_updated.connect(self.progress_bar.setValue)
        self.import_thread.status_updated.connect(self.status_label.setText)
        self.import_thread.import_complete.connect(self.import_finished)
        self.import_thread.start()
    
    def import_finished(self, added_count):
        """Handle import completion"""
        self.progress_bar.setVisible(False)
        
        if added_count > 0:
            self.status_label.setText(f"Import complete. Added {added_count} new minerals.")
            QMessageBox.information(self, "Import Complete", f"Successfully added {added_count} new minerals to the database.")
        else:
            self.status_label.setText("Import complete. No new minerals added (may be duplicates).")
        
        self.update_database_stats()
    
    def search_type_changed(self, search_type):
        """Handle search type change"""
        if search_type == "Elements":
            self.search_input.setPlaceholderText("Enter elements separated by commas (e.g., Mg, S, O)")
        elif search_type == "Chemical Formula":
            self.search_input.setPlaceholderText("Enter formula (e.g., MgSO4)")
        elif search_type == "Space Group":
            self.search_input.setPlaceholderText("Enter space group (e.g., P212121)")
        else:
            self.search_input.setPlaceholderText("Enter mineral name (e.g., epsomite)")
    
    def perform_search(self):
        """Perform database search"""
        search_term = self.search_input.text().strip()
        if not search_term:
            return
        
        search_type = self.search_type.currentText()
        max_results = self.max_results.value()
        
        try:
            if search_type == "Mineral Name":
                results = self.db_manager.search_by_mineral_name(search_term, max_results)
            elif search_type == "Chemical Formula":
                results = self.db_manager.search_by_formula(search_term, max_results)
            elif search_type == "Elements":
                elements = [e.strip() for e in search_term.split(',')]
                results = self.db_manager.search_by_elements(elements, exact_match=False, limit=max_results)
            elif search_type == "Space Group":
                # Use formula search for space group (could be improved)
                results = self.db_manager.search_by_formula(search_term, max_results)
            else:
                results = []
            
            self.display_search_results(results)
            self.status_label.setText(f"Found {len(results)} results for '{search_term}'")
            
        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"Error performing search: {e}")
            self.status_label.setText(f"Search error: {e}")
    
    def display_search_results(self, results):
        """Display search results in table"""
        self.search_results = results
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(result.get('mineral_name', 'Unknown')))
            self.results_table.setItem(i, 1, QTableWidgetItem(result.get('chemical_formula', 'Unknown')))
            self.results_table.setItem(i, 2, QTableWidgetItem(result.get('space_group', 'Unknown')))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.get('cell_a', 0):.3f}" if result.get('cell_a') else 'N/A'))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{result.get('cell_b', 0):.3f}" if result.get('cell_b') else 'N/A'))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{result.get('cell_c', 0):.3f}" if result.get('cell_c') else 'N/A'))
            self.results_table.setItem(i, 6, QTableWidgetItem(result.get('authors', 'Unknown')))
            self.results_table.setItem(i, 7, QTableWidgetItem(result.get('amcsd_id', 'N/A')))
        
        self.results_table.resizeColumnsToContents()
    
    def show_mineral_details(self):
        """Show details for selected mineral"""
        current_row = self.results_table.currentRow()
        if current_row < 0 or current_row >= len(self.search_results):
            return
        
        mineral = self.search_results[current_row]
        
        details = f"""
Mineral: {mineral.get('mineral_name', 'Unknown')}
Formula: {mineral.get('chemical_formula', 'Unknown')}
Space Group: {mineral.get('space_group', 'Unknown')}
Crystal System: {mineral.get('crystal_system', 'Unknown')}

Unit Cell:
  a = {mineral.get('cell_a', 'N/A')} Å
  b = {mineral.get('cell_b', 'N/A')} Å  
  c = {mineral.get('cell_c', 'N/A')} Å
  α = {mineral.get('cell_alpha', 'N/A')}°
  β = {mineral.get('cell_beta', 'N/A')}°
  γ = {mineral.get('cell_gamma', 'N/A')}°
  Volume = {mineral.get('cell_volume', 'N/A')} Å³

Publication:
  Authors: {mineral.get('authors', 'Unknown')}
  Journal: {mineral.get('journal', 'Unknown')}
  Year: {mineral.get('year', 'N/A')}
  DOI: {mineral.get('doi', 'N/A')}

Database:
  AMCSD ID: {mineral.get('amcsd_id', 'N/A')}
  Local ID: {mineral.get('id', 'N/A')}
        """.strip()
        
        self.details_text.setText(details)
        self.use_for_matching_btn.setEnabled(True)
        self.view_cif_btn.setEnabled(True)
    
    def use_for_matching(self):
        """Use selected mineral for phase matching"""
        current_row = self.results_table.currentRow()
        if current_row < 0 or current_row >= len(self.search_results):
            return
        
        mineral = self.search_results[current_row]
        
        # Convert to format expected by matching tab
        phase_data = {
            'id': mineral.get('id'),  # Include database ID for pre-calculated patterns
            'mineral': mineral.get('mineral_name', 'Unknown'),
            'formula': mineral.get('chemical_formula', 'Unknown'),
            'space_group': mineral.get('space_group', 'Unknown'),
            'cell_params': {
                'a': mineral.get('cell_a'),
                'b': mineral.get('cell_b'),
                'c': mineral.get('cell_c'),
                'alpha': mineral.get('cell_alpha'),
                'beta': mineral.get('cell_beta'),
                'gamma': mineral.get('cell_gamma')
            },
            'amcsd_id': mineral.get('amcsd_id'),
            'cif_content': mineral.get('cif_content'),
            'reference': f"{mineral.get('authors', 'Unknown')} - {mineral.get('journal', 'Unknown')}",
            'authors': mineral.get('authors', 'Unknown'),
            'journal': mineral.get('journal', 'Unknown'),
            'local_db': True  # Mark as from local database
        }
        
        # Emit signal to add to matching tab
        self.phases_selected.emit([phase_data])
        
        QMessageBox.information(self, "Added to Matching", 
                              f"Added {mineral.get('mineral_name')} to phase matching candidates.")
    
    def view_cif_content(self):
        """View CIF content for selected mineral"""
        current_row = self.results_table.currentRow()
        if current_row < 0 or current_row >= len(self.search_results):
            return
        
        mineral = self.search_results[current_row]
        cif_content = mineral.get('cif_content', 'No CIF content available')
        
        # Create dialog to show CIF content
        dialog = QMessageBox(self)
        dialog.setWindowTitle(f"CIF Content - {mineral.get('mineral_name', 'Unknown')}")
        dialog.setText("CIF file content:")
        dialog.setDetailedText(cif_content)
        dialog.exec()
    
    def calculate_all_diffraction_patterns(self):
        """Calculate diffraction patterns for all minerals in the database"""
        reply = QMessageBox.question(self, "Calculate Diffraction Patterns",
                                   "This will calculate diffraction patterns for all minerals in the database.\n"
                                   "This may take a long time depending on the number of minerals.\n\n"
                                   "Continue?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Start calculation in background thread
        self.calculation_thread = DiffractionCalculationThread(self.db_manager)
        self.calculation_thread.progress_updated.connect(self.progress_bar.setValue)
        self.calculation_thread.status_updated.connect(self.status_label.setText)
        self.calculation_thread.calculation_complete.connect(self.on_calculation_complete)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting diffraction pattern calculations...")
        
        self.calculation_thread.start()
    
    def calculate_common_minerals(self):
        """Calculate diffraction patterns for common minerals only"""
        reply = QMessageBox.question(self, "Calculate Common Minerals",
                                   "This will calculate diffraction patterns for ~1000 common minerals only.\n"
                                   "This is much faster and covers most phase identification needs.\n\n"
                                   "Continue?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Start calculation for common minerals only
        self.calculation_thread = CommonMineralsCalculationThread(self.db_manager)
        self.calculation_thread.progress_updated.connect(self.progress_bar.setValue)
        self.calculation_thread.status_updated.connect(self.status_label.setText)
        self.calculation_thread.calculation_complete.connect(self.on_calculation_complete)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Calculating patterns for common minerals...")
        
        self.calculation_thread.start()
    
    def on_calculation_complete(self, successful_count):
        """Handle completion of diffraction pattern calculation"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Calculation complete: {successful_count} patterns calculated")
        
        QMessageBox.information(self, "Calculation Complete",
                              f"Successfully calculated {successful_count} diffraction patterns.\n"
                              f"Phase matching will now be much faster!")
        
        # Update database stats to reflect new patterns
        self.update_database_stats()
    
    def show_diffraction_statistics(self):
        """Show statistics about stored diffraction patterns"""
        stats = self.db_manager.get_diffraction_statistics()
        
        if not stats:
            QMessageBox.warning(self, "Statistics Error", "Could not retrieve diffraction pattern statistics.")
            return
        
        # Format statistics message
        wavelength_info = ""
        for wavelength, count in stats.get('patterns_by_wavelength', {}).items():
            wavelength_info += f"  λ = {wavelength} Å: {count} patterns\n"
        
        if not wavelength_info:
            wavelength_info = "  No patterns calculated yet\n"
        
        message = f"""Diffraction Pattern Statistics:

Total Patterns: {stats.get('total_patterns', 0)} (Cu Kα reference)
Minerals with Patterns: {stats.get('minerals_with_patterns', 0)}
Total Minerals: {stats.get('total_minerals', 0)}
Coverage: {stats.get('coverage_percentage', 0):.1f}%

Patterns by Wavelength:
{wavelength_info}

Note: Only Cu Kα (1.5406Å) patterns are stored.
Other wavelengths are calculated on-demand using Bragg's law conversion.
This approach is much more efficient and provides the same accuracy!
        """.strip()
        
        QMessageBox.information(self, "Diffraction Pattern Statistics", message)
