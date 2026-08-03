"""
Local database management tab for CIF files
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QGroupBox,
                             QProgressBar, QTextEdit, QFileDialog, QMessageBox,
                             QSplitter, QLineEdit, QComboBox, QSpinBox, QRadioButton,
                             QDoubleSpinBox, QMenu, QApplication, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from utils.local_database import LocalCIFDatabase
from utils.cif_repository import get_cif_repository
from utils.conditions import (AMBIENT_MAX_PRESSURE_GPA, AMBIENT_MAX_TEMPERATURE_K,
                              AMBIENT_MIN_TEMPERATURE_K, Conditions)
from gui.dialogs.cif_dialog import CifViewerDialog

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
        self.cif_repo = get_cif_repository()
        self.search_results = []
        self.init_ui()
        self.update_database_stats()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self.create_database_management_section())
        layout.addWidget(self.create_search_section())
        layout.addWidget(self.create_results_section(), 1)

    def create_database_management_section(self):
        """Create database management controls"""
        group = QGroupBox("Database Management")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("Database: Loading...")
        stats_layout.addWidget(self.stats_label, 1)

        refresh_btn = QPushButton("Refresh Stats")
        refresh_btn.clicked.connect(self.update_database_stats)
        stats_layout.addWidget(refresh_btn)
        layout.addLayout(stats_layout)

        # 2-column action grid
        grid = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        single_dif_btn = QPushButton("Import Single DIF File")
        single_dif_btn.clicked.connect(self.import_single_dif)
        single_dif_btn.setToolTip("Import a single DIF file with diffraction pattern data")
        left_col.addWidget(single_dif_btn)

        bulk_dif_btn = QPushButton("Import DIF Directory")
        bulk_dif_btn.clicked.connect(self.import_dif_directory)
        bulk_dif_btn.setToolTip("Import all DIF files from a directory")
        left_col.addWidget(bulk_dif_btn)

        update_rir_btn = QPushButton("Update RIR from DIF")
        update_rir_btn.clicked.connect(self.update_rir_from_dif)
        update_rir_btn.setToolTip("Parse RIR/density from data/difdata.dif into existing minerals")
        left_col.addWidget(update_rir_btn)

        amcsd_bulk_btn = QPushButton("Import AMCSD Bulk DIF")
        amcsd_bulk_btn.setObjectName("primaryButton")
        amcsd_bulk_btn.clicked.connect(self.import_amcsd_bulk_dif)
        amcsd_bulk_btn.setToolTip("Import entire AMCSD bulk DIF file (all minerals)")
        right_col.addWidget(amcsd_bulk_btn)

        cif_to_dif_btn = QPushButton("Generate DIF from CIF")
        cif_to_dif_btn.clicked.connect(self.generate_dif_from_cif)
        cif_to_dif_btn.setToolTip("Convert CIF files to DIF format with pseudo-Voigt profiles")
        right_col.addWidget(cif_to_dif_btn)

        diffraction_stats_btn = QPushButton("Show Pattern Statistics")
        diffraction_stats_btn.clicked.connect(self.show_diffraction_statistics)
        diffraction_stats_btn.setToolTip("View statistics about stored diffraction patterns")
        right_col.addWidget(diffraction_stats_btn)

        grid.addLayout(left_col)
        grid.addLayout(right_col)
        layout.addLayout(grid)

        # Ultra-fast search index (Phases tab uses this; build/rebuild lives here only)
        index_row = QHBoxLayout()
        index_row.addWidget(QLabel("Index grid:"))
        self.index_grid_spin = QDoubleSpinBox()
        self.index_grid_spin.setRange(0.005, 0.1)
        self.index_grid_spin.setDecimals(3)
        self.index_grid_spin.setValue(0.02)
        self.index_grid_spin.setSuffix("°")
        self.index_grid_spin.setMaximumWidth(110)
        self.index_grid_spin.setToolTip("Grid resolution for the ultra-fast search index")
        index_row.addWidget(self.index_grid_spin)
        self.build_index_btn = QPushButton("Build Search Index")
        self.build_index_btn.setToolTip(
            "Build or rebuild the ultra-fast pattern search index used by Phases search"
        )
        self.build_index_btn.clicked.connect(self.build_search_index)
        index_row.addWidget(self.build_index_btn)
        self.index_status_label = QLabel("")
        self.index_status_label.setObjectName("mutedLabel")
        index_row.addWidget(self.index_status_label, 1)
        layout.addLayout(index_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")
        layout.addWidget(self.status_label)

        return group

    def create_search_section(self):
        """Create search controls"""
        group = QGroupBox("Search Local Database")
        layout = QVBoxLayout(group)

        search_type_layout = QHBoxLayout()
        search_type_layout.addWidget(QLabel("Search by:"))

        self.search_type = QComboBox()
        self.search_type.addItems(["Mineral Name", "Chemical Formula", "Elements", "Space Group"])
        self.search_type.currentTextChanged.connect(self.search_type_changed)
        search_type_layout.addWidget(self.search_type)
        search_type_layout.addStretch()
        layout.addLayout(search_type_layout)

        search_input_layout = QHBoxLayout()
        search_input_layout.addWidget(QLabel("Search term:"))

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search term...")
        self.search_input.returnPressed.connect(self.perform_search)
        search_input_layout.addWidget(self.search_input)

        search_btn = QPushButton("Search")
        search_btn.setObjectName("primaryButton")
        search_btn.clicked.connect(self.perform_search)
        search_input_layout.addWidget(search_btn)
        layout.addLayout(search_input_layout)

        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel("Max results:"))
        self.max_results = QSpinBox()
        self.max_results.setRange(10, 1000)
        self.max_results.setValue(100)
        options_layout.addWidget(self.max_results)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        self.ambient_only_check = QCheckBox("Ambient conditions only")
        self.ambient_only_check.setChecked(True)
        self.ambient_only_check.setToolTip(
            f"Hide structures measured above {AMBIENT_MAX_PRESSURE_GPA:g} GPa or outside "
            f"{AMBIENT_MIN_TEMPERATURE_K:.0f}–{AMBIENT_MAX_TEMPERATURE_K:.0f} K.\n"
            "Their compressed or expanded cells put lines at shifted 2θ, which\n"
            "produces false matches during phase identification."
        )
        self.ambient_only_check.toggled.connect(lambda _: self.perform_search())
        layout.addWidget(self.ambient_only_check)

        return group

    def create_results_section(self):
        """Create results display — flat details, no nested GroupBoxes."""
        group = QGroupBox("Search Results")
        layout = QVBoxLayout(group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            'Mineral', 'Formula', 'Space Group', 'a (Å)', 'b (Å)', 'c (Å)',
            'Conditions', 'Authors', 'AMCSD ID'
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.itemSelectionChanged.connect(self.show_mineral_details)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.show_results_context_menu)
        self.results_table.setToolTip("Right-click a row for CIF actions")
        layout.addWidget(self.results_table, 1)

        details_layout = QHBoxLayout()

        details_col = QVBoxLayout()
        details_label = QLabel("Mineral Details")
        details_label.setObjectName("mutedLabel")
        details_col.addWidget(details_label)

        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        details_col.addWidget(self.details_text)
        details_layout.addLayout(details_col, 3)

        actions_col = QVBoxLayout()
        actions_label = QLabel("Actions")
        actions_label.setObjectName("mutedLabel")
        actions_col.addWidget(actions_label)

        self.use_for_matching_btn = QPushButton("Use for Phase Matching")
        self.use_for_matching_btn.setObjectName("primaryButton")
        self.use_for_matching_btn.clicked.connect(self.use_for_matching)
        self.use_for_matching_btn.setEnabled(False)
        actions_col.addWidget(self.use_for_matching_btn)

        self.view_cif_btn = QPushButton("View CIF Content")
        self.view_cif_btn.clicked.connect(self.view_cif_content)
        self.view_cif_btn.setEnabled(False)
        actions_col.addWidget(self.view_cif_btn)
        actions_col.addStretch()
        details_layout.addLayout(actions_col, 1)

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

    def build_search_index(self):
        """Build or rebuild the ultra-fast pattern search index."""
        try:
            from utils.fast_pattern_search import FastPatternSearchEngine

            self.build_index_btn.setEnabled(False)
            self.build_index_btn.setText("Building…")
            self.index_status_label.setText("Building index…")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("Building search index…")

            engine = FastPatternSearchEngine()
            ok = engine.build_search_index(
                grid_resolution=self.index_grid_spin.value(),
                force_rebuild=True,
            )
            if ok:
                stats = engine.get_search_statistics()
                self.index_status_label.setText(
                    f"Ready · {stats.get('database_size', '?')} patterns · "
                    f"{stats.get('matrix_size_mb', 0):.0f} MB"
                )
                QMessageBox.information(
                    self,
                    "Index Built",
                    "Search index built successfully.\n\n"
                    f"Database size: {stats.get('database_size', '?')} patterns\n"
                    f"Grid points: {stats.get('grid_points', '?')}\n"
                    f"Build time: {stats.get('index_build_time_s', 0):.2f}s\n"
                    f"Memory usage: {stats.get('matrix_size_mb', 0):.1f} MB",
                )
                self.status_label.setText("Search index ready")
            else:
                self.index_status_label.setText("Build failed")
                QMessageBox.warning(
                    self, "Build Failed",
                    "Failed to build search index. Check the console for details.",
                )
                self.status_label.setText("Search index build failed")
        except Exception as e:
            self.index_status_label.setText("Build failed")
            QMessageBox.critical(self, "Build Error", f"Error building index:\n{e}")
            self.status_label.setText("Search index build failed")
        finally:
            self.build_index_btn.setEnabled(True)
            self.build_index_btn.setText("Build Search Index")
            self.progress_bar.setVisible(False)
    
    def import_single_dif(self):
        """Import a single DIF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DIF File", "", "DIF Files (*.dif);;All Files (*)"
        )
        
        if file_path:
            self.status_label.setText(f"Importing DIF file: {os.path.basename(file_path)}")
            try:
                added_count = self.db_manager.import_dif_file(file_path)
                if added_count > 0:
                    QMessageBox.information(self, "Import Complete", 
                                          f"Successfully imported {added_count} pattern(s) from DIF file.")
                    self.update_database_stats()
                else:
                    QMessageBox.warning(self, "Import Failed", 
                                      "No patterns were imported. The file may be invalid or already exists.")
                self.status_label.setText("Import complete")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Error importing DIF file: {str(e)}")
                self.status_label.setText("Import failed")
    
    def import_dif_directory(self):
        """Import all DIF files from a directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory with DIF Files")
        
        if dir_path:
            dif_files = list(Path(dir_path).glob("*.dif"))
            if not dif_files:
                QMessageBox.warning(self, "No DIF Files", "No DIF files found in the selected directory.")
                return
            
            reply = QMessageBox.question(
                self, "Directory Import", 
                f"Found {len(dif_files)} DIF files. Import all?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.import_directory_dif_files(dif_files)
    
    def import_directory_dif_files(self, dif_files):
        """Import multiple DIF files from directory"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(dif_files))
        
        total_added = 0
        for i, dif_file in enumerate(dif_files):
            try:
                added = self.db_manager.import_dif_file(str(dif_file))
                total_added += added
                self.progress_bar.setValue(i + 1)
                self.status_label.setText(f"Processing {dif_file.name}... ({i+1}/{len(dif_files)})")
            except Exception as e:
                print(f"Error importing {dif_file}: {e}")
        
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Directory import complete. Added {total_added} new patterns.")
        QMessageBox.information(self, "Import Complete", 
                              f"Successfully imported {total_added} patterns from {len(dif_files)} DIF files.")
        self.update_database_stats()
    
    def import_amcsd_bulk_dif(self):
        """Import entire AMCSD bulk DIF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select AMCSD Bulk DIF File", "", "DIF Files (*.dif);;All Files (*)"
        )
        
        if file_path:
            reply = QMessageBox.question(
                self, "Bulk Import", 
                "This will import ALL minerals from the AMCSD bulk DIF file.\n"
                "This may take several minutes depending on file size.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.start_bulk_import(file_path)
    
    def start_bulk_import(self, file_path):
        """Start bulk import in background thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        
        self.bulk_import_thread = BulkDifImportThread(self.db_manager, file_path)
        self.bulk_import_thread.progress_updated.connect(self.progress_bar.setValue)
        self.bulk_import_thread.status_updated.connect(self.status_label.setText)
        self.bulk_import_thread.import_complete.connect(self.bulk_import_finished)
        self.bulk_import_thread.start()
    
    def bulk_import_finished(self, imported_count):
        """Handle bulk import completion"""
        self.progress_bar.setVisible(False)
        
        if imported_count > 0:
            self.status_label.setText(f"Bulk import complete. Imported {imported_count} patterns.")
            QMessageBox.information(self, "Import Complete", 
                                  f"Successfully imported {imported_count} diffraction patterns from AMCSD bulk file!\n\n"
                                  f"You can now build the ultra-fast search index in the Pattern Search tab.")
        else:
            self.status_label.setText("Bulk import complete. No new patterns added.")
            QMessageBox.warning(self, "Import Complete", 
                              "No new patterns were imported. They may already exist in the database.")
        
        self.update_database_stats()

    def update_rir_from_dif(self):
        """Update RIR/density fields from the AMCSD bulk DIF file"""
        from pathlib import Path
        default = Path(__file__).parent.parent / "data" / "difdata.dif"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select AMCSD Bulk DIF",
            str(default if default.exists() else Path.home()),
            "DIF Files (*.dif);;All Files (*)"
        )
        if not file_path:
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText("Updating RIR values...")
        try:
            stats = self.db_manager.update_rir_from_dif(
                file_path,
                progress_callback=self.progress_bar.setValue
            )
            QMessageBox.information(
                self, "RIR Update Complete",
                f"Updated: {stats.get('updated', 0)}\n"
                f"No RIR in block: {stats.get('no_rir', 0)}\n"
                f"Not in DB: {stats.get('missing', 0)}\n"
                f"Errors: {stats.get('errors', 0)}"
            )
            self.status_label.setText(f"RIR update complete ({stats.get('updated', 0)} minerals)")
        except Exception as e:
            QMessageBox.critical(self, "RIR Update Failed", str(e))
        finally:
            self.progress_bar.setVisible(False)
    
    def generate_dif_from_cif(self):
        """Generate DIF files from CIF files"""
        # Show conversion options dialog
        dialog = CifToDifDialog(self)
        if dialog.exec() == dialog.Accepted:
            cif_files = dialog.get_cif_files()
            output_dir = dialog.get_output_directory()
            wavelength = dialog.get_wavelength()
            
            if cif_files and output_dir:
                self.start_cif_to_dif_conversion(cif_files, output_dir, wavelength)
    
    def import_single_cif(self):
        """Import a single CIF file (deprecated - kept for compatibility)"""
        QMessageBox.warning(self, "Deprecated Feature", 
                          "CIF import is deprecated. Please use DIF import instead.\n\n"
                          "Use 'Generate DIF from CIF' to convert CIF files to DIF format first.")
        return
    
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
        ambient_only = self.ambient_only_check.isChecked()
        
        try:
            if search_type == "Mineral Name":
                results = self.db_manager.search_by_mineral_name(
                    search_term, max_results, ambient_only=ambient_only)
            elif search_type == "Chemical Formula":
                results = self.db_manager.search_by_formula(
                    search_term, max_results, ambient_only=ambient_only)
            elif search_type == "Elements":
                elements = [e.strip() for e in search_term.split(',')]
                results = self.db_manager.search_by_elements(
                    elements, exact_match=False, limit=max_results,
                    ambient_only=ambient_only)
            elif search_type == "Space Group":
                # Use formula search for space group (could be improved)
                results = self.db_manager.search_by_formula(
                    search_term, max_results, ambient_only=ambient_only)
            else:
                results = []
            
            self.display_search_results(results)
            status = f"Found {len(results)} results for '{search_term}'"
            if ambient_only:
                status += " (ambient conditions only)"
            self.status_label.setText(status)
            
        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"Error performing search: {e}")
            self.status_label.setText(f"Search error: {e}")
    
    def _cif_metadata_for(self, mineral):
        """CIF-derived fields for a result row, or {} when no CIF is available."""
        if not mineral:
            return {}
        return self.cif_repo.get_metadata(mineral.get('amcsd_id')) or {}

    def display_search_results(self, results):
        """Display search results in table"""
        self.search_results = results
        self.results_table.setRowCount(len(results))

        for i, result in enumerate(results):
            # The DIF-built database leaves formula/authors empty; fill from CIF
            meta = self._cif_metadata_for(result)

            def value(key, fallback='Unknown'):
                return result.get(key) or meta.get(key) or fallback

            authors = result.get('authors') or ', '.join(meta.get('authors') or [])

            def cell(key):
                number = result.get(key) or meta.get(key)
                return f"{number:.3f}" if number else 'N/A'

            self.results_table.setItem(i, 0, QTableWidgetItem(value('mineral_name')))
            self.results_table.setItem(i, 1, QTableWidgetItem(value('chemical_formula')))
            self.results_table.setItem(i, 2, QTableWidgetItem(value('space_group')))
            conditions = Conditions(
                pressure_gpa=result.get('pressure_gpa', meta.get('pressure_gpa')),
                temperature_k=result.get('temperature_k', meta.get('temperature_k')),
            )
            conditions_item = QTableWidgetItem(conditions.describe() or 'ambient')
            if not conditions.is_ambient:
                conditions_item.setForeground(QBrush(QColor('#B3541E')))
                conditions_item.setToolTip(
                    "Measured off-ambient: this cell is compressed or expanded, "
                    "so its lines sit at shifted 2θ"
                )

            self.results_table.setItem(i, 3, QTableWidgetItem(cell('cell_a')))
            self.results_table.setItem(i, 4, QTableWidgetItem(cell('cell_b')))
            self.results_table.setItem(i, 5, QTableWidgetItem(cell('cell_c')))
            self.results_table.setItem(i, 6, conditions_item)
            self.results_table.setItem(i, 7, QTableWidgetItem(authors or 'Unknown'))
            self.results_table.setItem(i, 8, QTableWidgetItem(result.get('amcsd_id', 'N/A')))

        self.results_table.resizeColumnsToContents()

    def _selected_mineral(self):
        """Mineral dict for the current row, or None."""
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.search_results):
            return None
        return self.search_results[row]

    def show_results_context_menu(self, position):
        """Right-click menu for the results table."""
        mineral = self._selected_mineral()
        if mineral is None:
            index = self.results_table.indexAt(position)
            if not index.isValid():
                return
            self.results_table.selectRow(index.row())
            mineral = self._selected_mineral()
            if mineral is None:
                return

        amcsd_id = mineral.get('amcsd_id')
        has_cif = self.cif_repo.has(amcsd_id)

        menu = QMenu(self)

        view_action = menu.addAction("View CIF…")
        view_action.setEnabled(has_cif)
        if not has_cif:
            view_action.setToolTip(f"No CIF in archive for AMCSD {amcsd_id}")

        copy_cif_action = menu.addAction("Copy CIF to Clipboard")
        copy_cif_action.setEnabled(has_cif)

        save_cif_action = menu.addAction("Save CIF As…")
        save_cif_action.setEnabled(has_cif)

        menu.addSeparator()
        match_action = menu.addAction("Use for Phase Matching")
        copy_name_action = menu.addAction("Copy Mineral Name")

        chosen = menu.exec_(self.results_table.viewport().mapToGlobal(position))
        if chosen is None:
            return

        if chosen == view_action:
            self.view_cif_content()
        elif chosen == copy_cif_action:
            text = self.cif_repo.get_cif_text(amcsd_id)
            if text:
                QApplication.clipboard().setText(text)
                self.status_label.setText(
                    f"Copied CIF for {mineral.get('mineral_name', 'mineral')}"
                )
        elif chosen == save_cif_action:
            self.save_cif_as(mineral)
        elif chosen == match_action:
            self.use_for_matching()
        elif chosen == copy_name_action:
            QApplication.clipboard().setText(str(mineral.get('mineral_name', '')))

    def save_cif_as(self, mineral=None):
        """Write the selected mineral's CIF to a chosen path."""
        mineral = mineral or self._selected_mineral()
        if mineral is None:
            return

        amcsd_id = mineral.get('amcsd_id')
        text = self.cif_repo.get_cif_text(amcsd_id)
        if not text:
            QMessageBox.warning(
                self, "No CIF",
                f"No CIF available in the archive for AMCSD ID {amcsd_id}."
            )
            return

        suggested = self.cif_repo.source_name(amcsd_id) or f"{amcsd_id}.cif"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CIF", suggested, "CIF files (*.cif);;All files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(text)
            self.status_label.setText(f"Saved CIF to {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    
    def show_mineral_details(self):
        """Show details for selected mineral, filling gaps from its CIF."""
        mineral = self._selected_mineral()
        if mineral is None:
            return

        meta = self._cif_metadata_for(mineral)
        amcsd_id = mineral.get('amcsd_id')
        has_cif = self.cif_repo.has(amcsd_id)

        def value(key, fallback='Unknown'):
            return mineral.get(key) or meta.get(key) or fallback

        def cell(key):
            number = mineral.get(key) or meta.get(key)
            return f"{number:g}" if number else 'N/A'

        authors = mineral.get('authors') or ', '.join(meta.get('authors') or [])
        volume = mineral.get('cell_volume') or meta.get('cell_volume')
        density = mineral.get('density') or meta.get('density')
        system = mineral.get('crystal_system') or meta.get('crystal_system')
        system_text = f"{system} (from cell metric)" if system and not mineral.get('crystal_system') else (system or 'Unknown')

        journal_bits = [b for b in (
            meta.get('journal'),
            f"v{meta['journal_volume']}" if meta.get('journal_volume') else '',
            f"p{meta['journal_page']}" if meta.get('journal_page') else '',
            meta.get('year'),
        ) if b]

        lines = [
            f"Mineral: {value('mineral_name')}",
            f"Formula: {value('chemical_formula')}",
            f"Space Group: {value('space_group')}",
            f"Crystal System: {system_text}",
            "",
            "Unit Cell:",
            f"  a = {cell('cell_a')} Å   b = {cell('cell_b')} Å   c = {cell('cell_c')} Å",
            f"  α = {cell('cell_alpha')}°   β = {cell('cell_beta')}°   γ = {cell('cell_gamma')}°",
            f"  Volume = {f'{volume:g} Å³' if volume else 'N/A'}"
            f"   Density = {f'{density:g} g/cm³' if density else 'N/A'}",
            "",
            "Publication:",
            f"  Authors: {authors or 'Unknown'}",
            f"  Journal: {' '.join(journal_bits) if journal_bits else 'Unknown'}",
        ]

        if meta.get('locality'):
            lines.append(f"  Locality: {meta['locality']}")

        conditions = Conditions(
            pressure_gpa=mineral.get('pressure_gpa', meta.get('pressure_gpa')),
            temperature_k=mineral.get('temperature_k', meta.get('temperature_k')),
        )
        lines += [
            "",
            "Measurement Conditions:",
            f"  {conditions.describe() or 'not annotated (assumed ambient)'}",
        ]
        if not conditions.is_ambient:
            lines.append(
                "  Off-ambient: cell parameters are shifted, so this entry is"
            )
            lines.append(
                "  excluded from matching unless high P/T entries are enabled."
            )
        if meta.get('title'):
            lines.append(f"  Source study: {meta['title']}")

        lines += [
            "",
            "Database:",
            f"  AMCSD ID: {amcsd_id or 'N/A'}   Local ID: {mineral.get('id', 'N/A')}",
            f"  CIF: {self.cif_repo.source_name(amcsd_id) if has_cif else 'not in archive'}",
        ]

        self.details_text.setText("\n".join(lines))
        self.use_for_matching_btn.setEnabled(True)
        self.view_cif_btn.setEnabled(has_cif)
        self.view_cif_btn.setToolTip(
            "" if has_cif else f"No CIF in archive for AMCSD {amcsd_id}"
        )
    
    def use_for_matching(self):
        """Use selected mineral for phase matching"""
        mineral = self._selected_mineral()
        if mineral is None:
            return

        meta = self._cif_metadata_for(mineral)
        authors = mineral.get('authors') or ', '.join(meta.get('authors') or []) or 'Unknown'
        journal = mineral.get('journal') or meta.get('journal') or 'Unknown'

        # Convert to format expected by matching tab
        phase_data = {
            'id': mineral.get('id'),  # Include database ID for pre-calculated patterns
            'mineral': mineral.get('mineral_name') or meta.get('mineral_name') or 'Unknown',
            'formula': mineral.get('chemical_formula') or meta.get('chemical_formula') or 'Unknown',
            'space_group': mineral.get('space_group') or meta.get('space_group') or 'Unknown',
            'cell_params': {
                'a': mineral.get('cell_a') or meta.get('cell_a'),
                'b': mineral.get('cell_b') or meta.get('cell_b'),
                'c': mineral.get('cell_c') or meta.get('cell_c'),
                'alpha': mineral.get('cell_alpha') or meta.get('cell_alpha'),
                'beta': mineral.get('cell_beta') or meta.get('cell_beta'),
                'gamma': mineral.get('cell_gamma') or meta.get('cell_gamma')
            },
            'amcsd_id': mineral.get('amcsd_id'),
            'cif_content': self.cif_repo.get_cif_text(mineral.get('amcsd_id')),
            'reference': f"{authors} - {journal}",
            'authors': authors,
            'journal': journal,
            'local_db': True  # Mark as from local database
        }
        
        # Emit signal to add to matching tab
        self.phases_selected.emit([phase_data])
        
        QMessageBox.information(self, "Added to Matching", 
                              f"Added {mineral.get('mineral_name')} to phase matching candidates.")
    
    def view_cif_content(self):
        """View CIF content for selected mineral"""
        mineral = self._selected_mineral()
        if mineral is None:
            return

        amcsd_id = mineral.get('amcsd_id')
        if not self.cif_repo.has(amcsd_id):
            QMessageBox.information(
                self, "No CIF",
                f"No CIF available in the archive for AMCSD ID {amcsd_id}."
            )
            return

        dialog = CifViewerDialog(
            amcsd_id, mineral.get('mineral_name', 'Unknown'), parent=self
        )
        dialog.exec_()
    
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
    
    def recalculate_all_patterns(self):
        """Recalculate all existing diffraction patterns with improved intensity calculations"""
        # Confirm with user first
        reply = QMessageBox.question(self, "Recalculate Patterns", 
                                   "This will recalculate all existing diffraction patterns with improved intensity calculations.\n\n"
                                   "This process may take some time depending on the number of patterns.\n"
                                   "The existing patterns will be replaced with more accurate ones.\n\n"
                                   "Do you want to continue?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        # Start recalculation in background thread
        self.recalc_thread = RecalculationThread(self.db_manager)
        self.recalc_thread.progress_updated.connect(self.progress_bar.setValue)
        self.recalc_thread.status_updated.connect(self.status_label.setText)
        self.recalc_thread.recalculation_complete.connect(self.on_recalculation_complete)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Recalculating diffraction patterns with improved intensities...")
        
        self.recalc_thread.start()
    
    def on_recalculation_complete(self, successful_count):
        """Handle completion of pattern recalculation"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Recalculation complete: {successful_count} patterns updated")
        
        QMessageBox.information(self, "Recalculation Complete",
                              f"Successfully recalculated {successful_count} diffraction patterns with improved intensities.\n\n"
                              f"Phase matching should now be more accurate with better intensity calculations!")
        
        # Update database stats
        self.update_database_stats()
    
    def validate_pattern_quality(self):
        """Validate pattern quality by comparing old vs new calculations"""
        self.status_label.setText("Validating pattern quality...")
        
        try:
            # Run validation on a sample of patterns
            validation_results = self.db_manager.validate_pattern_intensities(sample_size=20)
            
            if 'error' in validation_results:
                QMessageBox.warning(self, "Validation Error", f"Could not validate patterns: {validation_results['error']}")
                self.status_label.setText("")
                return
            
            # Format results message
            total = validation_results.get('total_validated', 0)
            improved = validation_results.get('improved_patterns', 0)
            similar = validation_results.get('similar_patterns', 0)
            failed = validation_results.get('failed_validations', 0)
            avg_improvement = validation_results.get('average_improvement', 0.0)
            
            message = f"""Pattern Quality Validation Results:

Patterns Validated: {total}
Significantly Improved: {improved} ({improved/max(total,1)*100:.1f}%)
Similar Quality: {similar} ({similar/max(total,1)*100:.1f}%)
Failed Validations: {failed}

Average Improvement: {avg_improvement:.1f}%

The improved intensity calculations use:
• Proper structure factors from atomic positions
• Enhanced scattering factor tables
• Lorentz-polarization corrections
• Thermal factors (Debye-Waller)
• Multiplicity considerations

This provides more realistic and accurate intensity distributions
for better phase identification results.
            """.strip()
            
            QMessageBox.information(self, "Pattern Quality Validation", message)
            self.status_label.setText("Validation complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Error during validation: {str(e)}")
            self.status_label.setText("Validation failed")
    
    def cleanup_non_cu_patterns(self):
        """Remove all non-Cu Kα patterns from the database"""
        # First show current wavelength distribution
        wavelength_stats = self.db_manager.get_wavelength_distribution()
        
        if not wavelength_stats or wavelength_stats.get('total_patterns', 0) == 0:
            QMessageBox.information(self, "No Patterns", "No diffraction patterns found in database.")
            return
        
        # Show current distribution
        distribution = wavelength_stats.get('wavelength_distribution', {})
        total = wavelength_stats.get('total_patterns', 0)
        
        current_info = "Current wavelength distribution:\n\n"
        cu_patterns = 0
        non_cu_patterns = 0
        
        for wavelength, count in sorted(distribution.items()):
            current_info += f"λ = {wavelength:.4f} Å: {count} patterns\n"
            if abs(wavelength - 1.5406) < 0.0001:  # Cu Kα
                cu_patterns = count
            else:
                non_cu_patterns += count
        
        current_info += f"\nTotal: {total} patterns"
        current_info += f"\nCu Kα patterns: {cu_patterns}"
        current_info += f"\nNon-Cu Kα patterns: {non_cu_patterns}"
        
        if non_cu_patterns == 0:
            QMessageBox.information(self, "Already Optimized", 
                                  f"{current_info}\n\nDatabase is already optimized with only Cu Kα patterns!")
            return
        
        # Confirm with user
        reply = QMessageBox.question(self, "Remove Non-Cu Kα Patterns", 
                                   f"{current_info}\n\n"
                                   f"This will remove {non_cu_patterns} non-Cu Kα patterns, keeping only {cu_patterns} Cu Kα patterns.\n\n"
                                   f"Other wavelengths will be calculated on-demand using Bragg's law conversion.\n"
                                   f"This will make recalculation much faster!\n\n"
                                   f"Do you want to continue?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        # Perform cleanup
        self.status_label.setText("Removing non-Cu Kα patterns...")
        
        try:
            removed_count = self.db_manager.cleanup_non_cu_patterns()
            
            if removed_count > 0:
                QMessageBox.information(self, "Cleanup Complete",
                                      f"Successfully removed {removed_count} non-Cu Kα patterns.\n\n"
                                      f"Database is now optimized for Cu Kα reference wavelength only.\n"
                                      f"Other wavelengths will be calculated on-demand using Bragg's law.\n\n"
                                      f"Recalculation will now be much faster!")
                
                # Update database stats
                self.update_database_stats()
                self.status_label.setText(f"Cleanup complete: {removed_count} patterns removed")
            else:
                self.status_label.setText("No patterns were removed")
                
        except Exception as e:
            QMessageBox.critical(self, "Cleanup Error", f"Error during cleanup: {str(e)}")
            self.status_label.setText("Cleanup failed")
    
    def show_wavelength_distribution(self):
        """Show the distribution of wavelengths in stored patterns"""
        self.status_label.setText("Getting wavelength distribution...")
        
        try:
            wavelength_stats = self.db_manager.get_wavelength_distribution()
            
            if not wavelength_stats or wavelength_stats.get('total_patterns', 0) == 0:
                QMessageBox.information(self, "No Patterns", "No diffraction patterns found in database.")
                self.status_label.setText("")
                return
            
            distribution = wavelength_stats.get('wavelength_distribution', {})
            total = wavelength_stats.get('total_patterns', 0)
            unique_wavelengths = wavelength_stats.get('unique_wavelengths', 0)
            
            # Format distribution message
            message = f"Wavelength Distribution in Database:\n\n"
            message += f"Total Patterns: {total}\n"
            message += f"Unique Wavelengths: {unique_wavelengths}\n\n"
            
            cu_patterns = 0
            non_cu_patterns = 0
            
            # Common wavelength names
            wavelength_names = {
                1.5406: "Cu Kα1",
                1.5418: "Cu Kα",
                1.7890: "Co Kα1", 
                1.7902: "Co Kα",
                1.9373: "Fe Kα1",
                2.2897: "Cr Kα1",
                0.7107: "Mo Kα1",
                0.2401: "Mo Kα"
            }
            
            for wavelength, count in sorted(distribution.items()):
                name = wavelength_names.get(wavelength, "Unknown")
                percentage = (count / total) * 100
                message += f"{name} (λ = {wavelength:.4f} Å): {count} patterns ({percentage:.1f}%)\n"
                
                if abs(wavelength - 1.5406) < 0.0001:  # Cu Kα
                    cu_patterns = count
                else:
                    non_cu_patterns += count
            
            message += f"\nSummary:\n"
            message += f"Cu Kα patterns: {cu_patterns} ({cu_patterns/total*100:.1f}%)\n"
            message += f"Non-Cu Kα patterns: {non_cu_patterns} ({non_cu_patterns/total*100:.1f}%)\n\n"
            
            if non_cu_patterns > 0:
                message += f"💡 Recommendation: Remove non-Cu Kα patterns to optimize database.\n"
                message += f"   Other wavelengths can be calculated on-demand using Bragg's law."
            else:
                message += f"✅ Database is optimized with only Cu Kα reference patterns!"
            
            QMessageBox.information(self, "Wavelength Distribution", message)
            self.status_label.setText("Distribution analysis complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Distribution Error", f"Error getting wavelength distribution: {str(e)}")
            self.status_label.setText("Distribution analysis failed")
    
    def start_cif_to_dif_conversion(self, cif_files, output_dir, wavelength):
        """Start CIF to DIF conversion in background thread"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        
        self.conversion_thread = CifToDifConversionThread(
            self.db_manager, cif_files, output_dir, wavelength
        )
        self.conversion_thread.progress_updated.connect(self.progress_bar.setValue)
        self.conversion_thread.status_updated.connect(self.status_label.setText)
        self.conversion_thread.conversion_complete.connect(self.conversion_finished)
        self.conversion_thread.start()
    
    def conversion_finished(self, converted_count, output_dir):
        """Handle CIF to DIF conversion completion"""
        self.progress_bar.setVisible(False)
        
        if converted_count > 0:
            self.status_label.setText(f"Conversion complete. Generated {converted_count} DIF files.")
            
            # Ask if user wants to import the generated DIF files
            reply = QMessageBox.question(
                self, "Conversion Complete", 
                f"Successfully converted {converted_count} CIF files to DIF format!\n\n"
                f"DIF files saved to: {output_dir}\n\n"
                f"Would you like to import these DIF files into the database now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.import_converted_dif_files(output_dir)
        else:
            self.status_label.setText("Conversion failed. No DIF files were generated.")
            QMessageBox.warning(
                self, "Conversion Failed", 
                "No DIF files were generated. Check the console for error details."
            )
    
    def import_converted_dif_files(self, output_dir):
        """Import the converted DIF files into the database"""
        from pathlib import Path
        
        dif_files = list(Path(output_dir).glob("*.dif"))
        if dif_files:
            self.import_directory_dif_files(dif_files)


class CifToDifConversionThread(QThread):
    """Thread for converting CIF files to DIF format"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    conversion_complete = pyqtSignal(int, str)  # count, output_path
    
    def __init__(self, db_manager, cif_files, output_dir, wavelength=1.5406):
        super().__init__()
        self.db_manager = db_manager
        self.cif_files = cif_files if isinstance(cif_files, list) else [cif_files]
        self.output_dir = output_dir
        self.wavelength = wavelength
    
    def run(self):
        """Run the CIF to DIF conversion in a separate thread"""
        try:
            self.status_updated.emit("Starting CIF to DIF conversion...")
            converted_count = 0
            
            for i, cif_file in enumerate(self.cif_files):
                try:
                    self.status_updated.emit(f"Converting {os.path.basename(cif_file)}...")
                    
                    # Convert CIF to DIF
                    dif_path = self.convert_cif_to_dif(cif_file)
                    if dif_path:
                        converted_count += 1
                    
                    # Update progress
                    progress = int((i + 1) / len(self.cif_files) * 100)
                    self.progress_updated.emit(progress)
                    
                except Exception as e:
                    self.status_updated.emit(f"Error converting {os.path.basename(cif_file)}: {str(e)}")
            
            self.conversion_complete.emit(converted_count, self.output_dir)
            
        except Exception as e:
            self.status_updated.emit(f"Conversion error: {str(e)}")
            self.conversion_complete.emit(0, "")
    
    def convert_cif_to_dif(self, cif_file_path):
        """Convert a single CIF file to DIF format"""
        try:
            from utils.cif_parser import CIFParser
            
            # Read CIF file
            with open(cif_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                cif_content = f.read()
            
            # Calculate XRD pattern
            cif_parser = CIFParser()
            pattern = cif_parser.calculate_xrd_pattern_from_cif(
                cif_content, 
                wavelength=self.wavelength,
                max_2theta=90.0,
                min_d=0.5
            )
            
            if not pattern or len(pattern.get('two_theta', [])) == 0:
                print(f"Failed to calculate pattern for {cif_file_path}")
                return None
            
            # Parse CIF for metadata
            cif_data = cif_parser.parse_content(cif_content)
            
            # Extract mineral information
            mineral_info = self.extract_mineral_info(cif_data, cif_file_path)
            
            # Generate DIF file
            dif_filename = os.path.splitext(os.path.basename(cif_file_path))[0] + '.dif'
            dif_path = os.path.join(self.output_dir, dif_filename)
            
            self.write_dif_file(dif_path, mineral_info, pattern, self.wavelength)
            
            return dif_path
            
        except Exception as e:
            print(f"Error converting {cif_file_path}: {e}")
            return None
    
    def extract_mineral_info(self, cif_data, cif_file_path):
        """Extract mineral information from parsed CIF data"""
        info = {
            'mineral_name': 'Unknown',
            'chemical_formula': 'Unknown',
            'space_group': 'Unknown',
            'cell_a': 0.0,
            'cell_b': 0.0,
            'cell_c': 0.0,
            'cell_alpha': 90.0,
            'cell_beta': 90.0,
            'cell_gamma': 90.0,
            'authors': 'Unknown',
            'journal': 'Unknown'
        }
        
        # Extract from first data block
        if cif_data:
            first_block = next(iter(cif_data.values()), {})
            
            # Mineral name
            for key in ['_chemical_name_mineral', '_pd_phase_name']:
                if key in first_block:
                    info['mineral_name'] = str(first_block[key]).strip().strip('"\'')
                    break
            
            # Chemical formula
            if '_chemical_formula_sum' in first_block:
                info['chemical_formula'] = str(first_block['_chemical_formula_sum']).strip().strip('"\'')
            
            # Space group
            for key in ['_symmetry_space_group_name_H-M', '_space_group_name_H-M_alt']:
                if key in first_block:
                    info['space_group'] = str(first_block[key]).strip().strip('"\'')
                    break
            
            # Unit cell parameters
            cell_params = {
                '_cell_length_a': 'cell_a',
                '_cell_length_b': 'cell_b', 
                '_cell_length_c': 'cell_c',
                '_cell_angle_alpha': 'cell_alpha',
                '_cell_angle_beta': 'cell_beta',
                '_cell_angle_gamma': 'cell_gamma'
            }
            
            for cif_key, info_key in cell_params.items():
                if cif_key in first_block:
                    try:
                        value = str(first_block[cif_key]).split('(')[0]  # Remove uncertainty
                        info[info_key] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            # Publication info
            if '_publ_author_name' in first_block:
                info['authors'] = str(first_block['_publ_author_name']).strip().strip('"\'')
            if '_journal_name_full' in first_block:
                info['journal'] = str(first_block['_journal_name_full']).strip().strip('"\'')
        
        # Use filename as fallback for mineral name
        if info['mineral_name'] == 'Unknown':
            info['mineral_name'] = os.path.splitext(os.path.basename(cif_file_path))[0]
        
        return info
    
    def write_dif_file(self, dif_path, mineral_info, pattern, wavelength):
        """Write DIF file with pseudo-Voigt profiles"""
        with open(dif_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# DIF file generated from CIF\n")
            f.write(f"# Mineral: {mineral_info['mineral_name']}\n")
            f.write(f"# Formula: {mineral_info['chemical_formula']}\n")
            f.write(f"# Wavelength: {wavelength:.4f} Angstrom (Cu Ka1)\n")
            f.write(f"#\n")
            
            # CIF-style metadata
            f.write(f"data_{mineral_info['mineral_name'].replace(' ', '_')}\n")
            f.write(f"\n")
            f.write(f"_pd_phase_name                         '{mineral_info['mineral_name']}'\n")
            f.write(f"_chemical_name_mineral                 '{mineral_info['mineral_name']}'\n")
            f.write(f"_chemical_formula_sum                  '{mineral_info['chemical_formula']}'\n")
            f.write(f"_symmetry_space_group_name_H-M         '{mineral_info['space_group']}'\n")
            f.write(f"\n")
            
            # Unit cell parameters
            f.write(f"_cell_length_a                         {mineral_info['cell_a']:.6f}\n")
            f.write(f"_cell_length_b                         {mineral_info['cell_b']:.6f}\n")
            f.write(f"_cell_length_c                         {mineral_info['cell_c']:.6f}\n")
            f.write(f"_cell_angle_alpha                      {mineral_info['cell_alpha']:.6f}\n")
            f.write(f"_cell_angle_beta                       {mineral_info['cell_beta']:.6f}\n")
            f.write(f"_cell_angle_gamma                      {mineral_info['cell_gamma']:.6f}\n")
            f.write(f"\n")
            
            # Radiation wavelength
            f.write(f"_diffrn_radiation_wavelength           {wavelength:.6f}\n")
            f.write(f"_diffrn_radiation_type                 'Cu Ka1'\n")
            f.write(f"\n")
            
            # Publication info
            f.write(f"_publ_author_name                      '{mineral_info['authors']}'\n")
            f.write(f"_journal_name_full                     '{mineral_info['journal']}'\n")
            f.write(f"\n")
            
            # Diffraction data loop
            f.write(f"loop_\n")
            f.write(f"_pd_proc_2theta_corrected\n")
            f.write(f"_pd_proc_d_spacing\n")
            f.write(f"_pd_proc_intensity_net\n")
            
            # Write diffraction data with pseudo-Voigt profiles
            two_theta = pattern['two_theta']
            d_spacing = pattern['d_spacing']
            intensity = pattern['intensity']
            
            # Generate pseudo-Voigt profile data
            profile_data = self.generate_pseudo_voigt_profile(
                two_theta, intensity, fwhm_base=0.1
            )
            
            # Write profile data
            for tt, d_val, int_val in zip(profile_data['two_theta'], 
                                        profile_data['d_spacing'], 
                                        profile_data['intensity']):
                f.write(f"{tt:8.4f} {d_val:8.6f} {int_val:8.2f}\n")
    
    def generate_pseudo_voigt_profile(self, peak_positions, peak_intensities, 
                                    fwhm_base=0.1, points_per_peak=21):
        """Generate pseudo-Voigt peak profiles for realistic DIF patterns"""
        import numpy as np
        
        # Calculate FWHM for each peak (angle-dependent)
        fwhm_values = fwhm_base + 0.001 * peak_positions  # Increase FWHM with angle
        
        # Generate profile points
        all_two_theta = []
        all_intensities = []
        
        for pos, intensity, fwhm in zip(peak_positions, peak_intensities, fwhm_values):
            # Generate points around each peak
            half_width = fwhm * 2.5  # Cover ±2.5 FWHM
            peak_range = np.linspace(pos - half_width, pos + half_width, points_per_peak)
            
            # Pseudo-Voigt profile (mix of Gaussian and Lorentzian)
            eta = 0.5  # Mixing parameter (0=pure Gaussian, 1=pure Lorentzian)
            
            # Gaussian component
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            gaussian = np.exp(-0.5 * ((peak_range - pos) / sigma) ** 2)
            
            # Lorentzian component  
            gamma = fwhm / 2
            lorentzian = gamma**2 / ((peak_range - pos)**2 + gamma**2)
            
            # Pseudo-Voigt combination
            profile = intensity * ((1 - eta) * gaussian + eta * lorentzian)
            
            all_two_theta.extend(peak_range)
            all_intensities.extend(profile)
        
        # Sort by 2theta and remove duplicates
        combined = list(zip(all_two_theta, all_intensities))
        combined.sort(key=lambda x: x[0])
        
        # Remove very close points and keep highest intensity
        filtered_data = []
        last_angle = -999
        
        for angle, intensity in combined:
            if angle - last_angle > 0.01:  # Minimum 0.01° separation
                filtered_data.append((angle, intensity))
                last_angle = angle
            else:
                # Keep higher intensity
                if intensity > filtered_data[-1][1]:
                    filtered_data[-1] = (angle, intensity)
        
        # Extract final arrays
        final_two_theta = np.array([x[0] for x in filtered_data])
        final_intensities = np.array([x[1] for x in filtered_data])
        
        # Calculate d-spacings using Bragg's law
        wavelength = 1.5406  # Cu Ka1
        final_d_spacing = wavelength / (2 * np.sin(np.radians(final_two_theta / 2)))
        
        return {
            'two_theta': final_two_theta,
            'd_spacing': final_d_spacing,
            'intensity': final_intensities
        }


class BulkDifImportThread(QThread):
    """Thread for bulk importing AMCSD DIF file"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    import_complete = pyqtSignal(int)
    
    def __init__(self, db_manager, dif_file_path):
        super().__init__()
        self.db_manager = db_manager
        self.dif_file_path = dif_file_path
    
    def run(self):
        """Run the bulk import in a separate thread"""
        try:
            self.status_updated.emit("Starting bulk AMCSD DIF import...")
            imported_count = self.db_manager.bulk_import_amcsd_dif(
                self.dif_file_path,
                progress_callback=self.progress_updated.emit
            )
            self.import_complete.emit(imported_count)
            
        except Exception as e:
            self.status_updated.emit(f"Import error: {str(e)}")
            self.import_complete.emit(0)


class RecalculationThread(QThread):
    """Thread for recalculating diffraction patterns with improved intensities"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    recalculation_complete = pyqtSignal(int)
    
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
    
    def run(self):
        """Run the recalculation in a separate thread"""
        try:
            self.status_updated.emit("Starting pattern recalculation with improved intensities...")
            successful_count = self.db_manager.recalculate_all_diffraction_patterns(
                progress_callback=self.progress_updated.emit
            )
            self.recalculation_complete.emit(successful_count)
            
        except Exception as e:
            self.status_updated.emit(f"Recalculation error: {str(e)}")
            self.recalculation_complete.emit(0)


class CifToDifDialog(QMessageBox):
    """Dialog for CIF to DIF conversion options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CIF to DIF Conversion")
        self.setText("Convert CIF files to DIF format with pseudo-Voigt profiles")
        
        # Create custom widget for options
        self.options_widget = QWidget()
        layout = QVBoxLayout(self.options_widget)
        
        # File selection
        file_group = QGroupBox("CIF Files")
        file_layout = QVBoxLayout(file_group)
        
        # Single file option
        self.single_file_radio = QRadioButton("Single CIF file")
        self.single_file_radio.setChecked(True)
        self.single_file_radio.toggled.connect(self.update_file_selection)
        file_layout.addWidget(self.single_file_radio)
        
        self.single_file_btn = QPushButton("Select CIF File...")
        self.single_file_btn.clicked.connect(self.select_single_file)
        file_layout.addWidget(self.single_file_btn)
        
        self.single_file_label = QLabel("No file selected")
        self.single_file_label.setObjectName("mutedLabel")
        file_layout.addWidget(self.single_file_label)
        
        # Multiple files option
        self.multiple_files_radio = QRadioButton("Multiple CIF files (directory)")
        self.multiple_files_radio.toggled.connect(self.update_file_selection)
        file_layout.addWidget(self.multiple_files_radio)
        
        self.multiple_files_btn = QPushButton("Select Directory...")
        self.multiple_files_btn.clicked.connect(self.select_directory)
        self.multiple_files_btn.setEnabled(False)
        file_layout.addWidget(self.multiple_files_btn)
        
        self.multiple_files_label = QLabel("No directory selected")
        self.multiple_files_label.setObjectName("mutedLabel")
        file_layout.addWidget(self.multiple_files_label)
        
        layout.addWidget(file_group)
        
        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QVBoxLayout(output_group)
        
        self.output_btn = QPushButton("Select Output Directory...")
        self.output_btn.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_btn)
        
        self.output_label = QLabel("No directory selected")
        self.output_label.setObjectName("mutedLabel")
        output_layout.addWidget(self.output_label)
        
        layout.addWidget(output_group)
        
        # Wavelength selection
        wavelength_group = QGroupBox("X-ray Wavelength")
        wavelength_layout = QHBoxLayout(wavelength_group)
        
        wavelength_layout.addWidget(QLabel("Wavelength (Å):"))
        
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.addItems([
            "1.5406 (Cu Kα1)",
            "1.5418 (Cu Kα average)", 
            "0.7107 (Mo Kα1)",
            "1.7890 (Co Kα1)",
            "Custom..."
        ])
        self.wavelength_combo.currentTextChanged.connect(self.wavelength_changed)
        wavelength_layout.addWidget(self.wavelength_combo)
        
        self.custom_wavelength = QLineEdit()
        self.custom_wavelength.setPlaceholderText("Enter wavelength...")
        self.custom_wavelength.setVisible(False)
        wavelength_layout.addWidget(self.custom_wavelength)
        
        layout.addWidget(wavelength_group)
        
        # Add to dialog
        self.layout().addWidget(self.options_widget, 1, 0, 1, self.layout().columnCount())
        
        # Set buttons
        self.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        
        # Initialize
        self.cif_files = []
        self.output_directory = ""
        
    def update_file_selection(self):
        """Update file selection UI based on radio button"""
        single_selected = self.single_file_radio.isChecked()
        
        self.single_file_btn.setEnabled(single_selected)
        self.multiple_files_btn.setEnabled(not single_selected)
        
        if single_selected:
            self.multiple_files_label.setText("No directory selected")
            self.multiple_files_label.setObjectName("mutedLabel")
            self.multiple_files_label.style().unpolish(self.multiple_files_label)
            self.multiple_files_label.style().polish(self.multiple_files_label)
        else:
            self.single_file_label.setText("No file selected")
            self.single_file_label.setObjectName("mutedLabel")
            self.single_file_label.style().unpolish(self.single_file_label)
            self.single_file_label.style().polish(self.single_file_label)
    
    def select_single_file(self):
        """Select a single CIF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CIF File", "", "CIF Files (*.cif);;All Files (*)"
        )
        
        if file_path:
            self.cif_files = [file_path]
            self.single_file_label.setText(os.path.basename(file_path))
            self.single_file_label.setObjectName("")
            self.single_file_label.setStyleSheet("")
            self.single_file_label.style().unpolish(self.single_file_label)
            self.single_file_label.style().polish(self.single_file_label)
    
    def select_directory(self):
        """Select directory with CIF files"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory with CIF Files"
        )
        
        if dir_path:
            from pathlib import Path
            cif_files = list(Path(dir_path).glob("*.cif"))
            
            if cif_files:
                self.cif_files = [str(f) for f in cif_files]
                self.multiple_files_label.setText(f"{len(cif_files)} CIF files found")
                self.multiple_files_label.setObjectName("")
                self.multiple_files_label.setStyleSheet("")
            else:
                self.multiple_files_label.setText("No CIF files found in directory")
                self.multiple_files_label.setObjectName("statusErrorLabel")
                self.multiple_files_label.setStyleSheet("")
                self.cif_files = []
            self.multiple_files_label.style().unpolish(self.multiple_files_label)
            self.multiple_files_label.style().polish(self.multiple_files_label)
    
    def select_output_directory(self):
        """Select output directory for DIF files"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for DIF Files"
        )
        
        if dir_path:
            self.output_directory = dir_path
            self.output_label.setText(dir_path)
            self.output_label.setObjectName("")
            self.output_label.setStyleSheet("")
            self.output_label.style().unpolish(self.output_label)
            self.output_label.style().polish(self.output_label)
    
    def wavelength_changed(self, text):
        """Handle wavelength selection change"""
        if "Custom" in text:
            self.custom_wavelength.setVisible(True)
            self.custom_wavelength.setFocus()
        else:
            self.custom_wavelength.setVisible(False)
    
    def get_cif_files(self):
        """Get selected CIF files"""
        return self.cif_files
    
    def get_output_directory(self):
        """Get selected output directory"""
        return self.output_directory
    
    def get_wavelength(self):
        """Get selected wavelength"""
        text = self.wavelength_combo.currentText()
        
        if "Custom" in text:
            try:
                return float(self.custom_wavelength.text())
            except ValueError:
                return 1.5406  # Default to Cu Kα1
        else:
            # Extract wavelength from text
            return float(text.split()[0])
    
    def accept(self):
        """Validate inputs before accepting"""
        if not self.cif_files:
            QMessageBox.warning(self, "No Files Selected", "Please select CIF files to convert.")
            return
        
        if not self.output_directory:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory for DIF files.")
            return
        
        # Validate custom wavelength if selected
        if "Custom" in self.wavelength_combo.currentText():
            try:
                wavelength = float(self.custom_wavelength.text())
                if wavelength <= 0:
                    raise ValueError("Wavelength must be positive")
            except ValueError:
                QMessageBox.warning(self, "Invalid Wavelength", "Please enter a valid positive wavelength value.")
                return
        
        super().accept()
