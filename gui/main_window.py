"""
Main window for XRD Phase Matching application
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QMenuBar, QStatusBar, QToolBar,
                             QAction, QFileDialog, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

from .pattern_tab import PatternTab
from .processing_tab import ProcessingTab
from .database_tab import DatabaseTab
from .local_database_tab import LocalDatabaseTab
from .pattern_search_tab import PatternSearchTab
from .matching_tab import MatchingTab
from .visualization_tab import VisualizationTab
from .settings_tab import SettingsTab

class XRDMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_statusbar()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("XRD Phase Matcher v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.pattern_tab = PatternTab()
        self.processing_tab = ProcessingTab()
        self.database_tab = DatabaseTab()
        self.local_database_tab = LocalDatabaseTab()
        self.pattern_search_tab = PatternSearchTab()
        self.matching_tab = MatchingTab()
        self.visualization_tab = VisualizationTab()
        self.settings_tab = SettingsTab()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.pattern_tab, "Pattern Data")
        self.tab_widget.addTab(self.processing_tab, "Data Processing")
        self.tab_widget.addTab(self.database_tab, "AMCSD Search")
        self.tab_widget.addTab(self.local_database_tab, "Local Database")
        self.tab_widget.addTab(self.pattern_search_tab, "Pattern Search")
        self.tab_widget.addTab(self.matching_tab, "Phase Matching")
        self.tab_widget.addTab(self.visualization_tab, "Visualization & Export")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        
        # Connect signals - order matters: reset first, then set new data
        # Reset tabs when new pattern is loaded to clear previous data
        self.pattern_tab.pattern_loaded.connect(self.matching_tab.reset_for_new_pattern)
        self.pattern_tab.pattern_loaded.connect(self.pattern_search_tab.reset_for_new_pattern)
        
        # Then set the new pattern data
        self.pattern_tab.pattern_loaded.connect(self.processing_tab.set_pattern_data)
        self.pattern_tab.pattern_loaded.connect(self.matching_tab.set_experimental_pattern)
        self.pattern_tab.pattern_loaded.connect(self.pattern_search_tab.set_experimental_pattern)
        
        self.processing_tab.pattern_processed.connect(self.matching_tab.set_experimental_pattern)
        self.processing_tab.pattern_processed.connect(self.pattern_search_tab.set_experimental_pattern)
        self.processing_tab.peaks_found.connect(self.matching_tab.set_experimental_peaks)
        self.processing_tab.peaks_found.connect(self.pattern_search_tab.set_experimental_peaks)
        self.database_tab.phases_selected.connect(self.matching_tab.add_reference_phases)
        self.local_database_tab.phases_selected.connect(self.matching_tab.add_reference_phases)
        self.pattern_search_tab.phases_found.connect(self.matching_tab.add_reference_phases)
        
        # Connect Le Bail refinement components
        # Share the multi-phase analyzer between tabs for refined phase caching
        self.pattern_search_tab.set_multi_phase_analyzer(self.matching_tab.multi_phase_analyzer)
        self.visualization_tab.set_multi_phase_analyzer(self.matching_tab.multi_phase_analyzer)
        
        # Connect visualization tab import button to matching tab data
        self.visualization_tab.import_btn.clicked.connect(self.import_to_visualization)
        
    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QAction('&Open Pattern...', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_pattern)
        file_menu.addAction(open_action)
        
        save_action = QAction('&Save Results...', self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        peak_find_action = QAction('&Find Peaks', self)
        peak_find_action.triggered.connect(self.processing_tab.find_peaks)
        tools_menu.addAction(peak_find_action)
        
        tools_menu.addSeparator()
        
        pattern_search_action = QAction('&Pattern Search', self)
        pattern_search_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.pattern_search_tab))
        tools_menu.addAction(pattern_search_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_toolbar(self):
        """Setup application toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_pattern)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        find_peaks_action = QAction('Find Peaks', self)
        find_peaks_action.triggered.connect(self.processing_tab.find_peaks)
        toolbar.addAction(find_peaks_action)
        
    def setup_statusbar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def open_pattern(self):
        """Open a diffraction pattern file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Diffraction Pattern",
            "",
            "Data files (*.xy *.xye *.txt *.dat *.csv);;All files (*.*)"
        )
        
        if file_path:
            self.pattern_tab.load_pattern(file_path)
            self.status_bar.showMessage(f"Loaded pattern: {file_path}")
            
    def save_results(self):
        """Save matching results"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            self.matching_tab.save_results(file_path)
            self.status_bar.showMessage(f"Results saved: {file_path}")
            
    def import_to_visualization(self):
        """Import data from matching tab to visualization tab"""
        # Get experimental pattern and matched phases from matching tab
        experimental_pattern = self.matching_tab.experimental_pattern
        
        if not experimental_pattern:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load a pattern and perform phase matching first."
            )
            return
        
        # Get selected phases from matching tab
        matched_phases = self.matching_tab.get_selected_phases()
        
        if not matched_phases:
            QMessageBox.warning(
                self,
                "No Phases",
                "Please select phases in the Phase Matching tab first."
            )
            return
        
        # Import to visualization tab
        self.visualization_tab.set_data(experimental_pattern, matched_phases)
        
        # Switch to visualization tab
        self.tab_widget.setCurrentWidget(self.visualization_tab)
        
        self.status_bar.showMessage(f"Imported {len(matched_phases)} phase(s) to visualization")
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About XRD Phase Matcher",
            """<h3>XRD Phase Matcher v1.0</h3>
            <p>A comprehensive X-ray diffraction phase matching application</p>
            <p><b>Features:</b></p>
            <ul>
            <li>Load and analyze diffraction patterns</li>
            <li>Search AMCSD crystal structure database</li>
            <li>Multiple wavelength support</li>
            <li>Automated phase matching</li>
            <li>Le Bail refinement</li>
            <li>Advanced visualization and export</li>
            </ul>
            <p>Built with PyQt5 and scientific Python libraries</p>"""
        )
