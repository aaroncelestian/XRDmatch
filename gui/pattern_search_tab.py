"""
Pattern Search tab for XRD phase identification
Implements both peak-based and correlation-based pattern matching
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QGroupBox,
                             QDoubleSpinBox, QComboBox, QTextEdit, QSpinBox,
                             QSplitter, QProgressBar, QCheckBox, QMessageBox,
                             QTabWidget, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from utils.pattern_search import PatternSearchEngine
from utils.fast_pattern_search import FastPatternSearchEngine
import time

class PatternSearchThread(QThread):
    """Thread for pattern search operations"""
    
    search_complete = pyqtSignal(list)
    progress_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, search_engine, experimental_data, search_method, search_params):
        super().__init__()
        self.search_engine = search_engine
        self.experimental_data = experimental_data
        self.search_method = search_method
        self.search_params = search_params
        
    def run(self):
        """Run pattern search in separate thread"""
        try:
            self.progress_updated.emit("Starting pattern search...")
            
            if self.search_method == 'peaks':
                results = self.search_engine.search_by_peaks(
                    self.experimental_data,
                    **self.search_params
                )
            elif self.search_method == 'correlation':
                results = self.search_engine.search_by_correlation(
                    self.experimental_data,
                    **self.search_params
                )
            elif self.search_method == 'combined':
                results = self.search_engine.combined_search(
                    self.experimental_data,
                    **self.search_params
                )
            else:
                raise ValueError(f"Unknown search method: {self.search_method}")
            
            self.progress_updated.emit(f"Search complete: {len(results)} matches found")
            self.search_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class PatternSearchTab(QWidget):
    """Tab for pattern-based phase identification"""
    
    # Signal to send results to matching tab
    phases_found = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.experimental_pattern = None
        self.experimental_peaks = None
        self.search_engine = PatternSearchEngine()
        self.fast_search_engine = FastPatternSearchEngine()
        self.search_results = []
        self.multi_phase_analyzer = None  # Will be set by main window
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready for pattern search")
        layout.addWidget(self.status_label)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left side: Results
        results_panel = self.create_results_panel()
        main_splitter.addWidget(results_panel)
        
        # Right side: Plot
        plot_panel = self.create_plot_panel()
        main_splitter.addWidget(plot_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([500, 500])
        
    def create_control_panel(self):
        """Create the control panel with search parameters"""
        group = QGroupBox("Pattern Search Parameters")
        layout = QVBoxLayout(group)
        
        # Create tab widget for different search methods
        self.search_tabs = QTabWidget()
        layout.addWidget(self.search_tabs)
        
        # Peak-based search tab
        peak_tab = QWidget()
        peak_layout = QFormLayout(peak_tab)
        
        self.peak_tolerance_spin = QDoubleSpinBox()
        self.peak_tolerance_spin.setRange(0.01, 2.0)
        self.peak_tolerance_spin.setDecimals(2)
        self.peak_tolerance_spin.setValue(0.20)
        self.peak_tolerance_spin.setSuffix("°")
        self.peak_tolerance_spin.setToolTip("2θ tolerance for peak matching")
        peak_layout.addRow("2θ Tolerance:", self.peak_tolerance_spin)
        
        self.min_matches_spin = QSpinBox()
        self.min_matches_spin.setRange(1, 20)
        self.min_matches_spin.setValue(3)
        self.min_matches_spin.setToolTip("Minimum number of peak matches required")
        peak_layout.addRow("Min. Matches:", self.min_matches_spin)
        
        self.intensity_weight_spin = QDoubleSpinBox()
        self.intensity_weight_spin.setRange(0.0, 1.0)
        self.intensity_weight_spin.setDecimals(2)
        self.intensity_weight_spin.setValue(0.3)
        self.intensity_weight_spin.setToolTip("Weight for intensity similarity (0=position only, 1=intensity only)")
        peak_layout.addRow("Intensity Weight:", self.intensity_weight_spin)
        
        self.peak_max_results_spin = QSpinBox()
        self.peak_max_results_spin.setRange(10, 200)
        self.peak_max_results_spin.setValue(50)
        peak_layout.addRow("Max Results:", self.peak_max_results_spin)
        
        self.search_tabs.addTab(peak_tab, "Peak-Based")
        
        # Correlation-based search tab
        corr_tab = QWidget()
        corr_layout = QFormLayout(corr_tab)
        
        self.min_correlation_spin = QDoubleSpinBox()
        self.min_correlation_spin.setRange(0.1, 1.0)
        self.min_correlation_spin.setDecimals(2)
        self.min_correlation_spin.setValue(0.5)
        self.min_correlation_spin.setToolTip("Minimum correlation coefficient")
        corr_layout.addRow("Min. Correlation:", self.min_correlation_spin)
        
        self.corr_max_results_spin = QSpinBox()
        self.corr_max_results_spin.setRange(10, 200)
        self.corr_max_results_spin.setValue(50)
        corr_layout.addRow("Max Results:", self.corr_max_results_spin)
        
        # 2θ range for correlation
        range_layout = QHBoxLayout()
        self.min_2theta_corr_spin = QDoubleSpinBox()
        self.min_2theta_corr_spin.setRange(0, 180)
        self.min_2theta_corr_spin.setValue(5.0)
        self.min_2theta_corr_spin.setSuffix("°")
        range_layout.addWidget(self.min_2theta_corr_spin)
        
        range_layout.addWidget(QLabel("to"))
        
        self.max_2theta_corr_spin = QDoubleSpinBox()
        self.max_2theta_corr_spin.setRange(0, 180)
        self.max_2theta_corr_spin.setValue(60.0)
        self.max_2theta_corr_spin.setSuffix("°")
        range_layout.addWidget(self.max_2theta_corr_spin)
        
        corr_layout.addRow("2θ Range:", range_layout)
        
        self.search_tabs.addTab(corr_tab, "Correlation-Based")
        
        # Combined search tab
        combined_tab = QWidget()
        combined_layout = QFormLayout(combined_tab)
        
        self.combined_peak_tolerance_spin = QDoubleSpinBox()
        self.combined_peak_tolerance_spin.setRange(0.01, 2.0)
        self.combined_peak_tolerance_spin.setDecimals(2)
        self.combined_peak_tolerance_spin.setValue(0.20)
        self.combined_peak_tolerance_spin.setSuffix("°")
        combined_layout.addRow("Peak Tolerance:", self.combined_peak_tolerance_spin)
        
        self.combined_min_correlation_spin = QDoubleSpinBox()
        self.combined_min_correlation_spin.setRange(0.1, 1.0)
        self.combined_min_correlation_spin.setDecimals(2)
        self.combined_min_correlation_spin.setValue(0.3)
        combined_layout.addRow("Min. Correlation:", self.combined_min_correlation_spin)
        
        self.peak_weight_spin = QDoubleSpinBox()
        self.peak_weight_spin.setRange(0.0, 1.0)
        self.peak_weight_spin.setDecimals(2)
        self.peak_weight_spin.setValue(0.6)
        self.peak_weight_spin.setToolTip("Weight for peak-based score")
        combined_layout.addRow("Peak Weight:", self.peak_weight_spin)
        
        self.correlation_weight_spin = QDoubleSpinBox()
        self.correlation_weight_spin.setRange(0.0, 1.0)
        self.correlation_weight_spin.setDecimals(2)
        self.correlation_weight_spin.setValue(0.4)
        self.correlation_weight_spin.setToolTip("Weight for correlation score")
        combined_layout.addRow("Correlation Weight:", self.correlation_weight_spin)
        
        self.combined_max_results_spin = QSpinBox()
        self.combined_max_results_spin.setRange(10, 100)
        self.combined_max_results_spin.setValue(30)
        combined_layout.addRow("Max Results:", self.combined_max_results_spin)
        
        self.search_tabs.addTab(combined_tab, "Combined")
        
        # Ultra-Fast search tab
        fast_tab = QWidget()
        fast_layout = QFormLayout(fast_tab)
        
        # Index status and controls
        self.index_status_label = QLabel("Index Status: Not Built")
        fast_layout.addRow("Status:", self.index_status_label)
        
        # Check if index is already loaded
        self.update_index_status()
        
        # Build index button
        index_button_layout = QHBoxLayout()
        self.build_index_btn = QPushButton("Build Search Index")
        self.build_index_btn.clicked.connect(self.build_search_index)
        self.build_index_btn.setToolTip("Build optimized search index for ultra-fast searching")
        index_button_layout.addWidget(self.build_index_btn)
        
        self.benchmark_btn = QPushButton("Benchmark")
        self.benchmark_btn.clicked.connect(self.benchmark_search)
        self.benchmark_btn.setEnabled(False)
        self.benchmark_btn.setToolTip("Test search speed performance")
        index_button_layout.addWidget(self.benchmark_btn)
        
        fast_layout.addRow("Index:", index_button_layout)
        
        # Fast search parameters
        self.fast_min_correlation_spin = QDoubleSpinBox()
        self.fast_min_correlation_spin.setRange(0.1, 1.0)
        self.fast_min_correlation_spin.setDecimals(2)
        self.fast_min_correlation_spin.setValue(0.3)
        self.fast_min_correlation_spin.setToolTip("Minimum correlation for ultra-fast search")
        fast_layout.addRow("Min. Correlation:", self.fast_min_correlation_spin)
        
        self.fast_max_results_spin = QSpinBox()
        self.fast_max_results_spin.setRange(10, 200)
        self.fast_max_results_spin.setValue(50)
        fast_layout.addRow("Max Results:", self.fast_max_results_spin)
        
        # Grid resolution for index building
        self.grid_resolution_spin = QDoubleSpinBox()
        self.grid_resolution_spin.setRange(0.005, 0.1)
        self.grid_resolution_spin.setDecimals(3)
        self.grid_resolution_spin.setValue(0.02)
        self.grid_resolution_spin.setSuffix("°")
        self.grid_resolution_spin.setToolTip("Grid resolution for search index (smaller = more accurate but larger)")
        fast_layout.addRow("Grid Resolution:", self.grid_resolution_spin)
        
        # Performance info
        self.performance_label = QLabel("Performance: Not tested")
        fast_layout.addRow("Speed:", self.performance_label)
        
        self.search_tabs.addTab(fast_tab, "🚀 Ultra-Fast")
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.search_btn = QPushButton("Start Search")
        self.search_btn.clicked.connect(self.start_search)
        self.search_btn.setEnabled(False)
        button_layout.addWidget(self.search_btn)
        
        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("Export to Matching")
        self.export_btn.clicked.connect(self.export_to_matching)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return group
    
    def create_results_panel(self):
        """Create the results panel"""
        group = QGroupBox("Search Results")
        layout = QVBoxLayout(group)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Mineral', 'Formula', 'Space Group', 'Score', 'Method', 'Matches/Corr', 'Coverage', 'Select'
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.itemSelectionChanged.connect(self.show_result_details)
        layout.addWidget(self.results_table)
        
        # Result details
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        layout.addWidget(self.details_text)
        
        return group
    
    def create_plot_panel(self):
        """Create the plot panel"""
        group = QGroupBox("Pattern Comparison")
        layout = QVBoxLayout(group)
        
        # Plot canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create subplot
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.ax.set_xlabel('2θ (degrees)')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title('Experimental Pattern')
        self.ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        
        return group
    
    def set_experimental_pattern(self, pattern_data):
        """Set the experimental pattern data"""
        self.experimental_pattern = pattern_data
        self.update_search_availability()
        self.plot_experimental_pattern()
        
    def set_experimental_peaks(self, peak_data):
        """Set experimental peak data"""
        self.experimental_peaks = peak_data
        self.update_search_availability()
        
    def update_search_availability(self):
        """Update whether search can be performed"""
        has_pattern = self.experimental_pattern is not None
        has_peaks = self.experimental_peaks is not None
        
        # Peak-based search needs peaks
        # Correlation-based search needs full pattern
        # Combined search needs both
        
        current_tab = self.search_tabs.currentIndex()
        if current_tab == 0:  # Peak-based
            can_search = has_peaks
        elif current_tab == 1:  # Correlation-based
            can_search = has_pattern
        else:  # Combined
            can_search = has_pattern and has_peaks
            
        self.search_btn.setEnabled(can_search)
        
        # Update status
        if not has_pattern and not has_peaks:
            self.status_label.setText("Load experimental data to enable search")
        elif current_tab == 0 and not has_peaks:
            self.status_label.setText("Peak detection required for peak-based search")
        elif current_tab == 1 and not has_pattern:
            self.status_label.setText("Full pattern data required for correlation search")
        elif current_tab == 2 and not (has_pattern and has_peaks):
            self.status_label.setText("Both pattern and peaks required for combined search")
        else:
            self.status_label.setText("Ready for pattern search")
    
    def plot_experimental_pattern(self):
        """Plot the experimental pattern"""
        self.ax.clear()
        
        if self.experimental_pattern:
            self.ax.plot(self.experimental_pattern['two_theta'], 
                        self.experimental_pattern['intensity'],
                        'b-', linewidth=1, label='Experimental')
            
            # Plot peaks if available
            if self.experimental_peaks:
                peak_2theta = self.experimental_peaks['two_theta']
                peak_intensity = self.experimental_peaks['intensity']
                self.ax.plot(peak_2theta, peak_intensity, 'ro', 
                           markersize=4, label='Detected Peaks')
            
            self.ax.set_xlabel('2θ (degrees)')
            self.ax.set_ylabel('Intensity')
            self.ax.set_title('Experimental Pattern')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
        
        self.canvas.draw()
    
    def start_search(self):
        """Start the pattern search"""
        current_tab = self.search_tabs.currentIndex()
        
        # Prepare experimental data
        if current_tab == 0:  # Peak-based
            if not self.experimental_peaks:
                QMessageBox.warning(self, "No Peak Data", 
                                  "Peak detection is required for peak-based search.")
                return
            experimental_data = self.experimental_peaks
            search_method = 'peaks'
            search_params = {
                'tolerance': self.peak_tolerance_spin.value(),
                'min_matches': self.min_matches_spin.value(),
                'intensity_weight': self.intensity_weight_spin.value(),
                'max_results': self.peak_max_results_spin.value()
            }
        elif current_tab == 1:  # Correlation-based
            if not self.experimental_pattern:
                QMessageBox.warning(self, "No Pattern Data", 
                                  "Full pattern data is required for correlation search.")
                return
            experimental_data = self.experimental_pattern
            search_method = 'correlation'
            search_params = {
                'min_correlation': self.min_correlation_spin.value(),
                'max_results': self.corr_max_results_spin.value(),
                'two_theta_range': (self.min_2theta_corr_spin.value(), 
                                  self.max_2theta_corr_spin.value())
            }
        elif current_tab == 2:  # Combined
            if not (self.experimental_pattern and self.experimental_peaks):
                QMessageBox.warning(self, "Incomplete Data", 
                                  "Both pattern and peak data are required for combined search.")
                return
            # Combine pattern and peak data
            experimental_data = self.experimental_pattern.copy()
            experimental_data.update(self.experimental_peaks)
            search_method = 'combined'
            search_params = {
                'peak_tolerance': self.combined_peak_tolerance_spin.value(),
                'min_correlation': self.combined_min_correlation_spin.value(),
                'peak_weight': self.peak_weight_spin.value(),
                'correlation_weight': self.correlation_weight_spin.value(),
                'max_results': self.combined_max_results_spin.value()
            }
        elif current_tab == 3:  # Ultra-Fast
            self.start_ultra_fast_search()
            return
        else:  # Fallback for Combined (old index)
            if not (self.experimental_pattern and self.experimental_peaks):
                QMessageBox.warning(self, "Incomplete Data", 
                                  "Both pattern and peak data are required for combined search.")
                return
            # Combine pattern and peak data
            experimental_data = self.experimental_pattern.copy()
            experimental_data.update(self.experimental_peaks)
            search_method = 'combined'
            search_params = {
                'peak_tolerance': self.combined_peak_tolerance_spin.value(),
                'min_correlation': self.combined_min_correlation_spin.value(),
                'peak_weight': self.peak_weight_spin.value(),
                'correlation_weight': self.correlation_weight_spin.value(),
                'max_results': self.combined_max_results_spin.value()
            }
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.search_btn.setEnabled(False)
        self.status_label.setText("Searching database...")
        
        # Start search thread
        self.search_thread = PatternSearchThread(
            self.search_engine, experimental_data, search_method, search_params
        )
        self.search_thread.search_complete.connect(self.display_search_results)
        self.search_thread.progress_updated.connect(self.update_progress)
        self.search_thread.error_occurred.connect(self.handle_search_error)
        self.search_thread.start()
    
    def display_search_results(self, results):
        """Display search results"""
        # Prioritize refined phases if available
        prioritized_results = self.prioritize_refined_phases(results)
        
        self.search_results = prioritized_results
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.export_btn.setEnabled(len(prioritized_results) > 0)
        
        # Update results table
        self.results_table.setRowCount(len(prioritized_results))
        
        for i, result in enumerate(prioritized_results):
            # Mineral name - add [Refined] indicator if applicable
            mineral_name = result['mineral_name']
            if result.get('refined', False):
                mineral_name += " [Refined]"
            name_item = QTableWidgetItem(mineral_name)
            if result.get('refined', False):
                name_item.setBackground(Qt.lightGreen)
            self.results_table.setItem(i, 0, name_item)
            
            # Formula
            formula = result.get('chemical_formula', 'Unknown')
            self.results_table.setItem(i, 1, QTableWidgetItem(formula))
            
            # Space group
            space_group = result.get('space_group', 'Unknown')
            self.results_table.setItem(i, 2, QTableWidgetItem(space_group))
            
            # Score (depends on search method)
            if 'combined_score' in result:
                score = f"{result['combined_score']:.3f}"
            elif 'correlation' in result:
                score = f"{result['correlation']:.3f}"
            else:
                score = f"{result['match_score']:.3f}"
            self.results_table.setItem(i, 3, QTableWidgetItem(score))
            
            # Method
            method = result.get('search_method', 'unknown')
            self.results_table.setItem(i, 4, QTableWidgetItem(method))
            
            # Matches/Correlation
            if 'num_matches' in result:
                matches_corr = str(result['num_matches'])
            elif 'correlation' in result:
                matches_corr = f"{result['correlation']:.3f}"
            else:
                matches_corr = "N/A"
            self.results_table.setItem(i, 5, QTableWidgetItem(matches_corr))
            
            # Coverage
            coverage = result.get('coverage', 0)
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{coverage:.3f}"))
            
            # Select checkbox
            select_checkbox = QCheckBox()
            select_checkbox.setChecked(i < 5)  # Select top 5 by default
            self.results_table.setCellWidget(i, 7, select_checkbox)
        
        self.status_label.setText(f"Search complete: {len(results)} matches found")
    
    def show_result_details(self):
        """Show detailed information for selected result"""
        current_row = self.results_table.currentRow()
        if current_row < 0 or current_row >= len(self.search_results):
            return
        
        result = self.search_results[current_row]
        
        # Format detailed information
        details = f"Mineral: {result['mineral_name']}\n"
        details += f"Formula: {result.get('chemical_formula', 'Unknown')}\n"
        details += f"Space Group: {result.get('space_group', 'Unknown')}\n"
        details += f"Search Method: {result.get('search_method', 'Unknown')}\n\n"
        
        if 'match_score' in result:
            details += f"Match Score: {result['match_score']:.3f}\n"
        if 'correlation' in result:
            details += f"Correlation: {result['correlation']:.3f}\n"
        if 'combined_score' in result:
            details += f"Combined Score: {result['combined_score']:.3f}\n"
        if 'num_matches' in result:
            details += f"Peak Matches: {result['num_matches']}\n"
        if 'coverage' in result:
            details += f"Coverage: {result['coverage']:.3f}\n"
        if 'r_squared' in result:
            details += f"R²: {result['r_squared']:.3f}\n"
        if 'rms_error' in result:
            details += f"RMS Error: {result['rms_error']:.3f}\n"
        
        self.details_text.setText(details)
    
    def update_progress(self, message):
        """Update progress message"""
        self.status_label.setText(message)
    
    def handle_search_error(self, error_message):
        """Handle search errors"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.status_label.setText("Search failed")
        QMessageBox.critical(self, "Search Error", f"Search failed: {error_message}")
    
    def clear_results(self):
        """Clear search results"""
        self.results_table.setRowCount(0)
        self.details_text.clear()
        self.search_results = []
        self.export_btn.setEnabled(False)
        self.status_label.setText("Results cleared")
    
    def export_to_matching(self):
        """Export selected results to matching tab"""
        selected_results = []
        
        for i in range(self.results_table.rowCount()):
            checkbox = self.results_table.cellWidget(i, 7)
            if checkbox and checkbox.isChecked():
                result = self.search_results[i]
                
                # Convert to format expected by matching tab
                phase_data = {
                    'id': result['mineral_id'],
                    'mineral': result['mineral_name'],
                    'formula': result.get('chemical_formula', 'Unknown'),
                    'space_group': result.get('space_group', 'Unknown'),
                    'local_db': True,  # Mark as from local database
                    'search_score': result.get('combined_score', 
                                             result.get('correlation', 
                                                       result.get('match_score', 0)))
                }
                selected_results.append(phase_data)
        
        if selected_results:
            self.phases_found.emit(selected_results)
            QMessageBox.information(self, "Export Complete", 
                                  f"Exported {len(selected_results)} phases to matching tab.")
        else:
            QMessageBox.warning(self, "No Selection", 
                              "Please select phases to export using the checkboxes.")
    
    def reset_for_new_pattern(self):
        """Reset the pattern search tab when a new pattern is loaded"""
        # Clear search results
        self.search_results = []
        
        # Clear results table
        self.results_table.setRowCount(0)
        
        # Reset progress bar
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        # Enable search button if conditions are met
        self.update_search_availability()
        
        # Clear plot except for experimental pattern (which will be updated)
        # The experimental pattern will be updated via set_experimental_pattern
        
        print("Pattern search tab reset for new pattern")
    
    def build_search_index(self):
        """Build the ultra-fast search index"""
        try:
            self.build_index_btn.setEnabled(False)
            self.build_index_btn.setText("Building...")
            self.index_status_label.setText("Building index...")
            
            # Get parameters
            grid_resolution = self.grid_resolution_spin.value()
            
            # Build index in separate thread to avoid UI freezing
            success = self.fast_search_engine.build_search_index(
                grid_resolution=grid_resolution,
                force_rebuild=True
            )
            
            if success:
                # Update the status display
                self.update_index_status()
                
                # Get stats for the message
                stats = self.fast_search_engine.get_search_statistics()
                QMessageBox.information(self, "Index Built", 
                    f"Search index built successfully!\n\n"
                    f"Database size: {stats['database_size']} patterns\n"
                    f"Grid points: {stats['grid_points']}\n"
                    f"Build time: {stats['index_build_time_s']:.2f}s\n"
                    f"Memory usage: {stats['matrix_size_mb']:.1f} MB")
            else:
                self.index_status_label.setText("Build failed")
                QMessageBox.warning(self, "Build Failed", 
                    "Failed to build search index. Check console for details.")
                
        except Exception as e:
            self.index_status_label.setText("Build failed")
            QMessageBox.critical(self, "Build Error", f"Error building index: {str(e)}")
            
        finally:
            self.build_index_btn.setEnabled(True)
            self.build_index_btn.setText("Build Search Index")
    
    def benchmark_search(self):
        """Benchmark the ultra-fast search performance"""
        if not self.experimental_pattern:
            QMessageBox.warning(self, "No Pattern", 
                              "Load an experimental pattern first to benchmark search speed.")
            return
            
        try:
            self.benchmark_btn.setEnabled(False)
            self.benchmark_btn.setText("Benchmarking...")
            
            # Run benchmark
            benchmark_results = self.fast_search_engine.benchmark_search_speed(
                self.experimental_pattern, num_iterations=5
            )
            
            # Update performance display
            avg_time = benchmark_results['average_time_ms']
            patterns_per_sec = benchmark_results['patterns_per_second']
            
            self.performance_label.setText(f"{avg_time:.1f}ms ({patterns_per_sec:.0f} patterns/s)")
            
            # Show detailed results
            QMessageBox.information(self, "Benchmark Results",
                f"Ultra-Fast Search Performance:\n\n"
                f"Average search time: {avg_time:.1f}ms\n"
                f"Min/Max time: {benchmark_results['min_time_ms']:.1f}/{benchmark_results['max_time_ms']:.1f}ms\n"
                f"Database size: {benchmark_results['database_size']} patterns\n"
                f"Search rate: {patterns_per_sec:.0f} patterns/second\n\n"
                f"This is searching through {benchmark_results['database_size']} patterns\n"
                f"in under {avg_time:.0f} milliseconds!")
                
        except Exception as e:
            QMessageBox.critical(self, "Benchmark Error", f"Benchmark failed: {str(e)}")
            
        finally:
            self.benchmark_btn.setEnabled(True)
            self.benchmark_btn.setText("Benchmark")
    
    def start_ultra_fast_search(self):
        """Start ultra-fast correlation search"""
        if not self.experimental_pattern:
            QMessageBox.warning(self, "No Pattern Data", 
                              "Full pattern data is required for ultra-fast search.")
            return
            
        if self.fast_search_engine.search_index is None:
            reply = QMessageBox.question(self, "Index Required", 
                "Search index not built. Build it now?\n\n"
                "This is a one-time setup that enables instant searching.",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.build_search_index()
                if self.fast_search_engine.search_index is None:
                    return  # Build failed
            else:
                return
        
        try:
            # Show progress (though it should be very fast)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.search_btn.setEnabled(False)
            self.status_label.setText("Ultra-fast searching...")
            
            # Get parameters
            min_correlation = self.fast_min_correlation_spin.value()
            max_results = self.fast_max_results_spin.value()
            
            # Perform ultra-fast search
            results = self.fast_search_engine.ultra_fast_correlation_search(
                self.experimental_pattern,
                min_correlation=min_correlation,
                max_results=max_results
            )
            
            # Display results
            self.display_search_results(results)
            
            # Update performance info
            search_time = self.fast_search_engine.last_search_time * 1000
            self.performance_label.setText(f"Last search: {search_time:.1f}ms")
            
        except Exception as e:
            self.handle_search_error(str(e))
            
        finally:
            self.progress_bar.setVisible(False)
            self.search_btn.setEnabled(True)
            
    def set_multi_phase_analyzer(self, analyzer):
        """Set the multi-phase analyzer for refined phase integration"""
        self.multi_phase_analyzer = analyzer
        
    def prioritize_refined_phases(self, results):
        """Prioritize refined phases in search results"""
        if not self.multi_phase_analyzer:
            return results
            
        refined_phases = self.multi_phase_analyzer.get_refined_phases_for_search()
        if not refined_phases:
            return results
            
        # Create a mapping of phase IDs to refined phases
        refined_phase_map = {}
        for refined_phase in refined_phases:
            phase_id = refined_phase['phase'].get('id')
            if phase_id:
                refined_phase_map[phase_id] = refined_phase
                
        # Separate refined and non-refined results
        refined_results = []
        regular_results = []
        
        for result in results:
            phase_id = result.get('phase_id') or result.get('id')
            if phase_id in refined_phase_map:
                # This is a refined phase - boost its score and add refinement info
                refined_phase = refined_phase_map[phase_id]
                result = result.copy()
                result['refined'] = True
                result['refinement_quality'] = refined_phase['refinement_quality']
                result['search_priority'] = refined_phase['search_priority']
                
                # Boost the score based on refinement quality
                original_score = result.get('combined_score', result.get('correlation', 0))
                refinement_boost = min(0.2, refined_phase['search_priority'] / 10.0)
                result['boosted_score'] = original_score + refinement_boost
                
                refined_results.append(result)
            else:
                result = result.copy()
                result['refined'] = False
                result['boosted_score'] = result.get('combined_score', result.get('correlation', 0))
                regular_results.append(result)
                
        # Sort refined phases by boosted score, regular phases by original score
        refined_results.sort(key=lambda x: x['boosted_score'], reverse=True)
        regular_results.sort(key=lambda x: x['boosted_score'], reverse=True)
        
        # Combine with refined phases first
        prioritized_results = refined_results + regular_results
        
        if refined_results:
            print(f"Prioritized {len(refined_results)} refined phases in search results")
            
        return prioritized_results
        
    def update_search_with_refinement_feedback(self):
        """Update search engines with refined phase feedback"""
        if not self.multi_phase_analyzer:
            return
            
        refined_phases = self.multi_phase_analyzer.get_refined_phases_for_search()
        if not refined_phases:
            return
            
        # Update search engines with refined parameters
        for refined_phase in refined_phases:
            phase_id = refined_phase['phase'].get('id')
            if phase_id:
                # Update theoretical peaks with refined parameters
                refined_peaks = refined_phase['theoretical_peaks']
                refinement_quality = refined_phase['refinement_quality']
                
                # Inform search engines about the refined phase
                if hasattr(self.search_engine, 'update_refined_phase'):
                    self.search_engine.update_refined_phase(phase_id, refined_peaks, refinement_quality)
                    
                if hasattr(self.fast_search_engine, 'update_refined_phase'):
                    self.fast_search_engine.update_refined_phase(phase_id, refined_peaks, refinement_quality)
    
    def update_index_status(self):
        """Update the index status display"""
        try:
            if self.fast_search_engine.search_index is not None:
                stats = self.fast_search_engine.get_search_statistics()
                self.index_status_label.setText(f"Ready: {stats['database_size']} patterns")
                self.benchmark_btn.setEnabled(True)
                self.build_index_btn.setText("Rebuild Index")
                
                # Update performance info if available
                if stats.get('index_build_time_s', 0) > 0:
                    build_time = stats['index_build_time_s']
                    self.performance_label.setText(f"Index loaded ({build_time:.2f}s build time)")
            else:
                self.index_status_label.setText("Index Status: Not Built")
                self.benchmark_btn.setEnabled(False)
                self.build_index_btn.setText("Build Search Index")
                self.performance_label.setText("Performance: Not tested")
                
        except Exception as e:
            print(f"Error updating index status: {e}")
            self.index_status_label.setText("Index Status: Error")
