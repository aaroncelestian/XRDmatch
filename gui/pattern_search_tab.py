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
        self.search_results = []
        
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
        else:  # Combined
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
        self.search_results = results
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.export_btn.setEnabled(len(results) > 0)
        
        # Update results table
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            # Mineral name
            self.results_table.setItem(i, 0, QTableWidgetItem(result['mineral_name']))
            
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
