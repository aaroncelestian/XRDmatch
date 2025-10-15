"""
Visualization and Export Tab for XRD Phase Matching
Provides advanced plotting, customization, and Le Bail refinement visualization
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QCheckBox, QColorDialog, QFileDialog,
                             QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
                             QSplitter, QTextEdit, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from typing import Dict, List, Optional
import json


class VisualizationTab(QWidget):
    """Tab for advanced visualization, export, and Le Bail refinement"""
    
    def __init__(self):
        super().__init__()
        self.experimental_pattern = None
        self.matched_phases = []
        self.lebail_results = None
        self.multi_phase_analyzer = None
        
        # Visualization settings
        self.plot_settings = {
            'exp_color': '#1f77b4',
            'exp_linewidth': 1.5,
            'calc_color': '#ff7f0e',
            'calc_linewidth': 1.5,
            'diff_color': '#2ca02c',
            'diff_linewidth': 1.0,
            'phase_colors': {},
            'phase_linewidths': {},
            'waterfall_offset': 0.0,
            'show_legend': True,
            'show_grid': True,
            'title': 'XRD Pattern Analysis',
            'xlabel': '2θ (degrees)',
            'ylabel': 'Intensity (a.u.)',
            'dpi': 300
        }
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QHBoxLayout(self)
        
        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        
        # Data source group
        data_group = QGroupBox("Data Source")
        data_layout = QVBoxLayout(data_group)
        
        self.import_btn = QPushButton("Import from Phase Matching")
        self.import_btn.clicked.connect(self.import_from_matching_tab)
        data_layout.addWidget(self.import_btn)
        
        self.data_status = QLabel("No data loaded")
        self.data_status.setWordWrap(True)
        data_layout.addWidget(self.data_status)
        
        left_layout.addWidget(data_group)
        
        # Le Bail refinement group
        lebail_group = QGroupBox("Le Bail Refinement")
        lebail_layout = QVBoxLayout(lebail_group)
        
        self.lebail_btn = QPushButton("Run Le Bail Refinement")
        self.lebail_btn.clicked.connect(self.run_lebail_refinement)
        self.lebail_btn.setEnabled(False)
        lebail_layout.addWidget(self.lebail_btn)
        
        # Refinement settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Max Iterations:"))
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(3, 50)
        self.max_iter_spin.setValue(10)  # Reduced from 15 for faster refinement
        self.max_iter_spin.setToolTip("Fewer iterations = faster refinement. 10 is usually sufficient.")
        settings_layout.addWidget(self.max_iter_spin)
        lebail_layout.addLayout(settings_layout)
        
        # Initial FWHM setting
        fwhm_layout = QHBoxLayout()
        fwhm_layout.addWidget(QLabel("Initial FWHM (°):"))
        self.initial_fwhm_spin = QDoubleSpinBox()
        self.initial_fwhm_spin.setRange(0.005, 1.0)
        self.initial_fwhm_spin.setValue(0.1)
        self.initial_fwhm_spin.setSingleStep(0.005)
        self.initial_fwhm_spin.setDecimals(3)
        self.initial_fwhm_spin.setToolTip("Peak width estimate. Synchrotron: 0.01-0.02°, Lab XRD: 0.08-0.15°")
        self.initial_fwhm_spin.setSuffix("°")
        self.initial_fwhm_spin.setToolTip("Starting peak width (FWHM) in degrees. Smaller = narrower peaks.")
        fwhm_layout.addWidget(self.initial_fwhm_spin)
        lebail_layout.addLayout(fwhm_layout)
        
        # Scale factor bounds
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Max Scale:"))
        self.max_scale_spin = QDoubleSpinBox()
        self.max_scale_spin.setRange(1.0, 1000.0)
        self.max_scale_spin.setValue(100.0)
        self.max_scale_spin.setSingleStep(10.0)
        self.max_scale_spin.setToolTip("Maximum scale factor - higher allows peaks to be taller")
        scale_layout.addWidget(self.max_scale_spin)
        lebail_layout.addLayout(scale_layout)
        
        # Profile shape parameter
        eta_layout = QHBoxLayout()
        eta_layout.addWidget(QLabel("Peak Shape (η):"))
        self.eta_spin = QDoubleSpinBox()
        self.eta_spin.setRange(0.0, 1.0)
        self.eta_spin.setValue(0.5)
        self.eta_spin.setSingleStep(0.1)
        self.eta_spin.setToolTip("0=Gaussian, 1=Lorentzian, 0.5=50/50 mix")
        eta_layout.addWidget(self.eta_spin)
        lebail_layout.addLayout(eta_layout)
        
        # Refinement options
        self.refine_cell_check = QCheckBox("Refine Unit Cell")
        self.refine_cell_check.setChecked(True)
        self.refine_cell_check.setToolTip("Allow unit cell parameters to refine")
        lebail_layout.addWidget(self.refine_cell_check)
        
        self.refine_profile_check = QCheckBox("Refine Peak Profile")
        self.refine_profile_check.setChecked(True)
        self.refine_profile_check.setToolTip("Allow peak width parameters to refine")
        lebail_layout.addWidget(self.refine_profile_check)
        
        self.refine_intensities_check = QCheckBox("Refine Peak Intensities (Pawley) - SLOW!")
        self.refine_intensities_check.setChecked(False)
        self.refine_intensities_check.setToolTip("⚠️ WARNING: Very slow and often unstable!\nOnly use for preferred orientation/texture.\nLeave unchecked for normal Le Bail refinement.")
        self.refine_intensities_check.setStyleSheet("QCheckBox { color: #c60; }")
        lebail_layout.addWidget(self.refine_intensities_check)
        
        # 2-theta range settings
        range_label = QLabel("2θ Range (optional):")
        lebail_layout.addWidget(range_label)
        
        range_layout = QHBoxLayout()
        self.use_range_check = QCheckBox("Limit range")
        self.use_range_check.setChecked(False)
        self.use_range_check.stateChanged.connect(self._toggle_range_inputs)
        range_layout.addWidget(self.use_range_check)
        lebail_layout.addLayout(range_layout)
        
        range_inputs_layout = QHBoxLayout()
        range_inputs_layout.addWidget(QLabel("Min:"))
        self.min_2theta_spin = QDoubleSpinBox()
        self.min_2theta_spin.setRange(0.0, 180.0)
        self.min_2theta_spin.setValue(10.0)
        self.min_2theta_spin.setSingleStep(1.0)
        self.min_2theta_spin.setSuffix("°")
        self.min_2theta_spin.setEnabled(False)
        range_inputs_layout.addWidget(self.min_2theta_spin)
        
        range_inputs_layout.addWidget(QLabel("Max:"))
        self.max_2theta_spin = QDoubleSpinBox()
        self.max_2theta_spin.setRange(0.0, 180.0)
        self.max_2theta_spin.setValue(90.0)
        self.max_2theta_spin.setSingleStep(1.0)
        self.max_2theta_spin.setSuffix("°")
        self.max_2theta_spin.setEnabled(False)
        range_inputs_layout.addWidget(self.max_2theta_spin)
        lebail_layout.addLayout(range_inputs_layout)
        
        self.lebail_progress = QProgressBar()
        self.lebail_progress.setVisible(False)
        lebail_layout.addWidget(self.lebail_progress)
        
        self.lebail_status = QLabel("")
        self.lebail_status.setWordWrap(True)
        lebail_layout.addWidget(self.lebail_status)
        
        left_layout.addWidget(lebail_group)
        
        # Plot customization group
        custom_group = QGroupBox("Plot Customization")
        custom_layout = QVBoxLayout(custom_group)
        
        # Experimental data settings
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exp. Data:"))
        self.exp_color_btn = QPushButton("Color")
        self.exp_color_btn.clicked.connect(lambda: self.choose_color('exp_color'))
        exp_layout.addWidget(self.exp_color_btn)
        self.exp_width_spin = QDoubleSpinBox()
        self.exp_width_spin.setRange(0.1, 10.0)
        self.exp_width_spin.setValue(1.5)
        self.exp_width_spin.setSingleStep(0.5)
        self.exp_width_spin.valueChanged.connect(lambda v: self.update_setting('exp_linewidth', v))
        exp_layout.addWidget(self.exp_width_spin)
        custom_layout.addLayout(exp_layout)
        
        # Calculated pattern settings
        calc_layout = QHBoxLayout()
        calc_layout.addWidget(QLabel("Calc. Pattern:"))
        self.calc_color_btn = QPushButton("Color")
        self.calc_color_btn.clicked.connect(lambda: self.choose_color('calc_color'))
        calc_layout.addWidget(self.calc_color_btn)
        self.calc_width_spin = QDoubleSpinBox()
        self.calc_width_spin.setRange(0.1, 10.0)
        self.calc_width_spin.setValue(1.5)
        self.calc_width_spin.setSingleStep(0.5)
        self.calc_width_spin.valueChanged.connect(lambda v: self.update_setting('calc_linewidth', v))
        calc_layout.addWidget(self.calc_width_spin)
        custom_layout.addLayout(calc_layout)
        
        # Difference pattern settings
        diff_layout = QHBoxLayout()
        diff_layout.addWidget(QLabel("Difference:"))
        self.diff_color_btn = QPushButton("Color")
        self.diff_color_btn.clicked.connect(lambda: self.choose_color('diff_color'))
        diff_layout.addWidget(self.diff_color_btn)
        self.diff_width_spin = QDoubleSpinBox()
        self.diff_width_spin.setRange(0.1, 10.0)
        self.diff_width_spin.setValue(1.0)
        self.diff_width_spin.setSingleStep(0.5)
        self.diff_width_spin.valueChanged.connect(lambda v: self.update_setting('diff_linewidth', v))
        calc_layout.addWidget(self.diff_width_spin)
        custom_layout.addLayout(diff_layout)
        
        # Waterfall plot settings
        waterfall_layout = QHBoxLayout()
        waterfall_layout.addWidget(QLabel("Waterfall Offset:"))
        self.waterfall_spin = QDoubleSpinBox()
        self.waterfall_spin.setRange(0.0, 1000.0)
        self.waterfall_spin.setValue(0.0)
        self.waterfall_spin.setSingleStep(10.0)
        self.waterfall_spin.valueChanged.connect(lambda v: self.update_setting('waterfall_offset', v))
        waterfall_layout.addWidget(self.waterfall_spin)
        custom_layout.addLayout(waterfall_layout)
        
        # Plot options
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        self.legend_check.stateChanged.connect(lambda: self.update_setting('show_legend', self.legend_check.isChecked()))
        custom_layout.addWidget(self.legend_check)
        
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(lambda: self.update_setting('show_grid', self.grid_check.isChecked()))
        custom_layout.addWidget(self.grid_check)
        
        # Update plot button
        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self.update_plot)
        custom_layout.addWidget(self.update_plot_btn)
        
        left_layout.addWidget(custom_group)
        
        # Export group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        # DPI setting
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.valueChanged.connect(lambda v: self.update_setting('dpi', v))
        dpi_layout.addWidget(self.dpi_spin)
        export_layout.addLayout(dpi_layout)
        
        self.export_png_btn = QPushButton("Export as PNG")
        self.export_png_btn.clicked.connect(lambda: self.export_plot('png'))
        export_layout.addWidget(self.export_png_btn)
        
        self.export_pdf_btn = QPushButton("Export as PDF")
        self.export_pdf_btn.clicked.connect(lambda: self.export_plot('pdf'))
        export_layout.addWidget(self.export_pdf_btn)
        
        self.export_svg_btn = QPushButton("Export as SVG")
        self.export_svg_btn.clicked.connect(lambda: self.export_plot('svg'))
        export_layout.addWidget(self.export_svg_btn)
        
        self.export_data_btn = QPushButton("Export Data (CSV)")
        self.export_data_btn.clicked.connect(self.export_data)
        export_layout.addWidget(self.export_data_btn)
        
        left_layout.addWidget(export_group)
        
        left_layout.addStretch()
        
        # Right panel - visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create tab widget for different views
        self.viz_tabs = QTabWidget()
        
        # Main plot tab
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        self.viz_tabs.addTab(plot_widget, "Main Plot")
        
        # Refinement results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        self.viz_tabs.addTab(results_widget, "Refinement Results")
        
        # Phase table tab
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        
        self.phase_table = QTableWidget()
        self.phase_table.setColumnCount(5)
        self.phase_table.setHorizontalHeaderLabels(['Phase', 'Formula', 'Scale Factor', 'Rwp (%)', 'Color'])
        phase_layout.addWidget(self.phase_table)
        
        phase_customize_btn = QPushButton("Customize Phase Colors")
        phase_customize_btn.clicked.connect(self.customize_phase_colors)
        phase_layout.addWidget(phase_customize_btn)
        
        self.viz_tabs.addTab(phase_widget, "Phase Details")
        
        right_layout.addWidget(self.viz_tabs)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    def _toggle_range_inputs(self):
        """Enable/disable 2-theta range input fields"""
        enabled = self.use_range_check.isChecked()
        self.min_2theta_spin.setEnabled(enabled)
        self.max_2theta_spin.setEnabled(enabled)
        
    def set_multi_phase_analyzer(self, analyzer):
        """Set the multi-phase analyzer for Le Bail refinement"""
        self.multi_phase_analyzer = analyzer
        
    def import_from_matching_tab(self):
        """Import data from the phase matching tab"""
        # This will be connected by the main window
        pass
        
    def set_data(self, experimental_pattern, matched_phases):
        """Set experimental pattern and matched phases"""
        self.experimental_pattern = experimental_pattern
        self.matched_phases = matched_phases
        
        # Auto-adjust FWHM based on wavelength
        wavelength = experimental_pattern.get('wavelength', 1.5406)
        if wavelength < 0.5:  # Synchrotron
            suggested_fwhm = 0.015
            print(f"Synchrotron data detected (λ={wavelength:.5f} Å)")
            print(f"  → Setting initial FWHM to {suggested_fwhm}° (typical for synchrotron)")
        elif wavelength < 1.0:  # Mo or other short wavelength
            suggested_fwhm = 0.05
            print(f"Short wavelength detected (λ={wavelength:.4f} Å)")
            print(f"  → Setting initial FWHM to {suggested_fwhm}°")
        else:  # Lab XRD (Cu, Co, etc.)
            suggested_fwhm = 0.1
            print(f"Lab XRD detected (λ={wavelength:.4f} Å)")
            print(f"  → Setting initial FWHM to {suggested_fwhm}°")
        
        self.initial_fwhm_spin.setValue(suggested_fwhm)
        
        # Enable refinement button
        self.lebail_btn.setEnabled(True)
        
        # Update status
        num_phases = len(matched_phases)
        self.data_status.setText(f"Loaded: {num_phases} phase(s) from matching")
        
        # Update plot
        # Create initial plot
        self.update_plot()
        
    def update_phase_table(self):
        """Update the phase details table"""
        self.phase_table.setRowCount(len(self.matched_phases))
        
        for i, phase in enumerate(self.matched_phases):
            # Get phase info
            if 'phase' in phase:
                phase_info = phase['phase']
                mineral = phase_info.get('mineral', 'Unknown')
                formula = phase_info.get('formula', 'N/A')
            else:
                mineral = phase.get('mineral', 'Unknown')
                formula = phase.get('formula', 'N/A')
            
            # Phase name
            self.phase_table.setItem(i, 0, QTableWidgetItem(mineral))
            
            # Formula
            self.phase_table.setItem(i, 1, QTableWidgetItem(formula))
            
            # Scale factor (if available)
            scale = phase.get('optimized_scaling', 1.0)
            self.phase_table.setItem(i, 2, QTableWidgetItem(f"{scale:.3f}"))
            
            # Rwp (if available from refinement)
            rwp = phase.get('rwp', 'N/A')
            if isinstance(rwp, (int, float)):
                rwp = f"{rwp:.2f}"
            self.phase_table.setItem(i, 3, QTableWidgetItem(str(rwp)))
            
            # Color indicator
            phase_id = phase_info.get('id', f'phase_{i}') if 'phase' in phase else f'phase_{i}'
            color = self.plot_settings['phase_colors'].get(phase_id, self._get_default_color(i))
            color_item = QTableWidgetItem()
            color_item.setBackground(QColor(color))
            self.phase_table.setItem(i, 4, color_item)
            
    def _get_default_color(self, index: int) -> str:
        """Get default color for phase by index"""
        colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                 '#bcbd22', '#17becf', '#ff9896', '#c5b0d5', '#c49c94']
        return colors[index % len(colors)]
        
    def run_lebail_refinement(self):
        """Run Le Bail refinement on current phases"""
        if not self.experimental_pattern or not self.matched_phases:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
            
        if not self.multi_phase_analyzer:
            QMessageBox.warning(self, "No Analyzer", "Multi-phase analyzer not available.")
            return
        
        try:
            self.lebail_btn.setEnabled(False)
            self.lebail_progress.setVisible(True)
            self.lebail_progress.setRange(0, 0)  # Indeterminate
            self.lebail_status.setText("Running Le Bail refinement...")
            
            # Prepare experimental data
            # IMPORTANT: Use background-subtracted data for Le Bail refinement
            # Background should be removed before refinement to avoid fitting the background
            experimental_data = {
                'two_theta': self.experimental_pattern['two_theta'],
                'intensity': self.experimental_pattern['intensity'],
                'wavelength': self.experimental_pattern.get('wavelength', 1.5406),
                'errors': self.experimental_pattern.get('intensity_error')
            }
            
            # Get 2-theta range if specified
            two_theta_range = None
            if self.use_range_check.isChecked():
                min_2theta = self.min_2theta_spin.value()
                max_2theta = self.max_2theta_spin.value()
                
                # Validate range
                if min_2theta >= max_2theta:
                    QMessageBox.warning(self, "Invalid Range", 
                                      "Minimum 2θ must be less than maximum 2θ.")
                    return
                    
                two_theta_range = (min_2theta, max_2theta)
                print(f"Using 2θ range: {min_2theta}° - {max_2theta}°")
            
            # Set up real-time plotting callback
            from utils.lebail_refinement import LeBailRefinement
            LeBailRefinement.plot_callback = self._realtime_plot_callback
            
            # Auto-adjust FWHM based on wavelength if still at default
            wavelength = experimental_data.get('wavelength', 1.5406)
            current_fwhm = self.initial_fwhm_spin.value()
            
            # If FWHM is at default 0.1° but wavelength suggests otherwise, auto-adjust
            if abs(current_fwhm - 0.1) < 0.001:  # Still at default
                if wavelength < 0.5:  # Synchrotron
                    suggested_fwhm = 0.015
                    self.initial_fwhm_spin.setValue(suggested_fwhm)
                    print(f"⚠️  Auto-adjusted FWHM: 0.100° → {suggested_fwhm}° (synchrotron data)")
            
            # Get user-defined refinement parameters
            initial_fwhm = self.initial_fwhm_spin.value()
            max_scale = self.max_scale_spin.value()
            initial_eta = self.eta_spin.value()
            refine_cell = self.refine_cell_check.isChecked()
            refine_profile = self.refine_profile_check.isChecked()
            refine_intensities = self.refine_intensities_check.isChecked()
            
            # Warn if Pawley is enabled
            if refine_intensities:
                print("⚠️  WARNING: Pawley refinement enabled - this may be slow and unstable!")
                print("   Consider disabling 'Refine Peak Intensities' for faster, more stable refinement")
            
            # Calculate initial U, V, W from FWHM
            # FWHM² ≈ U·tan²θ + V·tanθ + W, at low angles W dominates
            initial_w = (initial_fwhm ** 2)
            initial_u = initial_w * 0.1  # U is typically smaller
            initial_v = 0.0  # V is often near zero
            
            print(f"Initial parameters: FWHM={initial_fwhm:.3f}°, W={initial_w:.6f}, U={initial_u:.6f}")
            print(f"Max scale factor: {max_scale}, η={initial_eta:.2f}")
            print(f"Refine cell: {refine_cell}, Refine profile: {refine_profile}")
            print(f"Refine intensities (Pawley): {refine_intensities}")
            
            # Prepare refinement parameters
            refinement_params = {
                'initial_u': initial_u,
                'initial_v': initial_v,
                'initial_w': initial_w,
                'initial_eta': initial_eta,
                'max_scale': max_scale,
                'refine_cell': refine_cell,
                'refine_profile': refine_profile,
                'refine_intensities': refine_intensities
            }
            
            # Run refinement
            max_iter = self.max_iter_spin.value()
            self.lebail_results = self.multi_phase_analyzer.perform_lebail_refinement(
                experimental_data, 
                self.matched_phases,
                max_iterations=max_iter,
                two_theta_range=two_theta_range,
                refinement_params=refinement_params
            )
            
            # Clear callback after refinement
            LeBailRefinement.plot_callback = None
            
            if self.lebail_results['success']:
                # Check if refinement quality is acceptable
                r_factors = self.lebail_results.get('r_factors', {})
                rwp = r_factors.get('Rwp', 999)
                
                print(f"\n=== Refinement Complete ===")
                print(f"Final Rwp: {rwp:.2f}%")
                print(f"Refinement results keys: {self.lebail_results.keys()}")
                
                # Check refinement data
                refinement_data = self.lebail_results.get('refinement_results', {})
                calc_pattern = refinement_data.get('calculated_pattern', None)
                if calc_pattern is not None:
                    print(f"Calculated pattern: {len(calc_pattern)} points, range {np.min(calc_pattern):.2f} - {np.max(calc_pattern):.2f}")
                else:
                    print("⚠️  No calculated pattern in results!")
                
                if rwp > 50:
                    print(f"⚠️  WARNING: Very poor fit (Rwp={rwp:.1f}%)")
                    print(f"   This usually means:")
                    print(f"   - Wrong phase identified")
                    print(f"   - FWHM too large/small (current: {self.initial_fwhm_spin.value():.3f}°)")
                    print(f"   - Wavelength mismatch")
                    print(f"   - Try adjusting FWHM or checking phase identity")
                
                self.lebail_status.setText(f"✓ Refinement complete (Rwp={rwp:.2f}%)")
                self.display_lebail_results()
                
                # Force update plot with final results
                print("Updating final plot...")
                self.update_plot()
                print("Plot update complete")
            else:
                error_msg = self.lebail_results.get('error', 'Unknown error')
                self.lebail_status.setText(f"✗ Refinement failed: {error_msg}")
                QMessageBox.critical(self, "Refinement Failed", f"Error: {error_msg}")
                
        except Exception as e:
            self.lebail_status.setText(f"✗ Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Refinement error:\n{str(e)}")
            
        finally:
            self.lebail_progress.setVisible(False)
            self.lebail_btn.setEnabled(True)
            
    def display_lebail_results(self):
        """Display Le Bail refinement results"""
        if not self.lebail_results or not self.lebail_results['success']:
            return
            
        # Display in results text area
        report = self.lebail_results.get('refinement_report', 'No report available')
        self.results_text.setText(report)
        
        # Update phase table with refined parameters
        refined_phases = self.lebail_results.get('refined_phases', [])
        for i, refined_phase in enumerate(refined_phases):
            if i < self.phase_table.rowCount():
                # Update scale factor
                quality = refined_phase.get('refinement_quality', {})
                scale = quality.get('scale_factor', 1.0)
                self.phase_table.setItem(i, 2, QTableWidgetItem(f"{scale:.3f}"))
                
        # Show R-factors
        r_factors = self.lebail_results.get('r_factors', {})
        rwp = r_factors.get('Rwp', 0)
        gof = r_factors.get('GoF', 0)
        
        self.lebail_status.setText(
            f"✓ Refinement complete! Rwp = {rwp:.2f}%, GoF = {gof:.2f}"
        )
        
        # Switch to results tab
        self.viz_tabs.setCurrentIndex(1)
        
    def update_plot(self):
        """Update the main plot with current data and settings"""
        if not self.experimental_pattern:
            return
            
        self.figure.clear()
        
        # Determine if we're doing waterfall plot
        waterfall_offset = self.plot_settings['waterfall_offset']
        
        if waterfall_offset > 0:
            self._create_waterfall_plot()
        else:
            self._create_standard_plot()
            
        self.canvas.draw()
        
    def _create_standard_plot(self):
        """Create standard overlay plot"""
        ax = self.figure.add_subplot(111)
        
        # If we have refinement results, use the normalized data from refinement
        # Otherwise use original experimental data
        if self.lebail_results and self.lebail_results['success']:
            refinement_data = self.lebail_results['refinement_results']
            
            # Use normalized data from refinement (both exp and calc are on same scale)
            two_theta = refinement_data.get('two_theta', self.experimental_pattern['two_theta'])
            intensity = refinement_data.get('experimental_intensity', self.experimental_pattern['intensity'])
            calculated_pattern = refinement_data['calculated_pattern']
            
            # Plot experimental data (normalized)
            ax.plot(two_theta, intensity, 
                   color=self.plot_settings['exp_color'],
                   linewidth=self.plot_settings['exp_linewidth'],
                   label='Experimental',
                   alpha=0.8)
            
            # Plot calculated pattern (normalized, same scale as experimental)
            ax.plot(two_theta, calculated_pattern,
                   color=self.plot_settings['calc_color'],
                   linewidth=self.plot_settings['calc_linewidth'],
                   label='Calculated (Le Bail)',
                   alpha=0.8)
            
            # Plot difference
            difference = intensity - calculated_pattern
            ax.plot(two_theta, difference,
                   color=self.plot_settings['diff_color'],
                   linewidth=self.plot_settings['diff_linewidth'],
                   label='Difference',
                   alpha=0.6)
            
            # Add R-factors to title
            r_factors = self.lebail_results['r_factors']
            title = f"{self.plot_settings['title']} (Rwp = {r_factors['Rwp']:.2f}%, GoF = {r_factors['GoF']:.2f})"
            ax.set_title(title)
        else:
            # No refinement results, plot original experimental data
            two_theta = self.experimental_pattern['two_theta']
            intensity = self.experimental_pattern['intensity']
            
            # Plot experimental data
            ax.plot(two_theta, intensity, 
                   color=self.plot_settings['exp_color'],
                   linewidth=self.plot_settings['exp_linewidth'],
                   label='Experimental',
                   alpha=0.8)
            
            ax.set_title(self.plot_settings['title'])
        
        ax.set_xlabel(self.plot_settings['xlabel'])
        ax.set_ylabel(self.plot_settings['ylabel'])
        
        if self.plot_settings['show_legend']:
            ax.legend()
            
        if self.plot_settings['show_grid']:
            ax.grid(True, alpha=0.3)
            
        self.figure.tight_layout()
        
    def _create_waterfall_plot(self):
        """Create waterfall plot with individual phases"""
        ax = self.figure.add_subplot(111)
        
        two_theta = self.experimental_pattern['two_theta']
        intensity = self.experimental_pattern['intensity']
        offset = self.plot_settings['waterfall_offset']
        
        # Plot experimental data at top
        current_offset = offset * (len(self.matched_phases) + 1)
        ax.plot(two_theta, intensity + current_offset,
               color=self.plot_settings['exp_color'],
               linewidth=self.plot_settings['exp_linewidth'],
               label='Experimental',
               alpha=0.8)
        
        # Plot each phase
        if self.lebail_results and self.lebail_results['success']:
            refined_phases = self.lebail_results['refinement_results']['refined_phases']
            
            for i, phase in enumerate(reversed(refined_phases)):
                current_offset = offset * i
                
                # Get phase info
                phase_data = phase['data']
                if 'phase' in phase_data:
                    phase_info = phase_data['phase']
                    mineral = phase_info.get('mineral', f'Phase {i+1}')
                    phase_id = phase_info.get('id', f'phase_{i}')
                else:
                    mineral = phase_data.get('mineral', f'Phase {i+1}')
                    phase_id = f'phase_{i}'
                
                # Get color
                color = self.plot_settings['phase_colors'].get(phase_id, self._get_default_color(i))
                
                # Calculate phase pattern
                # Note: This is simplified - in practice you'd extract individual phase contributions
                # For now, just show the theoretical peaks
                theo_peaks = phase['theoretical_peaks']
                if len(theo_peaks.get('two_theta', [])) > 0:
                    # Create stick pattern
                    for pos, int_val in zip(theo_peaks['two_theta'], theo_peaks['intensity']):
                        ax.plot([pos, pos], [current_offset, current_offset + int_val * phase['parameters']['scale_factor']],
                               color=color, linewidth=2, alpha=0.7)
                    
                    # Add label
                    ax.text(two_theta[0], current_offset + offset * 0.3, mineral,
                           fontsize=9, color=color)
        
        ax.set_xlabel(self.plot_settings['xlabel'])
        ax.set_ylabel('Intensity (offset)')
        ax.set_title(f"{self.plot_settings['title']} (Waterfall)")
        
        if self.plot_settings['show_grid']:
            ax.grid(True, alpha=0.3)
            
        self.figure.tight_layout()
        
    def choose_color(self, setting_key: str):
        """Open color picker dialog"""
        current_color = QColor(self.plot_settings[setting_key])
        color = QColorDialog.getColor(current_color, self, "Choose Color")
        
        if color.isValid():
            self.plot_settings[setting_key] = color.name()
            self.update_plot()
            
    def customize_phase_colors(self):
        """Customize colors for individual phases"""
        # This could open a more detailed dialog
        QMessageBox.information(self, "Phase Colors", 
                               "Click on phase rows in the table and use the color buttons to customize.")
        
    def update_setting(self, key: str, value):
        """Update a plot setting"""
        self.plot_settings[key] = value
    
    def _realtime_plot_callback(self, iteration_result, experimental_data):
        """Callback for real-time plotting during refinement"""
        from PyQt5.QtWidgets import QApplication
        
        # Update plot every iteration
        two_theta = experimental_data['two_theta']
        intensity = experimental_data['intensity']
        calculated = iteration_result['calculated_pattern']
        r_factors = iteration_result['r_factors']
        
        # Clear and redraw
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot experimental
        ax.plot(two_theta, intensity, 'b-', label='Experimental', alpha=0.7, linewidth=1.5)
        
        # Plot calculated
        ax.plot(two_theta, calculated, 'r-', label='Calculated', alpha=0.7, linewidth=1.5)
        
        # Plot difference
        difference = intensity - calculated
        ax.plot(two_theta, difference, 'g-', label='Difference', alpha=0.5, linewidth=1.0)
        
        # Add iteration info
        iteration = iteration_result['iteration']
        stage = iteration_result.get('stage', 0)
        stage_label = f"Stage {stage} - " if stage > 0 else ""
        title = f"{stage_label}Iteration {iteration}: Rwp={r_factors['Rwp']:.2f}%, GoF={r_factors.get('GoF', 0):.2f}"
        ax.set_title(title)
        ax.set_xlabel('2θ (degrees)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Process events to update GUI
        QApplication.processEvents()
        
    def export_plot(self, format: str):
        """Export the current plot"""
        if not self.experimental_pattern:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
            
        # Get save filename
        filters = {
            'png': "PNG Image (*.png)",
            'pdf': "PDF Document (*.pdf)",
            'svg': "SVG Vector (*.svg)"
        }
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Export Plot as {format.upper()}", 
            f"xrd_plot.{format}",
            filters[format]
        )
        
        if filename:
            try:
                dpi = self.plot_settings['dpi']
                self.figure.savefig(filename, dpi=dpi, bbox_inches='tight')
                QMessageBox.information(self, "Export Successful", 
                                       f"Plot saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", 
                                    f"Error saving plot:\n{str(e)}")
                
    def export_data(self):
        """Export data as CSV"""
        if not self.experimental_pattern:
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data as CSV",
            "xrd_data.csv",
            "CSV File (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    # Header
                    f.write("2theta,intensity_exp")
                    
                    if self.lebail_results and self.lebail_results['success']:
                        f.write(",intensity_calc,difference")
                    
                    f.write("\n")
                    
                    # Data
                    two_theta = self.experimental_pattern['two_theta']
                    intensity = self.experimental_pattern['intensity']
                    
                    if self.lebail_results and self.lebail_results['success']:
                        calc = self.lebail_results['refinement_results']['calculated_pattern']
                        diff = intensity - calc
                        
                        for tt, ie, ic, id in zip(two_theta, intensity, calc, diff):
                            f.write(f"{tt},{ie},{ic},{id}\n")
                    else:
                        for tt, ie in zip(two_theta, intensity):
                            f.write(f"{tt},{ie}\n")
                
                QMessageBox.information(self, "Export Successful",
                                       f"Data saved to:\n{filename}")
                                       
            except Exception as e:
                QMessageBox.critical(self, "Export Failed",
                                    f"Error saving data:\n{str(e)}")
