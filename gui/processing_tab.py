"""
Data processing tab for XRD pattern preprocessing and background subtraction
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QSlider, QCheckBox, QSplitter,
                             QMessageBox, QProgressBar, QTabWidget, QGridLayout,
                             QFrame, QToolButton, QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class ProcessingTab(QWidget):
    """Tab for data processing and background subtraction"""
    
    pattern_processed = pyqtSignal(dict)  # Signal emitted when pattern is processed
    peaks_found = pyqtSignal(dict)  # Signal emitted when peaks are found
    
    def __init__(self):
        super().__init__()
        self.pattern_data = None
        self.original_pattern_data = None
        self.background_data = None
        self.processed_pattern_data = None
        self.peaks = None
        self.manual_peaks = []  # User-added peaks
        self.removed_peaks = []  # User-removed peaks
        self.wavelength = 1.5406  # Default Cu Ka1
        self.peak_editing_mode = False  # Toggle for peak editing
        
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_background_preview)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - controls
        controls_widget = self.create_controls_panel()
        splitter.addWidget(controls_widget)
        
        # Right panel - plot
        plot_widget = self.create_plot_widget()
        splitter.addWidget(plot_widget)
        
        # Set splitter proportions (30% controls, 70% plot)
        splitter.setSizes([400, 900])
        
    def create_controls_panel(self):
        """Create the controls panel with tabbed interface"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Pattern info (always visible)
        info_group = self.create_pattern_info_group()
        layout.addWidget(info_group)
        
        # Create tabbed interface for different processing options
        tab_widget = QTabWidget()
        
        # Background subtraction tab
        bg_tab = QWidget()
        bg_layout = QVBoxLayout(bg_tab)
        bg_group = self.create_background_group()
        bg_layout.addWidget(bg_group)
        bg_layout.addStretch()
        tab_widget.addTab(bg_tab, "Background")
        
        # Corrections tab
        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)
        corr_group = self.create_corrections_group()
        corr_layout.addWidget(corr_group)
        corr_layout.addStretch()
        tab_widget.addTab(corr_tab, "Corrections")
        
        # Peak detection tab
        peak_tab = QWidget()
        peak_layout = QVBoxLayout(peak_tab)
        peak_group = self.create_peak_detection_group()
        peak_layout.addWidget(peak_group)
        peak_layout.addStretch()
        tab_widget.addTab(peak_tab, "Peak Detection")
        
        # Additional processing tab
        filter_tab = QWidget()
        filter_layout = QVBoxLayout(filter_tab)
        filter_group = self.create_filtering_group()
        filter_layout.addWidget(filter_group)
        filter_layout.addStretch()
        tab_widget.addTab(filter_tab, "Filtering")
        
        layout.addWidget(tab_widget)
        
        # Action buttons (always visible at bottom)
        actions_group = self.create_actions_group()
        layout.addWidget(actions_group)
        
        return widget
        
    def create_pattern_info_group(self):
        """Create pattern information group"""
        group = QGroupBox("Pattern Information")
        layout = QVBoxLayout(group)
        
        self.pattern_info_label = QLabel("No pattern loaded")
        layout.addWidget(self.pattern_info_label)
        
        self.data_points_label = QLabel("Data points: -")
        layout.addWidget(self.data_points_label)
        
        self.range_label = QLabel("2θ range: -")
        layout.addWidget(self.range_label)
        
        return group
        
    def create_background_group(self):
        """Create compact background subtraction controls"""
        group = QGroupBox("ALS Background Subtraction")
        layout = QVBoxLayout(group)
        
        # Enable/disable background subtraction
        self.enable_bg_subtraction = QCheckBox("Enable Background Subtraction")
        self.enable_bg_subtraction.stateChanged.connect(self.on_bg_enable_changed)
        layout.addWidget(self.enable_bg_subtraction)
        
        # Parameters in grid layout for compactness
        params_layout = QGridLayout()
        
        # Lambda parameter (smoothness)
        params_layout.addWidget(QLabel("Smoothness (λ):"), 0, 0)
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(2, 8)  # 10^2 to 10^8
        self.lambda_slider.setValue(5)  # 10^5
        self.lambda_slider.valueChanged.connect(self.on_lambda_changed)
        params_layout.addWidget(self.lambda_slider, 0, 1)
        self.lambda_value_label = QLabel("1e5")
        params_layout.addWidget(self.lambda_value_label, 0, 2)
        
        # P parameter (asymmetry)
        params_layout.addWidget(QLabel("Asymmetry (p):"), 1, 0)
        self.p_spinbox = QDoubleSpinBox()
        self.p_spinbox.setRange(0.001, 0.1)
        self.p_spinbox.setValue(0.01)
        self.p_spinbox.setDecimals(3)
        self.p_spinbox.setSingleStep(0.001)
        self.p_spinbox.valueChanged.connect(self.on_parameter_changed)
        params_layout.addWidget(self.p_spinbox, 1, 1, 1, 2)
        
        # Iterations
        params_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(5, 50)
        self.iterations_spinbox.setValue(10)
        self.iterations_spinbox.valueChanged.connect(self.on_parameter_changed)
        params_layout.addWidget(self.iterations_spinbox, 2, 1, 1, 2)
        
        layout.addLayout(params_layout)
        
        # Preview options in horizontal layout
        preview_layout = QHBoxLayout()
        self.show_background = QCheckBox("Show BG")
        self.show_background.setChecked(True)
        self.show_background.stateChanged.connect(self.update_plot)
        preview_layout.addWidget(self.show_background)
        
        self.show_original = QCheckBox("Show Orig")
        self.show_original.stateChanged.connect(self.update_plot)
        preview_layout.addWidget(self.show_original)
        
        self.realtime_preview = QCheckBox("Real-time")
        self.realtime_preview.setChecked(True)
        preview_layout.addWidget(self.realtime_preview)
        layout.addLayout(preview_layout)
        
        # Progress bar for processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Initially disabled
        self.set_bg_controls_enabled(False)
        
        return group
        
    def create_corrections_group(self):
        """Create sample displacement and other corrections"""
        group = QGroupBox("Sample Displacement & Corrections")
        layout = QVBoxLayout(group)
        
        # Sample displacement correction
        disp_layout = QHBoxLayout()
        disp_layout.addWidget(QLabel("2θ offset (°):"))
        
        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(-2.0, 2.0)
        self.displacement_spin.setDecimals(4)
        self.displacement_spin.setValue(0.0000)
        self.displacement_spin.setSingleStep(0.0010)
        self.displacement_spin.setToolTip("Sample displacement correction - shifts all 2θ values by this amount")
        self.displacement_spin.valueChanged.connect(self.apply_displacement_correction)
        disp_layout.addWidget(self.displacement_spin)
        
        # Auto-correct button
        auto_correct_btn = QPushButton("Auto-Correct")
        auto_correct_btn.setToolTip("Automatically estimate displacement using peak positions")
        auto_correct_btn.clicked.connect(self.auto_correct_displacement)
        disp_layout.addWidget(auto_correct_btn)
        
        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset displacement to zero")
        reset_btn.clicked.connect(self.reset_displacement)
        disp_layout.addWidget(reset_btn)
        
        layout.addLayout(disp_layout)
        
        # Info label
        self.displacement_info = QLabel("Sample displacement can cause systematic peak shifts.\nAdjust the offset to align experimental peaks with reference patterns.")
        self.displacement_info.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        self.displacement_info.setWordWrap(True)
        layout.addWidget(self.displacement_info)
        
        # Initially disabled
        self.set_correction_controls_enabled(False)
        
        return group
        
    def create_filtering_group(self):
        """Create filtering and smoothing controls"""
        group = QGroupBox("Additional Processing")
        layout = QVBoxLayout(group)
        
        # Smoothing
        self.enable_smoothing = QCheckBox("Enable Smoothing")
        self.enable_smoothing.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.enable_smoothing)
        
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Window Size:"))
        
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(3, 21)
        self.smooth_window.setValue(5)
        self.smooth_window.setSingleStep(2)  # Keep odd numbers
        self.smooth_window.valueChanged.connect(self.on_parameter_changed)
        smooth_layout.addWidget(self.smooth_window)
        layout.addLayout(smooth_layout)
        
        # Noise reduction
        self.enable_noise_reduction = QCheckBox("Noise Reduction")
        self.enable_noise_reduction.stateChanged.connect(self.on_parameter_changed)
        layout.addWidget(self.enable_noise_reduction)
        
        return group
        
    def create_peak_detection_group(self):
        """Create compact peak detection controls"""
        group = QGroupBox("Peak Detection & Manual Editing")
        layout = QVBoxLayout(group)
        
        # Peak editing mode toggle
        edit_layout = QHBoxLayout()
        self.peak_edit_btn = QPushButton("Enable Peak Editing")
        self.peak_edit_btn.setCheckable(True)
        self.peak_edit_btn.clicked.connect(self.toggle_peak_editing)
        edit_layout.addWidget(self.peak_edit_btn)
        
        self.clear_manual_btn = QPushButton("Clear Manual")
        self.clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        edit_layout.addWidget(self.clear_manual_btn)
        layout.addLayout(edit_layout)
        
        # Instructions
        instructions = QLabel("Click plot to add peaks, right-click to remove")
        instructions.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(instructions)
        
        # Compact parameter grid
        params_layout = QGridLayout()
        
        # Height and prominence
        params_layout.addWidget(QLabel("Min Height:"), 0, 0)
        self.min_height = QSpinBox()
        self.min_height.setRange(1, 10000)
        self.min_height.setValue(50)
        params_layout.addWidget(self.min_height, 0, 1)
        
        params_layout.addWidget(QLabel("Min Prominence:"), 0, 2)
        self.min_prominence = QSpinBox()
        self.min_prominence.setRange(1, 1000)
        self.min_prominence.setValue(10)
        params_layout.addWidget(self.min_prominence, 0, 3)
        
        # Width and distance
        params_layout.addWidget(QLabel("Min Width:"), 1, 0)
        self.min_width = QSpinBox()
        self.min_width.setRange(1, 20)
        self.min_width.setValue(1)
        params_layout.addWidget(self.min_width, 1, 1)
        
        params_layout.addWidget(QLabel("Min Distance:"), 1, 2)
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 50)
        self.min_distance.setValue(3)
        params_layout.addWidget(self.min_distance, 1, 3)
        
        layout.addLayout(params_layout)
        
        # Sensitivity and options
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity = QComboBox()
        self.sensitivity.addItems(["High", "Medium", "Low"])
        self.sensitivity.setCurrentIndex(0)
        sens_layout.addWidget(self.sensitivity)
        
        self.show_all_candidates = QCheckBox("Show candidates")
        self.show_all_candidates.stateChanged.connect(self.update_plot)
        sens_layout.addWidget(self.show_all_candidates)
        layout.addLayout(sens_layout)
        
        # Find peaks button
        self.find_peaks_btn = QPushButton("Find Peaks")
        self.find_peaks_btn.clicked.connect(self.find_peaks)
        self.find_peaks_btn.setEnabled(False)
        layout.addWidget(self.find_peaks_btn)
        
        return group
        
    def create_actions_group(self):
        """Create compact action buttons"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        
        # Main action buttons in horizontal layout
        main_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply Processing")
        self.apply_btn.clicked.connect(self.apply_processing)
        self.apply_btn.setEnabled(False)
        main_layout.addWidget(self.apply_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_to_original)
        self.reset_btn.setEnabled(False)
        main_layout.addWidget(self.reset_btn)
        
        layout.addLayout(main_layout)
        
        # Export button
        self.export_btn = QPushButton("Export Processed Data")
        self.export_btn.clicked.connect(self.export_processed_data)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        return group
        
    def create_plot_widget(self):
        """Create the matplotlib plot widget"""
        group = QGroupBox("Pattern Preview")
        layout = QVBoxLayout(group)
        
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('2θ (degrees)')
        self.ax.set_ylabel('Intensity (counts)')
        self.ax.set_title('XRD Pattern Processing Preview')
        self.ax.grid(True, alpha=0.3)
        
        # Connect mouse events for peak editing
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        return group
        
    def toggle_peak_editing(self):
        """Toggle peak editing mode"""
        self.peak_editing_mode = self.peak_edit_btn.isChecked()
        if self.peak_editing_mode:
            self.peak_edit_btn.setText("Disable Peak Editing")
            self.peak_edit_btn.setStyleSheet("QPushButton { background-color: #ffcccc; }")
        else:
            self.peak_edit_btn.setText("Enable Peak Editing")
            self.peak_edit_btn.setStyleSheet("")
            
    def clear_manual_peaks(self):
        """Clear all manually added/removed peaks"""
        self.manual_peaks.clear()
        self.removed_peaks.clear()
        self.update_plot()
        
    def on_plot_click(self, event):
        """Handle mouse clicks on the plot for peak editing"""
        if not self.peak_editing_mode or event.inaxes != self.ax:
            return
            
        if self.processed_pattern_data is None:
            return
            
        # Get click position
        click_2theta = event.xdata
        if click_2theta is None:
            return
            
        # Find closest data point
        two_theta = self.processed_pattern_data['two_theta']
        intensity = self.processed_pattern_data['intensity']
        
        # Find closest index
        closest_idx = np.argmin(np.abs(two_theta - click_2theta))
        closest_2theta = two_theta[closest_idx]
        closest_intensity = intensity[closest_idx]
        
        if event.button == 1:  # Left click - add peak
            # Check if peak already exists (within tolerance)
            tolerance = 0.05  # 0.05 degrees tolerance
            existing_peak = False
            
            # Check automatic peaks
            if self.peaks is not None:
                for peak_idx in self.peaks:
                    if abs(two_theta[peak_idx] - closest_2theta) < tolerance:
                        existing_peak = True
                        break
                        
            # Check manual peaks
            for manual_peak in self.manual_peaks:
                if abs(manual_peak['two_theta'] - closest_2theta) < tolerance:
                    existing_peak = True
                    break
                    
            if not existing_peak:
                # Add manual peak
                manual_peak = {
                    'index': closest_idx,
                    'two_theta': closest_2theta,
                    'intensity': closest_intensity,
                    'd_spacing': self.wavelength / (2 * np.sin(np.radians(closest_2theta / 2)))
                }
                self.manual_peaks.append(manual_peak)
                print(f"Added manual peak at 2θ = {closest_2theta:.3f}°")
                
        elif event.button == 3:  # Right click - remove peak
            tolerance = 0.1  # Slightly larger tolerance for removal
            
            # Check if clicking on an automatic peak
            if self.peaks is not None:
                for peak_idx in self.peaks:
                    if abs(two_theta[peak_idx] - closest_2theta) < tolerance:
                        # Add to removed peaks list
                        removed_peak = {
                            'index': peak_idx,
                            'two_theta': two_theta[peak_idx],
                            'intensity': intensity[peak_idx]
                        }
                        if removed_peak not in self.removed_peaks:
                            self.removed_peaks.append(removed_peak)
                            print(f"Removed automatic peak at 2θ = {two_theta[peak_idx]:.3f}°")
                        break
                        
            # Check if clicking on a manual peak
            for i, manual_peak in enumerate(self.manual_peaks):
                if abs(manual_peak['two_theta'] - closest_2theta) < tolerance:
                    removed_peak = self.manual_peaks.pop(i)
                    print(f"Removed manual peak at 2θ = {removed_peak['two_theta']:.3f}°")
                    break
                    
        self.update_plot()
        
    def get_effective_peaks(self):
        """Get the effective peak list (automatic + manual - removed)"""
        effective_peaks = []
        two_theta = self.processed_pattern_data['two_theta'] if self.processed_pattern_data else []
        intensity = self.processed_pattern_data['intensity'] if self.processed_pattern_data else []
        
        # Add automatic peaks (excluding removed ones)
        if self.peaks is not None and len(two_theta) > 0:
            for peak_idx in self.peaks:
                # Check if this peak was manually removed
                is_removed = False
                for removed_peak in self.removed_peaks:
                    if removed_peak['index'] == peak_idx:
                        is_removed = True
                        break
                        
                if not is_removed and peak_idx < len(two_theta):
                    effective_peaks.append({
                        'index': peak_idx,
                        'two_theta': two_theta[peak_idx],
                        'intensity': intensity[peak_idx],
                        'd_spacing': self.wavelength / (2 * np.sin(np.radians(two_theta[peak_idx] / 2))),
                        'type': 'automatic'
                    })
                    
        # Add manual peaks
        for manual_peak in self.manual_peaks:
            manual_peak['type'] = 'manual'
            effective_peaks.append(manual_peak)
            
        # Sort by 2theta
        effective_peaks.sort(key=lambda x: x['two_theta'])
        
        return effective_peaks
        
    def set_pattern_data(self, pattern_data):
        """Set the pattern data for processing"""
        self.pattern_data = pattern_data.copy()
        self.original_pattern_data = pattern_data.copy()
        self.processed_pattern_data = pattern_data.copy()
        self.background_data = None
        self.peaks = None
        self.manual_peaks.clear()
        self.removed_peaks.clear()
        
        # Get wavelength from pattern data if available
        if 'wavelength' in pattern_data:
            self.wavelength = pattern_data['wavelength']
        
        # Update UI
        self.update_pattern_info()
        self.update_plot()
        self.apply_btn.setEnabled(True)
        self.find_peaks_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        self.export_btn.setEnabled(True)
        
        # Enable correction controls
        self.set_correction_controls_enabled(True)
        
    def update_pattern_info(self):
        """Update pattern information display"""
        if self.pattern_data is None:
            self.pattern_info_label.setText("No pattern loaded")
            self.data_points_label.setText("Data points: -")
            self.range_label.setText("2θ range: -")
            return
            
        file_name = self.pattern_data.get('file_path', 'Unknown').split('/')[-1]
        file_format = self.pattern_data.get('file_format', 'Unknown')
        
        self.pattern_info_label.setText(f"File: {file_name} ({file_format})")
        
        n_points = len(self.pattern_data['two_theta'])
        self.data_points_label.setText(f"Data points: {n_points}")
        
        min_2theta = np.min(self.pattern_data['two_theta'])
        max_2theta = np.max(self.pattern_data['two_theta'])
        self.range_label.setText(f"2θ range: {min_2theta:.2f}° - {max_2theta:.2f}°")
        
    def on_bg_enable_changed(self, state):
        """Handle background subtraction enable/disable"""
        enabled = state == Qt.CheckState.Checked
        self.set_bg_controls_enabled(enabled)
        
        if enabled and self.realtime_preview.isChecked():
            self.start_update_timer()
        else:
            self.update_plot()
            
    def set_bg_controls_enabled(self, enabled):
        """Enable/disable background subtraction controls"""
        self.lambda_slider.setEnabled(enabled)
        self.p_spinbox.setEnabled(enabled)
        self.iterations_spinbox.setEnabled(enabled)
        self.show_background.setEnabled(enabled)
        
    def on_lambda_changed(self, value):
        """Handle lambda slider change"""
        lambda_val = 10**value
        self.lambda_value_label.setText(f"1e{value}")
        
        if self.realtime_preview.isChecked():
            self.start_update_timer()
            
    def on_parameter_changed(self):
        """Handle parameter changes"""
        if self.realtime_preview.isChecked():
            self.start_update_timer()
            
    def start_update_timer(self):
        """Start the update timer for real-time preview"""
        self.update_timer.stop()
        self.update_timer.start(500)  # 500ms delay
        
    def update_background_preview(self):
        """Update background subtraction preview"""
        if not self.enable_bg_subtraction.isChecked() or self.pattern_data is None:
            return
            
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Calculate background
            lambda_val = 10**self.lambda_slider.value()
            p_val = self.p_spinbox.value()
            n_iter = self.iterations_spinbox.value()
            
            self.background_data = self.als_baseline(
                self.original_pattern_data['intensity'],
                lam=lambda_val, p=p_val, niter=n_iter
            )
            
            # Apply processing
            self.apply_current_processing()
            self.update_plot()
            
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            print(f"Error in background preview: {e}")
            
    def als_baseline(self, y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares (ALS) baseline correction
        """
        try:
            L = len(y)
            D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            D = lam * D.dot(D.transpose())
            
            w = np.ones(L)
            W = diags(w, 0, shape=(L, L))
            
            for i in range(niter):
                W.setdiag(w)
                Z = W + D
                z = spsolve(Z, w*y)
                w = p * (y > z) + (1-p) * (y < z)
                
            return z
            
        except Exception as e:
            print(f"Error in ALS baseline correction: {e}")
            return np.zeros_like(y)
            
    def apply_current_processing(self):
        """Apply current processing settings to create processed pattern"""
        if self.original_pattern_data is None:
            return
            
        # Start with original data
        processed_intensity = self.original_pattern_data['intensity'].copy()
        
        # Apply background subtraction
        if self.enable_bg_subtraction.isChecked() and self.background_data is not None:
            processed_intensity = processed_intensity - self.background_data
            processed_intensity = np.maximum(processed_intensity, 0)  # No negative values
            
        # Apply smoothing
        if self.enable_smoothing.isChecked():
            from scipy.ndimage import uniform_filter1d
            window_size = self.smooth_window.value()
            processed_intensity = uniform_filter1d(processed_intensity, size=window_size)
            
        # Apply noise reduction (simple median filter)
        if self.enable_noise_reduction.isChecked():
            from scipy.ndimage import median_filter
            processed_intensity = median_filter(processed_intensity, size=3)
            
        # Update processed pattern data
        self.processed_pattern_data = self.original_pattern_data.copy()
        self.processed_pattern_data['intensity'] = processed_intensity
        
    def update_plot(self):
        """Update the plot with current data"""
        if self.pattern_data is None:
            return
            
        self.ax.clear()
        
        # Plot original pattern if requested
        if self.show_original.isChecked():
            self.ax.plot(self.original_pattern_data['two_theta'], 
                        self.original_pattern_data['intensity'], 
                        'lightblue', linewidth=1, alpha=0.7, label='Original')
        
        # Plot processed pattern
        if self.processed_pattern_data is not None:
            if self.processed_pattern_data.get('intensity_error') is not None:
                self.ax.errorbar(
                    self.processed_pattern_data['two_theta'],
                    self.processed_pattern_data['intensity'],
                    yerr=self.processed_pattern_data['intensity_error'],
                    fmt='b-', linewidth=1, elinewidth=0.5, capsize=0, alpha=0.8,
                    label='Processed'
                )
            else:
                self.ax.plot(self.processed_pattern_data['two_theta'], 
                           self.processed_pattern_data['intensity'], 
                           'b-', linewidth=1, label='Processed')
        
        # Plot automatic peaks (excluding removed ones)
        if self.peaks is not None and self.processed_pattern_data is not None:
            auto_peaks = []
            for peak_idx in self.peaks:
                # Check if this peak was manually removed
                is_removed = any(removed['index'] == peak_idx for removed in self.removed_peaks)
                if not is_removed:
                    auto_peaks.append(peak_idx)
                    
            if auto_peaks:
                peak_positions = self.processed_pattern_data['two_theta'][auto_peaks]
                peak_intensities = self.processed_pattern_data['intensity'][auto_peaks]
                self.ax.plot(peak_positions, peak_intensities, 'ro', markersize=5, label='Auto Peaks')
                
        # Plot manual peaks
        if self.manual_peaks:
            manual_positions = [peak['two_theta'] for peak in self.manual_peaks]
            manual_intensities = [peak['intensity'] for peak in self.manual_peaks]
            self.ax.plot(manual_positions, manual_intensities, 'go', markersize=6, 
                        marker='s', label='Manual Peaks')
                        
        # Plot removed peaks (grayed out)
        if self.removed_peaks and self.processed_pattern_data is not None:
            removed_positions = []
            removed_intensities = []
            for removed in self.removed_peaks:
                if removed['index'] < len(self.processed_pattern_data['two_theta']):
                    removed_positions.append(self.processed_pattern_data['two_theta'][removed['index']])
                    removed_intensities.append(self.processed_pattern_data['intensity'][removed['index']])
            if removed_positions:
                self.ax.plot(removed_positions, removed_intensities, 'x', color='gray', 
                           markersize=6, alpha=0.5, label='Removed Peaks')
            
        # Plot candidate peaks if requested and available
        if (hasattr(self, 'candidate_peaks') and self.candidate_peaks is not None and 
            self.show_all_candidates.isChecked() and self.processed_pattern_data is not None):
            candidate_positions = self.processed_pattern_data['two_theta'][self.candidate_peaks]
            candidate_intensities = self.processed_pattern_data['intensity'][self.candidate_peaks]
            self.ax.plot(candidate_positions, candidate_intensities, 'yo', markersize=3, 
                        alpha=0.7, label='All Candidates')
        
        # Plot background if requested and available
        if (self.show_background.isChecked() and 
            self.enable_bg_subtraction.isChecked() and 
            self.background_data is not None):
            self.ax.plot(self.original_pattern_data['two_theta'], 
                        self.background_data, 
                        'g--', linewidth=1, alpha=0.7, label='Background')
        
        self.ax.set_xlabel('2θ (degrees)')
        self.ax.set_ylabel('Intensity (counts)')
        
        # Update title based on processing status
        title = 'XRD Pattern Processing Preview'
        if self.enable_bg_subtraction.isChecked():
            title += ' (Background Subtracted)'
        if self.enable_smoothing.isChecked():
            title += ' (Smoothed)'
        if self.enable_noise_reduction.isChecked():
            title += ' (Noise Reduced)'
        if self.peak_editing_mode:
            title += ' - Peak Editing Mode'
            
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
        
    def apply_processing(self):
        """Apply current processing and emit signal"""
        if self.processed_pattern_data is None:
            return
            
        try:
            # Apply current processing
            self.apply_current_processing()
            
            # Update main pattern data and mark as processed
            self.pattern_data = self.processed_pattern_data.copy()
            self.pattern_data['processed'] = True
            
            # Enable reset button
            self.reset_btn.setEnabled(True)
            
            # Emit processed pattern
            self.pattern_processed.emit(self.pattern_data)
            
            QMessageBox.information(self, "Success", "Processing applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not apply processing:\n{str(e)}")
            
    def reset_to_original(self):
        """Reset to original pattern data"""
        if self.original_pattern_data is None:
            return
            
        try:
            self.pattern_data = self.original_pattern_data.copy()
            self.processed_pattern_data = self.original_pattern_data.copy()
            self.background_data = None
            
            # Reset UI controls
            self.enable_bg_subtraction.setChecked(False)
            self.enable_smoothing.setChecked(False)
            self.enable_noise_reduction.setChecked(False)
            
            self.update_plot()
            self.reset_btn.setEnabled(False)
            
            # Emit original pattern
            self.pattern_processed.emit(self.pattern_data)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not reset pattern:\n{str(e)}")
    
    def find_peaks(self):
        """Find peaks in the processed pattern with improved filtering"""
        if self.processed_pattern_data is None:
            return
            
        try:
            intensity = self.processed_pattern_data['intensity']
            two_theta = self.processed_pattern_data['two_theta']
            
            # Ensure arrays are numpy arrays with proper types
            intensity = np.asarray(intensity, dtype=float)
            two_theta = np.asarray(two_theta, dtype=float)
            
            if len(intensity) == 0 or len(two_theta) == 0:
                QMessageBox.warning(self, "Warning", "No data available for peak detection")
                return
            
            # Get user-defined parameters
            min_height = self.min_height.value()
            min_prominence = self.min_prominence.value()
            min_width = self.min_width.value()
            min_distance = self.min_distance.value()
            sensitivity = self.sensitivity.currentIndex()  # 0=High, 1=Medium, 2=Low
            
            # Adjust parameters based on sensitivity
            if sensitivity == 0:  # High sensitivity for small peaks
                # Use user values directly, minimal noise filtering
                height_threshold = min_height
                prominence_threshold = min_prominence
                width_threshold = min_width
                distance_threshold = min_distance
            elif sensitivity == 1:  # Medium sensitivity
                # Add some noise-based adjustment
                noise_level = np.std(intensity[:min(100, len(intensity) // 10)])
                height_threshold = max(min_height, noise_level * 2)
                prominence_threshold = max(min_prominence, noise_level * 0.5)
                width_threshold = min_width
                distance_threshold = min_distance
            else:  # Low sensitivity - only large peaks
                noise_level = np.std(intensity[:min(100, len(intensity) // 10)])
                height_threshold = max(min_height, noise_level * 5)
                prominence_threshold = max(min_prominence, noise_level * 2)
                width_threshold = max(min_width, 2)
                distance_threshold = max(min_distance, 5)
            
            print(f"Peak detection parameters: height={height_threshold}, prominence={prominence_threshold}, width={width_threshold}, distance={distance_threshold}")
            
            # Find peaks using scipy with user-controlled criteria
            peaks, properties = find_peaks(
                intensity,
                height=height_threshold,
                distance=distance_threshold,
                prominence=prominence_threshold,
                width=width_threshold
            )
            
            # Ensure peaks are integers
            peaks = np.asarray(peaks, dtype=int)
            
            # Store candidate peaks for visualization
            self.candidate_peaks = peaks.copy()
            
            if len(peaks) == 0:
                QMessageBox.warning(self, "Warning", "No peaks found with current parameters.\nTry lowering the height, prominence, or width thresholds.")
                return
            
            # Minimal additional filtering based on sensitivity
            filtered_peaks = []
            peak_heights = intensity[peaks]
            
            for i, peak_idx in enumerate(peaks):
                # Ensure peak_idx is within bounds
                if peak_idx < 0 or peak_idx >= len(intensity):
                    continue
                    
                peak_height = peak_heights[i]
                peak_2theta = two_theta[peak_idx]
                
                # Skip peaks at very low angles (likely artifacts) - but be more lenient
                if peak_2theta < 3.0:  # Reduced from 5.0 to allow more peaks
                    continue
                
                # For high sensitivity mode, skip most additional filtering
                if sensitivity == 0:  # High sensitivity - minimal filtering
                    filtered_peaks.append(int(peak_idx))
                else:
                    # For medium/low sensitivity, apply some filtering
                    window_size = 5  # Smaller window for local analysis
                    start_idx = max(0, int(peak_idx) - window_size)
                    end_idx = min(len(intensity), int(peak_idx) + window_size)
                    
                    # Ensure we have valid indices
                    if start_idx >= end_idx:
                        continue
                    
                    local_background = np.median(intensity[start_idx:end_idx])
                    noise_threshold = 1.5 if sensitivity == 1 else 3.0  # Less strict for medium
                    
                    if peak_height > local_background + noise_threshold:
                        filtered_peaks.append(int(peak_idx))
            
            # Convert to numpy array with integer type
            filtered_peaks = np.array(filtered_peaks, dtype=int)
            
            # Adjust peak limit based on sensitivity
            max_peaks = 100 if sensitivity == 0 else (75 if sensitivity == 1 else 50)
            
            # Limit to most significant peaks
            if len(filtered_peaks) > max_peaks:
                peak_intensities = intensity[filtered_peaks]
                sorted_indices = np.argsort(peak_intensities)[::-1]  # Sort descending
                filtered_peaks = filtered_peaks[sorted_indices[:max_peaks]]
                filtered_peaks = np.sort(filtered_peaks)  # Sort by position
            
            self.peaks = filtered_peaks
            
            # Get effective peaks (automatic + manual - removed)
            effective_peaks = self.get_effective_peaks()
            
            # Calculate peak data for matching using effective peaks
            if len(effective_peaks) > 0:
                # Convert to arrays for consistency
                peak_positions = np.array([p['two_theta'] for p in effective_peaks])
                peak_intensities = np.array([p['intensity'] for p in effective_peaks])
                peak_d_spacings = np.array([p['d_spacing'] for p in effective_peaks])
                
                # Calculate relative intensities
                max_intensity = np.max(peak_intensities)
                rel_intensities = (peak_intensities / max_intensity) * 100
                
                peak_data = {
                    'two_theta': peak_positions,
                    'intensity': peak_intensities,
                    'd_spacing': peak_d_spacings,
                    'rel_intensity': rel_intensities,
                    'wavelength': self.wavelength,
                    'manual_count': len(self.manual_peaks),
                    'removed_count': len(self.removed_peaks)
                }
                
                # Emit effective peak data signal
                self.peaks_found.emit(peak_data)
                
                auto_count = len(filtered_peaks) - len(self.removed_peaks)
                manual_count = len(self.manual_peaks)
                total_count = len(effective_peaks)
                
                message = f"Found {total_count} effective peaks!\n"
                message += f"({auto_count} automatic"
                if manual_count > 0:
                    message += f" + {manual_count} manual"
                if len(self.removed_peaks) > 0:
                    message += f" - {len(self.removed_peaks)} removed"
                message += f", filtered from {len(peaks)} initial candidates)"
                
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.warning(self, "Warning", "No effective peaks found after filtering and manual editing")
            
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not find peaks:\n{str(e)}")
            
    def export_processed_data(self):
        """Export processed pattern data"""
        if self.processed_pattern_data is None:
            QMessageBox.warning(self, "Warning", "No processed data to export")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Processed Pattern",
            "",
            "XY files (*.xy);;XYE files (*.xye);;Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Determine format based on extension
                if file_path.endswith('.xye') and self.processed_pattern_data.get('intensity_error') is not None:
                    # Export XYE format
                    data = np.column_stack([
                        self.processed_pattern_data['two_theta'],
                        self.processed_pattern_data['intensity'],
                        self.processed_pattern_data['intensity_error']
                    ])
                    header = "# 2theta\tIntensity\tError\n# Processed XRD pattern"
                else:
                    # Export XY format
                    data = np.column_stack([
                        self.processed_pattern_data['two_theta'],
                        self.processed_pattern_data['intensity']
                    ])
                    header = "# 2theta\tIntensity\n# Processed XRD pattern"
                
                np.savetxt(file_path, data, delimiter='\t', header=header)
                QMessageBox.information(self, "Success", f"Data exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not export data:\n{str(e)}")
    
    def set_correction_controls_enabled(self, enabled):
        """Enable/disable correction controls"""
        self.displacement_spin.setEnabled(enabled)
    
    def apply_displacement_correction(self):
        """Apply sample displacement correction to the pattern"""
        if self.original_pattern_data is None:
            return
        
        # Get displacement value
        displacement = self.displacement_spin.value()
        
        # Apply displacement to 2theta values (always from original data)
        corrected_two_theta = self.original_pattern_data['two_theta'] + displacement
        
        # Update pattern data
        self.pattern_data = self.original_pattern_data.copy()
        self.pattern_data['two_theta'] = corrected_two_theta
        self.pattern_data['displacement_correction'] = displacement
        
        # Update processed pattern if it exists - apply displacement to original processed data
        if hasattr(self, 'processed_pattern_data') and self.processed_pattern_data:
            # Store original processed data if not already stored
            if not hasattr(self, 'original_processed_pattern_data'):
                self.original_processed_pattern_data = self.processed_pattern_data.copy()
            
            # Apply displacement to original processed data
            self.processed_pattern_data = self.original_processed_pattern_data.copy()
            self.processed_pattern_data['two_theta'] = self.original_processed_pattern_data['two_theta'] + displacement
            self.processed_pattern_data['displacement_correction'] = displacement
        
        # Replot and emit signal
        self.update_plot()
        
        # Emit the corrected pattern
        pattern_to_emit = self.processed_pattern_data if hasattr(self, 'processed_pattern_data') and self.processed_pattern_data else self.pattern_data
        if pattern_to_emit:
            pattern_to_emit['processed'] = hasattr(self, 'processed_pattern_data') and self.processed_pattern_data is not None
            self.pattern_processed.emit(pattern_to_emit)
    
    def auto_correct_displacement(self):
        """Automatically estimate sample displacement correction"""
        if self.pattern_data is None:
            QMessageBox.warning(self, "Warning", "No pattern loaded")
            return
        
        # This is a simplified auto-correction
        # In practice, you might compare with known reference peaks
        QMessageBox.information(self, "Auto-Correction", 
            "Auto-correction requires reference peak positions.\n"
            "For now, manually adjust the 2θ offset based on known peak positions.\n\n"
            "Tip: If you know a peak should be at a specific 2θ value,\n"
            "calculate the difference and enter it as the offset.")
    
    def reset_displacement(self):
        """Reset displacement correction to zero"""
        self.displacement_spin.setValue(0.0)
        
        # Reset processed pattern data to original if it exists
        if hasattr(self, 'original_processed_pattern_data'):
            self.processed_pattern_data = self.original_processed_pattern_data.copy()
            delattr(self, 'original_processed_pattern_data')
        
        # This will trigger apply_displacement_correction with 0.0 displacement
