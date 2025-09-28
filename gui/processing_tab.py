"""
Data processing tab for XRD pattern preprocessing and background subtraction
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
                             QGroupBox, QSlider, QCheckBox, QSplitter,
                             QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
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
        self.wavelength = 1.5406  # Default Cu Ka1
        
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
        """Create the controls panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Pattern info
        info_group = self.create_pattern_info_group()
        layout.addWidget(info_group)
        
        # Background subtraction controls
        bg_group = self.create_background_group()
        layout.addWidget(bg_group)
        
        # Smoothing and filtering controls
        filter_group = self.create_filtering_group()
        layout.addWidget(filter_group)
        
        # Processing actions
        actions_group = self.create_actions_group()
        layout.addWidget(actions_group)
        
        layout.addStretch()
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
        """Create background subtraction controls"""
        group = QGroupBox("Background Subtraction (ALS)")
        layout = QVBoxLayout(group)
        
        # Enable/disable background subtraction
        self.enable_bg_subtraction = QCheckBox("Enable Background Subtraction")
        self.enable_bg_subtraction.stateChanged.connect(self.on_bg_enable_changed)
        layout.addWidget(self.enable_bg_subtraction)
        
        # Lambda parameter (smoothness)
        lambda_layout = QHBoxLayout()
        lambda_layout.addWidget(QLabel("Smoothness (λ):"))
        
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(2, 8)  # 10^2 to 10^8
        self.lambda_slider.setValue(5)  # 10^5
        self.lambda_slider.valueChanged.connect(self.on_lambda_changed)
        lambda_layout.addWidget(self.lambda_slider)
        
        self.lambda_value_label = QLabel("1e5")
        lambda_layout.addWidget(self.lambda_value_label)
        layout.addLayout(lambda_layout)
        
        # P parameter (asymmetry)
        p_layout = QHBoxLayout()
        p_layout.addWidget(QLabel("Asymmetry (p):"))
        
        self.p_spinbox = QDoubleSpinBox()
        self.p_spinbox.setRange(0.001, 0.1)
        self.p_spinbox.setValue(0.01)
        self.p_spinbox.setDecimals(3)
        self.p_spinbox.setSingleStep(0.001)
        self.p_spinbox.valueChanged.connect(self.on_parameter_changed)
        p_layout.addWidget(self.p_spinbox)
        layout.addLayout(p_layout)
        
        # Iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(5, 50)
        self.iterations_spinbox.setValue(10)
        self.iterations_spinbox.valueChanged.connect(self.on_parameter_changed)
        iter_layout.addWidget(self.iterations_spinbox)
        layout.addLayout(iter_layout)
        
        # Preview options
        self.show_background = QCheckBox("Show Background")
        self.show_background.setChecked(True)
        self.show_background.stateChanged.connect(self.update_plot)
        layout.addWidget(self.show_background)
        
        self.show_original = QCheckBox("Show Original")
        self.show_original.stateChanged.connect(self.update_plot)
        layout.addWidget(self.show_original)
        
        # Real-time preview
        self.realtime_preview = QCheckBox("Real-time Preview")
        self.realtime_preview.setChecked(True)
        layout.addWidget(self.realtime_preview)
        
        # Progress bar for processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Initially disabled
        self.set_bg_controls_enabled(False)
        
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
        
    def create_actions_group(self):
        """Create action buttons"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        
        # Apply processing
        self.apply_btn = QPushButton("Apply Processing")
        self.apply_btn.clicked.connect(self.apply_processing)
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)
        
        # Peak finding controls
        peak_group = QGroupBox("Peak Detection")
        peak_layout = QVBoxLayout(peak_group)
        
        # Min height control
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Min Height:"))
        self.min_height = QSpinBox()
        self.min_height.setRange(1, 10000)
        self.min_height.setValue(50)  # Lower default for small peaks
        height_layout.addWidget(self.min_height)
        peak_layout.addLayout(height_layout)
        
        # Prominence control
        prom_layout = QHBoxLayout()
        prom_layout.addWidget(QLabel("Min Prominence:"))
        self.min_prominence = QSpinBox()
        self.min_prominence.setRange(1, 1000)
        self.min_prominence.setValue(10)  # Low default for small peaks
        prom_layout.addWidget(self.min_prominence)
        peak_layout.addLayout(prom_layout)
        
        # Width control
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Min Width (pts):"))
        self.min_width = QSpinBox()
        self.min_width.setRange(1, 20)
        self.min_width.setValue(1)  # Allow very narrow peaks
        width_layout.addWidget(self.min_width)
        peak_layout.addLayout(width_layout)
        
        # Distance control
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Min Distance (pts):"))
        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 50)
        self.min_distance.setValue(3)  # Allow close peaks
        dist_layout.addWidget(self.min_distance)
        peak_layout.addLayout(dist_layout)
        
        # Sensitivity mode
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity = QComboBox()
        self.sensitivity.addItems(["High (small peaks)", "Medium", "Low (large peaks only)"])
        self.sensitivity.setCurrentIndex(0)  # Default to high sensitivity
        sens_layout.addWidget(self.sensitivity)
        peak_layout.addLayout(sens_layout)
        
        # Show all candidates option
        self.show_all_candidates = QCheckBox("Show all candidates (before filtering)")
        self.show_all_candidates.stateChanged.connect(self.update_plot)
        peak_layout.addWidget(self.show_all_candidates)
        
        # Find peaks button
        self.find_peaks_btn = QPushButton("Find Peaks")
        self.find_peaks_btn.clicked.connect(self.find_peaks)
        self.find_peaks_btn.setEnabled(False)
        peak_layout.addWidget(self.find_peaks_btn)
        
        layout.addWidget(peak_group)
        
        # Reset to original
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.clicked.connect(self.reset_to_original)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn)
        
        # Export processed data
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
        
        return group
        
    def set_pattern_data(self, pattern_data):
        """Set the pattern data for processing"""
        self.pattern_data = pattern_data.copy()
        self.original_pattern_data = pattern_data.copy()
        self.processed_pattern_data = pattern_data.copy()
        self.background_data = None
        self.peaks = None
        
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
        
        # Plot peaks if they exist
        if self.peaks is not None and self.processed_pattern_data is not None:
            peak_positions = self.processed_pattern_data['two_theta'][self.peaks]
            peak_intensities = self.processed_pattern_data['intensity'][self.peaks]
            self.ax.plot(peak_positions, peak_intensities, 'ro', markersize=4, label='Final Peaks')
            
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
            
            # Calculate peak data for matching
            if len(filtered_peaks) > 0:
                peak_positions = two_theta[filtered_peaks]
                peak_intensities = intensity[filtered_peaks]
                
                # Calculate d-spacings
                d_spacings = self.wavelength / (2 * np.sin(np.radians(peak_positions / 2)))
                
                # Calculate relative intensities
                max_intensity = np.max(peak_intensities)
                rel_intensities = (peak_intensities / max_intensity) * 100
                
                peak_data = {
                    'two_theta': peak_positions,
                    'intensity': peak_intensities,
                    'd_spacing': d_spacings,
                    'rel_intensity': rel_intensities,
                    'wavelength': self.wavelength
                }
                
                # Emit peak data signal
                self.peaks_found.emit(peak_data)
                
                QMessageBox.information(self, "Success", 
                    f"Found {len(filtered_peaks)} significant peaks!\n"
                    f"(Filtered from {len(peaks)} initial candidates)")
            else:
                QMessageBox.warning(self, "Warning", "No significant peaks found after filtering")
            
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
