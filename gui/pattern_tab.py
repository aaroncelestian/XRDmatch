"""
Pattern data tab for loading and analyzing diffraction patterns
"""

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                             QTableWidget, QTableWidgetItem, QGroupBox, QFileDialog,
                             QMessageBox, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class PatternTab(QWidget):
    """Tab for handling diffraction pattern data"""
    
    pattern_loaded = pyqtSignal(dict)  # Signal emitted when pattern is loaded
    
    def __init__(self):
        super().__init__()
        self.pattern_data = None
        self.original_pattern_data = None  # Store original data before background subtraction
        self.background_data = None  # Store calculated background
        self.peaks = None
        self.wavelength = 1.5406  # Cu Ka1 default
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Splitter for plot and peak table
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Plot widget
        self.plot_widget = self.create_plot_widget()
        splitter.addWidget(self.plot_widget)
        
        # Peak table
        self.peak_table = self.create_peak_table()
        splitter.addWidget(self.peak_table)
        
        # Set splitter proportions
        splitter.setSizes([700, 300])
        
    def create_control_panel(self):
        """Create the control panel"""
        group = QGroupBox("Pattern Controls")
        layout = QHBoxLayout(group)
        
        # File controls
        self.load_btn = QPushButton("Load Pattern")
        self.load_btn.clicked.connect(self.load_pattern_dialog)
        layout.addWidget(self.load_btn)
        
        # Wavelength controls - placed immediately after Load button for visibility
        layout.addWidget(QLabel("  λ:"))
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.addItems([
            "Cu Kα1 (1.5406)",
            "Cu Kα (1.5418)",
            "Co Kα1 (1.7890)",
            "Fe Kα1 (1.9373)",
            "Cr Kα1 (2.2897)",
            "Mo Kα1 (0.7107)",
            "17 BM (0.24105)",
            "Custom"
        ])
        self.wavelength_combo.currentTextChanged.connect(self.wavelength_changed)
        self.wavelength_combo.setToolTip("Select X-ray wavelength - IMPORTANT: Must match your experimental data!")
        layout.addWidget(self.wavelength_combo)
        
        self.custom_wavelength = QDoubleSpinBox()
        self.custom_wavelength.setRange(0.1, 10.0)
        self.custom_wavelength.setDecimals(4)
        self.custom_wavelength.setValue(1.5406)
        self.custom_wavelength.setVisible(False)
        self.custom_wavelength.valueChanged.connect(self.custom_wavelength_changed)
        self.custom_wavelength.setToolTip("Enter custom wavelength in Ångströms")
        layout.addWidget(self.custom_wavelength)
        
        # File status label
        self.file_label = QLabel("No file loaded - Drag & drop files here!")
        self.file_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(self.file_label)
        
        layout.addStretch()
        
        return group
        
    def create_plot_widget(self):
        """Create the matplotlib plot widget"""
        group = QGroupBox("Diffraction Pattern")
        layout = QVBoxLayout(group)
        
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('2θ (degrees)')
        self.ax.set_ylabel('Intensity (counts)')
        self.ax.set_title('X-ray Diffraction Pattern')
        self.ax.grid(True, alpha=0.3)
        
        return group
        
    def create_peak_table(self):
        """Create the peak table widget"""
        group = QGroupBox("Detected Peaks")
        layout = QVBoxLayout(group)
        
        self.peak_table_widget = QTableWidget()
        self.peak_table_widget.setColumnCount(4)
        self.peak_table_widget.setHorizontalHeaderLabels(['2θ', 'Intensity', 'd-spacing', 'Rel. Int.'])
        layout.addWidget(self.peak_table_widget)
        
        return group
        
    def wavelength_changed(self, text):
        """Handle wavelength selection change"""
        if "Custom" in text:
            self.custom_wavelength.setVisible(True)
            self.wavelength = self.custom_wavelength.value()
        else:
            self.custom_wavelength.setVisible(False)
            # Extract wavelength from text
            wavelength_str = text.split('(')[1].split(')')[0]
            self.wavelength = float(wavelength_str)
            
        self.update_d_spacings()
        
    def custom_wavelength_changed(self, value):
        """Handle custom wavelength change"""
        self.wavelength = value
        self.update_d_spacings()
        
    def update_d_spacings(self):
        """Update d-spacing calculations when wavelength changes"""
        if self.pattern_data is not None:
            # Update wavelength in pattern data
            self.pattern_data['wavelength'] = self.wavelength
            
            # Update peak table if peaks exist
            if self.peaks is not None and len(self.peaks) > 0:
                self.update_peak_table()
                
            # Emit updated pattern data
            self.pattern_loaded.emit(self.pattern_data)
    
    def update_peak_table(self):
        """Update the peak table with current peak data"""
        if self.peaks is None or self.pattern_data is None:
            self.peak_table_widget.setRowCount(0)
            return
            
        peak_positions = self.pattern_data['two_theta'][self.peaks]
        peak_intensities = self.pattern_data['intensity'][self.peaks]
        
        # Calculate d-spacings using Bragg's law: d = λ / (2 * sin(θ))
        # where θ is half of 2θ (in radians)
        d_spacings = self.wavelength / (2 * np.sin(np.radians(peak_positions / 2)))
        
        # Calculate relative intensities
        max_intensity = np.max(peak_intensities)
        rel_intensities = (peak_intensities / max_intensity) * 100
        
        # Update table
        self.peak_table_widget.setRowCount(len(self.peaks))
        
        for i, (pos, intensity, d_spacing, rel_int) in enumerate(
            zip(peak_positions, peak_intensities, d_spacings, rel_intensities)
        ):
            self.peak_table_widget.setItem(i, 0, QTableWidgetItem(f"{pos:.3f}"))
            self.peak_table_widget.setItem(i, 1, QTableWidgetItem(f"{intensity:.0f}"))
            self.peak_table_widget.setItem(i, 2, QTableWidgetItem(f"{d_spacing:.4f}"))
            self.peak_table_widget.setItem(i, 3, QTableWidgetItem(f"{rel_int:.1f}%"))
        
    def load_pattern_dialog(self):
        """Open file dialog to load pattern"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Diffraction Pattern",
            "",
            "Data files (*.xy *.xye *.chi *.xml *.txt *.dat *.csv);;All files (*.*)"
        )
        
        if file_path:
            self.load_pattern(file_path)
            
    def load_pattern(self, file_path):
        """Load diffraction pattern from file"""
        try:
            # Check if it's an XML file
            if file_path.lower().endswith('.xml'):
                two_theta, intensity, intensity_error, wavelength = self.parse_xml_file(file_path)
                file_format = 'XML'
                # Update wavelength if found in XML
                if wavelength:
                    self.wavelength = wavelength
                    # Update the wavelength combo to show custom
                    self.wavelength_combo.setCurrentText("Custom")
                    self.custom_wavelength.setValue(wavelength)
                    self.custom_wavelength.setVisible(True)
            else:
                # Handle text-based formats (XY, XYE, etc.)
                two_theta, intensity, intensity_error = self.parse_text_file(file_path)
                file_format = 'XYE' if intensity_error is not None else 'XY'
            
            self.pattern_data = {
                'two_theta': two_theta,
                'intensity': intensity,
                'intensity_error': intensity_error,
                'file_path': file_path,
                'file_format': file_format,
                'wavelength': self.wavelength
            }
            
            # Store original data for background subtraction
            self.original_pattern_data = self.pattern_data.copy()
            self.background_data = None
            
            self.plot_pattern()
            format_info = f" ({file_format} format)"
            self.file_label.setText(f"Loaded: {file_path.split('/')[-1]}{format_info}")
            self.file_label.setStyleSheet("QLabel { color: #000; font-style: normal; }")
            
            # Emit signal
            self.pattern_loaded.emit(self.pattern_data)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load pattern:\n{str(e)}")
            
    def plot_pattern(self):
        """Plot the diffraction pattern"""
        if self.pattern_data is None:
            return
            
        self.ax.clear()
        
        # Plot with or without error bars
        if self.pattern_data.get('intensity_error') is not None:
            # Plot with error bars for XYE format
            self.ax.errorbar(
                self.pattern_data['two_theta'], 
                self.pattern_data['intensity'],
                yerr=self.pattern_data['intensity_error'],
                fmt='b-', linewidth=1, elinewidth=0.5, capsize=0, alpha=0.8
            )
        else:
            # Plot without error bars for XY format
            self.ax.plot(self.pattern_data['two_theta'], self.pattern_data['intensity'], 'b-', linewidth=1)
        
        # Plot peaks if they exist
        if self.peaks is not None:
            peak_positions = self.pattern_data['two_theta'][self.peaks]
            peak_intensities = self.pattern_data['intensity'][self.peaks]
            self.ax.plot(peak_positions, peak_intensities, 'ro', markersize=4)
            
        self.ax.set_xlabel('2θ (degrees)')
        self.ax.set_ylabel('Intensity (counts)')
        
        # Update title to show format
        format_info = self.pattern_data.get('file_format', 'XY')
        self.ax.set_title(f'X-ray Diffraction Pattern ({format_info} format)')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            # Check if any of the URLs are files with supported extensions
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if any(file_path.lower().endswith(ext) for ext in ['.xy', '.xye', '.chi', '.txt', '.dat', '.csv']):
                        event.acceptProposedAction()
                        return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if any(file_path.lower().endswith(ext) for ext in ['.xy', '.xye', '.chi', '.xml', '.txt', '.dat', '.csv']):
                        self.load_pattern(file_path)
                        event.acceptProposedAction()
                        return
        event.ignore()
    
    def parse_xml_file(self, file_path):
        """Parse XML file format"""
        import xml.etree.ElementTree as ET
        
        two_theta = []
        intensity = []
        intensity_error = []
        wavelength = None
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract wavelength if available
            w_elem = root.find('w')
            if w_elem is not None:
                try:
                    wavelength = float(w_elem.text)
                except (ValueError, TypeError):
                    pass
            
            # Extract intensity data points
            for intensity_elem in root.findall('intensity'):
                try:
                    x_val = float(intensity_elem.get('X'))
                    y_val = float(intensity_elem.get('Y'))
                    t_val = float(intensity_elem.get('T', 1.0))  # Default to 1 if T not present
                    
                    two_theta.append(x_val)
                    intensity.append(y_val)
                    
                    # Calculate error as sqrt(counts) if T (time) is available
                    if t_val > 0:
                        error = np.sqrt(y_val) if y_val > 0 else 0
                        intensity_error.append(error)
                    
                except (ValueError, TypeError):
                    continue
            
            if not two_theta:
                raise ValueError("No valid intensity data found in XML file")
            
            # Convert to numpy arrays
            two_theta = np.array(two_theta)
            intensity = np.array(intensity)
            intensity_error = np.array(intensity_error) if intensity_error else None
            
            # Sort by 2theta
            sort_idx = np.argsort(two_theta)
            two_theta = two_theta[sort_idx]
            intensity = intensity[sort_idx]
            if intensity_error is not None:
                intensity_error = intensity_error[sort_idx]
            
            return two_theta, intensity, intensity_error, wavelength
            
        except ET.ParseError as e:
            raise ValueError(f"Could not parse XML file: {str(e)}")
    
    def detect_xye_format(self, file_path):
        """Detect the specific XYE format by examining the first few lines"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [f.readline().strip() for _ in range(10)]
        
        # Check for C-style comments (/* */) - indicates commented XYE format
        has_c_comments = any('/*' in line for line in first_lines)
        
        # Check for hash comments - indicates standard format
        has_hash_comments = any(line.startswith('#') for line in first_lines)
        
        # Look for data pattern in first non-comment line
        data_line = None
        for line in first_lines:
            if line and not line.startswith('#') and '/*' not in line:
                data_line = line
                break
        
        format_info = {
            'has_c_comments': has_c_comments,
            'has_hash_comments': has_hash_comments,
            'data_line': data_line,
            'format_type': 'standard'  # default
        }
        
        # Determine format type
        if has_c_comments:
            format_info['format_type'] = 'commented_xye'
        elif has_hash_comments:
            format_info['format_type'] = 'standard_xye'
        elif data_line:
            # Check number of columns in first data line
            values = data_line.split()
            if len(values) == 3:
                format_info['format_type'] = 'simple_xye'
            elif len(values) == 2:
                format_info['format_type'] = 'xy'
        
        return format_info
    
    def parse_text_file(self, file_path):
        """Parse text-based file formats (XY, XYE, etc.) with format detection"""
        # Detect format for XYE files
        if file_path.lower().endswith('.xye'):
            format_info = self.detect_xye_format(file_path)
            print(f"Detected XYE format: {format_info['format_type']}")
            
            if format_info['format_type'] == 'commented_xye':
                return self.parse_commented_xye(file_path)
            else:
                return self.parse_standard_text_file(file_path)
        else:
            return self.parse_standard_text_file(file_path)
    
    def parse_commented_xye(self, file_path):
        """Parse XYE format with C-style comments"""
        processed_lines = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            in_c_comment = False
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Handle C-style comments /* */
                # Check for complete comment on single line first
                if '/*' in line and '*/' in line:
                    # Remove the comment but keep any content before/after
                    start_comment = line.find('/*')
                    end_comment = line.find('*/') + 2
                    line = line[:start_comment] + line[end_comment:]
                elif '/*' in line:
                    in_c_comment = True
                    line = line[:line.find('/*')]
                elif '*/' in line:
                    in_c_comment = False
                    line = line[line.find('*/') + 2:]
                    
                # Skip lines inside C-style comments
                if in_c_comment:
                    continue
                    
                # If line has content after comment removal, add it
                if line.strip():
                    processed_lines.append(line)
        
        return self.parse_numeric_data(processed_lines)
    
    def parse_standard_text_file(self, file_path):
        """Parse standard text file formats (XY, standard XYE, etc.)"""
        processed_lines = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Skip regular comments starting with #
                if line.startswith('#'):
                    continue
                    
                # Add data lines
                processed_lines.append(line)
        
        return self.parse_numeric_data(processed_lines)
    
    def parse_numeric_data(self, processed_lines):
        """Parse numeric data from processed text lines"""
        # Parse the data manually to handle multiple whitespace properly
        parsed_data = []
        for line in processed_lines:
            # Split on any whitespace and filter out empty strings
            values = [x for x in line.split() if x]
            
            if len(values) >= 2:
                try:
                    # Convert to float
                    numeric_values = [float(v) for v in values]
                    parsed_data.append(numeric_values)
                except ValueError:
                    # Skip lines that can't be converted to numbers
                    continue
        
        if not parsed_data:
            raise ValueError("No valid numeric data found in file")
        
        # Convert to numpy array and then to DataFrame
        data_array = np.array(parsed_data)
        data = pd.DataFrame(data_array)
        
        if data.shape[1] < 2:
            raise ValueError(f"Could not parse file - need at least 2 columns, found {data.shape[1]} columns")
            
        # Extract 2theta and intensity
        two_theta = data.iloc[:, 0].values
        intensity = data.iloc[:, 1].values
        
        # Check if we have error data (XYE format)
        intensity_error = None
        if data.shape[1] >= 3:
            intensity_error = data.iloc[:, 2].values
            # Remove NaN values from error column too
            error_mask = ~np.isnan(intensity_error)
            if np.any(error_mask):
                intensity_error = intensity_error[error_mask]
            else:
                intensity_error = None
        
        # Remove any NaN values
        mask = ~(np.isnan(two_theta) | np.isnan(intensity))
        two_theta = two_theta[mask]
        intensity = intensity[mask]
        
        # Apply mask to error data if it exists
        if intensity_error is not None:
            if len(intensity_error) == len(mask):
                intensity_error = intensity_error[mask]
            elif len(intensity_error) != len(two_theta):
                # If error array doesn't match, disable error bars
                intensity_error = None
        
        # Sort by 2theta
        sort_idx = np.argsort(two_theta)
        two_theta = two_theta[sort_idx]
        intensity = intensity[sort_idx]
        
        if intensity_error is not None:
            if len(intensity_error) == len(sort_idx):
                intensity_error = intensity_error[sort_idx]
            else:
                # If error array doesn't match after sorting, disable error bars
                intensity_error = None
        
        return two_theta, intensity, intensity_error
