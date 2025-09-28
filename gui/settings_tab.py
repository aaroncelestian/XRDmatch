"""
Settings tab for application configuration
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
                             QComboBox, QCheckBox, QPushButton, QFileDialog,
                             QTextEdit, QTabWidget, QFormLayout)
from PyQt5.QtCore import Qt, pyqtSignal
import json
import os

class SettingsTab(QWidget):
    """Tab for application settings and configuration"""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.settings = self.load_default_settings()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different setting categories
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create setting tabs
        self.create_general_tab()
        self.create_wavelength_tab()
        self.create_database_tab()
        self.create_matching_tab()
        self.create_display_tab()
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Settings")
        self.load_btn.clicked.connect(self.load_settings)
        button_layout.addWidget(self.load_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def create_general_tab(self):
        """Create general settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File handling settings
        file_group = QGroupBox("File Handling")
        file_layout = QFormLayout(file_group)
        
        self.default_data_dir = QLineEdit(self.settings['general']['default_data_dir'])
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_data_dir)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.default_data_dir)
        dir_layout.addWidget(browse_btn)
        
        file_layout.addRow("Default Data Directory:", dir_layout)
        
        self.auto_save = QCheckBox("Auto-save results")
        self.auto_save.setChecked(self.settings['general']['auto_save'])
        file_layout.addRow(self.auto_save)
        
        self.backup_files = QCheckBox("Create backup files")
        self.backup_files.setChecked(self.settings['general']['backup_files'])
        file_layout.addRow(self.backup_files)
        
        layout.addWidget(file_group)
        
        # Performance settings
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        self.max_threads = QSpinBox()
        self.max_threads.setRange(1, 16)
        self.max_threads.setValue(self.settings['general']['max_threads'])
        perf_layout.addRow("Maximum Threads:", self.max_threads)
        
        self.cache_size = QSpinBox()
        self.cache_size.setRange(10, 1000)
        self.cache_size.setValue(self.settings['general']['cache_size_mb'])
        self.cache_size.setSuffix(" MB")
        perf_layout.addRow("Cache Size:", self.cache_size)
        
        layout.addWidget(perf_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "General")
        
    def create_wavelength_tab(self):
        """Create wavelength settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Common wavelengths
        common_group = QGroupBox("Common X-ray Sources")
        common_layout = QVBoxLayout(common_group)
        
        wavelengths_text = QTextEdit()
        wavelengths_text.setPlainText(
            "Cu Kα1: 1.5406 Å\n"
            "Cu Kα: 1.5418 Å\n"
            "Co Kα1: 1.7890 Å\n"
            "Fe Kα1: 1.9373 Å\n"
            "Cr Kα1: 2.2897 Å\n"
            "Mo Kα1: 0.7107 Å\n"
            "Ag Kα1: 0.5594 Å"
        )
        wavelengths_text.setMaximumHeight(150)
        wavelengths_text.setReadOnly(True)
        common_layout.addWidget(wavelengths_text)
        
        layout.addWidget(common_group)
        
        # Custom wavelengths
        custom_group = QGroupBox("Custom Wavelengths")
        custom_layout = QFormLayout(custom_group)
        
        self.custom_wavelengths = QTextEdit()
        self.custom_wavelengths.setPlainText(self.settings['wavelengths']['custom'])
        self.custom_wavelengths.setPlaceholderText("Enter custom wavelengths in format:\nName: wavelength Å")
        custom_layout.addRow("Custom Sources:", self.custom_wavelengths)
        
        layout.addWidget(custom_group)
        
        # Default wavelength
        default_group = QGroupBox("Default Settings")
        default_layout = QFormLayout(default_group)
        
        self.default_wavelength = QComboBox()
        self.default_wavelength.addItems([
            "Cu Kα1 (1.5406)",
            "Cu Kα (1.5418)",
            "Co Kα1 (1.7890)",
            "Fe Kα1 (1.9373)",
            "Cr Kα1 (2.2897)",
            "Mo Kα1 (0.7107)"
        ])
        default_layout.addRow("Default Wavelength:", self.default_wavelength)
        
        layout.addWidget(default_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Wavelengths")
        
    def create_database_tab(self):
        """Create database settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # AMCSD settings
        amcsd_group = QGroupBox("AMCSD Database")
        amcsd_layout = QFormLayout(amcsd_group)
        
        self.amcsd_url = QLineEdit(self.settings['database']['amcsd_url'])
        amcsd_layout.addRow("Base URL:", self.amcsd_url)
        
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 120)
        self.timeout.setValue(self.settings['database']['timeout'])
        self.timeout.setSuffix(" seconds")
        amcsd_layout.addRow("Request Timeout:", self.timeout)
        
        self.max_results = QSpinBox()
        self.max_results.setRange(10, 1000)
        self.max_results.setValue(self.settings['database']['max_results'])
        amcsd_layout.addRow("Maximum Results:", self.max_results)
        
        layout.addWidget(amcsd_group)
        
        # Local database
        local_group = QGroupBox("Local Database")
        local_layout = QFormLayout(local_group)
        
        self.local_db_path = QLineEdit(self.settings['database']['local_db_path'])
        browse_db_btn = QPushButton("Browse")
        browse_db_btn.clicked.connect(self.browse_local_db)
        
        db_layout = QHBoxLayout()
        db_layout.addWidget(self.local_db_path)
        db_layout.addWidget(browse_db_btn)
        
        local_layout.addRow("Local Database:", db_layout)
        
        self.use_local_db = QCheckBox("Use local database when available")
        self.use_local_db.setChecked(self.settings['database']['use_local'])
        local_layout.addRow(self.use_local_db)
        
        layout.addWidget(local_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Database")
        
    def create_matching_tab(self):
        """Create matching settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Matching parameters
        matching_group = QGroupBox("Matching Parameters")
        matching_layout = QFormLayout(matching_group)
        
        self.default_tolerance = QDoubleSpinBox()
        self.default_tolerance.setRange(0.001, 1.0)
        self.default_tolerance.setDecimals(3)
        self.default_tolerance.setValue(self.settings['matching']['default_tolerance'])
        self.default_tolerance.setSuffix(" Å")
        matching_layout.addRow("Default d-spacing Tolerance:", self.default_tolerance)
        
        self.min_match_score = QDoubleSpinBox()
        self.min_match_score.setRange(0.0, 1.0)
        self.min_match_score.setDecimals(2)
        self.min_match_score.setValue(self.settings['matching']['min_match_score'])
        matching_layout.addRow("Minimum Match Score:", self.min_match_score)
        
        self.min_peak_intensity = QSpinBox()
        self.min_peak_intensity.setRange(1, 1000)
        self.min_peak_intensity.setValue(self.settings['matching']['min_peak_intensity'])
        matching_layout.addRow("Minimum Peak Intensity:", self.min_peak_intensity)
        
        layout.addWidget(matching_group)
        
        # Peak finding
        peak_group = QGroupBox("Peak Finding")
        peak_layout = QFormLayout(peak_group)
        
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0.1, 100.0)
        self.peak_prominence.setValue(self.settings['matching']['peak_prominence'])
        peak_layout.addRow("Peak Prominence:", self.peak_prominence)
        
        self.peak_distance = QSpinBox()
        self.peak_distance.setRange(1, 50)
        self.peak_distance.setValue(self.settings['matching']['peak_distance'])
        peak_layout.addRow("Minimum Peak Distance:", self.peak_distance)
        
        layout.addWidget(peak_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Matching")
        
    def create_display_tab(self):
        """Create display settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Plot settings
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout(plot_group)
        
        self.plot_dpi = QSpinBox()
        self.plot_dpi.setRange(50, 300)
        self.plot_dpi.setValue(self.settings['display']['plot_dpi'])
        plot_layout.addRow("Plot DPI:", self.plot_dpi)
        
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.5, 5.0)
        self.line_width.setDecimals(1)
        self.line_width.setValue(self.settings['display']['line_width'])
        plot_layout.addRow("Line Width:", self.line_width)
        
        self.marker_size = QSpinBox()
        self.marker_size.setRange(2, 20)
        self.marker_size.setValue(self.settings['display']['marker_size'])
        plot_layout.addRow("Marker Size:", self.marker_size)
        
        layout.addWidget(plot_group)
        
        # Color scheme
        color_group = QGroupBox("Color Scheme")
        color_layout = QFormLayout(color_group)
        
        self.color_scheme = QComboBox()
        self.color_scheme.addItems(["Default", "Dark", "Colorblind-friendly", "High-contrast"])
        color_layout.addRow("Color Scheme:", self.color_scheme)
        
        self.show_grid = QCheckBox("Show grid")
        self.show_grid.setChecked(self.settings['display']['show_grid'])
        color_layout.addRow(self.show_grid)
        
        self.show_legend = QCheckBox("Show legend")
        self.show_legend.setChecked(self.settings['display']['show_legend'])
        color_layout.addRow(self.show_legend)
        
        self.show_error_bars = QCheckBox("Show error bars (XYE files)")
        self.show_error_bars.setChecked(self.settings['display']['show_error_bars'])
        color_layout.addRow(self.show_error_bars)
        
        layout.addWidget(color_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "Display")
        
    def load_default_settings(self):
        """Load default settings"""
        return {
            'general': {
                'default_data_dir': os.path.expanduser('~/XRD_Data'),
                'auto_save': True,
                'backup_files': True,
                'max_threads': 4,
                'cache_size_mb': 100
            },
            'wavelengths': {
                'default': 'Cu Kα1 (1.5406)',
                'custom': ''
            },
            'database': {
                'amcsd_url': 'https://rruff.geo.arizona.edu/AMS',
                'timeout': 30,
                'max_results': 100,
                'local_db_path': '',
                'use_local': False
            },
            'matching': {
                'default_tolerance': 0.02,
                'min_match_score': 0.1,
                'min_peak_intensity': 5,
                'peak_prominence': 1.0,
                'peak_distance': 5
            },
            'display': {
                'plot_dpi': 100,
                'line_width': 1.0,
                'marker_size': 6,
                'color_scheme': 'Default',
                'show_grid': True,
                'show_legend': True,
                'show_error_bars': True
            }
        }
        
    def browse_data_dir(self):
        """Browse for default data directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Default Data Directory", self.default_data_dir.text()
        )
        if directory:
            self.default_data_dir.setText(directory)
            
    def browse_local_db(self):
        """Browse for local database file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Local Database", "", "Database files (*.db *.sqlite);;All files (*.*)"
        )
        if file_path:
            self.local_db_path.setText(file_path)
            
    def get_current_settings(self):
        """Get current settings from UI"""
        return {
            'general': {
                'default_data_dir': self.default_data_dir.text(),
                'auto_save': self.auto_save.isChecked(),
                'backup_files': self.backup_files.isChecked(),
                'max_threads': self.max_threads.value(),
                'cache_size_mb': self.cache_size.value()
            },
            'wavelengths': {
                'default': self.default_wavelength.currentText(),
                'custom': self.custom_wavelengths.toPlainText()
            },
            'database': {
                'amcsd_url': self.amcsd_url.text(),
                'timeout': self.timeout.value(),
                'max_results': self.max_results.value(),
                'local_db_path': self.local_db_path.text(),
                'use_local': self.use_local_db.isChecked()
            },
            'matching': {
                'default_tolerance': self.default_tolerance.value(),
                'min_match_score': self.min_match_score.value(),
                'min_peak_intensity': self.min_peak_intensity.value(),
                'peak_prominence': self.peak_prominence.value(),
                'peak_distance': self.peak_distance.value()
            },
            'display': {
                'plot_dpi': self.plot_dpi.value(),
                'line_width': self.line_width.value(),
                'marker_size': self.marker_size.value(),
                'color_scheme': self.color_scheme.currentText(),
                'show_grid': self.show_grid.isChecked(),
                'show_legend': self.show_legend.isChecked(),
                'show_error_bars': self.show_error_bars.isChecked()
            }
        }
        
    def save_settings(self):
        """Save settings to file"""
        settings = self.get_current_settings()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "xrd_settings.json", "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(settings, f, indent=2)
                self.settings = settings
                self.settings_changed.emit(settings)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Could not save settings:\n{str(e)}")
                
    def load_settings(self):
        """Load settings from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                self.apply_settings(settings)
                self.settings = settings
                self.settings_changed.emit(settings)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Could not load settings:\n{str(e)}")
                
    def apply_settings(self, settings):
        """Apply settings to UI"""
        # General settings
        general = settings.get('general', {})
        self.default_data_dir.setText(general.get('default_data_dir', ''))
        self.auto_save.setChecked(general.get('auto_save', True))
        self.backup_files.setChecked(general.get('backup_files', True))
        self.max_threads.setValue(general.get('max_threads', 4))
        self.cache_size.setValue(general.get('cache_size_mb', 100))
        
        # Wavelength settings
        wavelengths = settings.get('wavelengths', {})
        self.custom_wavelengths.setPlainText(wavelengths.get('custom', ''))
        
        # Database settings
        database = settings.get('database', {})
        self.amcsd_url.setText(database.get('amcsd_url', ''))
        self.timeout.setValue(database.get('timeout', 30))
        self.max_results.setValue(database.get('max_results', 100))
        self.local_db_path.setText(database.get('local_db_path', ''))
        self.use_local_db.setChecked(database.get('use_local', False))
        
        # Matching settings
        matching = settings.get('matching', {})
        self.default_tolerance.setValue(matching.get('default_tolerance', 0.02))
        self.min_match_score.setValue(matching.get('min_match_score', 0.1))
        self.min_peak_intensity.setValue(matching.get('min_peak_intensity', 5))
        self.peak_prominence.setValue(matching.get('peak_prominence', 1.0))
        self.peak_distance.setValue(matching.get('peak_distance', 5))
        
        # Display settings
        display = settings.get('display', {})
        self.plot_dpi.setValue(display.get('plot_dpi', 100))
        self.line_width.setValue(display.get('line_width', 1.0))
        self.marker_size.setValue(display.get('marker_size', 6))
        self.show_grid.setChecked(display.get('show_grid', True))
        self.show_legend.setChecked(display.get('show_legend', True))
        self.show_error_bars.setChecked(display.get('show_error_bars', True))
        
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        defaults = self.load_default_settings()
        self.apply_settings(defaults)
        self.settings = defaults
        self.settings_changed.emit(defaults)
