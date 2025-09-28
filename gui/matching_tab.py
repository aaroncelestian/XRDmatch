"""
Phase matching tab for comparing experimental and reference patterns
"""

import numpy as np
import pandas as pd
import requests
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QTableWidget, QTableWidgetItem, QGroupBox,
                             QSlider, QDoubleSpinBox, QComboBox, QTextEdit,
                             QSplitter, QProgressBar, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from utils.cif_parser import CIFParser
from scipy.special import wofz

class PhaseMatchingThread(QThread):
    """Thread for phase matching calculations"""
    
    matching_complete = pyqtSignal(list)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, experimental_peaks, reference_phases, tolerance):
        super().__init__()
        self.experimental_peaks = experimental_peaks
        self.reference_phases = reference_phases
        self.tolerance = tolerance
        
    def run(self):
        """Run phase matching in separate thread"""
        results = []
        
        for i, phase in enumerate(self.reference_phases):
            match_result = self.match_phase(phase)
            results.append(match_result)
            
            # Update progress
            progress = int((i + 1) / len(self.reference_phases) * 100)
            self.progress_updated.emit(progress)
            
        self.matching_complete.emit(results)
        
    def match_phase(self, phase):
        """Match a single phase against experimental data"""
        if not self.experimental_peaks:
            return {
                'phase': phase,
                'matches': [],
                'match_score': 0.0,
                'coverage': 0.0
            }
            
        exp_dspacings = self.experimental_peaks['d_spacing']
        exp_intensities = self.experimental_peaks['intensity']
        
        # Generate theoretical pattern for this phase
        theoretical_peaks = self.generate_theoretical_pattern(phase)
        
        if not theoretical_peaks:
            return {
                'phase': phase,
                'matches': [],
                'match_score': 0.0,
                'coverage': 0.0
            }
            
        # Find matches
        matches = []
        matched_exp_peaks = set()
        
        for theo_d, theo_int in zip(theoretical_peaks['d_spacing'], theoretical_peaks['intensity']):
            # Find closest experimental peak
            differences = np.abs(exp_dspacings - theo_d)
            min_idx = np.argmin(differences)
            min_diff = differences[min_idx]
            
            if min_diff <= self.tolerance and min_idx not in matched_exp_peaks:
                matches.append({
                    'exp_d': exp_dspacings[min_idx],
                    'theo_d': theo_d,
                    'exp_int': exp_intensities[min_idx],
                    'theo_int': theo_int,
                    'difference': min_diff
                })
                matched_exp_peaks.add(min_idx)
                
        # Calculate match score and coverage
        match_score = len(matches) / len(theoretical_peaks) if theoretical_peaks else 0
        coverage = len(matched_exp_peaks) / len(exp_dspacings) if exp_dspacings.size > 0 else 0
        
        return {
            'phase': phase,
            'matches': matches,
            'match_score': match_score,
            'coverage': coverage,
            'theoretical_peaks': theoretical_peaks
        }
        
    def generate_theoretical_pattern(self, phase):
        """Generate theoretical diffraction pattern for a phase using DIF data with wavelength conversion"""
        try:
            cif_parser = CIFParser()
            
            # Get experimental wavelength
            experimental_wavelength = 1.5406  # Default Cu Kα
            if hasattr(self, 'experimental_peaks') and self.experimental_peaks:
                experimental_wavelength = self.experimental_peaks.get('wavelength', 1.5406)
            
            print(f"Using experimental wavelength: {experimental_wavelength} Å")
            print(f"Phase: {phase.get('mineral', 'Unknown')} (AMCSD ID: {phase.get('amcsd_id', 'Unknown')})")
            
            # Check if this phase is from local database and has pre-calculated patterns
            if phase.get('local_db') and phase.get('id'):
                from utils.local_database import LocalCIFDatabase
                local_db = LocalCIFDatabase()
                
                # Try to get pre-calculated pattern first
                pre_calculated = local_db.get_diffraction_pattern(
                    phase['id'], experimental_wavelength
                )
                
                if pre_calculated:
                    print(f"✅ Using pre-calculated diffraction pattern ({len(pre_calculated['two_theta'])} peaks)")
                    return pre_calculated
                else:
                    print("No pre-calculated pattern found, calculating from CIF...")
            
            # Prioritize local CIF content for XRD calculation
            cif_content = phase.get('cif_content')  # Check for local CIF content first
            cif_url = phase.get('cif_url')
            
            # Try CIF-based calculation (fallback method)
            if cif_content or cif_url:
                print(f"Attempting CIF-based XRD calculation...")
                try:
                    # Use local CIF content if available, otherwise download
                    if cif_content:
                        print("Using local CIF content from database")
                    else:
                        print(f"Downloading CIF from: {cif_url}")
                        response = requests.get(cif_url, timeout=10)
                        if response.status_code == 200:
                            cif_content = response.text
                        else:
                            print(f"Failed to download CIF file: HTTP {response.status_code}")
                            cif_content = None
                    
                    if cif_content:
                        # Calculate XRD pattern from CIF
                        # Use wider range for low-angle data
                        max_2theta = min(90.0, 180.0 / np.pi * np.arcsin(experimental_wavelength / (2 * 0.5)))
                        cif_pattern = cif_parser.calculate_xrd_pattern_from_cif(
                            cif_content, experimental_wavelength, max_2theta=max_2theta, min_d=0.5
                        )
                        
                        if len(cif_pattern.get('two_theta', [])) > 0:
                            print(f"CIF calculation successful: {len(cif_pattern['two_theta'])} peaks")
                            return cif_pattern
                        else:
                            print("CIF calculation failed - no valid peaks generated")
                            
                except Exception as e:
                    print(f"Error in CIF-based calculation: {e}")
            
            # Return empty pattern if CIF calculation failed
            print("No theoretical pattern could be generated for this phase - CIF calculation failed")
            return {
                'd_spacing': np.array([]),
                'intensity': np.array([]),
                'two_theta': np.array([])
            }
            
        except Exception as e:
            print(f"Error fetching DIF data: {e}")
            return {
                'd_spacing': np.array([]),
                'intensity': np.array([]),
                'two_theta': np.array([])
            }
    
    def convert_dif_to_wavelength(self, dif_data, target_wavelength):
        """Convert DIF data from Cu Kα to target wavelength"""
        try:
            # DIF files are typically calculated for Cu Kα (1.5406 Å)
            dif_wavelength = 1.5406
            
            # Get d-spacings from DIF data (these are wavelength-independent)
            d_spacings = dif_data['d_spacing']
            intensities = dif_data['intensity']
            
            # Calculate new 2θ values for the target wavelength using Bragg's law
            # λ = 2d sin(θ) → θ = arcsin(λ / 2d)
            new_two_theta = []
            valid_d_spacings = []
            valid_intensities = []
            
            for d, intensity in zip(d_spacings, intensities):
                if d > 0:
                    sin_theta = target_wavelength / (2 * d)
                    if sin_theta <= 1.0:  # Valid reflection
                        theta_rad = np.arcsin(sin_theta)
                        two_theta_deg = 2 * np.degrees(theta_rad)
                        
                        # Only include peaks in reasonable 2θ range (adjust for short wavelengths)
                        min_2theta = 1.0 if target_wavelength < 1.0 else 5.0
                        if min_2theta <= two_theta_deg <= 90:
                            new_two_theta.append(two_theta_deg)
                            valid_d_spacings.append(d)
                            valid_intensities.append(intensity)
            
            print(f"Converted {len(valid_d_spacings)} peaks from λ={dif_wavelength:.4f}Å to λ={target_wavelength:.4f}Å")
            if len(valid_d_spacings) > 0:
                print(f"2θ range after conversion: {np.min(new_two_theta):.2f}° to {np.max(new_two_theta):.2f}°")
                print(f"d-spacing range: {np.min(valid_d_spacings):.3f}Å to {np.max(valid_d_spacings):.3f}Å")
            
            return {
                'd_spacing': np.array(valid_d_spacings),
                'intensity': np.array(valid_intensities),
                'two_theta': np.array(new_two_theta)
            }
            
        except Exception as e:
            print(f"Error converting DIF wavelength: {e}")
            return {
                'd_spacing': np.array([]),
                'intensity': np.array([]),
                'two_theta': np.array([])
            }

class MatchingTab(QWidget):
    """Tab for phase matching analysis"""
    
    def __init__(self):
        super().__init__()
        self.experimental_pattern = None  # Raw pattern data
        self.processed_pattern = None     # Background-subtracted processed data
        self.experimental_peaks = None
        self.reference_phases = []
        self.matching_results = []
        
        # Cache for calculated theoretical patterns to avoid recalculation
        self.pattern_cache = {}  # Key: (phase_id, wavelength), Value: theoretical pattern data
        self.continuous_pattern_cache = {}  # Key: (phase_id, wavelength, fwhm, x_range_hash), Value: continuous pattern
        
        self.init_ui()
    
    def pseudo_voigt(self, x, center, fwhm, intensity, eta=0.5):
        """
        Generate a pseudo-Voigt peak profile
        
        Args:
            x: array of x values (2theta)
            center: peak center position
            fwhm: full width at half maximum
            intensity: peak intensity
            eta: mixing parameter (0=pure Gaussian, 1=pure Lorentzian)
        
        Returns:
            array of y values for the peak
        """
        # Ensure reasonable parameters
        if fwhm <= 0:
            fwhm = 0.1  # Default minimum FWHM
        if intensity <= 0:
            return np.zeros_like(x)
            
        # Convert FWHM to standard deviations
        sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Gaussian component
        gamma_l = fwhm / 2  # Lorentzian component
        
        # Avoid division by zero
        if sigma_g <= 0:
            sigma_g = 0.01
        if gamma_l <= 0:
            gamma_l = 0.01
        
        # Gaussian component
        gaussian = np.exp(-0.5 * ((x - center) / sigma_g) ** 2)
        
        # Lorentzian component
        lorentzian = 1 / (1 + ((x - center) / gamma_l) ** 2)
        
        # Pseudo-Voigt is a linear combination
        profile = intensity * ((1 - eta) * gaussian + eta * lorentzian)
        
        return profile
    
    def generate_theoretical_pattern_profile(self, two_theta_peaks, intensities, fwhm, x_range):
        """
        Generate a continuous theoretical diffraction pattern from peak positions and intensities
        
        Args:
            two_theta_peaks: array of peak positions in 2theta
            intensities: array of peak intensities
            fwhm: full width at half maximum for peaks
            x_range: array of 2theta values for the pattern
            
        Returns:
            array of intensity values for the continuous pattern
        """
        pattern = np.zeros_like(x_range)
        
        # Add some debugging to understand peak generation
        peaks_added = 0
        total_intensity_added = 0
        
        for center, intensity in zip(two_theta_peaks, intensities):
            # Only add peaks that are within the x_range with some buffer
            x_min, x_max = np.min(x_range), np.max(x_range)
            # Use a more generous buffer to ensure peaks near edges are included
            buffer = max(5*fwhm, 2.0)  # At least 2 degrees buffer
            if (x_min - buffer) <= center <= (x_max + buffer):
                # Ensure intensity is reasonable
                if intensity > 0 and not np.isnan(intensity) and not np.isinf(intensity):
                    peak = self.pseudo_voigt(x_range, center, fwhm, intensity, eta=0.3)
                    # Check if peak generation was successful
                    if not np.any(np.isnan(peak)) and not np.any(np.isinf(peak)):
                        pattern += peak
                        peaks_added += 1
                        total_intensity_added += intensity
        
        # Debug output
        if peaks_added > 0:
            max_pattern_intensity = np.max(pattern) if len(pattern) > 0 else 0
            print(f"  Generated theoretical pattern: {peaks_added} peaks, total input intensity: {total_intensity_added:.2f}, max pattern intensity: {max_pattern_intensity:.2f}")
            
            # Additional debugging for pattern generation
            if max_pattern_intensity == 0 and total_intensity_added > 0:
                print(f"  Warning: Non-zero input intensities but zero pattern intensity!")
                print(f"  X-range: {np.min(x_range):.2f} to {np.max(x_range):.2f}")
                print(f"  Peak centers: {two_theta_peaks[:5]}...")  # Show first 5 peak positions
                print(f"  FWHM: {fwhm}")
        else:
            print(f"  Warning: No peaks added to theoretical pattern")
            if len(two_theta_peaks) > 0:
                print(f"  Input had {len(two_theta_peaks)} peaks with centers: {two_theta_peaks[:5]}...")
                print(f"  X-range: {np.min(x_range):.2f} to {np.max(x_range):.2f}")
        
        return pattern
        
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
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left side: Plot
        plot_panel = self.create_plot_panel()
        main_splitter.addWidget(plot_panel)
        
        # Right side: Results
        results_panel = self.create_results_panel()
        main_splitter.addWidget(results_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 400])
        
    def create_control_panel(self):
        """Create the control panel"""
        group = QGroupBox("Matching Parameters")
        layout = QHBoxLayout(group)
        
        # Tolerance setting
        layout.addWidget(QLabel("d-spacing tolerance (Å):"))
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 1.0)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setValue(0.02)
        self.tolerance_spin.setSingleStep(0.001)
        layout.addWidget(self.tolerance_spin)
        
        # Minimum match score
        layout.addWidget(QLabel("Min. match score:"))
        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0)
        self.min_score_spin.setDecimals(2)
        self.min_score_spin.setValue(0.1)
        self.min_score_spin.setSingleStep(0.05)
        layout.addWidget(self.min_score_spin)
        
        # Theoretical pattern scaling
        layout.addWidget(QLabel("Theory scale (%):"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(10, 200)
        self.scale_spin.setDecimals(0)
        self.scale_spin.setValue(80)
        self.scale_spin.setSingleStep(10)
        self.scale_spin.setSuffix("%")
        self.scale_spin.valueChanged.connect(self.update_plot)
        layout.addWidget(self.scale_spin)
        
        # Peak width control for theoretical patterns
        layout.addWidget(QLabel("Peak FWHM (°):"))
        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(0.01, 2.0)
        self.fwhm_spin.setDecimals(2)
        self.fwhm_spin.setValue(0.1)
        self.fwhm_spin.setSingleStep(0.01)
        self.fwhm_spin.setToolTip("Full Width at Half Maximum for theoretical peak profiles")
        self.fwhm_spin.valueChanged.connect(self.update_plot)
        layout.addWidget(self.fwhm_spin)
        
        layout.addStretch()
        
        # Action buttons
        self.match_btn = QPushButton("Start Matching")
        self.match_btn.clicked.connect(self.start_matching)
        self.match_btn.setEnabled(False)
        layout.addWidget(self.match_btn)
        
        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(self.clear_btn)
        
        return group
        
    def create_plot_panel(self):
        """Create the plot panel"""
        group = QGroupBox("Pattern Comparison")
        layout = QVBoxLayout(group)
        
        # Plot canvas
        self.figure = Figure(figsize=(8, 10))
        self.canvas = FigureCanvas(self.figure)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Create subplots with adjusted height ratios (main plot larger, diff plot smaller)
        self.ax_main = self.figure.add_subplot(3, 1, (1, 2))  # Takes rows 1-2 (2/3 of space)
        self.ax_diff = self.figure.add_subplot(3, 1, 3)      # Takes row 3 (1/3 of space)
        
        self.ax_main.set_xlabel('2θ (degrees)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Experimental vs Reference Patterns')
        self.ax_main.grid(True, alpha=0.3)
        
        self.ax_diff.set_xlabel('d-spacing (Å)')
        self.ax_diff.set_ylabel('Phase')
        self.ax_diff.set_title('Peak Matching Overview')
        self.ax_diff.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        
        return group
        
    def create_results_panel(self):
        """Create the results panel"""
        group = QGroupBox("Matching Results")
        layout = QVBoxLayout(group)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            'Phase', 'Match Score', 'Coverage', 'Matches', 'Show'
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.results_table)
        
        # Detailed match info
        detail_group = QGroupBox("Match Details")
        detail_layout = QVBoxLayout(detail_group)
        
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(7)
        self.detail_table.setHorizontalHeaderLabels([
            'Exp. d (Å)', 'Theo. d (Å)', 'Difference', 'Exp. Int.', 'Theo. Int.', 'Exp. 2θ (°)', 'Match Quality'
        ])
        detail_layout.addWidget(self.detail_table)
        
        layout.addWidget(detail_group)
        
        # Connect table selection
        self.results_table.itemSelectionChanged.connect(self.show_match_details)
        
        return group
        
    def set_experimental_pattern(self, pattern_data):
        """Set the experimental pattern data (prioritize processed data)"""
        # Check if this is processed data (from processing tab)
        if pattern_data.get('processed', False) or hasattr(pattern_data, 'processed'):
            self.processed_pattern = pattern_data
            print("Received processed pattern data for matching")
        else:
            # This is raw pattern data (from pattern tab)
            if self.processed_pattern is None:
                self.experimental_pattern = pattern_data
                print("Received raw pattern data for matching")
        
        self.update_matching_availability()
        self.update_plot()
        
    def set_experimental_peaks(self, peak_data):
        """Set experimental peak data"""
        self.experimental_peaks = peak_data
        self.update_matching_availability()
        
    def add_reference_phases(self, phases):
        """Add reference phases from database search"""
        self.reference_phases.extend(phases)
        self.update_matching_availability()
        
    def update_matching_availability(self):
        """Update whether matching can be performed"""
        has_pattern = (self.processed_pattern is not None or 
                      self.experimental_pattern is not None)
        can_match = (self.experimental_peaks is not None and 
                    len(self.reference_phases) > 0 and 
                    has_pattern)
        self.match_btn.setEnabled(can_match)
        
    def start_matching(self):
        """Start the phase matching process"""
        if not self.experimental_peaks or not self.reference_phases:
            return
            
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.match_btn.setEnabled(False)
        
        # Start matching thread
        tolerance = self.tolerance_spin.value()
        self.matching_thread = PhaseMatchingThread(
            self.experimental_peaks, 
            self.reference_phases, 
            tolerance
        )
        self.matching_thread.matching_complete.connect(self.display_matching_results)
        self.matching_thread.progress_updated.connect(self.progress_bar.setValue)
        self.matching_thread.start()
        
    def display_matching_results(self, results):
        """Display matching results"""
        self.matching_results = results
        self.progress_bar.setVisible(False)
        self.match_btn.setEnabled(True)
        
        # Clear pattern caches when new results come in
        self.pattern_cache.clear()
        self.continuous_pattern_cache.clear()
        
        # Check how many phases have theoretical data
        phases_with_data = sum(1 for r in results if len(r.get('theoretical_peaks', {}).get('d_spacing', [])) > 0)
        phases_without_data = len(results) - phases_with_data
        
        if phases_without_data > 0:
            QMessageBox.information(self, "DIF Data Availability", 
                f"Note: {phases_without_data} of {len(results)} phases have no DIF data available.\n"
                f"Only {phases_with_data} phases can be compared.\n\n"
                f"DIF files contain the actual diffraction patterns needed for comparison.\n"
                f"You can manually download DIF files from the AMCSD website if available.")
        
        # Filter results by minimum score
        min_score = self.min_score_spin.value()
        filtered_results = [r for r in results if r['match_score'] >= min_score]
        
        # Sort by match score (descending)
        filtered_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Update results table
        self.results_table.setRowCount(len(filtered_results))
        
        for i, result in enumerate(filtered_results):
            phase = result['phase']
            
            # Phase name
            phase_name = phase.get('mineral', 'Unknown')
            self.results_table.setItem(i, 0, QTableWidgetItem(phase_name))
            
            # Match score
            score_item = QTableWidgetItem(f"{result['match_score']:.3f}")
            self.results_table.setItem(i, 1, score_item)
            
            # Coverage
            coverage_item = QTableWidgetItem(f"{result['coverage']:.3f}")
            self.results_table.setItem(i, 2, coverage_item)
            
            # Number of matches
            matches_item = QTableWidgetItem(str(len(result['matches'])))
            self.results_table.setItem(i, 3, matches_item)
            
            # Show checkbox
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(i < 3)  # Show top 3 by default
            show_checkbox.stateChanged.connect(self.update_plot)
            self.results_table.setCellWidget(i, 4, show_checkbox)
            
        self.update_plot()
        
    def show_match_details(self):
        """Show detailed matching information for selected phase"""
        current_row = self.results_table.currentRow()
        if current_row < 0:
            return
        
        # Get filtered results that match the current table display
        min_score = self.min_score_spin.value()
        filtered_results = [r for r in self.matching_results if r['match_score'] >= min_score]
        filtered_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        if current_row >= len(filtered_results):
            return
            
        result = filtered_results[current_row]
        matches = result['matches']
        
        # Update detail table
        self.detail_table.setRowCount(len(matches))
        
        for i, match in enumerate(matches):
            # Calculate 2θ from d-spacing using current wavelength
            wavelength = 1.5406  # Default Cu Kα
            if self.experimental_peaks and 'wavelength' in self.experimental_peaks:
                wavelength = self.experimental_peaks['wavelength']
            
            # Calculate 2θ using Bragg's law: λ = 2d sin(θ) → θ = arcsin(λ/2d)
            try:
                sin_theta = wavelength / (2 * match['exp_d'])
                if sin_theta <= 1.0:
                    theta_rad = np.arcsin(sin_theta)
                    two_theta_deg = 2 * np.degrees(theta_rad)
                else:
                    two_theta_deg = 0.0
            except:
                two_theta_deg = 0.0
            
            # Calculate match quality based on difference and intensity
            rel_diff = abs(match['difference']) / match['exp_d'] * 100  # Relative difference as percentage
            if rel_diff < 0.1:
                quality = "Excellent"
            elif rel_diff < 0.5:
                quality = "Very Good"
            elif rel_diff < 1.0:
                quality = "Good"
            elif rel_diff < 2.0:
                quality = "Fair"
            else:
                quality = "Poor"
            
            self.detail_table.setItem(i, 0, QTableWidgetItem(f"{match['exp_d']:.4f}"))
            self.detail_table.setItem(i, 1, QTableWidgetItem(f"{match['theo_d']:.4f}"))
            self.detail_table.setItem(i, 2, QTableWidgetItem(f"{match['difference']:.4f}"))
            self.detail_table.setItem(i, 3, QTableWidgetItem(f"{match['exp_int']:.0f}"))
            self.detail_table.setItem(i, 4, QTableWidgetItem(f"{match['theo_int']:.0f}"))
            self.detail_table.setItem(i, 5, QTableWidgetItem(f"{two_theta_deg:.2f}"))
            self.detail_table.setItem(i, 6, QTableWidgetItem(quality))
            
    def update_plot(self):
        """Update the comparison plot with normalized intensities"""
        self.ax_main.clear()
        self.ax_diff.clear()
        
        # Plot experimental pattern (prefer processed over raw)
        pattern_to_plot = self.processed_pattern or self.experimental_pattern
        
        # Normalize experimental data to 100 for easier comparison
        normalized_exp_intensity = None
        normalized_exp_error = None
        
        if pattern_to_plot:
            max_exp_intensity = np.max(pattern_to_plot['intensity'])
            if max_exp_intensity > 0:
                # Normalize experimental intensities to 100
                normalized_exp_intensity = (pattern_to_plot['intensity'] / max_exp_intensity) * 100
                
                # Normalize error bars if present
                if pattern_to_plot.get('intensity_error') is not None:
                    normalized_exp_error = (pattern_to_plot['intensity_error'] / max_exp_intensity) * 100
            else:
                normalized_exp_intensity = pattern_to_plot['intensity']
                normalized_exp_error = pattern_to_plot.get('intensity_error')
            
            # Plot normalized experimental pattern (without error bars)
            self.ax_main.plot(
                pattern_to_plot['two_theta'],
                normalized_exp_intensity,
                'b-', linewidth=1.5, 
                label='Experimental (Normalized to 100)' + (' - Processed' if self.processed_pattern else ''),
                alpha=0.8
            )
            
        # Plot selected reference patterns
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        y_offset = 0
        
        # Get filtered results that match the current table display
        min_score = self.min_score_spin.value()
        filtered_results = [r for r in self.matching_results if r['match_score'] >= min_score]
        filtered_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        for i in range(self.results_table.rowCount()):
            checkbox = self.results_table.cellWidget(i, 4)
            if checkbox and checkbox.isChecked():
                # Use the filtered results to match the table display
                result = filtered_results[i] if i < len(filtered_results) else None
                if result is None:
                    continue
                phase_name = result['phase'].get('mineral', f'Phase {i+1}')
                color = colors[i % len(colors)]
                
                # Plot theoretical pattern using cached data
                if 'theoretical_peaks' in result:
                    theo_peaks = result['theoretical_peaks']
                    
                    if len(theo_peaks.get('two_theta', [])) > 0:
                        # Use pre-calculated 2theta values (already converted for experimental wavelength)
                        two_theta = theo_peaks['two_theta']
                        intensities = theo_peaks['intensity']
                    else:
                        continue
                    
                    # Generate or retrieve cached continuous pattern
                    if pattern_to_plot and len(intensities) > 0:
                        # Create cache key for this specific pattern configuration
                        phase_id = result['phase'].get('id', f"phase_{i}")
                        wavelength = pattern_to_plot.get('wavelength', 1.5406)
                        fwhm = self.fwhm_spin.value()
                        scale_percentage = self.scale_spin.value() / 100.0
                        
                        # Create x-range for theoretical pattern based on experimental data range
                        x_min = np.min(pattern_to_plot['two_theta'])
                        x_max = np.max(pattern_to_plot['two_theta'])
                        x_range = np.linspace(x_min, x_max, len(pattern_to_plot['two_theta']) * 2)  # Higher resolution
                        
                        # Create cache key including all parameters that affect the pattern
                        x_range_hash = hash((x_min, x_max, len(x_range)))
                        cache_key = (phase_id, wavelength, fwhm, x_range_hash, scale_percentage)
                        
                        # Check if we have this pattern cached
                        if cache_key in self.continuous_pattern_cache:
                            theoretical_pattern = self.continuous_pattern_cache[cache_key]
                            print(f"Using cached pattern for {phase_name}")
                        else:
                            # Calculate and cache the pattern
                            print(f"Calculating new pattern for {phase_name}")
                            
                            # Normalize theoretical intensities properly
                            max_theo_intensity = np.max(intensities)
                            
                            print(f"  Raw theoretical intensities for {phase_name}:")
                            print(f"    Min: {np.min(intensities):.2f}, Max: {max_theo_intensity:.2f}")
                            print(f"    Mean: {np.mean(intensities):.2f}, Std: {np.std(intensities):.2f}")
                            print(f"    Number of peaks: {len(intensities)}")
                            
                            if max_theo_intensity > 0:
                                # Normalize theoretical to 0-100 scale, then apply user scaling
                                normalized_theo_intensities = (intensities / max_theo_intensity) * 100 * scale_percentage
                                
                                # Ensure no theoretical peaks exceed reasonable limits
                                normalized_theo_intensities = np.clip(normalized_theo_intensities, 0, 200)
                                
                                print(f"  After normalization:")
                                print(f"    Max theoretical intensity before: {max_theo_intensity:.2f}")
                                print(f"    Max theoretical intensity after: {np.max(normalized_theo_intensities):.2f}")
                                print(f"    Scale percentage: {scale_percentage:.2f}")
                                print(f"    Expected max after scaling: {100 * scale_percentage:.2f}")
                            else:
                                normalized_theo_intensities = intensities
                                print(f"  Warning: Zero max intensity for {phase_name}")
                            
                            # Generate continuous pattern
                            theoretical_pattern = self.generate_theoretical_pattern_profile(
                                two_theta, normalized_theo_intensities, fwhm, x_range
                            )
                            
                            # Debug the final pattern
                            if len(theoretical_pattern) > 0:
                                print(f"  Final continuous pattern for {phase_name}:")
                                print(f"    Max intensity: {np.max(theoretical_pattern):.2f}")
                                print(f"    Non-zero points: {np.count_nonzero(theoretical_pattern)}")
                            else:
                                print(f"  Warning: Empty theoretical pattern generated for {phase_name}")
                            
                            # Cache the result
                            self.continuous_pattern_cache[cache_key] = theoretical_pattern
                        
                        # Plot as continuous line
                        self.ax_main.plot(x_range, theoretical_pattern, 
                                        color=color, linewidth=1.5, alpha=0.8,
                                        label=phase_name)
                    
                    # Plot theoretical peaks in difference plot as bars close to experimental
                    if len(theo_peaks.get('d_spacing', [])) > 0:
                        theo_d_spacings = theo_peaks['d_spacing']
                        
                        # Plot theoretical peaks as bars slightly offset from experimental
                        y_position = -0.8 + (i * 0.15)  # Offset each phase slightly
                        for d_spacing in theo_d_spacings:
                            self.ax_diff.axvline(x=d_spacing, ymin=0.15 + (i * 0.1), ymax=0.25 + (i * 0.1), 
                                               color=color, linewidth=2, alpha=0.8)
                        
                        # Add legend entry
                        self.ax_diff.scatter([theo_d_spacings[0]], [y_position], 
                                           c=color, marker='|', s=100, label=phase_name)
                    
                y_offset += 0.1
                
        # Plot experimental peaks in difference plot as bars
        if self.experimental_peaks and len(self.experimental_peaks.get('d_spacing', [])) > 0:
            exp_d_spacings = self.experimental_peaks['d_spacing']
            exp_y_pos = [-1] * len(exp_d_spacings)
            
            # Plot as vertical bars instead of circles
            for d_spacing in exp_d_spacings:
                self.ax_diff.axvline(x=d_spacing, ymin=0.0, ymax=0.1, 
                                   color='blue', linewidth=2, alpha=0.8)
            
            # Add a single legend entry for experimental peaks
            self.ax_diff.scatter([exp_d_spacings[0]], [-1], 
                               c='blue', marker='|', s=100, label='Experimental')
            
        # Format plots
        self.ax_main.set_xlabel('2θ (degrees)')
        self.ax_main.set_ylabel('Normalized Intensity (0-100)')
        title = 'Experimental vs Theoretical Patterns (Pseudo-Voigt Profiles)'
        if self.processed_pattern:
            title += ' - Background Subtracted'
        self.ax_main.set_title(title)
        self.ax_main.grid(True, alpha=0.3)
        if self.ax_main.get_legend_handles_labels()[0]:  # Only show legend if there are items
            self.ax_main.legend(loc='upper right')
        
        self.ax_diff.set_xlabel('d-spacing (Å)')
        self.ax_diff.set_ylabel('Phase')
        self.ax_diff.set_title('Peak Position Comparisons')
        self.ax_diff.grid(True, alpha=0.3)
        
        # Invert x-axis so larger d-spacings are on the left
        self.ax_diff.invert_xaxis()
        
        # Set y-axis limits to show the bars properly
        self.ax_diff.set_ylim(-1.2, 1.0)
        
        # Remove y-axis ticks since they're not meaningful for this plot
        self.ax_diff.set_yticks([])
        
        if self.ax_diff.get_legend_handles_labels()[0]:  # Only show legend if there are items
            self.ax_diff.legend(loc='upper right')
        
        # Set reasonable axis limits with normalized scale
        if pattern_to_plot:
            # Determine appropriate 2θ range based on experimental data
            min_2theta = np.min(pattern_to_plot['two_theta'])
            max_2theta = np.max(pattern_to_plot['two_theta'])
            
            # Add some padding
            range_padding = (max_2theta - min_2theta) * 0.05
            self.ax_main.set_xlim(max(0, min_2theta - range_padding), max_2theta + range_padding)
            
            # Set y-axis to normalized scale (0-110 to give some headroom)
            self.ax_main.set_ylim(0, 110)
        
        self.canvas.draw()
        
    def clear_results(self):
        """Clear matching results"""
        self.matching_results = []
        self.results_table.setRowCount(0)
        self.detail_table.setRowCount(0)
        
        # Clear pattern caches
        self.pattern_cache.clear()
        self.continuous_pattern_cache.clear()
        
        self.update_plot()
        
    def save_results(self, file_path):
        """Save matching results to file"""
        if not self.matching_results:
            return
            
        # Prepare data for saving
        data = []
        for result in self.matching_results:
            phase = result['phase']
            data.append({
                'Phase': phase.get('mineral', 'Unknown'),
                'Formula': phase.get('formula', 'Unknown'),
                'Match_Score': result['match_score'],
                'Coverage': result['coverage'],
                'Num_Matches': len(result['matches'])
            })
            
        # Save as CSV or text
        df = pd.DataFrame(data)
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_string(file_path, index=False)
