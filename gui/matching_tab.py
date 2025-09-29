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
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

class PhaseMatchingThread(QThread):
    """Thread for phase matching calculations"""
    
    matching_complete = pyqtSignal(list)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, experimental_peaks, reference_phases, tolerance, peak_weight=0.6, corr_weight=0.4):
        super().__init__()
        self.experimental_peaks = experimental_peaks
        self.reference_phases = reference_phases
        self.tolerance = tolerance
        self.peak_weight = peak_weight
        self.corr_weight = corr_weight
        
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
        """Match a single phase against experimental data with intensity-weighted scoring"""
        if not self.experimental_peaks:
            return {
                'phase': phase,
                'matches': [],
                'match_score': 0.0,
                'coverage': 0.0,
                'intensity_weighted_score': 0.0
            }
            
        exp_two_theta = self.experimental_peaks['two_theta']
        exp_dspacings = self.experimental_peaks['d_spacing']
        exp_intensities = self.experimental_peaks['intensity']
        
        # Generate theoretical pattern for this phase
        theoretical_peaks = self.generate_theoretical_pattern(phase)
        
        if not theoretical_peaks:
            return {
                'phase': phase,
                'matches': [],
                'match_score': 0.0,
                'coverage': 0.0,
                'intensity_weighted_score': 0.0
            }
            
        # Normalize intensities for comparison
        max_exp_int = np.max(exp_intensities) if len(exp_intensities) > 0 else 1
        max_theo_int = np.max(theoretical_peaks['intensity']) if len(theoretical_peaks['intensity']) > 0 else 1
        
        # Find matches with intensity consideration using 2θ tolerance
        matches = []
        matched_exp_peaks = set()
        total_intensity_weight = 0
        matched_intensity_weight = 0
        
        theo_two_theta = theoretical_peaks['two_theta']
        theo_dspacings = theoretical_peaks['d_spacing']
        theo_intensities = theoretical_peaks['intensity']
        
        for i, (theo_2theta, theo_d, theo_int) in enumerate(zip(theo_two_theta, theo_dspacings, theo_intensities)):
            # Normalize theoretical intensity (0-100 scale)
            norm_theo_int = (theo_int / max_theo_int) * 100
            total_intensity_weight += norm_theo_int
            
            # Find closest experimental peak using 2θ differences
            two_theta_differences = np.abs(exp_two_theta - theo_2theta)
            min_idx = np.argmin(two_theta_differences)
            min_2theta_diff = two_theta_differences[min_idx]
            
            if min_2theta_diff <= self.tolerance and min_idx not in matched_exp_peaks:
                # Calculate intensity similarity (0-1 scale, 1 = perfect match)
                norm_exp_int = (exp_intensities[min_idx] / max_exp_int) * 100
                intensity_ratio = min(norm_exp_int, norm_theo_int) / max(norm_exp_int, norm_theo_int) if max(norm_exp_int, norm_theo_int) > 0 else 0
                
                matches.append({
                    'exp_d': exp_dspacings[min_idx],
                    'theo_d': theo_d,
                    'exp_2theta': exp_two_theta[min_idx],
                    'theo_2theta': theo_2theta,
                    'exp_int': exp_intensities[min_idx],
                    'theo_int': theo_int,
                    'difference': min_2theta_diff,  # Now stores 2θ difference
                    'intensity_similarity': intensity_ratio,
                    'norm_exp_int': norm_exp_int,
                    'norm_theo_int': norm_theo_int
                })
                matched_exp_peaks.add(min_idx)
                matched_intensity_weight += norm_theo_int
                
        # MULTI-MATCH REQUIREMENT SCORING SYSTEM
        # Requires multiple good matches to score well - prevents single spurious matches
        
        # 1. Coverage Score: What fraction of experimental peaks are explained?
        coverage = len(matched_exp_peaks) / len(exp_dspacings) if exp_dspacings.size > 0 else 0
        
        # 2. Count high-quality matches and calculate score
        high_quality_matches = []
        total_quality_score = 0.0
        
        for match in matches:
            # Position accuracy
            two_theta_accuracy = max(0, 1.0 - (match['difference'] / self.tolerance))
            
            # Intensity similarity
            intensity_similarity = match['intensity_similarity']
            
            # Combined quality
            combined_quality = (0.7 * two_theta_accuracy) + (0.3 * intensity_similarity)
            
            # Only count reasonably good matches (> 0.4 threshold)
            if combined_quality > 0.4:
                high_quality_matches.append((match, combined_quality))
                total_quality_score += combined_quality
        
        # 3. MULTI-MATCH PENALTY: Phases need multiple good matches to score well
        num_good_matches = len(high_quality_matches)
        
        if num_good_matches == 0:
            match_score = 0.0
        elif num_good_matches == 1:
            # Single matches get heavily penalized (max score 0.1)
            base_score = total_quality_score / len(exp_dspacings)
            match_score = min(0.1, base_score * 0.2)  # Heavy penalty for single matches
        elif num_good_matches == 2:
            # Two matches get moderate penalty
            base_score = total_quality_score / len(exp_dspacings)
            match_score = base_score * 0.6
        else:
            # Three or more matches get full score (real phases should have multiple matches)
            base_score = total_quality_score / len(exp_dspacings)
            # Bonus for having many matches (real phases)
            match_bonus = min(1.5, 1.0 + (num_good_matches - 3) * 0.1)
            match_score = base_score * match_bonus
        
        # 4. Intensity-weighted score (for reference)
        intensity_weighted_score = matched_intensity_weight / total_intensity_weight if total_intensity_weight > 0 else 0
        
        # 5. CORRELATION-BASED SCORING
        # Calculate correlation between experimental and theoretical patterns
        correlation_result = self._calculate_pattern_correlation(
            exp_two_theta, exp_intensities, theo_two_theta, theo_intensities
        )
        
        # 6. COMBINED SCORING: Combine peak-based and correlation-based scores
        # Use configurable weights from UI
        combined_score = (self.peak_weight * match_score + 
                         self.corr_weight * correlation_result['correlation'])
        
        return {
            'phase': phase,
            'matches': matches,
            'match_score': match_score,
            'coverage': coverage,
            'intensity_weighted_score': intensity_weighted_score,
            'correlation': correlation_result['correlation'],
            'r_squared': correlation_result['r_squared'],
            'rms_error': correlation_result['rms_error'],
            'combined_score': combined_score,
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
    
    def _calculate_pattern_correlation(self, exp_two_theta, exp_intensity, 
                                     theo_two_theta, theo_intensity):
        """Calculate correlation between experimental and theoretical patterns"""
        try:
            # Determine common 2θ range
            min_2theta = max(np.min(exp_two_theta), np.min(theo_two_theta))
            max_2theta = min(np.max(exp_two_theta), np.max(theo_two_theta))
            
            if min_2theta >= max_2theta:
                return {'correlation': 0, 'r_squared': 0, 'rms_error': 1}
            
            # Create common 2θ grid with reasonable resolution
            num_points = min(1000, int((max_2theta - min_2theta) / 0.02))  # 0.02° resolution
            common_2theta = np.linspace(min_2theta, max_2theta, num_points)
            
            # Normalize experimental intensities
            max_exp_int = np.max(exp_intensity) if len(exp_intensity) > 0 else 1
            norm_exp_intensity = exp_intensity / max_exp_int
            
            # Create interpolation function for experimental data
            exp_interp = interp1d(exp_two_theta, norm_exp_intensity, 
                                bounds_error=False, fill_value=0, kind='linear')
            exp_pattern = exp_interp(common_2theta)
            
            # Generate theoretical continuous pattern using pseudo-Voigt peaks
            theo_pattern = self._generate_continuous_pattern(
                theo_two_theta, theo_intensity, common_2theta, fwhm=0.1
            )
            
            # Normalize theoretical pattern
            max_theo_int = np.max(theo_pattern) if np.max(theo_pattern) > 0 else 1
            theo_pattern = theo_pattern / max_theo_int
            
            # Remove NaN values
            valid_mask = ~(np.isnan(exp_pattern) | np.isnan(theo_pattern))
            if np.sum(valid_mask) < 10:  # Need minimum points for correlation
                return {'correlation': 0, 'r_squared': 0, 'rms_error': 1}
            
            exp_valid = exp_pattern[valid_mask]
            theo_valid = theo_pattern[valid_mask]
            
            # Calculate Pearson correlation
            if np.std(exp_valid) == 0 or np.std(theo_valid) == 0:
                correlation = 0
            else:
                correlation, _ = pearsonr(exp_valid, theo_valid)
                if np.isnan(correlation):
                    correlation = 0
            
            # R-squared
            r_squared = correlation ** 2
            
            # RMS error
            rms_error = np.sqrt(np.mean((exp_valid - theo_valid) ** 2))
            
            return {
                'correlation': abs(correlation),  # Use absolute value
                'r_squared': r_squared,
                'rms_error': rms_error
            }
            
        except Exception as e:
            print(f"Error calculating pattern correlation: {e}")
            return {'correlation': 0, 'r_squared': 0, 'rms_error': 1}
    
    def _generate_continuous_pattern(self, two_theta_peaks, intensities, 
                                   x_range, fwhm=0.1):
        """Generate continuous pattern from peak positions using pseudo-Voigt profiles"""
        pattern = np.zeros_like(x_range)
        
        # Normalize intensities
        max_intensity = np.max(intensities) if len(intensities) > 0 else 1
        norm_intensities = intensities / max_intensity
        
        for center, intensity in zip(two_theta_peaks, norm_intensities):
            if intensity > 0:
                # Pseudo-Voigt profile (30% Lorentzian, 70% Gaussian)
                sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
                gamma_l = fwhm / 2
                
                gaussian = np.exp(-0.5 * ((x_range - center) / sigma_g) ** 2)
                lorentzian = 1 / (1 + ((x_range - center) / gamma_l) ** 2)
                
                peak = intensity * (0.7 * gaussian + 0.3 * lorentzian)
                pattern += peak
        
        return pattern

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
        layout.addWidget(QLabel("2θ tolerance (°):"))
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.01, 2.0)
        self.tolerance_spin.setDecimals(2)
        self.tolerance_spin.setValue(0.20)
        self.tolerance_spin.setSingleStep(0.01)
        layout.addWidget(self.tolerance_spin)
        
        # Minimum match score
        layout.addWidget(QLabel("Min. match score:"))
        self.min_score_spin = QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0)
        self.min_score_spin.setDecimals(2)
        self.min_score_spin.setValue(0.01)
        self.min_score_spin.setSingleStep(0.05)
        layout.addWidget(self.min_score_spin)
        
        # Global theoretical pattern scaling (now affects all phases)
        layout.addWidget(QLabel("Global scale (%):"))
        self.global_scale_spin = QDoubleSpinBox()
        self.global_scale_spin.setRange(10, 200)
        self.global_scale_spin.setDecimals(0)
        self.global_scale_spin.setValue(80)
        self.global_scale_spin.setSingleStep(10)
        self.global_scale_spin.setSuffix("%")
        self.global_scale_spin.setToolTip("Apply this scaling to all phases (updates individual phase scales)")
        self.global_scale_spin.valueChanged.connect(self.apply_global_scaling)
        layout.addWidget(self.global_scale_spin)
        
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
        
        # Minimum intensity filter for theoretical peaks
        layout.addWidget(QLabel("Min. intensity (%):")); 
        self.min_intensity_spin = QDoubleSpinBox()
        self.min_intensity_spin.setRange(0, 100)
        self.min_intensity_spin.setDecimals(1)
        self.min_intensity_spin.setValue(1.0)
        self.min_intensity_spin.setSingleStep(1.0)
        self.min_intensity_spin.setSuffix("%")
        self.min_intensity_spin.setToolTip("Minimum intensity percentage for theoretical peaks to be shown and considered in matching")
        self.min_intensity_spin.valueChanged.connect(self.update_plot)
        layout.addWidget(self.min_intensity_spin)
        
        # 2θ display limits
        layout.addWidget(QLabel("2θ range:"))
        self.min_2theta_spin = QDoubleSpinBox()
        self.min_2theta_spin.setRange(0, 180)
        self.min_2theta_spin.setDecimals(1)
        self.min_2theta_spin.setValue(5.0)
        self.min_2theta_spin.setSingleStep(1.0)
        self.min_2theta_spin.setSuffix("°")
        self.min_2theta_spin.setToolTip("Minimum 2θ angle to display")
        self.min_2theta_spin.valueChanged.connect(self.update_plot)
        layout.addWidget(self.min_2theta_spin)
        
        layout.addWidget(QLabel("to"))
        
        self.max_2theta_spin = QDoubleSpinBox()
        self.max_2theta_spin.setRange(0, 180)
        self.max_2theta_spin.setDecimals(1)
        self.max_2theta_spin.setValue(60.0)
        self.max_2theta_spin.setSingleStep(1.0)
        self.max_2theta_spin.setSuffix("°")
        self.max_2theta_spin.setToolTip("Maximum 2θ angle to display")
        self.max_2theta_spin.valueChanged.connect(self.update_plot)
        layout.addWidget(self.max_2theta_spin)
        
        # Auto-range button
        self.auto_range_btn = QPushButton("Auto")
        self.auto_range_btn.setToolTip("Auto-set 2θ range based on experimental data")
        self.auto_range_btn.clicked.connect(self.auto_set_2theta_range)
        layout.addWidget(self.auto_range_btn)
        
        layout.addStretch()
        
        # Correlation weighting controls
        layout.addWidget(QLabel("Peak weight:"))
        self.peak_weight_spin = QDoubleSpinBox()
        self.peak_weight_spin.setRange(0.0, 1.0)
        self.peak_weight_spin.setDecimals(2)
        self.peak_weight_spin.setValue(0.6)
        self.peak_weight_spin.setSingleStep(0.1)
        self.peak_weight_spin.setToolTip("Weight for peak-based scoring in combined score")
        layout.addWidget(self.peak_weight_spin)
        
        layout.addWidget(QLabel("Corr. weight:"))
        self.corr_weight_spin = QDoubleSpinBox()
        self.corr_weight_spin.setRange(0.0, 1.0)
        self.corr_weight_spin.setDecimals(2)
        self.corr_weight_spin.setValue(0.4)
        self.corr_weight_spin.setSingleStep(0.1)
        self.corr_weight_spin.setToolTip("Weight for correlation-based scoring in combined score")
        layout.addWidget(self.corr_weight_spin)
        
        # Auto-normalize weights
        def normalize_weights():
            peak_weight = self.peak_weight_spin.value()
            corr_weight = self.corr_weight_spin.value()
            total = peak_weight + corr_weight
            if total > 0:
                self.peak_weight_spin.setValue(peak_weight / total)
                self.corr_weight_spin.setValue(corr_weight / total)
        
        self.peak_weight_spin.valueChanged.connect(normalize_weights)
        self.corr_weight_spin.valueChanged.connect(normalize_weights)
        
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
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels([
            'Phase', 'Peak Score', 'Corr. Score', 'Combined', 'Coverage', 'Matches', 'R²', 'Scale %', 'Show'
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.results_table)
        
        # Detailed match info
        detail_group = QGroupBox("Match Details")
        detail_layout = QVBoxLayout(detail_group)
        
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(8)
        self.detail_table.setHorizontalHeaderLabels([
            'Exp. 2θ (°)', 'Theo. 2θ (°)', 'Δ 2θ (°)', 'Exp. Int.', 'Theo. Int.', 'Int. Sim.', 'Exp. d (Å)', 'Match Quality'
        ])
        detail_layout.addWidget(self.detail_table)
        
        layout.addWidget(detail_group)
        
        # Connect table selection
        self.results_table.itemSelectionChanged.connect(self.show_match_details)
        
        return group
        
    def auto_set_2theta_range(self):
        """Automatically set 2θ range based on experimental data"""
        pattern_to_use = self.processed_pattern or self.experimental_pattern
        if pattern_to_use and 'two_theta' in pattern_to_use:
            min_2theta = np.min(pattern_to_use['two_theta'])
            max_2theta = np.max(pattern_to_use['two_theta'])
            
            # Add some padding
            range_padding = (max_2theta - min_2theta) * 0.02
            min_2theta = max(0, min_2theta - range_padding)
            max_2theta = min(180, max_2theta + range_padding)
            
            # Update the spinboxes
            self.min_2theta_spin.setValue(min_2theta)
            self.max_2theta_spin.setValue(max_2theta)
            
            print(f"Auto-set 2θ range to {min_2theta:.1f}° - {max_2theta:.1f}°")
        
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
        
        # Auto-set 2θ range when first pattern is loaded
        if pattern_data and 'two_theta' in pattern_data:
            # Only auto-set if ranges are still at defaults
            if (self.min_2theta_spin.value() == 5.0 and 
                self.max_2theta_spin.value() == 60.0):
                self.auto_set_2theta_range()
        
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
        peak_weight = self.peak_weight_spin.value()
        corr_weight = self.corr_weight_spin.value()
        self.matching_thread = PhaseMatchingThread(
            self.experimental_peaks, 
            self.reference_phases, 
            tolerance,
            peak_weight,
            corr_weight
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
        
        # Sort by combined score (descending), fallback to match score
        filtered_results.sort(key=lambda x: x.get('combined_score', x['match_score']), reverse=True)
        
        # Update results table
        self.results_table.setRowCount(len(filtered_results))
        
        for i, result in enumerate(filtered_results):
            phase = result['phase']
            
            # Phase name
            phase_name = phase.get('mineral', 'Unknown')
            self.results_table.setItem(i, 0, QTableWidgetItem(phase_name))
            
            # Peak-based score
            peak_score_item = QTableWidgetItem(f"{result['match_score']:.3f}")
            self.results_table.setItem(i, 1, peak_score_item)
            
            # Correlation score
            corr_score = result.get('correlation', 0)
            corr_score_item = QTableWidgetItem(f"{corr_score:.3f}")
            self.results_table.setItem(i, 2, corr_score_item)
            
            # Combined score
            combined_score = result.get('combined_score', result['match_score'])
            combined_score_item = QTableWidgetItem(f"{combined_score:.3f}")
            self.results_table.setItem(i, 3, combined_score_item)
            
            # Coverage
            coverage_item = QTableWidgetItem(f"{result['coverage']:.3f}")
            self.results_table.setItem(i, 4, coverage_item)
            
            # Number of matches
            matches_item = QTableWidgetItem(str(len(result['matches'])))
            self.results_table.setItem(i, 5, matches_item)
            
            # R-squared
            r_squared = result.get('r_squared', 0)
            r_squared_item = QTableWidgetItem(f"{r_squared:.3f}")
            self.results_table.setItem(i, 6, r_squared_item)
            
            # Individual phase scaling control
            scale_spin = QDoubleSpinBox()
            scale_spin.setRange(1, 500)
            scale_spin.setDecimals(0)
            scale_spin.setValue(80)  # Default scaling
            scale_spin.setSingleStep(5)
            scale_spin.setSuffix("%")
            scale_spin.setToolTip(f"Individual scaling for {phase_name}")
            scale_spin.valueChanged.connect(self.update_plot)
            self.results_table.setCellWidget(i, 7, scale_spin)
            
            # Show checkbox
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(i < 3)  # Show top 3 by default
            show_checkbox.stateChanged.connect(self.update_plot)
            self.results_table.setCellWidget(i, 8, show_checkbox)
            
        self.update_plot()
        
    def apply_global_scaling(self):
        """Apply global scaling to all individual phase scales"""
        global_scale = self.global_scale_spin.value()
        
        # Update all individual phase scaling controls
        for i in range(self.results_table.rowCount()):
            scale_widget = self.results_table.cellWidget(i, 7)  # Updated column index
            if scale_widget and isinstance(scale_widget, QDoubleSpinBox):
                scale_widget.setValue(global_scale)
        
        print(f"Applied global scaling of {global_scale}% to all phases")
        
    def get_phase_scaling(self, row_index):
        """Get the individual scaling factor for a specific phase"""
        scale_widget = self.results_table.cellWidget(row_index, 7)  # Updated column index
        if scale_widget and isinstance(scale_widget, QDoubleSpinBox):
            return scale_widget.value() / 100.0  # Convert percentage to decimal
        return 0.8  # Default 80% scaling
        
    def show_match_details(self):
        """Show detailed matching information for selected phase"""
        current_row = self.results_table.currentRow()
        if current_row < 0:
            return
        
        # Get filtered results that match the current table display
        min_score = self.min_score_spin.value()
        filtered_results = [r for r in self.matching_results if r['match_score'] >= min_score]
        filtered_results.sort(key=lambda x: x.get('combined_score', x['match_score']), reverse=True)
        
        if current_row >= len(filtered_results):
            return
            
        result = filtered_results[current_row]
        matches = result['matches']
        
        # Update detail table
        self.detail_table.setRowCount(len(matches))
        
        for i, match in enumerate(matches):
            # Use stored 2θ values from matching algorithm
            exp_two_theta_deg = match.get('exp_2theta', 0.0)
            theo_two_theta_deg = match.get('theo_2theta', 0.0)
            two_theta_difference = match.get('difference', 0.0)  # This is now 2θ difference
            
            # Calculate match quality based on 2θ difference and intensity
            rel_diff = two_theta_difference / exp_two_theta_deg * 100 if exp_two_theta_deg > 0 else 100  # Relative difference as percentage
            intensity_sim = match.get('intensity_similarity', 0)
            
            # Combined quality score (position + intensity)
            if rel_diff < 0.1 and intensity_sim > 0.8:
                quality = "Excellent"
            elif rel_diff < 0.5 and intensity_sim > 0.6:
                quality = "Very Good"
            elif rel_diff < 1.0 and intensity_sim > 0.4:
                quality = "Good"
            elif rel_diff < 2.0 and intensity_sim > 0.2:
                quality = "Fair"
            else:
                quality = "Poor"
            
            # Populate table with 2θ values first, then d-spacing
            self.detail_table.setItem(i, 0, QTableWidgetItem(f"{exp_two_theta_deg:.3f}"))      # Exp. 2θ (°)
            self.detail_table.setItem(i, 1, QTableWidgetItem(f"{theo_two_theta_deg:.3f}"))     # Theo. 2θ (°)
            self.detail_table.setItem(i, 2, QTableWidgetItem(f"{two_theta_difference:.3f}"))   # Δ 2θ (°)
            self.detail_table.setItem(i, 3, QTableWidgetItem(f"{match['exp_int']:.0f}"))       # Exp. Int.
            self.detail_table.setItem(i, 4, QTableWidgetItem(f"{match['theo_int']:.0f}"))      # Theo. Int.
            self.detail_table.setItem(i, 5, QTableWidgetItem(f"{intensity_sim:.3f}"))          # Int. Sim.
            self.detail_table.setItem(i, 6, QTableWidgetItem(f"{match['exp_d']:.4f}"))         # Exp. d (Å)
            self.detail_table.setItem(i, 7, QTableWidgetItem(quality))                         # Match Quality
            
    def update_plot(self):
        """Update the comparison plot with normalized intensities"""
        self.ax_main.clear()
        self.ax_diff.clear()
        
        # Get user-defined 2θ range for consistent plotting
        min_2theta = self.min_2theta_spin.value()
        max_2theta = self.max_2theta_spin.value()
        
        # Ensure valid range
        if min_2theta >= max_2theta:
            max_2theta = min_2theta + 10
            self.max_2theta_spin.setValue(max_2theta)
        
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
        filtered_results.sort(key=lambda x: x.get('combined_score', x['match_score']), reverse=True)
        
        for i in range(self.results_table.rowCount()):
            checkbox = self.results_table.cellWidget(i, 8)  # Updated column index
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
                        
                        # Apply intensity filtering
                        min_intensity_percent = self.min_intensity_spin.value()
                        max_intensity = np.max(intensities) if len(intensities) > 0 else 1
                        intensity_threshold = (min_intensity_percent / 100.0) * max_intensity
                        
                        # Filter peaks by minimum intensity
                        intensity_mask = intensities >= intensity_threshold
                        two_theta = two_theta[intensity_mask]
                        intensities = intensities[intensity_mask]
                        
                        if len(intensities) == 0:
                            continue  # Skip if no peaks pass the intensity filter
                    else:
                        continue
                    
                    # Generate or retrieve cached continuous pattern
                    if pattern_to_plot and len(intensities) > 0:
                        # Create cache key for this specific pattern configuration
                        phase_id = result['phase'].get('id', f"phase_{i}")
                        wavelength = pattern_to_plot.get('wavelength', 1.5406)
                        fwhm = self.fwhm_spin.value()
                        scale_percentage = self.get_phase_scaling(i)  # Use individual phase scaling
                        min_intensity_percent = self.min_intensity_spin.value()
                        
                        # Create x-range for theoretical pattern based on experimental data range
                        x_min = np.min(pattern_to_plot['two_theta'])
                        x_max = np.max(pattern_to_plot['two_theta'])
                        x_range = np.linspace(x_min, x_max, len(pattern_to_plot['two_theta']) * 2)  # Higher resolution
                        
                        # Create cache key including all parameters that affect the pattern
                        x_range_hash = hash((x_min, x_max, len(x_range)))
                        cache_key = (phase_id, wavelength, fwhm, x_range_hash, scale_percentage, min_intensity_percent)
                        
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
                            
                            # CRITICAL FIX: Normalize the final continuous pattern to match expected scaling
                            if len(theoretical_pattern) > 0 and np.max(theoretical_pattern) > 0:
                                # The expected max should be the scale percentage (e.g., 80 for 80%)
                                expected_max = 100 * scale_percentage
                                actual_max = np.max(theoretical_pattern)
                                
                                # Normalize the continuous pattern to the expected maximum
                                theoretical_pattern = (theoretical_pattern / actual_max) * expected_max
                                
                                print(f"  Final continuous pattern for {phase_name}:")
                                print(f"    Before final normalization: {actual_max:.2f}")
                                print(f"    After final normalization: {np.max(theoretical_pattern):.2f}")
                                print(f"    Expected max: {expected_max:.2f}")
                                print(f"    Non-zero points: {np.count_nonzero(theoretical_pattern)}")
                            else:
                                print(f"  Warning: Empty theoretical pattern generated for {phase_name}")
                            
                            # Cache the result
                            self.continuous_pattern_cache[cache_key] = theoretical_pattern
                            
                        # Plot as continuous line
                        self.ax_main.plot(x_range, theoretical_pattern, 
                                        color=color, linewidth=1.5, alpha=0.8,
                                        label=phase_name)
                    # Plot theoretical peaks in difference plot as bars with intensity-based alpha
                    if len(theo_peaks.get('two_theta', [])) > 0:
                        theo_two_theta = theo_peaks['two_theta']
                        theo_intensities = theo_peaks['intensity']
                        
                        # Apply same intensity filtering as above
                        min_intensity_percent = self.min_intensity_spin.value()
                        max_intensity = np.max(theo_intensities) if len(theo_intensities) > 0 else 1
                        intensity_threshold = (min_intensity_percent / 100.0) * max_intensity
                        
                        # Filter peaks by minimum intensity
                        intensity_mask = theo_intensities >= intensity_threshold
                        filtered_two_theta = theo_two_theta[intensity_mask]
                        filtered_intensities = theo_intensities[intensity_mask]
                        
                        if len(filtered_intensities) > 0:
                            # Normalize intensities for alpha calculation (0.3 to 1.0 range)
                            norm_intensities = filtered_intensities / np.max(filtered_intensities)
                            alphas = 0.3 + 0.7 * norm_intensities  # Scale to 0.3-1.0 range
                            
                            # Plot theoretical peaks as bars with intensity-based alpha
                            y_position = -0.8 + (i * 0.15)  # Offset each phase slightly
                            for tt, alpha in zip(filtered_two_theta, alphas):
                                # Only plot peaks within the display range
                                if min_2theta <= tt <= max_2theta:
                                    self.ax_diff.axvline(x=tt, ymin=0.15 + (i * 0.1), ymax=0.25 + (i * 0.1), 
                                                       color=color, linewidth=2, alpha=alpha)
                            
                            # Add legend entry (use first peak within range)
                            visible_peaks = [(tt, alpha) for tt, alpha in zip(filtered_two_theta, alphas) if min_2theta <= tt <= max_2theta]
                            if visible_peaks:
                                self.ax_diff.scatter([visible_peaks[0][0]], [y_position], 
                                                   c=color, marker='|', s=100, label=phase_name, alpha=visible_peaks[0][1])
                    
                y_offset += 0.1
                
        # Plot experimental peaks in difference plot as bars with intensity-based alpha
        if self.experimental_peaks and len(self.experimental_peaks.get('two_theta', [])) > 0:
            exp_two_theta = self.experimental_peaks['two_theta']
            exp_intensities = self.experimental_peaks['intensity']
            
            # Normalize experimental intensities for alpha calculation
            max_exp_intensity = np.max(exp_intensities) if len(exp_intensities) > 0 else 1
            norm_exp_intensities = exp_intensities / max_exp_intensity
            exp_alphas = 0.3 + 0.7 * norm_exp_intensities  # Scale to 0.3-1.0 range
            
            # Plot as vertical bars with intensity-based alpha
            for tt, alpha in zip(exp_two_theta, exp_alphas):
                # Only plot peaks within the display range
                if min_2theta <= tt <= max_2theta:
                    self.ax_diff.axvline(x=tt, ymin=0.0, ymax=0.1, 
                                       color='blue', linewidth=2, alpha=alpha)
            
            # Add a single legend entry for experimental peaks (use first peak within range)
            visible_exp_data = [(tt, alpha) for tt, alpha in zip(exp_two_theta, exp_alphas) if min_2theta <= tt <= max_2theta]
            if visible_exp_data:
                self.ax_diff.scatter([visible_exp_data[0][0]], [-1], 
                                   c='blue', marker='|', s=100, label='Experimental', alpha=visible_exp_data[0][1])
            
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
        
        self.ax_diff.set_xlabel('2θ (degrees)')
        self.ax_diff.set_ylabel('Phase')
        min_int_text = f" (Min. Int.: {self.min_intensity_spin.value():.1f}%)"
        self.ax_diff.set_title('Peak Position Comparisons' + min_int_text)
        self.ax_diff.grid(True, alpha=0.3)
        
        # Set y-axis limits to show the bars properly
        self.ax_diff.set_ylim(-1.2, 1.0)
        
        # Remove y-axis ticks since they're not meaningful for this plot
        self.ax_diff.set_yticks([])
        
        if self.ax_diff.get_legend_handles_labels()[0]:  # Only show legend if there are items
            self.ax_diff.legend(loc='upper right')
        
        # Apply 2θ limits to both plots (synchronized)
        self.ax_main.set_xlim(min_2theta, max_2theta)
        self.ax_diff.set_xlim(min_2theta, max_2theta)
        
        # Set y-axis to normalized scale (0-110 to give some headroom)
        self.ax_main.set_ylim(0, 110)
        
        self.canvas.draw()
        
    def clear_results(self):
        """Clear matching results"""
        self.results_table.setRowCount(0)
        self.detail_table.setRowCount(0)
        
        # Clear pattern caches
        self.pattern_cache.clear()
        self.continuous_pattern_cache.clear()
        print("Cleared pattern caches - next plot update will recalculate with normalization fix")
        
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
    
    def reset_for_new_pattern(self):
        """Reset the matching tab when a new pattern is loaded"""
        # Clear reference phases and results
        self.reference_phases = []
        self.matching_results = []
        
        # Clear pattern cache
        self.pattern_cache.clear()
        self.continuous_pattern_cache.clear()
        
        # Clear results table
        self.results_table.setRowCount(0)
        
        # Clear match details table
        self.detail_table.setRowCount(0)
        
        # Reset progress bar
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        # Update availability
        self.update_matching_availability()
        
        # Clear and update plot
        self.update_plot()
        
        print("Matching tab reset for new pattern")
