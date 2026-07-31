"""
Le Bail refinement engine for XRD phase matching
Implements proper crystallographic refinement with profile functions and unit cell optimization
"""

import numpy as np
import itertools

from scipy.optimize import least_squares, nnls
from scipy.special import eval_legendre
from typing import Dict, List, Tuple, Optional
import copy
import warnings
warnings.filterwarnings('ignore')

class LeBailRefinement:
    """Le Bail refinement engine for multi-phase XRD analysis
    
    Features:
    - Profile function refinement (pseudo-Voigt, Pearson VII)
    - Unit cell parameter optimization
    - Peak position and intensity refinement
    - Multi-phase simultaneous refinement
    - Goodness-of-fit statistics (R-factors, chi-squared)
    """
    
    # Class variable for real-time plotting callback
    plot_callback = None
    
    def __init__(self):
        self.experimental_data = None
        self.phases = []
        self.refined_parameters = {}
        self.refinement_history = []
        self.r_factors = {}
        self.two_theta_range = None  # Optional 2-theta range (min, max)
        self.quiet = False
        self.mode = 'polish'  # 'trial' | 'polish'
        self.peak_intensity_cutoff = 0.0  # fraction of Imax; trial uses ~0.02
        self.extract_iterations = 3
        self._profile_cache = {}  # (phase_idx, cache_key) -> sparse profile parts
        self._mid_two_theta = 45.0  # anchor for angle-dependent corrections

        # Experimental grid geometry, cached for windowed profile evaluation
        self._x0 = 0.0
        self._dx = 0.02
        self._n = 0
        # Ceiling on reflections x window points in one profile array
        self._max_window_elements = 8_000_000
        # While set, phases reuse their last partitioned intensities instead of
        # re-partitioning. See _refine_single_phase.
        self._freeze_extracted = False

        # Instrument terms shared by every phase. Fitting these per phase lets
        # one phase absorb an instrument error and drift off its true cell.
        self.global_parameters = {
            'zero_shift': 0.0,      # constant 2-theta offset (degrees)
            'displacement': 0.0,    # specimen displacement: d(2-theta) = disp * cos(theta)
            'refine_zero_shift': True,
            'refine_displacement': False,
        }

        # 'extract': classic Le Bail, intensities partitioned out of the observed
        #            pattern. Best profile/cell fit, but scale and the intensity
        #            corrections are absorbed and cannot be determined.
        # 'fixed':   calculated intensities stay tied to the reference pattern and
        #            one scale per phase is refined. Required for quantification.
        self.intensity_model = 'extract'

    def _update_grid(self):
        """Cache grid geometry used for windowed profile evaluation"""
        x = self.experimental_data['two_theta'] if self.experimental_data else None
        if x is None or len(x) == 0:
            self._x0, self._dx, self._n = 0.0, 0.02, 0
            return
        self._x0 = float(x[0])
        self._n = len(x)
        self._dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.02
        if self._dx <= 0:
            self._dx = 0.02

    def set_global_parameters(self, **values):
        """Update shared instrument parameters and their refine flags."""
        for key, value in values.items():
            if key in self.global_parameters:
                self.global_parameters[key] = value

    def _uses_fixed_intensities(self, params: Optional[Dict] = None) -> bool:
        """True when calculated intensities come from the reference pattern."""
        if params is not None and params.get('_use_scaled_pattern'):
            return True
        return self.mode == 'trial' or self.intensity_model == 'fixed'
        
    def set_experimental_data(self, two_theta: np.ndarray, intensity: np.ndarray, 
                            errors: Optional[np.ndarray] = None,
                            two_theta_range: Optional[Tuple[float, float]] = None):
        """Set experimental diffraction data
        
        Args:
            two_theta: 2-theta values in degrees
            intensity: Intensity values
                      IMPORTANT: Should be background-subtracted intensity
                      Background subtraction must be performed before Le Bail refinement
                      to avoid fitting the background as part of the diffraction pattern
            errors: Optional error values (defaults to sqrt(intensity))
            two_theta_range: Optional (min, max) 2-theta range to limit refinement
        """
        # Normalize intensity to 0-100 scale for better numerical stability
        intensity = np.array(intensity)
        max_intensity = np.max(intensity)
        
        if max_intensity > 0:
            normalized_intensity = (intensity / max_intensity) * 100.0
            self._log(f"Normalized experimental intensity: {max_intensity:.0f} → 100.0")
        else:
            normalized_intensity = intensity
        
        # Scale errors proportionally
        if errors is not None:
            errors = np.array(errors)
            normalized_errors = (errors / max_intensity) * 100.0 if max_intensity > 0 else errors
        else:
            normalized_errors = np.sqrt(np.maximum(normalized_intensity, 1))
        
        self.experimental_data = {
            'two_theta': np.array(two_theta),
            'intensity': normalized_intensity,
            'errors': normalized_errors,
            'original_max_intensity': max_intensity  # Store for reference
        }
        self.two_theta_range = two_theta_range
        self._update_grid()
        
        # Apply 2-theta range filter if specified
        if self.two_theta_range is not None:
            self._apply_two_theta_filter()
        
        tt = self.experimental_data['two_theta']
        if len(tt) > 0:
            self._mid_two_theta = float(0.5 * (np.min(tt) + np.max(tt)))
        
    def add_phase(self, phase_data: Dict, initial_parameters: Optional[Dict] = None):
        """
        Add a phase for refinement
        
        Args:
            phase_data: Phase information including theoretical peaks
            initial_parameters: Initial refinement parameters
        """
        if 'theoretical_peaks' not in phase_data:
            raise ValueError("Phase data must include theoretical_peaks")
        
        # Estimate initial scale factor from intensity ratio
        initial_scale = self._estimate_initial_scale(phase_data['theoretical_peaks'])
            
        # Default refinement parameters
        default_params = {
            'scale_factor': initial_scale,
            'u_param': 0.01,      # Peak width parameter U
            'v_param': -0.001,    # Peak width parameter V  
            'w_param': 0.01,      # Peak width parameter W
            'eta_param': 0.5,     # Pseudo-Voigt mixing parameter
            'zero_shift': 0.0,    # legacy per-phase offset; the global term is used now
            'unit_cell': self._extract_unit_cell(phase_data),
            # Isotropic lattice dilation: every d-spacing scales by this factor.
            # Anisotropic a/b/c refinement needs Miller indices, which the stored
            # reference patterns do not carry.
            'lattice_scale': 1.0,
            'absorption': 0.0,        # angle-dependent intensity loss
            'harmonic_order': 0,      # even spherical-harmonic order (0, 2, 4, 6)
            'harmonic_coeffs': [],
            'refine_cell': True,
            'refine_profile': True,
            'refine_scale': True,
            'refine_absorption': False,
            'refine_harmonics': False,
            'refine_intensities': False  # Pawley-style intensity refinement
        }
        
        if initial_parameters:
            default_params.update(initial_parameters)
        
        # Initialize individual peak intensity multipliers for Pawley refinement
        n_peaks = len(phase_data['theoretical_peaks'].get('two_theta', []))
        if default_params.get('refine_intensities', False):
            # Start all intensity multipliers at 1.0
            default_params['peak_intensity_multipliers'] = np.ones(n_peaks)
        else:
            default_params['peak_intensity_multipliers'] = None

        order = int(default_params.get('harmonic_order', 0) or 0)
        n_harmonics = max(0, order // 2)
        coeffs = list(default_params.get('harmonic_coeffs') or [])
        if len(coeffs) != n_harmonics:
            coeffs = (coeffs + [0.0] * n_harmonics)[:n_harmonics]
        default_params['harmonic_coeffs'] = coeffs
        # Lattice dilation is refined against the starting cell, so keep a copy
        default_params['_base_unit_cell'] = dict(default_params['unit_cell'])
            
        phase = {
            'data': phase_data,
            'parameters': default_params,
            'theoretical_peaks': phase_data['theoretical_peaks'].copy()
        }
        
        self.phases.append(phase)
    
    def _apply_two_theta_filter(self):
        """Apply 2-theta range filter to experimental data"""
        if self.two_theta_range is None:
            return
        
        # Store original data if not already stored (to prevent double-filtering)
        if not hasattr(self, '_original_experimental_data'):
            self._original_experimental_data = {
                'two_theta': self.experimental_data['two_theta'].copy(),
                'intensity': self.experimental_data['intensity'].copy(),
                'errors': self.experimental_data['errors'].copy()
            }
            
        min_2theta, max_2theta = self.two_theta_range
        two_theta = self._original_experimental_data['two_theta']
        
        # Create mask for the specified range
        mask = (two_theta >= min_2theta) & (two_theta <= max_2theta)
        
        # Filter all data arrays from original data
        self.experimental_data['two_theta'] = two_theta[mask]
        self.experimental_data['intensity'] = self._original_experimental_data['intensity'][mask]
        self.experimental_data['errors'] = self._original_experimental_data['errors'][mask]
        self._update_grid()
        
        self._log(f"Applied 2-theta range filter: {min_2theta:.2f}° - {max_2theta:.2f}°")
        self._log(f"Data points: {len(two_theta)} → {len(self.experimental_data['two_theta'])}")

    def _log(self, message: str):
        """Print unless quiet mode is enabled"""
        if not self.quiet:
            print(message)

    def clear_phases(self):
        """Remove all phases and profile cache"""
        self.phases = []
        self._profile_cache = {}
        self._freeze_extracted = False
        self.refinement_history = []
        self.r_factors = {}
        
    def _estimate_initial_scale(self, theoretical_peaks: Dict) -> float:
        """
        Estimate initial scale factor by comparing experimental and theoretical intensities
        
        Both experimental and theoretical are now normalized to 0-100 scale, so the
        initial scale should be close to 1.0, but we still estimate it for better convergence.
        
        Args:
            theoretical_peaks: Dictionary with 'two_theta' and 'intensity' arrays
            
        Returns:
            Estimated scale factor
        """
        if not self.experimental_data:
            return 1.0
            
        exp_two_theta = self.experimental_data['two_theta']
        exp_intensity = self.experimental_data['intensity']
        theo_two_theta = np.array(theoretical_peaks.get('two_theta', []))
        theo_intensity = np.array(theoretical_peaks.get('intensity', []))
        
        if len(theo_two_theta) == 0 or len(exp_two_theta) == 0:
            return 1.0
        
        # Find overlapping 2θ range
        exp_min, exp_max = np.min(exp_two_theta), np.max(exp_two_theta)
        theo_min, theo_max = np.min(theo_two_theta), np.max(theo_two_theta)
        overlap_min = max(exp_min, theo_min)
        overlap_max = min(exp_max, theo_max)
        
        if overlap_min >= overlap_max:
            return 1.0
        
        # Get max intensities in overlapping region
        exp_mask = (exp_two_theta >= overlap_min) & (exp_two_theta <= overlap_max)
        theo_mask = (theo_two_theta >= overlap_min) & (theo_two_theta <= overlap_max)
        
        if not np.any(exp_mask) or not np.any(theo_mask):
            return 1.0
        
        exp_max_intensity = np.max(exp_intensity[exp_mask])
        theo_max_intensity = np.max(theo_intensity[theo_mask])
        
        if theo_max_intensity > 0:
            # Both are normalized to 0-100, so scale should be close to 1.0
            # Target 80% of experimental max for initial estimate
            initial_scale = (exp_max_intensity * 0.8) / theo_max_intensity
            self._log(f"Estimated initial scale factor: {initial_scale:.3f}")
            return initial_scale
        
        return 1.0
    
    def _extract_unit_cell(self, phase_data: Dict) -> Dict:
        """Extract unit cell parameters from phase data"""
        phase_info = phase_data.get('phase', {})
        
        # Try to get unit cell from phase data with proper defaults
        unit_cell = {
            'a': phase_info.get('cell_a', 10.0) if phase_info.get('cell_a') else 10.0,
            'b': phase_info.get('cell_b', 10.0) if phase_info.get('cell_b') else 10.0, 
            'c': phase_info.get('cell_c', 10.0) if phase_info.get('cell_c') else 10.0,
            'alpha': phase_info.get('cell_alpha', 90.0) if phase_info.get('cell_alpha') else 90.0,
            'beta': phase_info.get('cell_beta', 90.0) if phase_info.get('cell_beta') else 90.0,
            'gamma': phase_info.get('cell_gamma', 90.0) if phase_info.get('cell_gamma') else 90.0
        }
        unit_cell['volume'] = self._cell_volume(unit_cell)
        
        return unit_cell

    @staticmethod
    def _cell_volume(cell: Dict) -> float:
        """Triclinic cell volume, valid for every crystal system."""
        try:
            a, b, c = float(cell['a']), float(cell['b']), float(cell['c'])
            alpha, beta, gamma = (
                np.radians(float(cell.get('alpha', 90.0))),
                np.radians(float(cell.get('beta', 90.0))),
                np.radians(float(cell.get('gamma', 90.0))),
            )
        except (KeyError, TypeError, ValueError):
            return 0.0
        term = (
            1.0
            - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2
            + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        return float(a * b * c * np.sqrt(max(term, 0.0)))
        
    def refine_phases(self, max_iterations: int = 20, 
                     convergence_threshold: float = 1e-5,
                     two_theta_range: Optional[Tuple[float, float]] = None,
                     staged_refinement: bool = True,
                     mode: str = 'polish',
                     quiet: Optional[bool] = None) -> Dict:
        """
        Perform Le Bail refinement on all phases
        
        Args:
            max_iterations: Maximum number of refinement cycles
            convergence_threshold: Convergence criterion for R-factors
            two_theta_range: Optional (min, max) 2-theta range to limit refinement
            staged_refinement: Use staged refinement (unit cell first, then profile)
            mode: 'trial' for fast accept/reject fits, 'polish' for full refinement
            quiet: Suppress verbose logging (defaults True for trial, False for polish)
            
        Returns:
            Dictionary with refinement results
        """
        self.mode = mode if mode in ('trial', 'polish') else 'polish'
        if quiet is not None:
            self.quiet = quiet
        else:
            self.quiet = (self.mode == 'trial')

        if self.mode == 'trial':
            # Fast path defaults
            self.peak_intensity_cutoff = 0.02
            self.extract_iterations = 1
            max_iterations = min(max_iterations, 2)
            staged_refinement = False
            for phase in self.phases:
                phase['parameters']['refine_cell'] = False
                phase['parameters']['refine_profile'] = False
                phase['parameters']['refine_scale'] = True
                phase['parameters']['refine_intensities'] = False
                # Trial uses scaled theoretical intensities (not full Le Bail extract each step)
                phase['parameters']['_use_scaled_pattern'] = True
        else:
            self.peak_intensity_cutoff = 0.0
            self.extract_iterations = 5
            fixed = self.intensity_model == 'fixed'
            for phase in self.phases:
                phase['parameters']['_use_scaled_pattern'] = fixed
            self._log(
                "Intensity model: reference intensities with refined scale"
                if fixed else
                "Intensity model: Le Bail extraction (scale and corrections not determinable)"
            )

        self._profile_cache = {}
        self._global_scan_pass = 0
        self._freeze_extracted = False
        for phase in self.phases:
            phase.pop('_extracted_intensities', None)

        # Update 2-theta range if provided
        if two_theta_range is not None and two_theta_range != self.two_theta_range:
            self.two_theta_range = two_theta_range
            self._apply_two_theta_filter()
        if not self.experimental_data or not self.phases:
            raise ValueError("Must set experimental data and add phases before refinement")
            
        self._log(f"Starting Le Bail refinement ({self.mode}) with {len(self.phases)} phases")
        self._log(f"Experimental data: {len(self.experimental_data['two_theta'])} points")
        
        # Check for Pawley mode and warn if too many peaks
        total_pawley_params = 0
        for phase in self.phases:
            if phase['parameters'].get('refine_intensities', False):
                n_peaks = len(phase['theoretical_peaks'].get('two_theta', []))
                total_pawley_params += n_peaks
        
        if total_pawley_params > 0:
            self._log(f"Pawley mode enabled: refining {total_pawley_params} individual peak intensities")
            if total_pawley_params > 100:
                self._log(f"WARNING: {total_pawley_params} intensity parameters may cause slow/unstable refinement")
        
        if staged_refinement and self.mode == 'polish':
            self._log("Using staged refinement: unit cell → profile parameters")
        
        # Initialize refinement
        self.refinement_history = []
        previous_rwp = float('inf')
        rwp_change = float('inf')

        # Align the pattern before solving the scales. Fitting scales against
        # misaligned peaks drives a real phase towards zero, and a phase that
        # starts at zero contributes nothing for the position search to work with.
        self._refine_global_parameters()
        self._initialize_scales()
        
        # STAGE 1: Refine unit cell and zero shift only (if staged refinement)
        if staged_refinement and self.mode == 'polish':
            self._log("\n=== STAGE 1: Unit Cell & Zero Shift Refinement ===")
            saved_intensity_settings = []
            for phase in self.phases:
                phase['parameters']['refine_profile'] = False
                saved_intensity_settings.append(phase['parameters'].get('refine_intensities', False))
                phase['parameters']['refine_intensities'] = False
            
            stage1_iterations = max(3, max_iterations // 3)
            for iteration in range(stage1_iterations):
                self._log(f"\nStage 1 - Iteration {iteration + 1}/{stage1_iterations}")
                
                self._refine_global_parameters()
                for phase_idx, phase in enumerate(self.phases):
                    self._refine_single_phase(phase_idx)
                
                calculated_pattern = self._calculate_total_pattern()
                r_factors = self._calculate_r_factors(calculated_pattern)
                self._log(f"R-factors: Rp={r_factors['Rp']:.3f}, Rwp={r_factors['Rwp']:.3f}")
                
                iteration_result = {
                    'iteration': iteration + 1,
                    'stage': 1,
                    'r_factors': r_factors.copy(),
                    'parameters': copy.deepcopy([p['parameters'] for p in self.phases]),
                    'calculated_pattern': calculated_pattern.copy()
                }
                self.refinement_history.append(iteration_result)
            
            self._log("\n=== STAGE 2: Profile Parameter Refinement ===")
            for idx, phase in enumerate(self.phases):
                phase['parameters']['refine_profile'] = True
                if idx < len(saved_intensity_settings):
                    phase['parameters']['refine_intensities'] = saved_intensity_settings[idx]
            
            remaining_iterations = max_iterations - stage1_iterations
            start_iteration = stage1_iterations
        else:
            remaining_iterations = max_iterations
            start_iteration = 0
        
        # Main refinement loop
        for iteration in range(remaining_iterations):
            actual_iteration = start_iteration + iteration + 1
            stage_label = "Stage 2 - " if (staged_refinement and self.mode == 'polish') else ""
            self._log(f"\n=== {stage_label}Le Bail Iteration {actual_iteration} ===")
            
            self._refine_global_parameters()
            for phase_idx, phase in enumerate(self.phases):
                phase_name = phase['data']['phase'].get('mineral', f'Phase_{phase_idx}')
                self._log(f"Refining {phase_name}...")
                self._refine_single_phase(phase_idx)
                
            calculated_pattern = self._calculate_total_pattern()
            r_factors = self._calculate_r_factors(calculated_pattern)
            phase_contributions = self._calculate_phase_contributions()
            
            self._log(f"R-factors: Rp={r_factors['Rp']:.3f}, Rwp={r_factors['Rwp']:.3f}, "
                      f"GoF={r_factors['GoF']:.3f}")
            
            for phase_idx, phase in enumerate(self.phases):
                phase_name = phase['data']['phase'].get('mineral', f'Phase_{phase_idx}')
                scale = phase['parameters']['scale_factor']
                phase_rwp = phase_contributions[phase_idx]['rwp']
                contribution = phase_contributions[phase_idx]['contribution_percent']
                self._log(f"  {phase_name}: Scale={scale:.3f}, Rwp={phase_rwp:.2f}%, Contribution={contribution:.1f}%")
            
            iteration_result = {
                'iteration': actual_iteration,
                'stage': 2 if staged_refinement else 0,
                'mode': self.mode,
                'r_factors': r_factors.copy(),
                'parameters': copy.deepcopy([p['parameters'] for p in self.phases]),
                'calculated_pattern': calculated_pattern.copy()
            }
            self.refinement_history.append(iteration_result)
            
            if self.plot_callback is not None and self.mode == 'polish':
                try:
                    self.plot_callback(iteration_result, self.experimental_data)
                except Exception as e:
                    self._log(f"Warning: Plot callback failed: {e}")
            
            rwp_change = abs(previous_rwp - r_factors['Rwp'])
            if rwp_change < convergence_threshold:
                self._log(f"Converged after {iteration + 1} iterations (ΔRwp = {rwp_change:.6f})")
                break
                
            previous_rwp = r_factors['Rwp']
            
        final_results = {
            'converged': rwp_change < convergence_threshold,
            'iterations': len(self.refinement_history),
            'mode': self.mode,
            'final_r_factors': self.refinement_history[-1]['r_factors'],
            'refined_phases': copy.deepcopy(self.phases),
            'calculated_pattern': self.refinement_history[-1]['calculated_pattern'],
            'refinement_history': self.refinement_history,
            'two_theta': self.experimental_data['two_theta'].copy(),
            'experimental_intensity': self.experimental_data['intensity'].copy(),
            'global_parameters': dict(self.global_parameters),
            'intensity_model': self.intensity_model,
            'phase_summary': self.phase_summary(),
        }
        
        self.r_factors = final_results['final_r_factors']
        
        return final_results
        
    def _initialize_scales(self):
        """
        Solve every phase scale at once by non-negative least squares.

        The scales enter the calculated pattern linearly, so they can be solved
        outright instead of being searched for. Starting the nonlinear refinement
        from the right order of magnitude matters: with scales badly wrong, the
        first fit of the position parameters shifts peaks to compensate and the
        refinement settles into a poor minimum it never leaves.
        """
        if not self._uses_fixed_intensities() or not self.phases:
            return
        columns = []
        for index, phase in enumerate(self.phases):
            params = dict(phase['parameters'])
            params['scale_factor'] = 1.0
            columns.append(self._calculate_phase_pattern(index, params))
        design = np.column_stack(columns)
        if not np.any(design > 0):
            return
        try:
            scales, _ = nnls(design, self.experimental_data['intensity'])
        except Exception as e:
            self._log(f"Scale initialization failed: {e}")
            return
        for phase, scale in zip(self.phases, scales):
            if scale > 0:
                phase['parameters']['scale_factor'] = float(scale)
        self._log(
            "  Initial scales (least squares): "
            + ", ".join(f"{p['parameters']['scale_factor']:.4f}" for p in self.phases)
        )

    # Half-width of the global search window per pass, in degrees. It shrinks
    # because the phase lattices move underneath and the instrument terms have to
    # be re-searched, not just nudged, until both have settled.
    _GLOBAL_SCAN_SCHEDULE = (0.40, 0.10, 0.03, 0.01)

    def _refine_global_parameters(self):
        """
        Refine the instrument terms shared by all phases against the full pattern.

        Done in one step over the total calculated pattern rather than per phase,
        so a zero-point error cannot be soaked up by one phase's lattice.

        Each pass scans a window around the current values before polishing
        locally. Zero shift and specimen displacement are nearly collinear over a
        limited 2-theta range, since cos(theta) barely varies, so a local
        optimizer slides along that degenerate valley into a bound instead of
        finding the combination that fits both ends of the pattern.
        """
        names = []
        vector = []
        bounds = []
        if self.global_parameters.get('refine_zero_shift', False):
            names.append('zero_shift')
            vector.append(float(self.global_parameters.get('zero_shift', 0.0)))
            bounds.append((-0.5, 0.5))
        if self.global_parameters.get('refine_displacement', False):
            names.append('displacement')
            vector.append(float(self.global_parameters.get('displacement', 0.0)))
            bounds.append((-0.5, 0.5))
        if not names:
            return

        saved = {name: self.global_parameters[name] for name in names}
        observed = self.experimental_data['intensity']
        errors = self.experimental_data['errors']
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)

        def residuals(x):
            for name, value in zip(names, x):
                self.global_parameters[name] = float(value)
            return (observed - self._calculate_total_pattern()) / errors

        def chi2(x):
            return float(np.sum(residuals(x) ** 2))

        # The scan below evaluates the whole pattern hundreds of times, so the
        # partitioned intensities are fixed for the duration, as in a Le Bail step
        self._refresh_extracted()
        self._freeze_extracted = True
        try:
            scan_pass = getattr(self, '_global_scan_pass', 0)
            if scan_pass < len(self._GLOBAL_SCAN_SCHEDULE):
                half_width = self._GLOBAL_SCAN_SCHEDULE[scan_pass]
                # Eleven samples per axis is enough to find the valley; 21^n was
                # spending most of the refinement budget on the scan alone
                grids = [
                    np.clip(
                        np.linspace(center - half_width, center + half_width, 11),
                        low, high,
                    )
                    for center, (low, high) in zip(vector, bounds)
                ]
                best_value = chi2(vector)
                best_point = list(vector)
                for point in itertools.product(*grids):
                    value = chi2(point)
                    if value < best_value:
                        best_value, best_point = value, list(point)
                vector = best_point
                self._global_scan_pass = scan_pass + 1

            result = least_squares(
                residuals, np.asarray(vector, dtype=float),
                bounds=(lower, upper), method='trf',
                x_scale=np.maximum(upper - lower, 1e-8),
                diff_step=1e-4, ftol=1e-10, xtol=1e-10, gtol=1e-8,
                max_nfev=80,
            )
            best = result.x if chi2(result.x) <= chi2(vector) else vector
            for name, value in zip(names, best):
                self.global_parameters[name] = float(value)
            self._log(
                "  Global: zero shift="
                f"{self.global_parameters['zero_shift']:+.4f}°, "
                f"displacement={self.global_parameters['displacement']:+.4f}°"
            )
        except Exception as e:
            for name, value in saved.items():
                self.global_parameters[name] = value
            self._log(f"Global parameter refinement failed: {e}")
        finally:
            self._freeze_extracted = False
            self._refresh_extracted()

    # Parameters that move peak positions. Fitting them in the same least-squares
    # step as the profile widths makes the Jacobian so ill-conditioned that the
    # trust-region SVD hangs; they are refined in a separate pass instead.
    _POSITION_PARAM_NAMES = frozenset({'lattice_scale'})

    def _refine_single_phase(self, phase_idx: int):
        """Refine parameters for a single phase"""
        phase = self.phases[phase_idx]
        params = phase['parameters']

        param_vector, param_bounds, param_names = self._create_parameter_vector(params)
        if len(param_vector) == 0:
            return

        other_pattern = np.zeros_like(self.experimental_data['two_theta'])
        for i in range(len(self.phases)):
            if i != phase_idx:
                other_pattern += self._calculate_phase_pattern(i, self.phases[i]['parameters'])

        position_idx = [i for i, n in enumerate(param_names) if n in self._POSITION_PARAM_NAMES]
        intensity_idx = [i for i, n in enumerate(param_names) if n not in self._POSITION_PARAM_NAMES]
        # Position first so the profile fit starts with peaks already aligned
        groups = [g for g in (position_idx, intensity_idx) if g]

        try:
            # Le Bail step: partition once, then hold intensities fixed while the
            # profile and lattice are fitted against them
            self._refresh_extracted(phase_idx)
            self._freeze_extracted = True
            try:
                working = np.asarray(param_vector, dtype=float)
                for group in groups:
                    working = self._fit_parameter_group(
                        phase_idx, params, working, param_bounds, param_names,
                        group, other_pattern,
                    )
            finally:
                self._freeze_extracted = False

            optimized_params = self._vector_to_parameters(working, param_names, params)

            if 'u_param' in optimized_params:
                self._log(f"  Profile refined: U={optimized_params['u_param']:.6f}, "
                          f"V={optimized_params['v_param']:.6f}, W={optimized_params['w_param']:.6f}, "
                          f"η={optimized_params['eta_param']:.3f}")

            if 'lattice_scale' in optimized_params:
                cell = optimized_params.get('unit_cell', params.get('unit_cell', {}))
                self._log(
                    f"  Lattice scale: {optimized_params['lattice_scale']:.6f} "
                    f"(a={cell.get('a', 0.0):.4f}, b={cell.get('b', 0.0):.4f}, "
                    f"c={cell.get('c', 0.0):.4f} Å)"
                )

            if 'absorption' in optimized_params:
                self._log(f"  Absorption: {optimized_params['absorption']:.4f}")

            if 'harmonic_coeffs' in optimized_params:
                coeffs = ", ".join(f"{c:+.3f}" for c in optimized_params['harmonic_coeffs'])
                self._log(f"  Harmonic coefficients: {coeffs}")

            if 'scale_factor' in optimized_params:
                final_scale = optimized_params['scale_factor']
                initial_scale = params['scale_factor']
                if final_scale < initial_scale * 0.2:
                    self._log(f"  WARNING: Scale collapsed from {initial_scale:.3f} to {final_scale:.3f}")

            phase['parameters'].update(optimized_params)
            self._refresh_extracted(phase_idx)

        except Exception as e:
            self._freeze_extracted = False
            self._log(f"Optimization failed for phase {phase_idx}: {e}")

    def _fit_parameter_group(self, phase_idx: int, params: Dict,
                             working: np.ndarray, param_bounds: List,
                             param_names: List[str], group: List[int],
                             other_pattern: np.ndarray) -> np.ndarray:
        """least_squares on one disjoint subset of the free parameters."""
        observed = self.experimental_data['intensity']
        errors = self.experimental_data['errors']
        x0 = working[group]
        lower = np.array([param_bounds[i][0] for i in group], dtype=float)
        upper = np.array([param_bounds[i][1] for i in group], dtype=float)
        names = [param_names[i] for i in group]
        max_nfev = 30 if self.mode == 'trial' else 80
        ftol = 1e-4 if self.mode == 'trial' else 1e-10

        def residuals(x):
            trial = working.copy()
            trial[group] = x
            temp = self._vector_to_parameters(trial, param_names, params)
            merged = dict(params)
            merged.update(temp)
            if 'unit_cell' in temp:
                merged['unit_cell'] = temp['unit_cell']
            total = self._calculate_phase_pattern(phase_idx, merged) + other_pattern
            return (observed - total) / errors

        if len(group) == 1 and names[0] == 'lattice_scale':
            # A 1-D bracketed search is cheaper and more robust than TRF for the
            # single lattice dilation; peak sliding makes the Jacobian noisy
            best_x, best_val = float(x0[0]), float(np.sum(residuals(x0) ** 2))
            for trial in np.linspace(lower[0], upper[0], 21):
                value = float(np.sum(residuals(np.array([trial])) ** 2))
                if value < best_val:
                    best_x, best_val = float(trial), value
            x0 = np.array([best_x])

        result = least_squares(
            residuals, x0, bounds=(lower, upper), method='trf',
            x_scale=np.maximum(upper - lower, 1e-8),
            diff_step=1e-4, ftol=ftol, xtol=ftol, gtol=1e-8,
            max_nfev=max_nfev,
        )
        updated = working.copy()
        updated[group] = result.x
        return updated
            
    def _create_parameter_vector(self, params: Dict) -> Tuple[np.ndarray, List, List]:
        """Create parameter vector for optimization"""
        param_vector = []
        param_bounds = []
        param_names = []
        
        is_pawley = params.get('refine_intensities', False)
        use_scaled = self._uses_fixed_intensities(params)
        
        # Scale only means something when calculated intensities are tied to the
        # reference pattern; Le Bail extraction absorbs it entirely
        if params.get('refine_scale', True) and (is_pawley or use_scaled):
            initial_scale = params['scale_factor']
            param_vector.append(initial_scale)
            # Bounds must not be tied to the current value: a phase that starts
            # near zero because its peaks were not yet aligned would be locked
            # there for the rest of the refinement. Zero stays reachable so a
            # genuinely absent phase can still fall out.
            upper = max(params.get('max_scale_bound', 10.0), 20.0 * max(initial_scale, 1e-3))
            param_bounds.append((0.0, upper))
            param_names.append('scale_factor')
            self._log(f"  Scale bounds: 0 - {upper:.3f} (initial: {initial_scale:.4g})")
        elif params.get('refine_scale', True) and not use_scaled:
            self._log("  Scale factor: not refinable with Le Bail intensity extraction")
            
        if params.get('refine_profile', True):
            param_vector.extend([
                params['u_param'],
                params['v_param'], 
                params['w_param'],
                params['eta_param']
            ])
            param_bounds.extend([
                (0.0, 0.05),
                (-0.01, 0.01),
                (0.00001, 0.05),
                (0.0, 1.0)
            ])
            param_names.extend(['u_param', 'v_param', 'w_param', 'eta_param'])
            self._log(f"  Profile params: U={params['u_param']:.6f}, V={params['v_param']:.6f}, "
                      f"W={params['w_param']:.6f}, η={params['eta_param']:.3f}")
            
        # Zero shift and specimen displacement are refined globally, not here
        if params.get('refine_cell', True):
            param_vector.append(params.get('lattice_scale', 1.0))
            param_bounds.append((0.95, 1.05))
            param_names.append('lattice_scale')

        if params.get('refine_absorption', False) and use_scaled:
            param_vector.append(params.get('absorption', 0.0))
            param_bounds.append((-0.5, 0.5))
            param_names.append('absorption')

        coeffs = params.get('harmonic_coeffs') or []
        if params.get('refine_harmonics', False) and use_scaled and len(coeffs):
            for index in range(len(coeffs)):
                param_vector.append(coeffs[index])
                param_bounds.append((-0.9, 0.9))
                param_names.append(f'harmonic_{index}')

        if params.get('refine_intensities', False) and params.get('peak_intensity_multipliers') is not None:
            multipliers = params['peak_intensity_multipliers']
            param_vector.extend(multipliers)
            for i in range(len(multipliers)):
                param_bounds.append((0.1, 10.0))
                param_names.append(f'intensity_mult_{i}')
            
        return np.array(param_vector), param_bounds, param_names
        
    def _vector_to_parameters(self, vector: np.ndarray, names: List[str], 
                            original_params: Dict) -> Dict:
        """Convert parameter vector back to parameter dictionary"""
        params = {}
        intensity_multipliers = []
        harmonics = dict(enumerate(original_params.get('harmonic_coeffs') or []))
        harmonics_seen = False
        
        for i, name in enumerate(names):
            if name.startswith('cell_'):
                if 'unit_cell' not in params:
                    params['unit_cell'] = original_params['unit_cell'].copy()
                params['unit_cell'][name[5:]] = vector[i]
            elif name.startswith('intensity_mult_'):
                intensity_multipliers.append(vector[i])
            elif name.startswith('harmonic_'):
                harmonics[int(name.split('_')[1])] = float(vector[i])
                harmonics_seen = True
            else:
                params[name] = vector[i]
        
        if intensity_multipliers:
            params['peak_intensity_multipliers'] = np.array(intensity_multipliers)
        if harmonics_seen:
            params['harmonic_coeffs'] = [harmonics[k] for k in sorted(harmonics)]

        # A refined lattice dilation is only meaningful if it is also reported as
        # cell edges, so keep the stored cell in step with it
        if 'lattice_scale' in params:
            base = original_params.get('_base_unit_cell') or original_params.get('unit_cell')
            if base:
                scale = float(params['lattice_scale'])
                cell = dict(base)
                for edge in ('a', 'b', 'c'):
                    if cell.get(edge):
                        cell[edge] = float(base[edge]) * scale
                cell['volume'] = self._cell_volume(cell)
                params['unit_cell'] = cell
                
        return params
        
    def _filter_peaks(self, theo_peaks: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply intensity cutoff; return positions and intensities"""
        positions = np.asarray(theo_peaks.get('two_theta', []), dtype=float)
        intensities = np.asarray(theo_peaks.get('intensity', []), dtype=float)
        if len(positions) == 0:
            return positions, intensities
        if self.peak_intensity_cutoff > 0:
            imax = np.max(intensities) if np.max(intensities) > 0 else 1.0
            mask = intensities >= (self.peak_intensity_cutoff * imax)
            return positions[mask], intensities[mask]
        return positions, intensities

    def _shift_positions(self, positions: np.ndarray, parameters: Dict) -> np.ndarray:
        """
        Reference positions moved by the phase lattice and the shared instrument terms.

        An isotropic lattice dilation scales every d-spacing, so sin(theta) scales
        by 1/factor. Working in sin(theta) keeps this independent of wavelength.
        Specimen displacement in Bragg-Brentano geometry adds a cos(theta) term.
        """
        positions = np.asarray(positions, dtype=float)

        lattice_scale = float(parameters.get('lattice_scale', 1.0) or 1.0)
        if lattice_scale > 0 and abs(lattice_scale - 1.0) > 1e-12:
            sin_theta = np.sin(np.radians(positions / 2.0)) / lattice_scale
            positions = 2.0 * np.degrees(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))

        positions = positions + float(self.global_parameters.get('zero_shift', 0.0))

        displacement = float(self.global_parameters.get('displacement', 0.0))
        if displacement != 0.0:
            positions = positions + displacement * np.cos(np.radians(positions / 2.0))
        return positions

    def _intensity_corrections(self, positions: np.ndarray, parameters: Dict,
                               intensities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Per-phase multiplicative intensity corrections.

        Absorption: exp(-a / sin(theta)), the angular form of an absorption or
        microabsorption loss.

        Texture: an axially symmetric spherical-harmonic expansion over even
        orders in cos(theta). A full orientation distribution would need Miller
        indices per reflection, which the stored reference patterns do not carry.

        Both are renormalized so their intensity-weighted mean over this phase's
        reflections is one. Without that they can rescale the phase wholesale,
        which is indistinguishable from changing its abundance: the refinement
        then trades scale against absorption and the weight percents drift. A
        redistributing correction changes the shape of the phase's intensity
        envelope, which is what the data can actually determine from one pattern,
        and leaves the scale factor meaning what it should.
        """
        factor = np.ones(len(positions), dtype=float)
        if len(positions) == 0:
            return factor
        theta = np.radians(np.asarray(positions, dtype=float) / 2.0)

        absorption = float(parameters.get('absorption', 0.0) or 0.0)
        if absorption != 0.0:
            sin_theta = np.maximum(np.sin(theta), 1e-6)
            factor *= np.exp(-absorption / sin_theta)

        coeffs = parameters.get('harmonic_coeffs') or []
        if len(coeffs):
            cos_theta = np.cos(theta)
            harmonic = np.ones(len(positions), dtype=float)
            for index, coefficient in enumerate(coeffs):
                if coefficient:
                    harmonic += float(coefficient) * eval_legendre(2 * (index + 1), cos_theta)
            factor *= np.maximum(harmonic, 0.05)  # never let texture null a peak out

        if intensities is not None and len(intensities) == len(factor):
            weights = np.asarray(intensities, dtype=float)
            total = float(np.sum(weights))
            if total > 0:
                mean = float(np.sum(weights * factor)) / total
                if mean > 1e-6:
                    factor /= mean
        return factor

    def _calculate_phase_pattern(self, phase_idx: int, parameters: Dict) -> np.ndarray:
        """Calculate diffraction pattern for a single phase"""
        phase = self.phases[phase_idx]
        theo_peaks = phase['theoretical_peaks']
        
        if len(theo_peaks.get('two_theta', [])) == 0:
            return np.zeros_like(self.experimental_data['two_theta'])

        positions, intensities = self._filter_peaks(theo_peaks)
        if len(positions) == 0:
            return np.zeros_like(self.experimental_data['two_theta'])
            
        shifted_positions = self._shift_positions(positions, parameters)
        peak_widths = self._calculate_peak_widths(shifted_positions, parameters)
        scale_factor = parameters.get('scale_factor', 1.0)
        eta = parameters.get('eta_param', 0.5)
        intensity_multipliers = parameters.get('peak_intensity_multipliers')
        is_pawley = intensity_multipliers is not None
        use_scaled = self._uses_fixed_intensities(parameters)

        if use_scaled and not is_pawley:
            # Calculated intensities stay tied to the reference pattern, so scale
            # and the correction terms are determinable here
            effective = (
                intensities * scale_factor
                * self._intensity_corrections(shifted_positions, parameters, intensities)
            )
            return self._accumulate_pseudo_voigt(
                shifted_positions, peak_widths, effective, eta
            )

        if not is_pawley:
            effective = self._partitioned_intensities(
                phase, shifted_positions, peak_widths, eta
            )
        else:
            # Need full peak list for Pawley multipliers — use unfiltered theo peaks
            all_pos = self._shift_positions(theo_peaks['two_theta'], parameters)
            all_int = np.asarray(theo_peaks['intensity'], dtype=float)
            all_widths = self._calculate_peak_widths(all_pos, parameters)
            effective = (
                all_int * scale_factor * intensity_multipliers
                * self._intensity_corrections(all_pos, parameters, all_int)
            )
            return self._accumulate_pseudo_voigt(all_pos, all_widths, effective, eta)

        return self._accumulate_pseudo_voigt(shifted_positions, peak_widths, effective, eta)

    def _partitioned_intensities(self, phase: Dict, positions: np.ndarray,
                                 widths: np.ndarray, eta: float) -> np.ndarray:
        """
        This phase's Le Bail intensities, re-partitioned unless they are frozen.

        Le Bail alternates two steps: partition the observed intensity among the
        reflections at the current profile, then fit the profile and lattice
        against those intensities held fixed. Partitioning inside the objective
        instead would run it once per finite-difference probe of the gradient,
        hundreds of times per optimizer step, and would also let the intensities
        chase the parameters so that a worse profile can still reproduce the
        pattern, which flattens the very gradient being measured.
        """
        cached = phase.get('_extracted_intensities')
        if self._freeze_extracted and cached is not None and len(cached) == len(positions):
            return cached
        extracted = self._extract_lebail_intensities(positions, widths, eta)
        phase['_extracted_intensities'] = extracted
        return extracted

    def _refresh_extracted(self, phase_idx: Optional[int] = None):
        """Re-partition intensities at the current parameters"""
        targets = range(len(self.phases)) if phase_idx is None else [phase_idx]
        frozen, self._freeze_extracted = self._freeze_extracted, False
        try:
            for index in targets:
                self.phases[index].pop('_extracted_intensities', None)
                self._calculate_phase_pattern(index, self.phases[index]['parameters'])
        finally:
            self._freeze_extracted = frozen

    def _peak_windows(self, positions: np.ndarray, widths: np.ndarray,
                      eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unit-height pseudo-Voigt profiles for every reflection, in one array.

        A peak only reaches a few tenths of a degree, so each is evaluated on a
        window of grid points rather than the whole pattern. Giving every peak
        the same window length in points lets all of them live in one
        (reflections, window) array and be evaluated in a single pass, instead
        of a Python loop that repeats the same handful of array operations
        hundreds of times. The profile is rebuilt on every objective evaluation
        during refinement, so this is the inner loop of the whole engine.

        Returns the grid indices each window covers, the profiles, and the
        number of grid points under each unit-height peak.
        """
        n = self._n
        positions = np.asarray(positions, dtype=float)
        if n == 0 or positions.size == 0:
            return (np.zeros((0, 0), dtype=np.intp), np.zeros((0, 0)), np.zeros(0))

        x = self.experimental_data['two_theta']
        widths = np.maximum(np.asarray(widths, dtype=float), 1e-6)
        cutoff = 5.0 * widths

        half = int(np.ceil(float(np.max(cutoff)) / self._dx)) + 1
        # Broad peaks on a fine grid would otherwise make the array enormous
        budget = max(3, self._max_window_elements // (2 * positions.size))
        half = int(min(half, budget, n))

        centers = np.clip(np.searchsorted(x, positions), 0, n - 1).astype(np.intp)
        indices = centers[:, None] + np.arange(-half, half + 1)[None, :]
        # Windows near either end of the pattern run off the grid; those points
        # are clamped to stay valid indices and then masked out of the profile
        on_grid = (indices >= 0) & (indices < n)
        np.clip(indices, 0, n - 1, out=indices)

        offset = x[indices] - positions[:, None]
        sigma = widths[:, None] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = widths[:, None] / 2.0
        profiles = (
            (1.0 - eta) * np.exp(-0.5 * (offset / sigma) ** 2)
            + eta / (1.0 + (offset / gamma) ** 2)
        )
        profiles *= on_grid & (np.abs(offset) <= cutoff[:, None])

        return indices, profiles, profiles.sum(axis=1)

    def _accumulate_pseudo_voigt(self, positions: np.ndarray, widths: np.ndarray,
                                  intensities: np.ndarray, eta: float) -> np.ndarray:
        """Windowed accumulation of pseudo-Voigt peaks into a dense pattern"""
        if self._n == 0 or len(positions) == 0:
            return np.zeros(self._n)

        indices, profiles, _ = self._peak_windows(positions, widths, eta)
        heights = np.maximum(np.asarray(intensities, dtype=float), 0.0)
        contributions = profiles * heights[:, None]
        return np.bincount(
            indices.ravel(), weights=contributions.ravel(), minlength=self._n
        )
    
    def _extract_lebail_intensities(self, positions: np.ndarray, widths: np.ndarray, 
                                     eta: float) -> np.ndarray:
        """
        Le Bail intensity extraction using windowed partitioning.
        Avoids allocating full-length profile arrays per peak.

        Partitioning works with area-normalized profiles, but the caller builds
        the pattern from unit-height profiles, so the integrated intensities are
        converted back to heights before returning. Skipping that conversion
        overpredicts the pattern by the number of points under a peak.

        Each phase is partitioned against its own reflections only, so where two
        phases overlap both claim the same observed intensity. That is why this
        mode cannot be used for quantification.
        """
        observed = self.experimental_data['intensity']
        n_peaks = len(positions)
        n_pts = self._n

        if n_peaks == 0 or n_pts == 0:
            return np.zeros(n_peaks)

        indices, profiles, areas = self._peak_windows(positions, widths, eta)
        areas = np.maximum(areas, 1e-12)
        unit_area = profiles / areas[:, None]
        observed_window = observed[indices]

        # Starting values only need the right relative sizes: one partitioning
        # step is invariant to an overall rescaling of them
        centers = np.clip(
            np.searchsorted(self.experimental_data['two_theta'],
                            np.asarray(positions, dtype=float)),
            0, n_pts - 1,
        )
        extracted = np.maximum(observed[centers], 0.0) * areas

        for _ in range(max(1, int(self.extract_iterations))):
            contribution = extracted[:, None] * unit_area
            total = np.bincount(
                indices.ravel(), weights=contribution.ravel(), minlength=n_pts
            )
            overlap = total[indices]
            share = np.divide(
                contribution, overlap,
                out=np.zeros_like(contribution), where=overlap > 0,
            )
            extracted = np.sum(observed_window * share, axis=1)

        return extracted / areas
        
    def _calculate_peak_widths(self, two_theta: np.ndarray, parameters: Dict) -> np.ndarray:
        """Calculate peak widths using Caglioti function: FWHM² = U*tan²θ + V*tanθ + W"""
        U = parameters.get('u_param', 0.01)
        V = parameters.get('v_param', -0.001)
        W = parameters.get('w_param', 0.01)
        
        theta_rad = np.radians(two_theta / 2)
        tan_theta = np.tan(theta_rad)
        
        fwhm_squared = U * tan_theta**2 + V * tan_theta + W
        fwhm_squared = np.maximum(fwhm_squared, 0.001)
        return np.sqrt(fwhm_squared)
        
    def _pseudo_voigt_profile(self, x: np.ndarray, center: float, fwhm: float, 
                            intensity: float, eta: float) -> np.ndarray:
        """Generate pseudo-Voigt peak profile (windowed)"""
        if fwhm <= 0 or intensity <= 0:
            return np.zeros_like(x)
        
        cutoff = 5 * fwhm
        mask = np.abs(x - center) <= cutoff
        
        if not np.any(mask):
            return np.zeros_like(x)
        
        profile = np.zeros_like(x)
        x_local = x[mask]
        
        sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-0.5 * ((x_local - center) / sigma_g) ** 2)
        gamma_l = fwhm / 2
        lorentzian = 1 / (1 + ((x_local - center) / gamma_l) ** 2)
        profile[mask] = intensity * ((1 - eta) * gaussian + eta * lorentzian)
        
        return profile
        
    def refined_positions(self, phase_idx: int) -> np.ndarray:
        """Peak positions a phase currently predicts, corrections included."""
        phase = self.phases[phase_idx]
        return self._shift_positions(
            phase['theoretical_peaks'].get('two_theta', []), phase['parameters']
        )
        
    def _calculate_total_pattern(self) -> np.ndarray:
        """Calculate total calculated pattern from all phases"""
        total_pattern = np.zeros_like(self.experimental_data['two_theta'])
        
        for phase_idx in range(len(self.phases)):
            phase_pattern = self._calculate_phase_pattern(phase_idx, self.phases[phase_idx]['parameters'])
            total_pattern += phase_pattern
            
        return total_pattern
        
    def _calculate_phase_contributions(self) -> List[Dict]:
        """Calculate per-phase contributions and R-factors"""
        contributions = []
        obs = self.experimental_data['intensity']
        errors = self.experimental_data['errors']
        total_pattern = self._calculate_total_pattern()
        
        for phase_idx in range(len(self.phases)):
            phase_pattern = self._calculate_phase_pattern(phase_idx, self.phases[phase_idx]['parameters'])
            
            total_intensity = np.sum(total_pattern)
            phase_intensity = np.sum(phase_pattern)
            contribution_percent = (phase_intensity / total_intensity * 100) if total_intensity > 0 else 0
            
            residual = (obs - phase_pattern) / errors
            rwp_num = np.sum(residual ** 2)
            rwp_den = np.sum((obs / errors) ** 2)
            phase_rwp = np.sqrt(rwp_num / rwp_den) * 100 if rwp_den > 0 else float('inf')
            
            contributions.append({
                'phase_idx': phase_idx,
                'contribution_percent': contribution_percent,
                'rwp': phase_rwp,
                'scale_factor': self.phases[phase_idx]['parameters']['scale_factor'],
                'integrated_intensity': float(phase_intensity)
            })
        
        return contributions
    
    def _calculate_r_factors(self, calculated_pattern: np.ndarray) -> Dict[str, float]:
        """Calculate crystallographic R-factors"""
        obs = self.experimental_data['intensity']
        calc = calculated_pattern
        errors = self.experimental_data['errors']
        
        rp = np.sum(np.abs(obs - calc)) / np.sum(obs) if np.sum(obs) > 0 else float('inf')
        
        rwp_num = np.sum(((obs - calc) / errors) ** 2)
        rwp_den = np.sum((obs / errors) ** 2)
        rwp = np.sqrt(rwp_num / rwp_den) if rwp_den > 0 else float('inf')
        
        n_obs = len(obs)
        n_param = sum(len(self._create_parameter_vector(p['parameters'])[0]) for p in self.phases)
        r_exp = np.sqrt((n_obs - n_param) / rwp_den) if rwp_den > 0 and n_obs > n_param else float('inf')
        
        gof = rwp / r_exp if r_exp > 0 and not np.isinf(r_exp) else float('inf')
        chi_squared = rwp_num / (n_obs - n_param) if n_obs > n_param else float('inf')
        
        return {
            'Rp': rp * 100,
            'Rwp': rwp * 100,
            'Rexp': r_exp * 100,
            'GoF': gof,
            'chi_squared': chi_squared
        }
        
    def get_refined_phases_for_search(self) -> List[Dict]:
        """
        Get refined phase data optimized for ultra-fast pattern searching
        
        Returns:
            List of refined phase data with optimized parameters
        """
        refined_phases = []
        
        for phase in self.phases:
            # Create refined phase data
            refined_phase = {
                'phase': phase['data']['phase'].copy(),
                'theoretical_peaks': phase['theoretical_peaks'].copy(),
                'refinement_quality': {
                    'r_factors': self.r_factors,
                    'scale_factor': phase['parameters']['scale_factor'],
                    'profile_params': {
                        'u': phase['parameters']['u_param'],
                        'v': phase['parameters']['v_param'], 
                        'w': phase['parameters']['w_param'],
                        'eta': phase['parameters']['eta_param']
                    },
                    'zero_shift': self.global_parameters.get('zero_shift', 0.0),
                    'displacement': self.global_parameters.get('displacement', 0.0),
                    'lattice_scale': phase['parameters'].get('lattice_scale', 1.0),
                    'absorption': phase['parameters'].get('absorption', 0.0),
                    'harmonic_coeffs': list(phase['parameters'].get('harmonic_coeffs') or []),
                    'refined_unit_cell': phase['parameters']['unit_cell']
                },
                'search_priority': self._calculate_search_priority(phase)
            }
            
            refined_phases.append(refined_phase)
            
        # Sort by search priority (best fits first)
        refined_phases.sort(key=lambda x: x['search_priority'], reverse=True)
        
        return refined_phases
        
    def _calculate_search_priority(self, phase: Dict) -> float:
        """Calculate search priority based on refinement quality"""
        params = phase['parameters']
        
        # Base priority on scale factor and R-factors
        scale_factor = params['scale_factor']
        rwp = self.r_factors.get('Rwp', 100.0)
        gof = self.r_factors.get('GoF', 10.0)
        
        # Higher scale factor and lower R-factors = higher priority
        priority = scale_factor * (100.0 / max(rwp, 1.0)) * (1.0 / max(gof, 1.0))
        
        return priority
        
    def phase_summary(self) -> List[Dict]:
        """
        Per-phase refined values plus RIR weight percents, for display and export.

        Weight percents follow Chung: w_i = (I_i/RIR_i) / sum_j (I_j/RIR_j), with
        I_i the strongest-line intensity implied by the refined scale. They are
        only reported for the fixed-intensity model, because Le Bail extraction
        absorbs the scale factor and leaves nothing to quantify with.
        """
        quantitative = self.intensity_model == 'fixed'
        contributions = self._calculate_phase_contributions() if self.phases else []

        rows = []
        for index, phase in enumerate(self.phases):
            params = phase['parameters']
            info = phase['data'].get('phase', {}) if isinstance(phase.get('data'), dict) else {}
            theo_intensity = np.asarray(
                phase['theoretical_peaks'].get('intensity', []), dtype=float
            )
            reference_max = float(np.max(theo_intensity)) if len(theo_intensity) else 0.0

            rir = info.get('rir')
            try:
                rir = float(rir) if rir is not None else None
            except (TypeError, ValueError):
                rir = None
            if rir is not None and not (np.isfinite(rir) and rir > 0):
                rir = None

            scale = float(params.get('scale_factor', 1.0))
            rows.append({
                'name': info.get('mineral', info.get('mineral_name', f'Phase {index + 1}')),
                'formula': info.get('formula', info.get('chemical_formula', '')),
                'scale': scale,
                'line_intensity': scale * reference_max,
                'rir': rir,
                'lattice_scale': float(params.get('lattice_scale', 1.0)),
                'unit_cell': dict(params.get('unit_cell') or {}),
                'base_unit_cell': dict(params.get('_base_unit_cell') or {}),
                'absorption': float(params.get('absorption', 0.0) or 0.0),
                'harmonic_coeffs': list(params.get('harmonic_coeffs') or []),
                'profile': {
                    'u': float(params.get('u_param', 0.0)),
                    'v': float(params.get('v_param', 0.0)),
                    'w': float(params.get('w_param', 0.0)),
                    'eta': float(params.get('eta_param', 0.5)),
                },
                'contribution_percent': (
                    contributions[index]['contribution_percent'] if index < len(contributions) else None
                ),
                'integrated_intensity': (
                    contributions[index]['integrated_intensity'] if index < len(contributions) else None
                ),
            })

        terms = [
            row['line_intensity'] / row['rir']
            for row in rows
            if quantitative and row['rir'] and row['line_intensity'] > 0
        ]
        total = float(sum(terms))
        for row in rows:
            if quantitative and row['rir'] and row['line_intensity'] > 0 and total > 0:
                row['weight_percent'] = 100.0 * (row['line_intensity'] / row['rir']) / total
            else:
                row['weight_percent'] = None
        return rows

    def generate_refinement_report(self) -> str:
        """Generate detailed refinement report"""
        if not self.refinement_history:
            return "No refinement performed"
            
        report = []
        report.append("=== Le Bail Refinement Report ===\n")
        
        final_iteration = self.refinement_history[-1]
        
        report.append(f"Refinement completed after {len(self.refinement_history)} iterations")
        report.append(f"Final R-factors:")
        report.append(f"  Rp  = {final_iteration['r_factors']['Rp']:.3f}%")
        report.append(f"  Rwp = {final_iteration['r_factors']['Rwp']:.3f}%")
        report.append(f"  Rexp= {final_iteration['r_factors']['Rexp']:.3f}%")
        report.append(f"  GoF = {final_iteration['r_factors']['GoF']:.3f}")
        report.append("")

        report.append("Global parameters:")
        report.append(f"  Zero shift    = {self.global_parameters.get('zero_shift', 0.0):+.4f}°")
        report.append(f"  Displacement  = {self.global_parameters.get('displacement', 0.0):+.4f}°")
        report.append(
            "  Intensity model = "
            + ("reference intensities, scale refined" if self.intensity_model == 'fixed'
               else "Le Bail extraction")
        )
        report.append("")

        # Phase details
        for i, phase in enumerate(self.phases):
            phase_name = phase['data']['phase'].get('mineral', f'Phase_{i+1}')
            params = phase['parameters']
            
            report.append(f"Phase {i+1}: {phase_name}")
            report.append(f"  Scale factor: {params['scale_factor']:.4f}")
            report.append(f"  Profile parameters:")
            report.append(f"    U = {params['u_param']:.6f}")
            report.append(f"    V = {params['v_param']:.6f}")
            report.append(f"    W = {params['w_param']:.6f}")
            report.append(f"    η = {params['eta_param']:.3f}")

            if params.get('absorption'):
                report.append(f"  Absorption: {params['absorption']:+.4f}")
            coeffs = params.get('harmonic_coeffs') or []
            if any(coeffs):
                orders = ", ".join(
                    f"c{2 * (k + 1)}={c:+.3f}" for k, c in enumerate(coeffs)
                )
                report.append(f"  Spherical harmonics: {orders}")
            
            if params.get('refine_cell', True):
                cell = params['unit_cell']
                base = params.get('_base_unit_cell') or cell
                scale = params.get('lattice_scale', 1.0)
                report.append(f"  Lattice scale: {scale:.6f} ({(scale - 1.0) * 100:+.3f}%)")
                report.append(f"  Unit cell:")
                report.append(f"    a = {cell['a']:.4f} Å  (start {base.get('a', 0.0):.4f})")
                report.append(f"    b = {cell['b']:.4f} Å  (start {base.get('b', 0.0):.4f})")
                report.append(f"    c = {cell['c']:.4f} Å  (start {base.get('c', 0.0):.4f})")
                report.append(f"    α = {cell['alpha']:.3f}°")
                report.append(f"    β = {cell['beta']:.3f}°")
                report.append(f"    γ = {cell['gamma']:.3f}°")
                if cell.get('volume'):
                    report.append(f"    V = {cell['volume']:.3f} Å³")
            
            # Show space group if available
            space_group = phase['data']['phase'].get('space_group', 'Unknown')
            if space_group and space_group != 'Unknown':
                report.append(f"  Space group: {space_group}")
            report.append("")
            
        # Quality assessment
        rwp = final_iteration['r_factors']['Rwp']
        if rwp < 5.0:
            quality = "Excellent"
        elif rwp < 10.0:
            quality = "Very Good"
        elif rwp < 15.0:
            quality = "Good"
        elif rwp < 25.0:
            quality = "Acceptable"
        else:
            quality = "Poor"
            
        report.append(f"Refinement Quality: {quality}")
        
        return "\n".join(report)
