"""
Le Bail refinement engine for XRD phase matching
Implements proper crystallographic refinement with profile functions and unit cell optimization
"""

import numpy as np
import itertools

from scipy.optimize import least_squares, lsq_linear, nnls
from scipy.sparse import coo_matrix, diags
from scipy.special import eval_legendre
from typing import Dict, List, Tuple, Optional
import copy
import warnings

from utils.profile_functions import (
    MAX_ASYMMETRY, asymmetry_exponent, flank_widths, phase_widths,
    skew_description,
)

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

        # Optional hooks for a caller watching the refinement run. progress is
        # handed a snapshot after every cycle, log every message, and cancel is
        # consulted between cycles and between phases -- the points at which the
        # refinement is in a consistent state and can be stopped without leaving
        # a phase half-fitted.
        self.progress_callback = None
        self.log_callback = None
        self.cancel_check = None
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
        #
        # The profile terms are here, not on the phases, because peak width is
        # part instrument and part sample and nothing in a single pattern says
        # where the line falls. Held fixed, they give the sample terms a stable
        # baseline to work against; refined, they are three strongly correlated
        # coefficients of an unbounded polynomial that can be driven negative,
        # which is the usual cause of a profile refinement blowing up. So they
        # default to fixed, and refining them is opt-in.
        self.global_parameters = {
            'zero_shift': 0.0,      # constant 2-theta offset (degrees)
            'displacement': 0.0,    # specimen displacement: d(2-theta) = disp * cos(theta)
            'u_param': 0.002,       # Caglioti Gaussian, instrument resolution curve
            'v_param': -0.0005,
            'w_param': 0.004,
            'x_param': 0.0,         # instrument Lorentzian; normally zero
            'y_param': 0.0,
            'axial_asymmetry': 0.0,  # axial divergence skew, as cot(2-theta)
            'refine_zero_shift': True,
            'refine_displacement': False,
            'refine_instrument_profile': False,
            'refine_axial_asymmetry': False,
        }
        self.wavelength = 1.5406

        # Restrict the residual to the neighbourhood of the modelled reflections,
        # so the optimizer is not pulled by counting noise in the empty stretches
        # between them. Off by default: it changes the fit, not just the report.
        self.fit_peak_regions_only = False
        self._fit_mask = None

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
                            two_theta_range: Optional[Tuple[float, float]] = None,
                            wavelength: Optional[float] = None):
        """Set experimental diffraction data
        
        Args:
            two_theta: 2-theta values in degrees
            intensity: Intensity values
                      IMPORTANT: Should be background-subtracted intensity
                      Background subtraction must be performed before Le Bail refinement
                      to avoid fitting the background as part of the diffraction pattern
            errors: Optional error values (defaults to sqrt(intensity))
            two_theta_range: Optional (min, max) 2-theta range to limit refinement
            wavelength: Radiation wavelength in angstroms; used by the Scherrer
                        size term. Defaults to Cu K-alpha when omitted.
        """
        if wavelength is not None and wavelength > 0:
            self.wavelength = float(wavelength)

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
            # Sample broadening, the part of the width that belongs to this
            # phase. Both are strictly positive and carry distinct angular
            # signatures -- size goes as 1/cos(theta), strain as tan(theta) --
            # which is what lets them be told apart, and what makes them far
            # better behaved under refinement than the instrument polynomial.
            'crystallite_size': 1.0,   # micrometres; ~1 um broadens negligibly
            'microstrain': 1000.0,     # delta-d/d x 1e6
            'refine_size': False,      # size matters mainly for nanomaterials
            'refine_strain': True,     # microstrain is the usual dominant term
            # Skew belonging to this phase alone, as log(low flank / high flank).
            # Layer silicates with stacking disorder need it; most phases do not.
            'asymmetry': 0.0,
            'refine_asymmetry': False,
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

        # Callers that predate the instrument/sample split still hand the
        # Caglioti terms to the phase. They describe the instrument, so route
        # them there rather than silently ignoring the width the caller asked
        # for. The last phase to supply them wins, which is harmless because
        # every caller passes the same values to every phase.
        for legacy, target in (('u_param', 'u_param'), ('v_param', 'v_param'),
                               ('w_param', 'w_param')):
            if legacy in default_params:
                self.global_parameters[target] = float(default_params.pop(legacy))
        default_params.pop('eta_param', None)  # now derived, not fitted

        # Pawley intensities are solved, not searched, so they need no starting
        # vector here; the cache is filled on the first alternation step.
        default_params.pop('peak_intensity_multipliers', None)

        default_params['_locked'] = frozenset(default_params.get('_locked') or ())

        order = int(default_params.get('harmonic_order', 0) or 0)
        n_harmonics = max(0, order // 2)
        coeffs = list(default_params.get('harmonic_coeffs') or [])
        if len(coeffs) != n_harmonics:
            coeffs = (coeffs + [0.0] * n_harmonics)[:n_harmonics]
        default_params['harmonic_coeffs'] = coeffs
        # Lattice dilation is refined against the starting cell, so keep a copy
        default_params['_base_unit_cell'] = dict(default_params['unit_cell'])

        # A dilation carried over from an earlier run has to be applied to the
        # reported cell here, because the cell is otherwise only recomputed when
        # lattice_scale is among the parameters actually being refined.
        carried_scale = float(default_params.get('lattice_scale', 1.0) or 1.0)
        if abs(carried_scale - 1.0) > 1e-12:
            default_params['unit_cell'] = self._scaled_unit_cell(
                default_params['_base_unit_cell'], carried_scale
            )


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
        if self.log_callback is not None:
            try:
                self.log_callback(message)
            except Exception:
                pass  # a watcher must never be able to stop the refinement
        if not self.quiet:
            print(message)

    _skew_direction = staticmethod(skew_description)

    def _phase_name(self, phase_idx: int) -> str:
        try:
            info = self.phases[phase_idx]['data']['phase']
            return info.get('mineral') or info.get('mineral_name') or f'Phase {phase_idx + 1}'
        except (IndexError, KeyError, TypeError, AttributeError):
            return f'Phase {phase_idx + 1}'

    def _cancelled(self) -> bool:
        """True when the caller has asked the refinement to stop."""
        if self.cancel_check is None:
            return False
        try:
            return bool(self.cancel_check())
        except Exception:
            return False

    def _report_progress(self, **payload):
        """Hand a caller a snapshot of where the refinement has got to."""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception as e:
            self._log(f"Warning: progress callback failed: {e}")

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
        
        total_pawley_params = 0
        for phase in self.phases:
            if phase['parameters'].get('refine_intensities', False):
                n_peaks = len(phase['theoretical_peaks'].get('two_theta', []))
                total_pawley_params += n_peaks

        if total_pawley_params > 0:
            self._log(f"Pawley mode enabled: solving {total_pawley_params} peak intensities by NNLS")
        
        if staged_refinement and self.mode == 'polish':
            self._log("Using staged refinement: unit cell → profile parameters")
        
        # The fitted region is fixed here, once, from the starting positions. If
        # it were recomputed from the current model the optimizer could improve
        # the residual by sliding peaks until the awkward points fell outside it.
        self._fit_mask = None
        if self.fit_peak_regions_only:
            mask = self._peak_region_mask(reach=4.0)
            fraction = float(np.mean(mask))
            if mask.any():
                self._fit_mask = mask
                self._log(
                    f"Fitting the {fraction * 100:.1f}% of points near modelled "
                    "peaks; the empty stretches are excluded"
                )

        # Initialize refinement
        self.refinement_history = []
        previous_rwp = float('inf')
        rwp_change = float('inf')
        cancelled = False

        # Cycles a watcher can expect, so it can show a meaningful progress bar
        staged = staged_refinement and self.mode == 'polish'
        stage1_iterations = max(3, max_iterations // 3) if staged else 0
        total_cycles = (
            stage1_iterations + (max_iterations - stage1_iterations)
            if staged else max_iterations
        )
        self._report_progress(
            phase_of_work='start', stage=0, iteration=0,
            total_iterations=total_cycles, message='Aligning pattern…',
        )

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
            
            for iteration in range(stage1_iterations):
                if self._cancelled():
                    cancelled = True
                    break
                self._log(f"\nStage 1 - Iteration {iteration + 1}/{stage1_iterations}")
                
                self._refine_global_parameters()
                for phase_idx, phase in enumerate(self.phases):
                    if self._cancelled():
                        cancelled = True
                        break
                    self._report_progress(
                        phase_of_work='phase', stage=1, iteration=iteration + 1,
                        total_iterations=total_cycles,
                        message=f"Cell — {self._phase_name(phase_idx)}",
                    )
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
                self._report_progress(
                    phase_of_work='cycle', stage=1, iteration=iteration + 1,
                    total_iterations=total_cycles, r_factors=r_factors.copy(),
                    calculated_pattern=calculated_pattern.copy(),
                    message="Stage 1 — unit cell & zero shift",
                )
                if cancelled:
                    break

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
            if cancelled or self._cancelled():
                cancelled = True
                break
            actual_iteration = start_iteration + iteration + 1
            stage_label = "Stage 2 - " if (staged_refinement and self.mode == 'polish') else ""
            self._log(f"\n=== {stage_label}Le Bail Iteration {actual_iteration} ===")
            
            self._refine_global_parameters()
            for phase_idx, phase in enumerate(self.phases):
                if self._cancelled():
                    cancelled = True
                    break
                phase_name = phase['data']['phase'].get('mineral', f'Phase_{phase_idx}')
                self._log(f"Refining {phase_name}...")
                self._report_progress(
                    phase_of_work='phase', stage=2 if staged else 0,
                    iteration=actual_iteration, total_iterations=total_cycles,
                    message=f"Profile — {phase_name}",
                )
                self._refine_single_phase(phase_idx)
                
            calculated_pattern = self._calculate_total_pattern()
            r_factors = self._calculate_r_factors(calculated_pattern)
            phase_contributions = self._calculate_phase_contributions()
            
            self._log(f"R-factors: Rp={r_factors['Rp']:.3f}, Rwp={r_factors['Rwp']:.3f}, "
                      f"Rwp(peaks)={r_factors['Rwp_peak']:.3f}, "
                      f"GoF={r_factors['GoF']:.3f}, "
                      f"DW={r_factors['durbin_watson']:.2f}")
            
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

            self._report_progress(
                phase_of_work='cycle', stage=2 if staged else 0,
                iteration=actual_iteration, total_iterations=total_cycles,
                r_factors=r_factors.copy(),
                calculated_pattern=calculated_pattern.copy(),
                message="Stage 2 — profile & intensities" if staged else "Refining",
            )

            rwp_change = abs(previous_rwp - r_factors['Rwp'])
            if rwp_change < convergence_threshold:
                self._log(f"Converged after {iteration + 1} iterations (ΔRwp = {rwp_change:.6f})")
                break
                
            previous_rwp = r_factors['Rwp']

        if cancelled:
            self._log("Refinement stopped at the user's request")
        if not self.refinement_history:
            # Cancelled before the first cycle finished; report the state the
            # pre-alignment left behind rather than failing outright
            calculated_pattern = self._calculate_total_pattern()
            self.refinement_history.append({
                'iteration': 0,
                'stage': 0,
                'r_factors': self._calculate_r_factors(calculated_pattern),
                'parameters': copy.deepcopy([p['parameters'] for p in self.phases]),
                'calculated_pattern': calculated_pattern.copy(),
            })

        final_results = {
            'cancelled': cancelled,
            'converged': (not cancelled) and rwp_change < convergence_threshold,
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
            'fit_peak_regions_only': self._fit_mask is not None,
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
        # A pinned scale is a fixed quantity of a phase, often the whole point of
        # the run -- an internal standard weighed into the mount. Solving for it
        # here would discard that before the refinement had begun, so those
        # phases are held at their value and the rest are solved around them.
        pinned = [
            'scale_factor' in self._locked_parameters(p['parameters'])
            for p in self.phases
        ]
        columns = []
        for index, phase in enumerate(self.phases):
            params = dict(phase['parameters'])
            if not pinned[index]:
                params['scale_factor'] = 1.0
            columns.append(self._calculate_phase_pattern(index, params))
        design = np.column_stack(columns)
        if not np.any(design > 0):
            return

        target = self.experimental_data['intensity']
        if any(pinned):
            target = target - np.sum(
                [column for column, fixed in zip(columns, pinned) if fixed],
                axis=0, initial=0.0,
            )
            free = [i for i, fixed in enumerate(pinned) if not fixed]
            if not free:
                return
            design = design[:, free]
        try:
            scales, _ = nnls(design, np.maximum(target, 0.0))
        except Exception as e:
            self._log(f"Scale initialization failed: {e}")
            return

        targets = [i for i, fixed in enumerate(pinned) if not fixed] if any(pinned) \
            else list(range(len(self.phases)))
        for index, scale in zip(targets, scales):
            if scale > 0:
                self.phases[index]['parameters']['scale_factor'] = float(scale)
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
        # The scan below is a product grid, so it stays restricted to the two
        # near-degenerate shift terms. Asymmetry changes the peak shape rather
        # than its position and is not degenerate with either, so the local
        # optimizer finds it without help and a third scan axis would multiply
        # the cost of every pass by eleven for nothing.
        n_scanned = len(names)
        if self.global_parameters.get('refine_axial_asymmetry', False):
            names.append('axial_asymmetry')
            vector.append(float(self.global_parameters.get('axial_asymmetry', 0.0)))
            bounds.append((-0.2, 0.2))
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
            return self._fitted_residual(
                (observed - self._calculate_total_pattern()) / errors
            )

        def chi2(x):
            return float(np.sum(residuals(x) ** 2))

        # The scan below evaluates the whole pattern hundreds of times, so the
        # partitioned intensities are fixed for the duration, as in a Le Bail step
        self._refresh_extracted()
        self._freeze_extracted = True
        try:
            scan_pass = getattr(self, '_global_scan_pass', 0)
            if n_scanned and scan_pass < len(self._GLOBAL_SCAN_SCHEDULE):
                half_width = self._GLOBAL_SCAN_SCHEDULE[scan_pass]
                # Eleven samples per axis is enough to find the valley; 21^n was
                # spending most of the refinement budget on the scan alone
                grids = [
                    np.clip(
                        np.linspace(center - half_width, center + half_width, 11),
                        low, high,
                    )
                    for center, (low, high) in zip(vector[:n_scanned], bounds[:n_scanned])
                ]
                best_value = chi2(vector)
                best_point = list(vector)
                for point in itertools.product(*grids):
                    trial = list(point) + list(vector[n_scanned:])
                    value = chi2(trial)
                    if value < best_value:
                        best_value, best_point = value, trial
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
            message = (
                "  Global: zero shift="
                f"{self.global_parameters['zero_shift']:+.4f}°, "
                f"displacement={self.global_parameters['displacement']:+.4f}°"
            )
            if 'axial_asymmetry' in names:
                axial = self.global_parameters['axial_asymmetry']
                message += f", axial asymmetry={axial:+.4f}"
            self._log(message)
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

            if 'crystallite_size' in optimized_params or 'microstrain' in optimized_params:
                size = optimized_params.get('crystallite_size',
                                            params.get('crystallite_size', 0.0))
                strain = optimized_params.get('microstrain',
                                              params.get('microstrain', 0.0))
                self._log(f"  Sample broadening refined: size={size:.4g} um, "
                          f"microstrain={strain:.4g}")

            if 'lattice_scale' in optimized_params:
                cell = optimized_params.get('unit_cell', params.get('unit_cell', {}))
                self._log(
                    f"  Lattice scale: {optimized_params['lattice_scale']:.6f} "
                    f"(a={cell.get('a', 0.0):.4f}, b={cell.get('b', 0.0):.4f}, "
                    f"c={cell.get('c', 0.0):.4f} Å)"
                )

            if 'asymmetry' in optimized_params:
                self._log(f"  Phase asymmetry: {optimized_params['asymmetry']:+.4f} "
                          f"({self._skew_direction(optimized_params['asymmetry'])})")

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
            return self._fitted_residual((observed - total) / errors)

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
        # reference pattern; Le Bail extraction and the Pawley solve both absorb
        # it entirely, leaving it a flat direction if it were refined anyway
        if params.get('refine_scale', True) and use_scaled and not is_pawley:
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
        elif params.get('refine_scale', True):
            absorbed = "the Pawley solve" if is_pawley else "Le Bail intensity extraction"
            self._log(f"  Scale factor: not refinable, absorbed by {absorbed}")
            
        # Sample broadening. Bounds follow the range a diffractometer can
        # actually resolve: below 0.01 um the peaks are too broad to refine
        # against, above 10 um the broadening is imperceptible.
        if params.get('refine_profile', True):
            if params.get('refine_size', False):
                param_vector.append(float(params.get('crystallite_size', 1.0)))
                param_bounds.append((0.01, 10.0))
                param_names.append('crystallite_size')
            if params.get('refine_strain', True):
                param_vector.append(float(params.get('microstrain', 1000.0)))
                param_bounds.append((0.0, 50000.0))
                param_names.append('microstrain')
            if 'crystallite_size' in param_names or 'microstrain' in param_names:
                self._log(
                    f"  Sample broadening: size={params.get('crystallite_size', 1.0):.4g} um, "
                    f"microstrain={params.get('microstrain', 1000.0):.4g}"
                )
            if params.get('refine_asymmetry', False):
                param_vector.append(float(params.get('asymmetry', 0.0)))
                param_bounds.append((-MAX_ASYMMETRY, MAX_ASYMMETRY))
                param_names.append('asymmetry')


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

        # Pawley intensities are absent by design: they are linear in the pattern
        # and are solved by NNLS in the alternation step instead.

        # A pinned parameter is dropped here rather than at each flag above, so
        # that the staged refinement -- which switches whole groups of flags on
        # and off as it moves between stages -- cannot hand one back.
        locked = self._locked_parameters(params)
        if locked:
            keep = [i for i, name in enumerate(param_names)
                    if not self._is_locked(name, locked)]
            param_vector = [param_vector[i] for i in keep]
            param_bounds = [param_bounds[i] for i in keep]
            param_names = [param_names[i] for i in keep]

        return np.array(param_vector), param_bounds, param_names

    @staticmethod
    def _locked_parameters(params: Dict) -> frozenset:
        """Names the user has pinned, which nothing may refine or overwrite."""
        return frozenset(params.get('_locked') or ())

    @staticmethod
    def _is_locked(name: str, locked: frozenset) -> bool:
        if name in locked:
            return True
        # The harmonic terms are pinned as a group, being one correction
        return name.startswith('harmonic_') and 'harmonic_coeffs' in locked
        
    def _vector_to_parameters(self, vector: np.ndarray, names: List[str], 
                            original_params: Dict) -> Dict:
        """Convert parameter vector back to parameter dictionary"""
        params = {}
        harmonics = dict(enumerate(original_params.get('harmonic_coeffs') or []))
        harmonics_seen = False
        
        for i, name in enumerate(names):
            if name.startswith('cell_'):
                if 'unit_cell' not in params:
                    params['unit_cell'] = original_params['unit_cell'].copy()
                params['unit_cell'][name[5:]] = vector[i]
            elif name.startswith('harmonic_'):
                harmonics[int(name.split('_')[1])] = float(vector[i])
                harmonics_seen = True
            else:
                params[name] = vector[i]
        
        if harmonics_seen:
            params['harmonic_coeffs'] = [harmonics[k] for k in sorted(harmonics)]

        # A refined lattice dilation is only meaningful if it is also reported as
        # cell edges, so keep the stored cell in step with it
        if 'lattice_scale' in params:
            base = original_params.get('_base_unit_cell') or original_params.get('unit_cell')
            if base:
                params['unit_cell'] = self._scaled_unit_cell(
                    base, float(params['lattice_scale'])
                )

        return params

    def _scaled_unit_cell(self, base: Dict, scale: float) -> Dict:
        """The starting cell under an isotropic dilation; angles are unchanged."""
        cell = dict(base)
        for edge in ('a', 'b', 'c'):
            if cell.get(edge):
                cell[edge] = float(base[edge]) * float(scale)
        cell['volume'] = self._cell_volume(cell)
        return cell
        
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
        peak_widths, eta = self._calculate_peak_widths(shifted_positions, parameters)
        skew = self._peak_asymmetry(shifted_positions, parameters)
        scale_factor = parameters.get('scale_factor', 1.0)
        is_pawley = bool(parameters.get('refine_intensities', False))
        use_scaled = self._uses_fixed_intensities(parameters)

        if use_scaled and not is_pawley:
            # Calculated intensities stay tied to the reference pattern, so scale
            # and the correction terms are determinable here
            effective = (
                intensities * scale_factor
                * self._intensity_corrections(shifted_positions, parameters, intensities)
            )
            return self._accumulate_pseudo_voigt(
                shifted_positions, peak_widths, effective, eta, skew
            )

        if is_pawley:
            effective = self._pawley_intensities(
                phase_idx, shifted_positions, peak_widths, eta, skew
            )
        else:
            effective = self._partitioned_intensities(
                phase, shifted_positions, peak_widths, eta, skew
            )

        return self._accumulate_pseudo_voigt(
            shifted_positions, peak_widths, effective, eta, skew
        )

    def _pawley_intensities(self, phase_idx: int, positions: np.ndarray,
                            widths: np.ndarray, eta, asymmetry=0.0) -> np.ndarray:
        """
        This phase's Pawley intensities, re-solved unless they are frozen.

        Pawley shares Le Bail's alternation -- intensities at the current profile,
        then the profile against those intensities held fixed -- and differs only
        in how the intensities are obtained. Le Bail partitions the observed counts
        between overlapping reflections in the ratio the previous cycle gave them;
        Pawley solves for them outright.
        """
        phase = self.phases[phase_idx]
        cached = phase.get('_pawley_intensities')
        usable = cached is not None and len(cached) == len(positions)

        if self._freeze_extracted:
            # A stale or missing cache during a frozen pass means this phase is
            # only being evaluated as another phase's contribution; the reference
            # intensities stand in for it and stop the solve from re-entering.
            return cached if usable else self._reference_intensities(phase, positions)

        solved = self._solve_pawley_intensities(
            phase_idx, positions, widths, eta, asymmetry
        )
        phase['_pawley_intensities'] = solved
        return solved

    def _reference_intensities(self, phase: Dict, positions: np.ndarray) -> np.ndarray:
        """Scaled reference intensities, the starting point before any solve."""
        params = phase['parameters']
        _, intensities = self._filter_peaks(phase['theoretical_peaks'])
        if len(intensities) != len(positions):
            return np.ones(len(positions))
        return intensities * params.get('scale_factor', 1.0) * self._intensity_corrections(
            positions, params, intensities
        )

    def _solve_pawley_intensities(self, phase_idx: int, positions: np.ndarray,
                                  widths: np.ndarray, eta,
                                  asymmetry=0.0) -> np.ndarray:
        """
        Reflection intensities by non-negative least squares.

        Every reflection enters the calculated pattern linearly, so with the
        profile fixed the intensities are the solution of a bounded linear problem
        rather than something to hunt for with a general optimizer. Solving them
        directly costs one factorization; refining them nonlinearly costs one
        pattern rebuild per reflection per Jacobian, which is where Pawley
        refinement otherwise spends nearly all of its time.

        Non-negativity is the only constraint imposed. It matters because heavily
        overlapped reflections are individually underdetermined -- their sum is
        well constrained but the split between them is not -- and without a floor
        the solution runs off to a large positive intensity cancelled by a large
        negative one on its neighbour.
        """
        observed = self.experimental_data['intensity']
        errors = self.experimental_data.get('errors')
        if errors is None:
            errors = np.ones_like(observed)

        # Whatever the other phases contribute is not this phase's to fit. They
        # are evaluated frozen so that a second Pawley phase reuses its cache
        # instead of re-entering this solve.
        frozen, self._freeze_extracted = self._freeze_extracted, True
        try:
            others = np.zeros_like(observed)
            for index, other in enumerate(self.phases):
                if index != phase_idx:
                    others = others + self._calculate_phase_pattern(
                        index, other['parameters']
                    )
        finally:
            self._freeze_extracted = frozen

        target = observed - others

        indices, profiles, _ = self._peak_windows(positions, widths, eta, asymmetry)
        n_peaks = len(positions)
        n_points = len(observed)

        # Each column is one reflection's unit-height profile. The windows are
        # narrow, so the design is overwhelmingly sparse; coo_matrix sums the
        # duplicate entries that window clamping leaves at the pattern edges.
        rows = np.asarray(indices).ravel()
        cols = np.repeat(np.arange(n_peaks), np.asarray(indices).shape[1])
        design = coo_matrix(
            (np.asarray(profiles).ravel(), (rows, cols)), shape=(n_points, n_peaks)
        ).tocsr()

        weights = 1.0 / np.maximum(errors, 1e-9)
        design = diags(weights) @ design

        result = lsq_linear(
            design, target * weights, bounds=(0.0, np.inf),
            method='trf', tol=1e-8, max_iter=50, lsq_solver='lsmr', verbose=0,
        )
        return np.maximum(result.x, 0.0)

    def _partitioned_intensities(self, phase: Dict, positions: np.ndarray,
                                 widths: np.ndarray, eta,
                                 asymmetry=0.0) -> np.ndarray:
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
        extracted = self._extract_lebail_intensities(positions, widths, eta, asymmetry)
        phase['_extracted_intensities'] = extracted
        return extracted

    def _refresh_extracted(self, phase_idx: Optional[int] = None):
        """Re-partition (Le Bail) or re-solve (Pawley) intensities at the current parameters"""
        targets = range(len(self.phases)) if phase_idx is None else [phase_idx]
        frozen, self._freeze_extracted = self._freeze_extracted, False
        try:
            for index in targets:
                self.phases[index].pop('_extracted_intensities', None)
                self.phases[index].pop('_pawley_intensities', None)
                self._calculate_phase_pattern(index, self.phases[index]['parameters'])
        finally:
            self._freeze_extracted = frozen

    @staticmethod
    def _as_column(values, count: int):
        """Broadcast a scalar or per-reflection array against a window axis."""
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return float(array)
        if array.size == 1:
            return float(array.reshape(-1)[0])
        return array.reshape(count, 1)

    def _peak_windows(self, positions: np.ndarray, widths: np.ndarray,
                      eta, asymmetry=0.0
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Unit-height pseudo-Voigt profiles for every reflection, in one array.

        A peak only reaches a few tenths of a degree, so each is evaluated on a
        window of grid points rather than the whole pattern. Giving every peak
        the same window length in points lets all of them live in one
        (reflections, window) array and be evaluated in a single pass, instead
        of a Python loop that repeats the same handful of array operations
        hundreds of times. The profile is rebuilt on every objective evaluation
        during refinement, so this is the inner loop of the whole engine.

        `eta` and `asymmetry` may each be a scalar or one value per reflection:
        once the mixing is derived from the Gaussian and Lorentzian widths it
        varies with angle, and the skew from axial divergence varies with angle
        too, so every reflection carries its own.

        Returns the grid indices each window covers, the profiles, and the
        number of grid points under each unit-height peak.
        """
        n = self._n
        positions = np.asarray(positions, dtype=float)
        if n == 0 or positions.size == 0:
            return (np.zeros((0, 0), dtype=np.intp), np.zeros((0, 0)), np.zeros(0))

        x = self.experimental_data['two_theta']
        widths = np.maximum(np.asarray(widths, dtype=float), 1e-6)
        mixing = self._as_column(eta, positions.size)
        skew = self._as_column(asymmetry, positions.size)
        low_width, high_width = flank_widths(widths[:, None], skew)
        # How far a peak has to be followed depends on how Lorentzian it is. A
        # Gaussian is dead by three widths, but a Lorentzian still holds a few
        # percent of its area past five, and cutting it there leaves that
        # intensity out of the calculated pattern as an unfitted tail on every
        # strong peak. Scaling the window with eta buys the reach where it is
        # needed and hands it back on the near-Gaussian peaks.
        reach = 3.0 + 12.0 * mixing
        # The window also has to clear the broader flank, or a skewed tail is
        # cut off exactly where it was added to model something
        cutoff = (reach * np.maximum(low_width, high_width)).ravel()

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
        # A split profile: each flank keeps its own width, and the two agree at
        # the centre because both are unit height there
        width = np.where(offset < 0.0, low_width, high_width)
        sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma = width / 2.0
        profiles = (
            (1.0 - mixing) * np.exp(-0.5 * (offset / sigma) ** 2)
            + mixing / (1.0 + (offset / gamma) ** 2)
        )
        profiles *= on_grid & (np.abs(offset) <= cutoff[:, None])

        return indices, profiles, profiles.sum(axis=1)

    def _accumulate_pseudo_voigt(self, positions: np.ndarray, widths: np.ndarray,
                                  intensities: np.ndarray, eta,
                                  asymmetry=0.0) -> np.ndarray:
        """Windowed accumulation of pseudo-Voigt peaks into a dense pattern"""
        if self._n == 0 or len(positions) == 0:
            return np.zeros(self._n)

        indices, profiles, _ = self._peak_windows(positions, widths, eta, asymmetry)
        heights = np.maximum(np.asarray(intensities, dtype=float), 0.0)
        contributions = profiles * heights[:, None]
        return np.bincount(
            indices.ravel(), weights=contributions.ravel(), minlength=self._n
        )
    
    def _extract_lebail_intensities(self, positions: np.ndarray, widths: np.ndarray,
                                     eta, asymmetry=0.0, residual=None) -> np.ndarray:
        """
        Le Bail intensity extraction using windowed partitioning.
        Avoids allocating full-length profile arrays per peak.

        Partitioning works with area-normalized profiles, but the caller builds
        the pattern from unit-height profiles, so the integrated intensities are
        converted back to heights before returning. Skipping that conversion
        overpredicts the pattern by the number of points under a peak.

        Each phase is partitioned against its own reflections only, so where two
        phases overlap both claim the same observed intensity. That is why this
        mode cannot be used for quantification. Pass `residual` -- the other
        phases' calculated pattern -- to take their share out first, which is
        what an honest per-phase comparison needs.
        """
        observed = self.experimental_data['intensity']
        if residual is not None:
            observed = np.maximum(observed - np.asarray(residual, dtype=float), 0.0)
        n_peaks = len(positions)
        n_pts = self._n

        if n_peaks == 0 or n_pts == 0:
            return np.zeros(n_peaks)

        indices, profiles, areas = self._peak_windows(positions, widths, eta, asymmetry)
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
        
    def _calculate_peak_widths(self, two_theta: np.ndarray,
                               parameters: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Peak width and Gaussian/Lorentzian mixing for one phase's reflections.

        The instrument supplies a Gaussian width via the Caglioti curve, the
        phase supplies Lorentzian size and strain terms, and Thompson-Cox-
        Hastings combines them. The mixing parameter falls out of that
        combination instead of being fitted, which removes a parameter that
        would otherwise trade directly against the widths.
        """
        return phase_widths(
            two_theta,
            self.global_parameters,
            {
                'crystallite_size': parameters.get('crystallite_size', 0.0),
                'microstrain': parameters.get('microstrain', 0.0),
                'strain_extra': parameters.get('strain_extra'),
            },
            self.wavelength,
        )

    def _peak_asymmetry(self, two_theta: np.ndarray, parameters: Dict) -> np.ndarray:
        """
        Per-reflection peak skew, from the instrument and from this phase.

        The axial term belongs to the diffractometer and is shared; the sample
        term belongs to the phase, so a disordered layer silicate can be skewed
        without dragging the phases beside it out of shape.
        """
        return asymmetry_exponent(
            two_theta,
            axial=self.global_parameters.get('axial_asymmetry', 0.0),
            sample=parameters.get('asymmetry', 0.0),
        )
        
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

        weighted = (obs - calc) / errors
        rwp_num = np.sum(weighted ** 2)
        rwp_den = np.sum((obs / errors) ** 2)
        rwp = np.sqrt(rwp_num / rwp_den) if rwp_den > 0 else float('inf')

        n_obs = len(obs)
        n_param = sum(len(self._create_parameter_vector(p['parameters'])[0]) for p in self.phases)
        r_exp = np.sqrt((n_obs - n_param) / rwp_den) if rwp_den > 0 and n_obs > n_param else float('inf')

        gof = rwp / r_exp if r_exp > 0 and not np.isinf(r_exp) else float('inf')
        chi_squared = rwp_num / (n_obs - n_param) if n_obs > n_param else float('inf')

        factors = {
            'Rp': rp * 100,
            'Rwp': rwp * 100,
            'Rexp': r_exp * 100,
            'GoF': gof,
            'chi_squared': chi_squared,
            'Rwp_peak': self._peak_region_rwp(calc) * 100,
            'peak_coverage': float(np.mean(self._peak_region_mask())) * 100,
            'durbin_watson': self._durbin_watson(weighted),
            'R_Bragg': self._bragg_r_factor(),
        }
        return factors

    # How far either side of a reflection still counts as part of the peak. Two
    # widths holds essentially all of a Gaussian and the body of a Lorentzian,
    # without reaching so far that the gaps between peaks creep back in.
    _PEAK_REGION_WIDTHS = 2.0

    def _peak_region_mask(self, reach: Optional[float] = None) -> np.ndarray:
        """
        The points where the model actually predicts something.

        A pattern is mostly gaps. Because the data is background subtracted the
        gaps sit at zero, where the Poisson error model gives its smallest error
        and therefore its largest weight, so the emptiest parts of the pattern
        carry the most weight in Rwp while contributing almost nothing to its
        denominator. What comes out is dominated by counting noise between the
        peaks rather than by how well the peaks are fitted.
        """
        x = self.experimental_data['two_theta']
        mask = np.zeros(len(x), dtype=bool)
        reach = self._PEAK_REGION_WIDTHS if reach is None else float(reach)

        for phase in self.phases:
            params = phase['parameters']
            positions, _ = self._filter_peaks(phase['theoretical_peaks'])
            if len(positions) == 0:
                continue
            positions = self._shift_positions(positions, params)
            widths, _ = self._calculate_peak_widths(positions, params)
            span = reach * np.maximum(widths, 1e-6)
            starts = np.searchsorted(x, positions - span, side='left')
            stops = np.searchsorted(x, positions + span, side='right')
            for start, stop in zip(starts, stops):
                mask[start:stop] = True
        return mask

    def _strongest_line_area(self, phase_idx: int) -> Tuple[float, float]:
        """
        Profile area under a unit-height peak at this phase's strongest line,
        and how much of that line's width the sample terms are supplying.

        I/Ic is defined on integrated intensities, but a refined scale factor
        only gives a peak height. The two are proportional only at fixed width,
        and crystallite size and microstrain are refined per phase, so a phase
        whose peaks refine narrower gets a taller strongest line at unchanged
        area and is read as more abundant than it is. Multiplying by the area
        under the profile converts the height the scale factor carries into the
        integrated intensity the method is defined on.

        The width share is a diagnostic rather than a correction: when the
        sample terms are supplying most of the width, they are standing in for
        an instrument profile that was never calibrated, and the crystallite
        size that comes out is not a particle size.
        """
        phase = self.phases[phase_idx]
        params = phase['parameters']
        positions, reference = self._filter_peaks(phase['theoretical_peaks'])
        if len(positions) == 0:
            return 1.0, 0.0

        strongest = int(np.argmax(reference))
        shifted = self._shift_positions(positions, params)
        widths, eta = self._calculate_peak_widths(shifted, params)
        skew = self._peak_asymmetry(shifted, params)

        _, _, areas = self._peak_windows(shifted, widths, eta, skew)
        area = float(areas[strongest]) if len(areas) > strongest else 1.0

        # The same width with the sample contribution switched off tells us how
        # much of it the sample terms are carrying.
        bare = dict(params)
        bare['crystallite_size'] = 0.0
        bare['microstrain'] = 0.0
        instrument, _ = self._calculate_peak_widths(shifted, bare)
        total = float(widths[strongest])
        share = 0.0
        if total > 0:
            share = max(0.0, 1.0 - float(instrument[strongest]) / total)
        return area, share

    def _fitted_residual(self, residual: np.ndarray) -> np.ndarray:
        """The part of the residual the optimizer is being asked to minimise."""
        if self._fit_mask is None:
            return residual
        return residual[self._fit_mask]

    def _peak_region_rwp(self, calc: np.ndarray) -> float:
        """Rwp over the peaks alone, with the empty stretches left out."""
        mask = self._peak_region_mask()
        if not mask.any():
            return float('inf')
        obs = self.experimental_data['intensity'][mask]
        errors = self.experimental_data['errors'][mask]
        denominator = np.sum((obs / errors) ** 2)
        if denominator <= 0:
            return float('inf')
        numerator = np.sum(((obs - calc[mask]) / errors) ** 2)
        return float(np.sqrt(numerator / denominator))

    @staticmethod
    def _durbin_watson(weighted_residual: np.ndarray) -> float:
        """
        Serial correlation in the residuals, as a number near 2.

        Rwp says how large the misfit is; this says whether it has structure.
        Around 2 the residuals are uncorrelated point to point, which is what
        noise looks like and means the model has taken everything it can. Well
        below 2 they wander in runs -- the signature of an unmodelled peak
        shape, a wrong width or a missing phase -- and a smaller Rwp reached by
        adding parameters has not fixed it.
        """
        residual = np.asarray(weighted_residual, dtype=float)
        if residual.size < 2:
            return float('nan')
        denominator = float(np.sum(residual ** 2))
        if denominator <= 0:
            return float('nan')
        return float(np.sum(np.diff(residual) ** 2) / denominator)

    def _bragg_r_factor(self) -> Optional[float]:
        """
        Agreement between the observed and modelled integrated intensities.

        Reported only when the calculated intensities come from the structure.
        Le Bail sets them equal to the values it partitioned out of the observed
        pattern, so it scores exactly zero however wrong the model is, and
        Pawley fits them as free parameters against the same data. In either
        case the number would say nothing about the model. With reference
        intensities the comparison is real, and this is where preferred
        orientation or a misidentified polymorph shows itself.
        """
        if self.intensity_model != 'fixed' or not self.phases:
            return None
        if any(p['parameters'].get('refine_intensities') for p in self.phases):
            return None

        observed_total = 0.0
        difference_total = 0.0
        frozen, self._freeze_extracted = self._freeze_extracted, True
        try:
            # Each phase's pattern once, then subtract to get everyone else's,
            # rather than rebuilding the others from scratch for every phase.
            patterns = [self._calculate_phase_pattern(i, p['parameters'])
                        for i, p in enumerate(self.phases)]
            everything = np.sum(patterns, axis=0)

            for index, phase in enumerate(self.phases):
                params = phase['parameters']
                positions, reference = self._filter_peaks(phase['theoretical_peaks'])
                if len(positions) == 0:
                    continue
                positions = self._shift_positions(positions, params)
                widths, eta = self._calculate_peak_widths(positions, params)
                skew = self._peak_asymmetry(positions, params)

                modelled = (
                    reference * params.get('scale_factor', 1.0)
                    * self._intensity_corrections(positions, params, reference)
                )

                # What the data leaves for this phase once the others have taken
                # their share; without this every phase would claim the whole of
                # any overlapped intensity and score better than it deserves.
                others = everything - patterns[index]
                observed = self._extract_lebail_intensities(
                    positions, widths, eta, skew, residual=others
                )

                observed_total += float(np.sum(np.abs(observed)))
                difference_total += float(np.sum(np.abs(observed - modelled)))
        finally:
            self._freeze_extracted = frozen

        if observed_total <= 0:
            return None
        return 100.0 * difference_total / observed_total
        
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
                        'u': self.global_parameters.get('u_param', 0.0),
                        'v': self.global_parameters.get('v_param', 0.0),
                        'w': self.global_parameters.get('w_param', 0.0),
                        'crystallite_size': phase['parameters'].get('crystallite_size', 0.0),
                        'microstrain': phase['parameters'].get('microstrain', 0.0),
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
        I_i the strongest-line intensity implied by the refined scale. Where the
        database is missing an I/Ic for any phase they fall back to each phase's
        share of the fitted pattern, and `weight_percent_basis` says which of the
        two a row came from. They are only reported for the fixed-intensity
        model, because Le Bail extraction absorbs the scale factor and leaves
        nothing to quantify with.
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
            line_area, width_share = self._strongest_line_area(index)

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
                'line_area': scale * reference_max * line_area,
                'sample_width_share': width_share,
                'rir': rir,
                'lattice_scale': float(params.get('lattice_scale', 1.0)),
                'unit_cell': dict(params.get('unit_cell') or {}),
                'base_unit_cell': dict(params.get('_base_unit_cell') or {}),
                'absorption': float(params.get('absorption', 0.0) or 0.0),
                'harmonic_coeffs': list(params.get('harmonic_coeffs') or []),
                'profile': {
                    'u': float(self.global_parameters.get('u_param', 0.0)),
                    'v': float(self.global_parameters.get('v_param', 0.0)),
                    'w': float(self.global_parameters.get('w_param', 0.0)),
                },
                'crystallite_size': float(params.get('crystallite_size', 0.0)),
                'microstrain': float(params.get('microstrain', 0.0)),
                'asymmetry': float(params.get('asymmetry', 0.0) or 0.0),
                # What was free for this phase, so the next run can be set up
                # from where the last one actually stood rather than a default
                'refine_flags': {
                    key: bool(params.get(key, False))
                    for key in ('refine_scale', 'refine_strain', 'refine_size',
                                'refine_asymmetry', 'refine_cell',
                                'refine_absorption', 'refine_harmonics')
                },
                'locked': sorted(self._locked_parameters(params)),
                'contribution_percent': (
                    contributions[index]['contribution_percent'] if index < len(contributions) else None
                ),
                'integrated_intensity': (
                    contributions[index]['integrated_intensity'] if index < len(contributions) else None
                ),
            })

        # A Chung normalisation needs an I/Ic for every phase, because the sum
        # runs over all of them. Quantifying only the phases that have one
        # inflates them to fill 100% and hides the rest of the sample -- a
        # single RIR-bearing phase would come out at 100% however little of the
        # pattern it accounts for. When the set is incomplete, put every phase
        # on its share of the fitted pattern instead: that assumes equal
        # scattering power rather than silently dropping phases, and it keeps
        # all the numbers on one basis.
        complete_rir = bool(rows) and all(row['rir'] for row in rows)
        terms = [
            row['line_area'] / row['rir']
            for row in rows
            if row['rir'] and row['line_area'] > 0
        ]
        total = float(sum(terms))

        if not quantitative:
            basis = None
        elif complete_rir and total > 0:
            basis = 'rir'
        else:
            basis = 'contribution'

        for row in rows:
            row['weight_percent_basis'] = basis
            if basis == 'rir':
                row['weight_percent'] = (
                    100.0 * (row['line_area'] / row['rir']) / total
                    if row['line_area'] > 0 else 0.0
                )
            elif basis == 'contribution':
                row['weight_percent'] = row['contribution_percent']
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
            f"  Instrument profile: U={self.global_parameters.get('u_param', 0.0):.6f}, "
            f"V={self.global_parameters.get('v_param', 0.0):.6f}, "
            f"W={self.global_parameters.get('w_param', 0.0):.6f}"
        )
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
            report.append(
                f"  Sample broadening: size={params.get('crystallite_size', 0.0):.4g} um, "
                f"microstrain={params.get('microstrain', 0.0):.4g}"
            )

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
