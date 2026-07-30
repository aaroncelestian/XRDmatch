"""
Le Bail refinement engine for XRD phase matching
Implements proper crystallographic refinement with profile functions and unit cell optimization
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from scipy.special import wofz
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
        self.extract_iterations = 5
        self._profile_cache = {}  # (phase_idx, cache_key) -> sparse profile parts
        
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
        
        # Apply 2-theta range filter if specified
        if self.two_theta_range is not None:
            self._apply_two_theta_filter()
        
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
            'zero_shift': 0.0,    # Zero point shift
            'unit_cell': self._extract_unit_cell(phase_data),
            'refine_cell': True,
            'refine_profile': True,
            'refine_scale': True,
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
        
        return unit_cell
        
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
            for phase in self.phases:
                phase['parameters']['_use_scaled_pattern'] = False

        self._profile_cache = {}

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
            'experimental_intensity': self.experimental_data['intensity'].copy()
        }
        
        self.r_factors = final_results['final_r_factors']
        
        return final_results
        
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
        
        maxiter = 8 if self.mode == 'trial' else 50
        ftol = 1e-3 if self.mode == 'trial' else 1e-6
        
        def objective(x):
            temp_params = self._vector_to_parameters(x, param_names, params)
            # Merge with base params so non-refined keys remain
            merged = dict(params)
            merged.update(temp_params)
            if 'unit_cell' in temp_params:
                merged['unit_cell'] = temp_params['unit_cell']
            phase_pattern = self._calculate_phase_pattern(phase_idx, merged)
            total_pattern = phase_pattern + other_pattern
            residual = (self.experimental_data['intensity'] - total_pattern) / self.experimental_data['errors']
            return np.sum(residual ** 2)
            
        try:
            initial_obj = objective(param_vector)
            self._log(f"  Initial objective: {initial_obj:.2e}")
            
            result = minimize(
                objective,
                param_vector,
                bounds=param_bounds,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-5}
            )
            
            optimized_params = self._vector_to_parameters(result.x, param_names, params)
            
            if 'u_param' in optimized_params:
                self._log(f"  Profile refined: U={optimized_params['u_param']:.6f}, "
                          f"V={optimized_params['v_param']:.6f}, W={optimized_params['w_param']:.6f}, "
                          f"η={optimized_params['eta_param']:.3f}")
            
            if 'zero_shift' in optimized_params:
                self._log(f"  Zero shift: {optimized_params['zero_shift']:.4f}°")
            
            if 'cell_a' in optimized_params:
                cell = optimized_params['unit_cell']
                self._log(f"  Unit cell: a={cell['a']:.4f}, b={cell['b']:.4f}, c={cell['c']:.4f}")
            
            if 'scale_factor' in optimized_params:
                final_scale = optimized_params['scale_factor']
                initial_scale = params['scale_factor']
                if final_scale < initial_scale * 0.2:
                    self._log(f"  WARNING: Scale collapsed from {initial_scale:.3f} to {final_scale:.3f}")
            
            phase['parameters'].update(optimized_params)
            if self.mode == 'polish':
                self._update_theoretical_peaks(phase_idx)
                
        except Exception as e:
            self._log(f"Optimization failed for phase {phase_idx}: {e}")
            
    def _create_parameter_vector(self, params: Dict) -> Tuple[np.ndarray, List, List]:
        """Create parameter vector for optimization"""
        param_vector = []
        param_bounds = []
        param_names = []
        
        is_pawley = params.get('refine_intensities', False)
        use_scaled = params.get('_use_scaled_pattern', False)
        
        # Refine scale for Pawley OR trial scaled-pattern mode
        if params.get('refine_scale', True) and (is_pawley or use_scaled):
            initial_scale = params['scale_factor']
            param_vector.append(initial_scale)
            max_scale = params.get('max_scale_bound', 10.0)
            min_scale = max(0.01, initial_scale * 0.1)
            max_scale_adjusted = min(max_scale, initial_scale * 10.0)
            param_bounds.append((min_scale, max_scale_adjusted))
            param_names.append('scale_factor')
            self._log(f"  Scale bounds: {min_scale:.3f} - {max_scale_adjusted:.3f} (initial: {initial_scale:.3f})")
        elif params.get('refine_scale', True) and not use_scaled:
            self._log(f"  Scale factor: 1.0 (fixed, using observed intensities)")
            
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
            
        param_vector.append(params['zero_shift'])
        param_bounds.append((-0.1, 0.1))
        param_names.append('zero_shift')
        
        if params.get('refine_cell', True):
            unit_cell = params['unit_cell']
            param_vector.extend([
                unit_cell['a'],
                unit_cell['b'],
                unit_cell['c']
            ])
            a, b, c = unit_cell['a'], unit_cell['b'], unit_cell['c']
            param_bounds.extend([
                (a * 0.95, a * 1.05),
                (b * 0.95, b * 1.05), 
                (c * 0.95, c * 1.05)
            ])
            param_names.extend(['cell_a', 'cell_b', 'cell_c'])
        
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
        
        for i, name in enumerate(names):
            if name.startswith('cell_'):
                if 'unit_cell' not in params:
                    params['unit_cell'] = original_params['unit_cell'].copy()
                params['unit_cell'][name[5:]] = vector[i]
            elif name.startswith('intensity_mult_'):
                intensity_multipliers.append(vector[i])
            else:
                params[name] = vector[i]
        
        if intensity_multipliers:
            params['peak_intensity_multipliers'] = np.array(intensity_multipliers)
                
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

    def _calculate_phase_pattern(self, phase_idx: int, parameters: Dict) -> np.ndarray:
        """Calculate diffraction pattern for a single phase"""
        phase = self.phases[phase_idx]
        theo_peaks = phase['theoretical_peaks']
        
        if len(theo_peaks.get('two_theta', [])) == 0:
            return np.zeros_like(self.experimental_data['two_theta'])

        positions, intensities = self._filter_peaks(theo_peaks)
        if len(positions) == 0:
            return np.zeros_like(self.experimental_data['two_theta'])
            
        shifted_positions = positions + parameters.get('zero_shift', 0.0)
        peak_widths = self._calculate_peak_widths(shifted_positions, parameters)
        scale_factor = parameters.get('scale_factor', 1.0)
        eta = parameters.get('eta_param', 0.5)
        intensity_multipliers = parameters.get('peak_intensity_multipliers')
        is_pawley = intensity_multipliers is not None
        use_scaled = parameters.get('_use_scaled_pattern', False) or self.mode == 'trial'

        if use_scaled and not is_pawley:
            # Fast trial path: scale theoretical intensities (no Le Bail extract)
            effective = intensities * scale_factor
            return self._accumulate_pseudo_voigt(
                shifted_positions, peak_widths, effective, eta
            )

        if not is_pawley:
            extracted_intensities = self._extract_lebail_intensities(
                shifted_positions, peak_widths, eta
            )
            effective = extracted_intensities
        else:
            # Need full peak list for Pawley multipliers — use unfiltered theo peaks
            all_pos = np.asarray(theo_peaks['two_theta'], dtype=float) + parameters.get('zero_shift', 0.0)
            all_int = np.asarray(theo_peaks['intensity'], dtype=float)
            all_widths = self._calculate_peak_widths(all_pos, parameters)
            effective = all_int * scale_factor * intensity_multipliers
            return self._accumulate_pseudo_voigt(all_pos, all_widths, effective, eta)

        return self._accumulate_pseudo_voigt(shifted_positions, peak_widths, effective, eta)

    def _accumulate_pseudo_voigt(self, positions: np.ndarray, widths: np.ndarray,
                                  intensities: np.ndarray, eta: float) -> np.ndarray:
        """Windowed accumulation of pseudo-Voigt peaks into a dense pattern"""
        x = self.experimental_data['two_theta']
        pattern = np.zeros_like(x)
        if len(x) == 0:
            return pattern

        # Approximate dx for index windowing
        dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.02
        x0 = float(x[0])
        n = len(x)

        for pos, width, intensity in zip(positions, widths, intensities):
            if width <= 0 or intensity <= 0:
                continue
            cutoff = 5.0 * width
            i_lo = max(0, int((pos - cutoff - x0) / dx) - 1)
            i_hi = min(n, int((pos + cutoff - x0) / dx) + 2)
            if i_lo >= i_hi:
                continue
            x_local = x[i_lo:i_hi]
            sigma_g = width / (2 * np.sqrt(2 * np.log(2)))
            gaussian = np.exp(-0.5 * ((x_local - pos) / sigma_g) ** 2)
            gamma_l = width / 2.0
            lorentzian = 1.0 / (1.0 + ((x_local - pos) / gamma_l) ** 2)
            pattern[i_lo:i_hi] += intensity * ((1.0 - eta) * gaussian + eta * lorentzian)

        return pattern
    
    def _extract_lebail_intensities(self, positions: np.ndarray, widths: np.ndarray, 
                                     eta: float) -> np.ndarray:
        """
        Le Bail intensity extraction using windowed partitioning.
        Avoids allocating full-length profile arrays per peak.
        """
        exp_2theta = self.experimental_data['two_theta']
        exp_intensity = self.experimental_data['intensity']
        n_peaks = len(positions)
        n_pts = len(exp_2theta)
        extracted = np.zeros(n_peaks)

        if n_peaks == 0 or n_pts == 0:
            return extracted

        dx = float(np.median(np.diff(exp_2theta))) if n_pts > 1 else 0.02
        x0 = float(exp_2theta[0])

        # Sparse storage: list of (i_lo, i_hi, normalized_local_profile)
        sparse_profiles = []
        for pos, width in zip(positions, widths):
            if width <= 0:
                sparse_profiles.append(None)
                continue
            cutoff = 5.0 * width
            i_lo = max(0, int((pos - cutoff - x0) / dx) - 1)
            i_hi = min(n_pts, int((pos + cutoff - x0) / dx) + 2)
            if i_lo >= i_hi:
                sparse_profiles.append(None)
                continue
            x_local = exp_2theta[i_lo:i_hi]
            sigma_g = width / (2 * np.sqrt(2 * np.log(2)))
            gaussian = np.exp(-0.5 * ((x_local - pos) / sigma_g) ** 2)
            gamma_l = width / 2.0
            lorentzian = 1.0 / (1.0 + ((x_local - pos) / gamma_l) ** 2)
            profile = (1.0 - eta) * gaussian + eta * lorentzian
            psum = np.sum(profile)
            if psum > 0:
                profile = profile / psum
            sparse_profiles.append((i_lo, i_hi, profile))

            # Initial guess from nearest observed intensity
            closest_idx = int(np.clip(round((pos - x0) / dx), 0, n_pts - 1))
            extracted[len(sparse_profiles) - 1] = max(0.0, float(exp_intensity[closest_idx]))

        # Fix initial guesses (index was wrong above if some were None) — redo cleanly
        extracted = np.zeros(n_peaks)
        for idx, pos in enumerate(positions):
            closest_idx = int(np.clip(round((pos - x0) / dx), 0, n_pts - 1))
            extracted[idx] = max(0.0, float(exp_intensity[closest_idx]))

        n_iter = max(1, int(self.extract_iterations))
        for _ in range(n_iter):
            total_calc = np.zeros(n_pts)
            for idx, sp in enumerate(sparse_profiles):
                if sp is None:
                    continue
                i_lo, i_hi, profile = sp
                total_calc[i_lo:i_hi] += extracted[idx] * profile

            for idx, sp in enumerate(sparse_profiles):
                if sp is None:
                    continue
                i_lo, i_hi, profile = sp
                local_total = total_calc[i_lo:i_hi]
                local_obs = exp_intensity[i_lo:i_hi]
                contrib = extracted[idx] * profile
                with np.errstate(divide='ignore', invalid='ignore'):
                    fraction = np.where(local_total > 0, contrib / local_total, 0.0)
                extracted[idx] = float(np.sum(local_obs * fraction))

        return extracted
        
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
        
    def _update_theoretical_peaks(self, phase_idx: int):
        """Update theoretical peak positions based on refined unit cell"""
        phase = self.phases[phase_idx]
        params = phase['parameters']
        
        if not params.get('refine_cell', True):
            return
            
        zero_shift = params.get('zero_shift', 0.0)
        original_peaks = phase['data']['theoretical_peaks']
        
        phase['theoretical_peaks']['two_theta'] = (
            original_peaks['two_theta'] + zero_shift
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
                    'zero_shift': phase['parameters']['zero_shift'],
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
            report.append(f"  Zero shift: {params['zero_shift']:.4f}°")
            
            if params.get('refine_cell', True):
                cell = params['unit_cell']
                report.append(f"  Unit cell:")
                report.append(f"    a = {cell['a']:.4f} Å")
                report.append(f"    b = {cell['b']:.4f} Å") 
                report.append(f"    c = {cell['c']:.4f} Å")
                report.append(f"    α = {cell['alpha']:.3f}°")
                report.append(f"    β = {cell['beta']:.3f}°")
                report.append(f"    γ = {cell['gamma']:.3f}°")
            
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
