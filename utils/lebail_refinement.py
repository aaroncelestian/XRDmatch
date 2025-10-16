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
            print(f"Normalized experimental intensity: {max_intensity:.0f} → 100.0")
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
        
        print(f"Applied 2-theta range filter: {min_2theta:.2f}° - {max_2theta:.2f}°")
        print(f"Data points: {len(two_theta)} → {len(self.experimental_data['two_theta'])}")
        
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
            print(f"Estimated initial scale factor: {initial_scale:.3f}")
            print(f"  (Normalized exp max: {exp_max_intensity:.1f}, theo max: {theo_max_intensity:.1f})")
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
                     staged_refinement: bool = True) -> Dict:
        """
        Perform Le Bail refinement on all phases
        
        Args:
            max_iterations: Maximum number of refinement cycles (reduced for performance)
            convergence_threshold: Convergence criterion for R-factors
            two_theta_range: Optional (min, max) 2-theta range to limit refinement
            staged_refinement: Use staged refinement (unit cell first, then profile)
            
        Returns:
            Dictionary with refinement results
        """
        # Update 2-theta range if provided
        if two_theta_range is not None and two_theta_range != self.two_theta_range:
            self.two_theta_range = two_theta_range
            self._apply_two_theta_filter()
        if not self.experimental_data or not self.phases:
            raise ValueError("Must set experimental data and add phases before refinement")
            
        print(f"Starting Le Bail refinement with {len(self.phases)} phases")
        print(f"Experimental data: {len(self.experimental_data['two_theta'])} points")
        
        # Check for Pawley mode and warn if too many peaks
        total_pawley_params = 0
        for phase in self.phases:
            if phase['parameters'].get('refine_intensities', False):
                n_peaks = len(phase['theoretical_peaks'].get('two_theta', []))
                total_pawley_params += n_peaks
        
        if total_pawley_params > 0:
            print(f"⚠️  Pawley mode enabled: refining {total_pawley_params} individual peak intensities")
            if total_pawley_params > 100:
                print(f"⚠️  WARNING: {total_pawley_params} intensity parameters may cause slow/unstable refinement")
                print(f"   Consider using 2θ range to reduce number of peaks, or disable Pawley mode")
        
        if staged_refinement:
            print("Using staged refinement: unit cell → profile parameters")
            if total_pawley_params > 0:
                print("  Stage 1: Pawley intensities disabled (too many parameters)")
                print("  Stage 2: Pawley intensities enabled")
        
        # Initialize refinement
        self.refinement_history = []
        previous_rwp = float('inf')
        rwp_change = float('inf')  # Initialize to avoid reference error
        
        # STAGE 1: Refine unit cell and zero shift only (if staged refinement)
        if staged_refinement:
            print("\n=== STAGE 1: Unit Cell & Zero Shift Refinement ===")
            # Temporarily disable profile refinement AND Pawley intensities in stage 1
            # (too many parameters otherwise)
            saved_intensity_settings = []
            for phase in self.phases:
                phase['parameters']['refine_profile'] = False
                # Save and disable Pawley refinement for stage 1
                saved_intensity_settings.append(phase['parameters'].get('refine_intensities', False))
                phase['parameters']['refine_intensities'] = False
            
            # Run first stage (about 1/3 of iterations)
            stage1_iterations = max(3, max_iterations // 3)
            for iteration in range(stage1_iterations):
                print(f"\nStage 1 - Iteration {iteration + 1}/{stage1_iterations}")
                
                for phase_idx, phase in enumerate(self.phases):
                    phase_name = phase['data']['phase'].get('mineral', f'Phase_{phase_idx}')
                    self._refine_single_phase(phase_idx)
                
                calculated_pattern = self._calculate_total_pattern()
                r_factors = self._calculate_r_factors(calculated_pattern)
                print(f"R-factors: Rp={r_factors['Rp']:.3f}, Rwp={r_factors['Rwp']:.3f}")
                
                iteration_result = {
                    'iteration': iteration + 1,
                    'stage': 1,
                    'r_factors': r_factors.copy(),
                    'parameters': copy.deepcopy([p['parameters'] for p in self.phases]),
                    'calculated_pattern': calculated_pattern.copy()
                }
                self.refinement_history.append(iteration_result)
            
            # STAGE 2: Enable profile refinement and restore Pawley settings
            print("\n=== STAGE 2: Profile Parameter Refinement ===")
            print("Enabling profile parameter refinement (U, V, W, η)")
            for idx, phase in enumerate(self.phases):
                phase['parameters']['refine_profile'] = True
                # Restore Pawley intensity refinement setting
                if idx < len(saved_intensity_settings):
                    phase['parameters']['refine_intensities'] = saved_intensity_settings[idx]
                print(f"  Phase {idx}: refine_profile=True, refine_cell=True")
            
            # Adjust remaining iterations
            remaining_iterations = max_iterations - stage1_iterations
            start_iteration = stage1_iterations
        else:
            remaining_iterations = max_iterations
            start_iteration = 0
        
        # Main refinement loop (Stage 2 if staged, or full refinement if not)
        for iteration in range(remaining_iterations):
            actual_iteration = start_iteration + iteration + 1
            stage_label = "Stage 2 - " if staged_refinement else ""
            print(f"\n=== {stage_label}Le Bail Iteration {actual_iteration} ===")
            
            # Refine each phase sequentially
            for phase_idx, phase in enumerate(self.phases):
                phase_name = phase['data']['phase'].get('mineral', f'Phase_{phase_idx}')
                print(f"Refining {phase_name}...")
                
                # Optimize phase parameters
                self._refine_single_phase(phase_idx)
                
            # Calculate current fit quality
            calculated_pattern = self._calculate_total_pattern()
            r_factors = self._calculate_r_factors(calculated_pattern)
            
            # Calculate per-phase contributions and R-factors
            phase_contributions = self._calculate_phase_contributions()
            
            print(f"R-factors: Rp={r_factors['Rp']:.3f}, Rwp={r_factors['Rwp']:.3f}, "
                  f"GoF={r_factors['GoF']:.3f}")
            
            # Print per-phase information
            for phase_idx, phase in enumerate(self.phases):
                phase_name = phase['data']['phase'].get('mineral', f'Phase_{phase_idx}')
                scale = phase['parameters']['scale_factor']
                phase_rwp = phase_contributions[phase_idx]['rwp']
                contribution = phase_contributions[phase_idx]['contribution_percent']
                print(f"  {phase_name}: Scale={scale:.3f}, Rwp={phase_rwp:.2f}%, Contribution={contribution:.1f}%")
            
            # Store iteration results
            iteration_result = {
                'iteration': actual_iteration,
                'stage': 2 if staged_refinement else 0,
                'r_factors': r_factors.copy(),
                'parameters': copy.deepcopy([p['parameters'] for p in self.phases]),
                'calculated_pattern': calculated_pattern.copy()
            }
            self.refinement_history.append(iteration_result)
            
            # Real-time plotting callback
            if self.plot_callback is not None:
                try:
                    self.plot_callback(iteration_result, self.experimental_data)
                except Exception as e:
                    print(f"Warning: Plot callback failed: {e}")
            
            # Check convergence
            rwp_change = abs(previous_rwp - r_factors['Rwp'])
            if rwp_change < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations (ΔRwp = {rwp_change:.6f})")
                break
                
            previous_rwp = r_factors['Rwp']
            
        # Final results
        final_results = {
            'converged': rwp_change < convergence_threshold,
            'iterations': len(self.refinement_history),
            'final_r_factors': self.refinement_history[-1]['r_factors'],
            'refined_phases': copy.deepcopy(self.phases),
            'calculated_pattern': self.refinement_history[-1]['calculated_pattern'],
            'refinement_history': self.refinement_history,
            # Include the actual 2-theta and intensity arrays used in refinement
            'two_theta': self.experimental_data['two_theta'].copy(),
            'experimental_intensity': self.experimental_data['intensity'].copy()
        }
        
        self.r_factors = final_results['final_r_factors']
        
        return final_results
        
    def _refine_single_phase(self, phase_idx: int):
        """Refine parameters for a single phase"""
        phase = self.phases[phase_idx]
        params = phase['parameters']
        
        # Create parameter vector for optimization
        param_vector, param_bounds, param_names = self._create_parameter_vector(params)
        
        # Pre-calculate other phases pattern (doesn't change during this phase's optimization)
        other_pattern = np.zeros_like(self.experimental_data['two_theta'])
        for i, other_phase in enumerate(self.phases):
            if i != phase_idx:
                other_pattern += self._calculate_phase_pattern(i, other_phase['parameters'])
        
        # Define objective function
        def objective(x):
            # Update parameters
            temp_params = self._vector_to_parameters(x, param_names, params)
            
            # Calculate pattern for this phase
            phase_pattern = self._calculate_phase_pattern(phase_idx, temp_params)
                    
            # Total calculated pattern
            total_pattern = phase_pattern + other_pattern
            
            # Calculate weighted residual
            residual = (self.experimental_data['intensity'] - total_pattern) / self.experimental_data['errors']
            return np.sum(residual ** 2)
            
        # Optimize parameters
        try:
            # Print initial objective value
            initial_obj = objective(param_vector)
            print(f"  Initial objective: {initial_obj:.2e}")
            
            result = minimize(
                objective,
                param_vector,
                bounds=param_bounds,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-5}
            )
            
            if result.success:
                # Update phase parameters
                optimized_params = self._vector_to_parameters(result.x, param_names, params)
                
                # Show what changed
                if 'u_param' in optimized_params:
                    print(f"  Profile refined: U={optimized_params['u_param']:.6f}, V={optimized_params['v_param']:.6f}, W={optimized_params['w_param']:.6f}, η={optimized_params['eta_param']:.3f}")
                    # Calculate FWHM at 5 degrees for reference
                    import math
                    theta_rad = math.radians(5.0 / 2)
                    tan_theta = math.tan(theta_rad)
                    fwhm_sq = optimized_params['u_param'] * tan_theta**2 + optimized_params['v_param'] * tan_theta + optimized_params['w_param']
                    fwhm = math.sqrt(max(fwhm_sq, 0.00001))
                    print(f"  → FWHM at 2θ=5°: {fwhm:.4f}°")
                
                if 'zero_shift' in optimized_params:
                    print(f"  Zero shift: {optimized_params['zero_shift']:.4f}°")
                
                if 'cell_a' in optimized_params:
                    cell = optimized_params['unit_cell']
                    print(f"  Unit cell: a={cell['a']:.4f}, b={cell['b']:.4f}, c={cell['c']:.4f}")
                
                # Check if scale factor collapsed
                if 'scale_factor' in optimized_params:
                    final_scale = optimized_params['scale_factor']
                    initial_scale = params['scale_factor']
                    if final_scale < initial_scale * 0.2:
                        print(f"  ⚠️  WARNING: Scale collapsed from {initial_scale:.3f} to {final_scale:.3f}")
                        print(f"     This usually means wrong phase or FWHM mismatch")
                
                phase['parameters'].update(optimized_params)
                
                # Update theoretical peaks with new parameters
                self._update_theoretical_peaks(phase_idx)
                
        except Exception as e:
            print(f"Optimization failed for phase {phase_idx}: {e}")
            
    def _create_parameter_vector(self, params: Dict) -> Tuple[np.ndarray, List, List]:
        """Create parameter vector for optimization"""
        param_vector = []
        param_bounds = []
        param_names = []
        
        # Scale factor (use max_scale_bound if provided)
        # In Le Bail mode with intensity extraction, scale factor is not needed
        # Only refine scale in Pawley mode
        is_pawley = params.get('refine_intensities', False)
        
        if params.get('refine_scale', True) and is_pawley:
            initial_scale = params['scale_factor']
            param_vector.append(initial_scale)
            max_scale = params.get('max_scale_bound', 10.0)
            
            # With normalized data (0-100), scale should be close to 1.0
            # Set bounds relative to initial estimate to prevent collapse
            min_scale = max(0.01, initial_scale * 0.1)  # At least 10% of initial
            max_scale_adjusted = min(max_scale, initial_scale * 10.0)  # At most 10x initial
            
            param_bounds.append((min_scale, max_scale_adjusted))
            param_names.append('scale_factor')
            print(f"  Scale bounds: {min_scale:.3f} - {max_scale_adjusted:.3f} (initial: {initial_scale:.3f})")
        elif params.get('refine_scale', True):
            # Le Bail mode: scale factor fixed at 1.0 (not refined)
            print(f"  Scale factor: 1.0 (fixed, using observed intensities)")
            
        # Profile parameters
        if params.get('refine_profile', True):
            param_vector.extend([
                params['u_param'],
                params['v_param'], 
                params['w_param'],
                params['eta_param']
            ])
            # Reasonable bounds for synchrotron data
            # Allow enough flexibility to fit the actual peak shapes
            param_bounds.extend([
                (0.0, 0.05),      # U - can vary with sample
                (-0.01, 0.01),    # V - usually small
                (0.00001, 0.05),  # W - controls minimum FWHM, allow wider range
                (0.0, 1.0)        # eta - full range for Voigt mixing
            ])
            param_names.extend(['u_param', 'v_param', 'w_param', 'eta_param'])
            print(f"  Profile params: U={params['u_param']:.6f}, V={params['v_param']:.6f}, W={params['w_param']:.6f}, η={params['eta_param']:.3f}")
            
        # Zero shift (tighter bounds to prevent excessive shifts)
        param_vector.append(params['zero_shift'])
        param_bounds.append((-0.1, 0.1))  # Reduced from ±0.5 to ±0.1 degrees
        param_names.append('zero_shift')
        
        # Unit cell parameters
        if params.get('refine_cell', True):
            unit_cell = params['unit_cell']
            param_vector.extend([
                unit_cell['a'],
                unit_cell['b'],
                unit_cell['c']
            ])
            
            # Set reasonable bounds based on initial values
            a, b, c = unit_cell['a'], unit_cell['b'], unit_cell['c']
            param_bounds.extend([
                (a * 0.95, a * 1.05),
                (b * 0.95, b * 1.05), 
                (c * 0.95, c * 1.05)
            ])
            param_names.extend(['cell_a', 'cell_b', 'cell_c'])
        
        # Individual peak intensity multipliers (Pawley refinement)
        if params.get('refine_intensities', False) and params.get('peak_intensity_multipliers') is not None:
            multipliers = params['peak_intensity_multipliers']
            param_vector.extend(multipliers)
            # Allow intensities to vary from 0.1x to 10x the theoretical value
            # This handles preferred orientation
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
                params['unit_cell'][name[5:]] = vector[i]  # Remove 'cell_' prefix
            elif name.startswith('intensity_mult_'):
                # Collect intensity multipliers
                intensity_multipliers.append(vector[i])
            else:
                params[name] = vector[i]
        
        # Store intensity multipliers as array if any were found
        if intensity_multipliers:
            params['peak_intensity_multipliers'] = np.array(intensity_multipliers)
                
        return params
        
    def _calculate_phase_pattern(self, phase_idx: int, parameters: Dict) -> np.ndarray:
        """Calculate diffraction pattern for a single phase using Le Bail intensity extraction"""
        phase = self.phases[phase_idx]
        theo_peaks = phase['theoretical_peaks']
        
        if len(theo_peaks.get('two_theta', [])) == 0:
            return np.zeros_like(self.experimental_data['two_theta'])
            
        # Apply zero shift to peak positions
        shifted_positions = theo_peaks['two_theta'] + parameters.get('zero_shift', 0.0)
        
        # Calculate peak widths using Caglioti function
        peak_widths = self._calculate_peak_widths(shifted_positions, parameters)
        
        # Generate pattern using pseudo-Voigt profiles
        pattern = np.zeros_like(self.experimental_data['two_theta'])
        scale_factor = parameters.get('scale_factor', 1.0)
        eta = parameters.get('eta_param', 0.5)
        
        # Get intensity multipliers if using Pawley refinement
        intensity_multipliers = parameters.get('peak_intensity_multipliers')
        is_pawley = intensity_multipliers is not None
        
        # For Le Bail: extract intensities using proper partitioning
        if not is_pawley:
            extracted_intensities = self._extract_lebail_intensities(
                shifted_positions, peak_widths, eta, parameters
            )
            # Debug: check extracted intensities
            if np.sum(extracted_intensities) == 0:
                print(f"  ⚠️  WARNING: All extracted intensities are zero!")
                print(f"     Peak widths: min={np.min(peak_widths):.6f}, max={np.max(peak_widths):.6f}")
                print(f"     Positions: min={np.min(shifted_positions):.2f}, max={np.max(shifted_positions):.2f}")
        
        for idx, (pos, intensity, width) in enumerate(zip(shifted_positions, theo_peaks['intensity'], peak_widths)):
            if width > 0 and intensity > 0:
                if is_pawley:
                    # Pawley mode: free intensity parameters
                    effective_intensity = intensity * scale_factor * intensity_multipliers[idx]
                else:
                    # Le Bail mode: use extracted intensities
                    effective_intensity = extracted_intensities[idx]
                
                peak_profile = self._pseudo_voigt_profile(
                    self.experimental_data['two_theta'], pos, width, effective_intensity, eta
                )
                pattern += peak_profile
                
        return pattern
    
    def _extract_lebail_intensities(self, positions: np.ndarray, widths: np.ndarray, 
                                     eta: float, parameters: Dict) -> np.ndarray:
        """
        Le Bail intensity extraction using proper partitioning
        
        For each peak, calculate its contribution to the observed pattern
        by partitioning overlapping peaks based on their profile shapes.
        This is the correct Le Bail method.
        """
        exp_2theta = self.experimental_data['two_theta']
        exp_intensity = self.experimental_data['intensity']
        n_peaks = len(positions)
        
        # Initialize extracted intensities with a good first guess
        # Use peak height at each position as starting point
        extracted_intensities = np.zeros(n_peaks)
        for idx, pos in enumerate(positions):
            # Find closest experimental point
            closest_idx = np.argmin(np.abs(exp_2theta - pos))
            if closest_idx < len(exp_intensity):
                extracted_intensities[idx] = max(0, exp_intensity[closest_idx])
        
        # Pre-calculate profile shapes for all peaks (normalized to unit area)
        # We'll scale these by the extracted intensities
        profiles = []
        for pos, width in zip(positions, widths):
            if width > 0:
                # Create normalized profile (unit intensity)
                profile = self._pseudo_voigt_profile(exp_2theta, pos, width, 1.0, eta)
                # Normalize to unit area
                profile_sum = np.sum(profile)
                if profile_sum > 0:
                    profile = profile / profile_sum
                profiles.append(profile)
            else:
                profiles.append(np.zeros_like(exp_2theta))
        
        # Iterative Le Bail extraction (3-5 iterations for convergence)
        for iteration in range(5):
            # Calculate total calculated pattern with current intensities
            total_calc = np.zeros_like(exp_2theta)
            for idx, profile in enumerate(profiles):
                total_calc += extracted_intensities[idx] * profile
            
            # Extract intensities by partitioning
            for idx, profile in enumerate(profiles):
                if np.sum(profile) == 0:
                    continue
                
                # Calculate this peak's fraction of the total at each point
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    fraction = np.where(total_calc > 0, 
                                      extracted_intensities[idx] * profile / total_calc,
                                      0.0)
                    fraction = np.nan_to_num(fraction, 0.0)
                
                # Extract intensity: sum of (observed * fraction)
                # This partitions the observed intensity among overlapping peaks
                extracted_intensities[idx] = np.sum(exp_intensity * fraction)
        
        return extracted_intensities
        
    def _calculate_peak_widths(self, two_theta: np.ndarray, parameters: Dict) -> np.ndarray:
        """Calculate peak widths using Caglioti function: FWHM² = U*tan²θ + V*tanθ + W"""
        U = parameters.get('u_param', 0.01)
        V = parameters.get('v_param', -0.001)
        W = parameters.get('w_param', 0.01)
        
        # Convert to radians for calculation
        theta_rad = np.radians(two_theta / 2)
        tan_theta = np.tan(theta_rad)
        
        # Caglioti function
        fwhm_squared = U * tan_theta**2 + V * tan_theta + W
        
        # Ensure positive widths
        fwhm_squared = np.maximum(fwhm_squared, 0.001)
        fwhm = np.sqrt(fwhm_squared)
        
        return fwhm
        
    def _pseudo_voigt_profile(self, x: np.ndarray, center: float, fwhm: float, 
                            intensity: float, eta: float) -> np.ndarray:
        """Generate pseudo-Voigt peak profile (optimized)"""
        if fwhm <= 0 or intensity <= 0:
            return np.zeros_like(x)
        
        # Only calculate profile within ±5*FWHM of peak center (optimization)
        cutoff = 5 * fwhm
        mask = np.abs(x - center) <= cutoff
        
        if not np.any(mask):
            return np.zeros_like(x)
        
        profile = np.zeros_like(x)
        x_local = x[mask]
        
        # Gaussian component
        sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-0.5 * ((x_local - center) / sigma_g) ** 2)
        
        # Lorentzian component  
        gamma_l = fwhm / 2
        lorentzian = 1 / (1 + ((x_local - center) / gamma_l) ** 2)
        
        # Pseudo-Voigt mixing
        profile[mask] = intensity * ((1 - eta) * gaussian + eta * lorentzian)
        
        return profile
        
    def _update_theoretical_peaks(self, phase_idx: int):
        """Update theoretical peak positions based on refined unit cell"""
        phase = self.phases[phase_idx]
        params = phase['parameters']
        
        if not params.get('refine_cell', True):
            return
            
        # This is a simplified update - in practice, you'd recalculate
        # peak positions from Miller indices and refined unit cell
        # For now, we'll apply the zero shift
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
            # Calculate this phase's pattern
            phase_pattern = self._calculate_phase_pattern(phase_idx, self.phases[phase_idx]['parameters'])
            
            # Calculate contribution as fraction of total calculated intensity
            total_intensity = np.sum(total_pattern)
            phase_intensity = np.sum(phase_pattern)
            contribution_percent = (phase_intensity / total_intensity * 100) if total_intensity > 0 else 0
            
            # Calculate Rwp for this phase alone vs experimental
            residual = (obs - phase_pattern) / errors
            rwp_num = np.sum(residual ** 2)
            rwp_den = np.sum((obs / errors) ** 2)
            phase_rwp = np.sqrt(rwp_num / rwp_den) * 100 if rwp_den > 0 else float('inf')
            
            contributions.append({
                'phase_idx': phase_idx,
                'contribution_percent': contribution_percent,
                'rwp': phase_rwp,
                'scale_factor': self.phases[phase_idx]['parameters']['scale_factor']
            })
        
        return contributions
    
    def _calculate_r_factors(self, calculated_pattern: np.ndarray) -> Dict[str, float]:
        """Calculate crystallographic R-factors"""
        obs = self.experimental_data['intensity']
        calc = calculated_pattern
        errors = self.experimental_data['errors']
        
        # Profile R-factor
        rp = np.sum(np.abs(obs - calc)) / np.sum(obs) if np.sum(obs) > 0 else float('inf')
        
        # Weighted profile R-factor
        rwp_num = np.sum(((obs - calc) / errors) ** 2)
        rwp_den = np.sum((obs / errors) ** 2)
        rwp = np.sqrt(rwp_num / rwp_den) if rwp_den > 0 else float('inf')
        
        # Expected R-factor
        n_obs = len(obs)
        n_param = sum(len(self._create_parameter_vector(p['parameters'])[0]) for p in self.phases)
        r_exp = np.sqrt((n_obs - n_param) / rwp_den) if rwp_den > 0 and n_obs > n_param else float('inf')
        
        # Goodness of fit
        gof = rwp / r_exp if r_exp > 0 and not np.isinf(r_exp) else float('inf')
        
        # Chi-squared
        chi_squared = rwp_num / (n_obs - n_param) if n_obs > n_param else float('inf')
        
        return {
            'Rp': rp * 100,      # Convert to percentage
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
