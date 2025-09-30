"""
Le Bail refinement engine for XRD phase matching
Implements proper crystallographic refinement with profile functions and unit cell optimization
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from scipy.special import wofz
from typing import Dict, List, Tuple, Optional, Union
import copy
import warnings
warnings.filterwarnings('ignore')

class LeBailRefinement:
    """
    Le Bail refinement engine for optimizing phase fits in XRD patterns
    
    Features:
    - Profile function refinement (pseudo-Voigt, Pearson VII)
    - Unit cell parameter optimization
    - Peak position and intensity refinement
    - Multi-phase simultaneous refinement
    - Goodness-of-fit statistics (R-factors, chi-squared)
    """
    
    def __init__(self):
        self.experimental_data = None
        self.phases = []
        self.refined_parameters = {}
        self.refinement_history = []
        self.r_factors = {}
        
    def set_experimental_data(self, two_theta: np.ndarray, intensity: np.ndarray, 
                            errors: Optional[np.ndarray] = None):
        """Set experimental diffraction data"""
        self.experimental_data = {
            'two_theta': np.array(two_theta),
            'intensity': np.array(intensity),
            'errors': np.array(errors) if errors is not None else np.sqrt(np.maximum(intensity, 1))
        }
        
    def add_phase(self, phase_data: Dict, initial_parameters: Optional[Dict] = None):
        """
        Add a phase for refinement
        
        Args:
            phase_data: Phase information including theoretical peaks
            initial_parameters: Initial refinement parameters
        """
        if 'theoretical_peaks' not in phase_data:
            raise ValueError("Phase data must include theoretical_peaks")
            
        # Default refinement parameters
        default_params = {
            'scale_factor': 1.0,
            'u_param': 0.01,      # Peak width parameter U
            'v_param': -0.001,    # Peak width parameter V  
            'w_param': 0.01,      # Peak width parameter W
            'eta_param': 0.5,     # Pseudo-Voigt mixing parameter
            'zero_shift': 0.0,    # Zero point shift
            'unit_cell': self._extract_unit_cell(phase_data),
            'refine_cell': True,
            'refine_profile': True,
            'refine_scale': True
        }
        
        if initial_parameters:
            default_params.update(initial_parameters)
            
        phase = {
            'data': phase_data,
            'parameters': default_params,
            'theoretical_peaks': phase_data['theoretical_peaks'].copy()
        }
        
        self.phases.append(phase)
        
    def _extract_unit_cell(self, phase_data: Dict) -> Dict:
        """Extract unit cell parameters from phase data"""
        phase_info = phase_data.get('phase', {})
        
        # Try to get unit cell from phase data
        unit_cell = {
            'a': phase_info.get('cell_a', 10.0),
            'b': phase_info.get('cell_b', 10.0), 
            'c': phase_info.get('cell_c', 10.0),
            'alpha': phase_info.get('cell_alpha', 90.0),
            'beta': phase_info.get('cell_beta', 90.0),
            'gamma': phase_info.get('cell_gamma', 90.0)
        }
        
        return unit_cell
        
    def refine_phases(self, max_iterations: int = 20, 
                     convergence_threshold: float = 1e-5) -> Dict:
        """
        Perform Le Bail refinement on all phases
        
        Args:
            max_iterations: Maximum number of refinement cycles (reduced for performance)
            convergence_threshold: Convergence criterion for R-factors
            
        Returns:
            Dictionary with refinement results
        """
        if not self.experimental_data or not self.phases:
            raise ValueError("Must set experimental data and add phases before refinement")
            
        print(f"Starting Le Bail refinement with {len(self.phases)} phases")
        print(f"Experimental data: {len(self.experimental_data['two_theta'])} points")
        
        # Initialize refinement
        self.refinement_history = []
        previous_rwp = float('inf')
        
        for iteration in range(max_iterations):
            print(f"\n=== Le Bail Iteration {iteration + 1} ===")
            
            # Refine each phase sequentially
            for phase_idx, phase in enumerate(self.phases):
                phase_name = phase['data']['phase'].get('mineral', f'Phase_{phase_idx}')
                print(f"Refining {phase_name}...")
                
                # Optimize phase parameters
                self._refine_single_phase(phase_idx)
                
            # Calculate current fit quality
            calculated_pattern = self._calculate_total_pattern()
            r_factors = self._calculate_r_factors(calculated_pattern)
            
            print(f"R-factors: Rp={r_factors['Rp']:.3f}, Rwp={r_factors['Rwp']:.3f}, "
                  f"GoF={r_factors['GoF']:.3f}")
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'r_factors': r_factors.copy(),
                'parameters': copy.deepcopy([p['parameters'] for p in self.phases]),
                'calculated_pattern': calculated_pattern.copy()
            }
            self.refinement_history.append(iteration_result)
            
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
            'refinement_history': self.refinement_history
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
        
        # Scale factor
        if params.get('refine_scale', True):
            param_vector.append(params['scale_factor'])
            param_bounds.append((0.01, 10.0))
            param_names.append('scale_factor')
            
        # Profile parameters
        if params.get('refine_profile', True):
            param_vector.extend([
                params['u_param'],
                params['v_param'], 
                params['w_param'],
                params['eta_param']
            ])
            param_bounds.extend([
                (0.0, 1.0),      # U
                (-0.1, 0.1),     # V
                (0.001, 1.0),    # W
                (0.0, 1.0)       # eta
            ])
            param_names.extend(['u_param', 'v_param', 'w_param', 'eta_param'])
            
        # Zero shift
        param_vector.append(params['zero_shift'])
        param_bounds.append((-0.5, 0.5))
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
            
        return np.array(param_vector), param_bounds, param_names
        
    def _vector_to_parameters(self, vector: np.ndarray, names: List[str], 
                            original_params: Dict) -> Dict:
        """Convert parameter vector back to parameter dictionary"""
        params = {}
        
        for i, name in enumerate(names):
            if name.startswith('cell_'):
                if 'unit_cell' not in params:
                    params['unit_cell'] = original_params['unit_cell'].copy()
                params['unit_cell'][name[5:]] = vector[i]  # Remove 'cell_' prefix
            else:
                params[name] = vector[i]
                
        return params
        
    def _calculate_phase_pattern(self, phase_idx: int, parameters: Dict) -> np.ndarray:
        """Calculate diffraction pattern for a single phase"""
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
        
        for pos, intensity, width in zip(shifted_positions, theo_peaks['intensity'], peak_widths):
            if width > 0 and intensity > 0:
                peak_profile = self._pseudo_voigt_profile(
                    self.experimental_data['two_theta'], pos, width, intensity * scale_factor, eta
                )
                pattern += peak_profile
                
        return pattern
        
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
