"""
Multi-phase analysis system for sequential phase identification with residue analysis
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import copy
from .lebail_refinement import LeBailRefinement

class MultiPhaseAnalyzer:
    """
    Advanced multi-phase analysis system that performs sequential phase identification
    with residue-based analysis and overlapping peak handling
    """
    
    def __init__(self):
        self.identified_phases = []
        self.residue_history = []
        self.optimization_history = []
        self.lebail_engine = LeBailRefinement()
        self.refined_phases_cache = {}
        
    def sequential_phase_identification(self, experimental_data: Dict, 
                                     candidate_phases: List[Dict],
                                     max_phases: int = 5,
                                     residue_threshold: float = 0.05,
                                     use_lebail: bool = True) -> Dict:
        """
        Perform sequential phase identification with residue analysis
        
        Args:
            experimental_data: Dict with 'two_theta', 'intensity', 'wavelength'
            candidate_phases: List of phase dictionaries from matching
            max_phases: Maximum number of phases to identify
            residue_threshold: Stop when residue intensity drops below this fraction
            
        Returns:
            Dict with identified phases, residues, and optimization results
        """
        
        # Initialize with original experimental data
        current_residue = experimental_data['intensity'].copy()
        exp_two_theta = experimental_data['two_theta']
        
        identified_phases = []
        residue_history = [current_residue.copy()]
        
        print(f"Starting sequential phase identification with {len(candidate_phases)} candidates")
        print(f"Initial pattern intensity: max={np.max(current_residue):.0f}, mean={np.mean(current_residue):.0f}")
        
        for iteration in range(max_phases):
            print(f"\n=== Phase Identification Iteration {iteration + 1} ===")
            
            # Calculate current residue statistics
            max_residue = np.max(current_residue)
            mean_residue = np.mean(current_residue)
            residue_fraction = max_residue / np.max(experimental_data['intensity'])
            
            print(f"Current residue: max={max_residue:.0f}, mean={mean_residue:.0f}, fraction={residue_fraction:.3f}")
            
            # Check if residue is below threshold
            if residue_fraction < residue_threshold:
                print(f"Residue fraction {residue_fraction:.3f} below threshold {residue_threshold}")
                break
                
            # Find best matching phase for current residue
            best_phase, best_score, best_scaling = self._find_best_phase_for_residue(
                exp_two_theta, current_residue, candidate_phases, experimental_data['wavelength']
            )
            
            if best_phase is None or best_score < 0.1:
                print(f"No suitable phase found (best score: {best_score:.3f})")
                break
                
            print(f"Best phase: {best_phase['phase'].get('mineral', 'Unknown')} (score: {best_score:.3f}, scaling: {best_scaling:.3f})")
            
            # Generate theoretical pattern for best phase
            theoretical_pattern = self._generate_theoretical_pattern(
                best_phase, experimental_data['wavelength'], exp_two_theta
            )
            
            if theoretical_pattern is None:
                print("Failed to generate theoretical pattern")
                continue
                
            # Optimize phase scaling and subtract from residue
            optimized_scaling, subtracted_pattern = self._optimize_and_subtract_phase(
                exp_two_theta, current_residue, theoretical_pattern, best_scaling
            )
            
            # Store identified phase with optimization results
            phase_result = {
                'phase': best_phase['phase'],
                'theoretical_peaks': best_phase.get('theoretical_peaks'),
                'match_score': best_score,
                'optimized_scaling': optimized_scaling,
                'initial_scaling': best_scaling,
                'iteration': iteration + 1,
                'residue_before': current_residue.copy(),
                'contribution': current_residue - subtracted_pattern
            }
            
            identified_phases.append(phase_result)
            
            # Update residue for next iteration
            current_residue = subtracted_pattern
            residue_history.append(current_residue.copy())
            
            # Remove this phase from candidates to avoid re-identification
            candidate_phases = [p for p in candidate_phases if p['phase'].get('id') != best_phase['phase'].get('id')]
            
            print(f"Phase subtracted. New residue max: {np.max(current_residue):.0f}")
            
        print(f"\nSequential identification complete: {len(identified_phases)} phases identified")
        
        # Perform Le Bail refinement on identified phases if requested
        refinement_results = None
        if use_lebail and identified_phases:
            print("\n=== Starting Le Bail Refinement ===")
            refinement_results = self.perform_lebail_refinement(
                experimental_data, identified_phases
            )
            
        return {
            'identified_phases': identified_phases,
            'residue_history': residue_history,
            'final_residue': current_residue,
            'experimental_data': experimental_data,
            'residue_fraction': np.max(current_residue) / np.max(experimental_data['intensity']),
            'lebail_refinement': refinement_results
        }
    
    def _find_best_phase_for_residue(self, exp_two_theta: np.ndarray, 
                                   current_residue: np.ndarray,
                                   candidate_phases: List[Dict],
                                   wavelength: float) -> Tuple[Optional[Dict], float, float]:
        """
        Find the phase that best matches the current residue pattern
        """
        best_phase = None
        best_score = 0.0
        best_scaling = 1.0
        
        for phase in candidate_phases:
            # Generate theoretical pattern
            theoretical_pattern = self._generate_theoretical_pattern(
                phase, wavelength, exp_two_theta
            )
            
            if theoretical_pattern is None:
                continue
                
            # Calculate correlation with current residue
            score, optimal_scaling = self._calculate_residue_correlation(
                exp_two_theta, current_residue, theoretical_pattern
            )
            
            if score > best_score:
                best_score = score
                best_phase = phase
                best_scaling = optimal_scaling
                
        return best_phase, best_score, best_scaling
    
    def _calculate_residue_correlation(self, exp_two_theta: np.ndarray,
                                     residue: np.ndarray,
                                     theoretical_pattern: np.ndarray) -> Tuple[float, float]:
        """
        Calculate correlation between residue and theoretical pattern with optimal scaling
        """
        if len(theoretical_pattern) == 0 or np.max(theoretical_pattern) == 0:
            return 0.0, 1.0
            
        # Normalize patterns
        norm_residue = residue / np.max(residue) if np.max(residue) > 0 else residue
        norm_theoretical = theoretical_pattern / np.max(theoretical_pattern)
        
        # Find optimal scaling using least squares
        def objective(scaling):
            scaled_theoretical = norm_theoretical * scaling[0]
            # Only consider positive residue regions
            mask = norm_residue > 0.01 * np.max(norm_residue)
            if np.sum(mask) < 10:
                return 1.0
            return np.sum((norm_residue[mask] - scaled_theoretical[mask]) ** 2)
        
        # Optimize scaling factor
        result = minimize(objective, [1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
        optimal_scaling = result.x[0]
        
        # Calculate correlation with optimal scaling
        scaled_theoretical = norm_theoretical * optimal_scaling
        
        # Calculate correlation coefficient
        mask = (norm_residue > 0.01 * np.max(norm_residue)) & (scaled_theoretical > 0.01 * np.max(scaled_theoretical))
        if np.sum(mask) < 10:
            return 0.0, optimal_scaling
            
        correlation = np.corrcoef(norm_residue[mask], scaled_theoretical[mask])[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
            
        return abs(correlation), optimal_scaling
    
    def _generate_theoretical_pattern(self, phase_result: Dict, 
                                    wavelength: float,
                                    exp_two_theta: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate continuous theoretical pattern from phase data
        """
        if 'theoretical_peaks' not in phase_result:
            return None
            
        theo_peaks = phase_result['theoretical_peaks']
        if len(theo_peaks.get('two_theta', [])) == 0:
            return None
            
        # Use pseudo-Voigt profiles to generate continuous pattern
        return self._generate_pseudo_voigt_pattern(
            theo_peaks['two_theta'],
            theo_peaks['intensity'],
            exp_two_theta,
            fwhm=0.1
        )
    
    def _generate_pseudo_voigt_pattern(self, peak_positions: np.ndarray,
                                     peak_intensities: np.ndarray,
                                     x_range: np.ndarray,
                                     fwhm: float = 0.1) -> np.ndarray:
        """
        Generate continuous pattern using pseudo-Voigt peak profiles
        """
        pattern = np.zeros_like(x_range)
        
        for center, intensity in zip(peak_positions, peak_intensities):
            if intensity > 0:
                # Pseudo-Voigt profile (30% Lorentzian, 70% Gaussian)
                sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
                gamma_l = fwhm / 2
                
                gaussian = np.exp(-0.5 * ((x_range - center) / sigma_g) ** 2)
                lorentzian = 1 / (1 + ((x_range - center) / gamma_l) ** 2)
                
                peak = intensity * (0.7 * gaussian + 0.3 * lorentzian)
                pattern += peak
                
        return pattern
    
    def _optimize_and_subtract_phase(self, exp_two_theta: np.ndarray,
                                   current_residue: np.ndarray,
                                   theoretical_pattern: np.ndarray,
                                   initial_scaling: float) -> Tuple[float, np.ndarray]:
        """
        Optimize phase scaling and subtract from residue with overlap handling
        """
        
        def objective(scaling):
            scaled_theoretical = theoretical_pattern * scaling[0]
            # Prevent negative residues (physical constraint)
            subtracted = np.maximum(current_residue - scaled_theoretical, 0)
            # Minimize remaining intensity while preventing over-subtraction
            penalty = np.sum(np.maximum(scaled_theoretical - current_residue, 0) ** 2)
            return np.sum(subtracted ** 2) + 10 * penalty
        
        # Optimize scaling with physical constraints
        result = minimize(
            objective, 
            [initial_scaling], 
            bounds=[(0.01, 5.0)], 
            method='L-BFGS-B'
        )
        
        optimized_scaling = result.x[0]
        
        # Apply optimized subtraction
        scaled_theoretical = theoretical_pattern * optimized_scaling
        subtracted_residue = np.maximum(current_residue - scaled_theoretical, 0)
        
        return optimized_scaling, subtracted_residue
    
    def calculate_phase_fractions(self, multi_phase_result: Dict) -> Dict[str, float]:
        """
        Calculate quantitative phase fractions from identified phases
        """
        identified_phases = multi_phase_result['identified_phases']
        
        if not identified_phases:
            return {}
            
        # Calculate total contribution of all phases
        total_contribution = 0
        phase_contributions = {}
        
        for phase_result in identified_phases:
            contribution = np.sum(phase_result['contribution'])
            phase_name = phase_result['phase'].get('mineral', 'Unknown')
            phase_contributions[phase_name] = contribution
            total_contribution += contribution
            
        # Calculate fractions
        phase_fractions = {}
        for phase_name, contribution in phase_contributions.items():
            fraction = contribution / total_contribution if total_contribution > 0 else 0
            phase_fractions[phase_name] = fraction
            
        return phase_fractions
    
    def generate_residue_analysis_report(self, multi_phase_result: Dict) -> str:
        """
        Generate detailed analysis report
        """
        identified_phases = multi_phase_result['identified_phases']
        final_residue_fraction = multi_phase_result['residue_fraction']
        
        report = []
        report.append("=== Multi-Phase Analysis Report ===\n")
        
        report.append(f"Phases Identified: {len(identified_phases)}")
        report.append(f"Final Residue: {final_residue_fraction:.1%} of original intensity\n")
        
        # Phase details
        phase_fractions = self.calculate_phase_fractions(multi_phase_result)
        
        for i, phase_result in enumerate(identified_phases, 1):
            phase_name = phase_result['phase'].get('mineral', 'Unknown')
            fraction = phase_fractions.get(phase_name, 0)
            
            report.append(f"Phase {i}: {phase_name}")
            report.append(f"  - Match Score: {phase_result['match_score']:.3f}")
            report.append(f"  - Optimized Scaling: {phase_result['optimized_scaling']:.3f}")
            report.append(f"  - Estimated Fraction: {fraction:.1%}")
            report.append("")
            
        # Analysis quality assessment
        if final_residue_fraction < 0.05:
            report.append("Analysis Quality: Excellent - Very low residue")
        elif final_residue_fraction < 0.15:
            report.append("Analysis Quality: Good - Acceptable residue")
        elif final_residue_fraction < 0.30:
            report.append("Analysis Quality: Fair - Moderate residue remains")
        else:
            report.append("Analysis Quality: Poor - High residue suggests missing phases")
            
        return "\n".join(report)
        
    def perform_lebail_refinement(self, experimental_data: Dict, 
                                identified_phases: List[Dict]) -> Dict:
        """
        Perform Le Bail refinement on identified phases
        
        Args:
            experimental_data: Experimental diffraction data
            identified_phases: List of identified phases from sequential analysis
            
        Returns:
            Dictionary with refinement results
        """
        try:
            # Initialize Le Bail engine
            self.lebail_engine = LeBailRefinement()
            
            # Set experimental data
            errors = experimental_data.get('errors')
            if errors is None:
                # Estimate errors as sqrt(intensity) for Poisson statistics
                errors = np.sqrt(np.maximum(experimental_data['intensity'], 1))
                
            self.lebail_engine.set_experimental_data(
                experimental_data['two_theta'],
                experimental_data['intensity'],
                errors
            )
            
            # Add identified phases to refinement
            for phase_result in identified_phases:
                phase_data = {
                    'phase': phase_result['phase'],
                    'theoretical_peaks': phase_result['theoretical_peaks']
                }
                
                # Set initial parameters based on sequential analysis results
                initial_params = {
                    'scale_factor': phase_result.get('optimized_scaling', 1.0),
                    'u_param': 0.01,
                    'v_param': -0.001,
                    'w_param': 0.01,
                    'eta_param': 0.5,
                    'zero_shift': 0.0,
                    'refine_cell': True,
                    'refine_profile': True,
                    'refine_scale': True
                }
                
                self.lebail_engine.add_phase(phase_data, initial_params)
                
            # Perform refinement
            refinement_results = self.lebail_engine.refine_phases(
                max_iterations=30,
                convergence_threshold=1e-5
            )
            
            # Generate report
            refinement_report = self.lebail_engine.generate_refinement_report()
            
            # Get refined phases for future searches
            refined_phases = self.lebail_engine.get_refined_phases_for_search()
            
            # Cache refined phases for ultra-fast searching
            for refined_phase in refined_phases:
                phase_id = refined_phase['phase'].get('id')
                if phase_id:
                    self.refined_phases_cache[phase_id] = refined_phase
                    
            print("Le Bail refinement completed successfully")
            print(f"Final Rwp: {refinement_results['final_r_factors']['Rwp']:.3f}%")
            
            return {
                'success': True,
                'refinement_results': refinement_results,
                'refinement_report': refinement_report,
                'refined_phases': refined_phases,
                'r_factors': refinement_results['final_r_factors']
            }
            
        except Exception as e:
            print(f"Le Bail refinement failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'refinement_results': None,
                'refined_phases': []
            }
            
    def get_refined_phases_for_search(self) -> List[Dict]:
        """
        Get cached refined phases optimized for ultra-fast pattern searching
        
        Returns:
            List of refined phase data with optimized parameters
        """
        return list(self.refined_phases_cache.values())
        
    def update_candidate_phases_with_refinement(self, candidate_phases: List[Dict]) -> List[Dict]:
        """
        Update candidate phases with refined parameters from previous analyses
        
        Args:
            candidate_phases: Original candidate phases from database search
            
        Returns:
            Updated candidate phases with refined parameters where available
        """
        updated_phases = []
        
        for phase in candidate_phases:
            phase_id = phase['phase'].get('id')
            
            # Check if we have refined parameters for this phase
            if phase_id in self.refined_phases_cache:
                refined_phase = self.refined_phases_cache[phase_id]
                
                # Create updated phase with refined parameters
                updated_phase = phase.copy()
                updated_phase['theoretical_peaks'] = refined_phase['theoretical_peaks']
                updated_phase['refinement_quality'] = refined_phase['refinement_quality']
                updated_phase['search_priority'] = refined_phase['search_priority']
                updated_phase['refined'] = True
                
                print(f"Using refined parameters for {phase['phase'].get('mineral', 'Unknown')}")
                updated_phases.append(updated_phase)
            else:
                # Use original phase data
                updated_phase = phase.copy()
                updated_phase['refined'] = False
                updated_phases.append(updated_phase)
                
        # Sort by search priority (refined phases with good fits first)
        updated_phases.sort(
            key=lambda x: (x.get('refined', False), x.get('search_priority', 0)), 
            reverse=True
        )
        
        return updated_phases
        
    def clear_refined_cache(self):
        """Clear the refined phases cache"""
        self.refined_phases_cache.clear()
        print("Refined phases cache cleared")
