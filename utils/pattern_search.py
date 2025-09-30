"""
Pattern-based search functionality for XRD phase identification
Implements both peak-based and correlation-based matching algorithms
"""

import numpy as np
import sqlite3
import json
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from utils.local_database import LocalCIFDatabase
from utils.ima_mineral_database import get_ima_database
import time

class PatternSearchEngine:
    """
    Advanced pattern search engine for XRD phase identification
    Supports both peak-based and correlation-based matching
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the pattern search engine"""
        self.local_db = LocalCIFDatabase(db_path)
        self.ima_db = get_ima_database()
        
    def search_by_peaks(self, experimental_peaks: Dict, 
                       tolerance: float = 0.2, 
                       min_matches: int = 3,
                       intensity_weight: float = 0.3,
                       max_results: int = 50) -> List[Dict]:
        """
        Search for phases based on peak positions and intensities
        
        Args:
            experimental_peaks: Dict with 'two_theta', 'intensity', 'd_spacing' arrays
            tolerance: 2θ tolerance for peak matching (degrees)
            min_matches: Minimum number of peak matches required
            intensity_weight: Weight for intensity similarity (0-1, 0=position only)
            max_results: Maximum number of results to return
            
        Returns:
            List of matching phases with scores
        """
        print(f"🔍 Starting peak-based search...")
        print(f"   Experimental peaks: {len(experimental_peaks['two_theta'])}")
        print(f"   Tolerance: ±{tolerance}°")
        print(f"   Min matches: {min_matches}")
        print(f"   Intensity weight: {intensity_weight}")
        
        # Get experimental data
        exp_two_theta = np.array(experimental_peaks['two_theta'])
        exp_intensity = np.array(experimental_peaks['intensity'])
        exp_wavelength = experimental_peaks.get('wavelength', 1.5406)
        
        # Normalize experimental intensities
        max_exp_intensity = np.max(exp_intensity)
        norm_exp_intensity = exp_intensity / max_exp_intensity
        
        # Get all minerals with pre-calculated diffraction patterns
        conn = sqlite3.connect(self.local_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT m.id, m.mineral_name, m.chemical_formula, m.space_group,
                   m.cell_a, m.cell_b, m.cell_c, m.cell_alpha, m.cell_beta, m.cell_gamma,
                   dp.two_theta, dp.intensities, dp.d_spacings
            FROM minerals m
            JOIN diffraction_patterns dp ON m.id = dp.mineral_id
            WHERE dp.wavelength = 1.5406  -- Cu Kα reference patterns
        ''')
        
        results = []
        total_minerals = cursor.rowcount if cursor.rowcount else 0
        processed = 0
        
        for row in cursor.fetchall():
            mineral_id, mineral_name, formula, space_group, cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, two_theta_json, intensities_json, d_spacings_json = row
            
            try:
                # Parse JSON data
                theo_two_theta = np.array(json.loads(two_theta_json))
                theo_intensity = np.array(json.loads(intensities_json))
                theo_d_spacings = np.array(json.loads(d_spacings_json))
                
                # Convert to experimental wavelength if needed
                if abs(exp_wavelength - 1.5406) > 0.0001:
                    theo_two_theta = self._convert_wavelength(theo_d_spacings, exp_wavelength)
                
                # Calculate match score
                match_result = self._calculate_peak_match_score(
                    exp_two_theta, norm_exp_intensity,
                    theo_two_theta, theo_intensity,
                    tolerance, intensity_weight
                )
                
                if match_result['num_matches'] >= min_matches:
                    # Enhance with IMA database info
                    result = {
                        'mineral_id': mineral_id,
                        'mineral_name': mineral_name,
                        'chemical_formula': formula,
                        'space_group': space_group,
                        'cell_a': cell_a,
                        'cell_b': cell_b,
                        'cell_c': cell_c,
                        'cell_alpha': cell_alpha,
                        'cell_beta': cell_beta,
                        'cell_gamma': cell_gamma,
                        'match_score': match_result['match_score'],
                        'intensity_score': match_result['intensity_score'],
                        'num_matches': match_result['num_matches'],
                        'coverage': match_result['coverage'],
                        'matched_peaks': match_result['matched_peaks'],
                        'search_method': 'peak_based'
                    }
                    
                    # Cross-reference with IMA database for authoritative info
                    ima_info = self.ima_db.get_mineral_info(mineral_name)
                    if ima_info:
                        result['ima_chemistry'] = ima_info.get('chemistry', formula)
                        result['ima_space_group'] = ima_info.get('space_group', space_group)
                        result['ima_verified'] = True
                    else:
                        result['ima_verified'] = False
                    
                    results.append(result)
                
                processed += 1
                if processed % 100 == 0:
                    print(f"   Processed {processed} minerals...")
                    
            except Exception as e:
                print(f"   Error processing {mineral_name}: {e}")
                continue
        
        conn.close()
        
        # Sort by match score and limit results
        results.sort(key=lambda x: x['match_score'], reverse=True)
        results = results[:max_results]
        
        print(f"✅ Peak search complete: {len(results)} matches found from {processed} minerals")
        return results
    
    def search_by_correlation(self, experimental_pattern: Dict,
                            min_correlation: float = 0.5,
                            max_results: int = 50,
                            two_theta_range: Tuple[float, float] = None) -> List[Dict]:
        """
        Search for phases using correlation analysis of full diffraction patterns
        
        Args:
            experimental_pattern: Dict with 'two_theta' and 'intensity' arrays
            min_correlation: Minimum correlation coefficient (0-1)
            max_results: Maximum number of results to return
            two_theta_range: Optional (min, max) 2θ range for comparison
            
        Returns:
            List of matching phases with correlation scores
        """
        print(f"🔍 Starting correlation-based search...")
        print(f"   Experimental data points: {len(experimental_pattern['two_theta'])}")
        print(f"   Min correlation: {min_correlation}")
        
        # Prepare experimental data
        exp_two_theta = np.array(experimental_pattern['two_theta'])
        exp_intensity = np.array(experimental_pattern['intensity'])
        exp_wavelength = experimental_pattern.get('wavelength', 1.5406)
        
        # Apply 2θ range filter if specified
        if two_theta_range:
            mask = (exp_two_theta >= two_theta_range[0]) & (exp_two_theta <= two_theta_range[1])
            exp_two_theta = exp_two_theta[mask]
            exp_intensity = exp_intensity[mask]
            print(f"   Applied 2θ range {two_theta_range[0]:.1f}° - {two_theta_range[1]:.1f}°")
            print(f"   Filtered data points: {len(exp_two_theta)}")
        
        # Normalize experimental pattern
        exp_intensity = exp_intensity / np.max(exp_intensity)
        
        # Create interpolation function for experimental data
        exp_interp = interp1d(exp_two_theta, exp_intensity, 
                             bounds_error=False, fill_value=0, kind='linear')
        
        # Get all minerals with pre-calculated patterns
        conn = sqlite3.connect(self.local_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT m.id, m.mineral_name, m.chemical_formula, m.space_group,
                   m.cell_a, m.cell_b, m.cell_c, m.cell_alpha, m.cell_beta, m.cell_gamma,
                   dp.two_theta, dp.intensities, dp.d_spacings
            FROM minerals m
            JOIN diffraction_patterns dp ON m.id = dp.mineral_id
            WHERE dp.wavelength = 1.5406  -- Cu Kα reference patterns
        ''')
        
        results = []
        processed = 0
        
        for row in cursor.fetchall():
            mineral_id, mineral_name, formula, space_group, cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma, two_theta_json, intensities_json, d_spacings_json = row
            
            try:
                # Parse theoretical pattern
                theo_two_theta = np.array(json.loads(two_theta_json))
                theo_intensity = np.array(json.loads(intensities_json))
                theo_d_spacings = np.array(json.loads(d_spacings_json))
                
                # Convert to experimental wavelength if needed
                if abs(exp_wavelength - 1.5406) > 0.0001:
                    theo_two_theta = self._convert_wavelength(theo_d_spacings, exp_wavelength)
                
                # Calculate correlation
                correlation_result = self._calculate_pattern_correlation(
                    exp_two_theta, exp_intensity, exp_interp,
                    theo_two_theta, theo_intensity,
                    two_theta_range
                )
                
                if correlation_result['correlation'] >= min_correlation:
                    # Enhance with IMA database info
                    result = {
                        'mineral_id': mineral_id,
                        'mineral_name': mineral_name,
                        'chemical_formula': formula,
                        'space_group': space_group,
                        'cell_a': cell_a,
                        'cell_b': cell_b,
                        'cell_c': cell_c,
                        'cell_alpha': cell_alpha,
                        'cell_beta': cell_beta,
                        'cell_gamma': cell_gamma,
                        'correlation': correlation_result['correlation'],
                        'r_squared': correlation_result['r_squared'],
                        'overlap_fraction': correlation_result['overlap_fraction'],
                        'rms_error': correlation_result['rms_error'],
                        'search_method': 'correlation_based'
                    }
                    
                    # Cross-reference with IMA database for authoritative info
                    ima_info = self.ima_db.get_mineral_info(mineral_name)
                    if ima_info:
                        result['ima_chemistry'] = ima_info.get('chemistry', formula)
                        result['ima_space_group'] = ima_info.get('space_group', space_group)
                        result['ima_verified'] = True
                    else:
                        result['ima_verified'] = False
                    
                    results.append(result)
                
                processed += 1
                if processed % 100 == 0:
                    print(f"   Processed {processed} minerals...")
                    
            except Exception as e:
                print(f"   Error processing {mineral_name}: {e}")
                continue
        
        conn.close()
        
        # Sort by correlation and limit results
        results.sort(key=lambda x: x['correlation'], reverse=True)
        results = results[:max_results]
        
        print(f"✅ Correlation search complete: {len(results)} matches found from {processed} minerals")
        return results
    
    def combined_search(self, experimental_data: Dict,
                       peak_tolerance: float = 0.2,
                       min_correlation: float = 0.3,
                       peak_weight: float = 0.6,
                       correlation_weight: float = 0.4,
                       max_results: int = 30) -> List[Dict]:
        """
        Combined search using both peak-based and correlation-based methods
        
        Args:
            experimental_data: Dict with both peak data and full pattern
            peak_tolerance: 2θ tolerance for peak matching
            min_correlation: Minimum correlation for inclusion
            peak_weight: Weight for peak-based score (0-1)
            correlation_weight: Weight for correlation score (0-1)
            max_results: Maximum results to return
            
        Returns:
            Combined and weighted results
        """
        print(f"🔍 Starting combined search (peak + correlation)...")
        
        # Perform both searches
        peak_results = self.search_by_peaks(
            experimental_data, 
            tolerance=peak_tolerance,
            min_matches=2,  # Lower threshold for combined search
            max_results=100  # Get more candidates
        )
        
        correlation_results = self.search_by_correlation(
            experimental_data,
            min_correlation=min_correlation,
            max_results=100  # Get more candidates
        )
        
        # Combine results by mineral ID
        combined_results = {}
        
        # Add peak-based results
        for result in peak_results:
            mineral_id = result['mineral_id']
            combined_results[mineral_id] = result.copy()
            combined_results[mineral_id]['peak_score'] = result['match_score']
            combined_results[mineral_id]['correlation_score'] = 0.0
        
        # Add correlation scores
        for result in correlation_results:
            mineral_id = result['mineral_id']
            if mineral_id in combined_results:
                combined_results[mineral_id]['correlation_score'] = result['correlation']
                combined_results[mineral_id]['r_squared'] = result['r_squared']
            else:
                # Add correlation-only result
                combined_results[mineral_id] = result.copy()
                combined_results[mineral_id]['peak_score'] = 0.0
                combined_results[mineral_id]['correlation_score'] = result['correlation']
        
        # Calculate combined scores
        final_results = []
        for mineral_id, result in combined_results.items():
            peak_score = result.get('peak_score', 0.0)
            corr_score = result.get('correlation_score', 0.0)
            
            # Combined weighted score
            combined_score = (peak_weight * peak_score + 
                            correlation_weight * corr_score)
            
            result['combined_score'] = combined_score
            result['search_method'] = 'combined'
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        final_results = final_results[:max_results]
        
        print(f"✅ Combined search complete: {len(final_results)} results")
        return final_results
    
    def _calculate_peak_match_score(self, exp_two_theta: np.ndarray, exp_intensity: np.ndarray,
                                   theo_two_theta: np.ndarray, theo_intensity: np.ndarray,
                                   tolerance: float, intensity_weight: float) -> Dict:
        """Calculate peak-based matching score"""
        
        # Normalize theoretical intensities
        max_theo_intensity = np.max(theo_intensity) if len(theo_intensity) > 0 else 1
        norm_theo_intensity = theo_intensity / max_theo_intensity
        
        matches = []
        matched_exp_indices = set()
        total_theo_intensity = np.sum(norm_theo_intensity)
        matched_theo_intensity = 0
        
        # Find matches
        for i, (theo_2theta, theo_int) in enumerate(zip(theo_two_theta, norm_theo_intensity)):
            # Find closest experimental peak
            differences = np.abs(exp_two_theta - theo_2theta)
            min_idx = np.argmin(differences)
            min_diff = differences[min_idx]
            
            if min_diff <= tolerance and min_idx not in matched_exp_indices:
                exp_int = exp_intensity[min_idx]
                
                # Calculate intensity similarity
                intensity_sim = min(exp_int, theo_int) / max(exp_int, theo_int) if max(exp_int, theo_int) > 0 else 0
                
                matches.append({
                    'exp_2theta': exp_two_theta[min_idx],
                    'theo_2theta': theo_2theta,
                    'exp_intensity': exp_int,
                    'theo_intensity': theo_int,
                    'difference': min_diff,
                    'intensity_similarity': intensity_sim
                })
                
                matched_exp_indices.add(min_idx)
                matched_theo_intensity += theo_int
        
        # Calculate scores
        num_matches = len(matches)
        coverage = len(matched_exp_indices) / len(exp_two_theta) if len(exp_two_theta) > 0 else 0
        
        # Position-based score
        if num_matches > 0:
            position_score = num_matches / len(exp_two_theta)
        else:
            position_score = 0
        
        # Intensity-weighted score
        intensity_score = matched_theo_intensity / total_theo_intensity if total_theo_intensity > 0 else 0
        
        # Combined score
        match_score = ((1 - intensity_weight) * position_score + 
                      intensity_weight * intensity_score)
        
        return {
            'match_score': match_score,
            'intensity_score': intensity_score,
            'num_matches': num_matches,
            'coverage': coverage,
            'matched_peaks': matches
        }
    
    def _calculate_pattern_correlation(self, exp_two_theta: np.ndarray, exp_intensity: np.ndarray,
                                     exp_interp, theo_two_theta: np.ndarray, theo_intensity: np.ndarray,
                                     two_theta_range: Tuple[float, float] = None) -> Dict:
        """Calculate correlation between experimental and theoretical patterns"""
        
        # Determine common 2θ range
        if two_theta_range:
            min_2theta, max_2theta = two_theta_range
        else:
            min_2theta = max(np.min(exp_two_theta), np.min(theo_two_theta))
            max_2theta = min(np.max(exp_two_theta), np.max(theo_two_theta))
        
        # Create common 2θ grid
        common_2theta = np.linspace(min_2theta, max_2theta, 1000)
        
        # Interpolate experimental data
        exp_interp_values = exp_interp(common_2theta)
        
        # Generate theoretical pattern on same grid using pseudo-Voigt peaks
        theo_pattern = self._generate_continuous_pattern(
            theo_two_theta, theo_intensity, common_2theta, fwhm=0.1
        )
        
        # Normalize both patterns
        theo_pattern = theo_pattern / np.max(theo_pattern) if np.max(theo_pattern) > 0 else theo_pattern
        
        # Calculate correlation
        valid_mask = ~(np.isnan(exp_interp_values) | np.isnan(theo_pattern))
        if np.sum(valid_mask) < 10:  # Need minimum points for correlation
            return {'correlation': 0, 'r_squared': 0, 'overlap_fraction': 0, 'rms_error': 1}
        
        exp_valid = exp_interp_values[valid_mask]
        theo_valid = theo_pattern[valid_mask]
        
        # Pearson correlation
        correlation, _ = pearsonr(exp_valid, theo_valid)
        if np.isnan(correlation):
            correlation = 0
        
        # R-squared
        r_squared = correlation ** 2
        
        # RMS error
        rms_error = np.sqrt(np.mean((exp_valid - theo_valid) ** 2))
        
        # Overlap fraction
        overlap_fraction = np.sum(valid_mask) / len(common_2theta)
        
        return {
            'correlation': abs(correlation),  # Use absolute value
            'r_squared': r_squared,
            'overlap_fraction': overlap_fraction,
            'rms_error': rms_error
        }
    
    def _generate_continuous_pattern(self, two_theta_peaks: np.ndarray, intensities: np.ndarray,
                                   x_range: np.ndarray, fwhm: float = 0.1) -> np.ndarray:
        """Generate continuous pattern from peak positions using pseudo-Voigt profiles"""
        pattern = np.zeros_like(x_range)
        
        for center, intensity in zip(two_theta_peaks, intensities):
            if intensity > 0:
                # Pseudo-Voigt profile (30% Lorentzian, 70% Gaussian)
                sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
                gamma_l = fwhm / 2
                
                gaussian = np.exp(-0.5 * ((x_range - center) / sigma_g) ** 2)
                lorentzian = 1 / (1 + ((x_range - center) / gamma_l) ** 2)
                
                peak = intensity * (0.7 * gaussian + 0.3 * lorentzian)
                pattern += peak
        
        return pattern
    
    def _convert_wavelength(self, d_spacings: np.ndarray, target_wavelength: float) -> np.ndarray:
        """Convert d-spacings to 2θ values for target wavelength using Bragg's law"""
        two_theta_values = []
        
        for d in d_spacings:
            if d > 0:
                sin_theta = target_wavelength / (2 * d)
                if sin_theta <= 1.0:
                    theta_rad = np.arcsin(sin_theta)
                    two_theta_deg = 2 * np.degrees(theta_rad)
                    two_theta_values.append(two_theta_deg)
        
        return np.array(two_theta_values)
