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
from utils.conditions import ambient_sql_filter
import time


# Reference patterns are stored at whatever Cu Kα the source used: AMCSD DIF
# files give 1.541838, other calculators give 1.5406. Asking for one exact
# value silently selects nothing, so accept the whole Cu Kα window and convert
# from the wavelength each row was actually calculated at.
CU_KALPHA_MIN = 1.5400
CU_KALPHA_MAX = 1.5420

def parse_series(text) -> np.ndarray:
    """
    One stored pattern column as floats.

    Older rows hold a JSON list and the AMCSD import writes a comma-separated
    string, sometimes with a trailing separator, so both have to be accepted.
    """
    if text is None:
        return np.array([], dtype=float)
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=float)
    try:
        return np.asarray(json.loads(text), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.array(
            [float(x) for x in str(text).split(',') if x.strip()], dtype=float
        )


REFERENCE_PATTERN_SQL = '''
    SELECT m.id, m.mineral_name, m.chemical_formula, m.space_group,
           m.cell_a, m.cell_b, m.cell_c, m.cell_alpha, m.cell_beta, m.cell_gamma,
           dp.two_theta, dp.intensities, dp.d_spacings, dp.wavelength
    FROM minerals m
    JOIN diffraction_patterns dp ON m.id = dp.mineral_id
    WHERE dp.wavelength BETWEEN ? AND ?
    {ambient_clause}
    ORDER BY m.id,
             CASE dp.calculation_method
                 WHEN 'AMCSD_DIF' THEN 1
                 WHEN 'pymatgen' THEN 2
                 ELSE 3
             END
'''


class PatternSearchEngine:
    """
    Advanced pattern search engine for XRD phase identification
    Supports both peak-based and correlation-based matching
    """
    
    def __init__(self, db_path: str = None):
        """Initialize the pattern search engine"""
        self.local_db = LocalCIFDatabase(db_path)
        self.ima_db = get_ima_database()

    def _iter_reference_patterns(self, cursor, ambient_only: bool):
        """
        One Cu Kα reference pattern per mineral, best calculation method first.

        Yields the raw rows of REFERENCE_PATTERN_SQL, skipping a mineral once
        it has been seen so a database holding several Cu Kα patterns for the
        same phase cannot enter it into the results twice.
        """
        ambient_clause, ambient_params = '', []
        if ambient_only:
            clause, ambient_params = ambient_sql_filter('m')
            ambient_clause = f'AND {clause}'

        cursor.execute(
            REFERENCE_PATTERN_SQL.format(ambient_clause=ambient_clause),
            [CU_KALPHA_MIN, CU_KALPHA_MAX] + list(ambient_params),
        )

        seen = set()
        for row in cursor.fetchall():
            if row[0] in seen:
                continue
            seen.add(row[0])
            yield row

    def _at_wavelength(self, two_theta: np.ndarray, intensity: np.ndarray,
                       d_spacings: np.ndarray, ref_wavelength: float,
                       exp_wavelength: float):
        """
        Reference lines placed where the measured wavelength puts them.

        Reflections Bragg's law cannot reach at the new wavelength are dropped
        from the positions and the intensities together, so the two arrays stay
        the same length and keep pointing at the same lines.
        """
        try:
            ref_wavelength = float(ref_wavelength)
        except (TypeError, ValueError):
            ref_wavelength = CU_KALPHA_MIN
        if abs(float(exp_wavelength) - ref_wavelength) <= 0.0001:
            return two_theta, intensity

        n = min(len(d_spacings), len(intensity))
        converted = self._convert_wavelength(d_spacings[:n], exp_wavelength)
        keep = np.isfinite(converted)
        return converted[keep], np.asarray(intensity[:n])[keep]
        
    def search_by_peaks(self, experimental_peaks: Dict, 
                       tolerance: float = 0.2, 
                       min_matches: int = 3,
                       intensity_weight: float = 0.3,
                       max_results: int = 50,
                       ambient_only: bool = True) -> List[Dict]:
        """
        Search for phases based on peak positions and intensities
        
        Args:
            experimental_peaks: Dict with 'two_theta', 'intensity', 'd_spacing' arrays
            tolerance: 2θ tolerance for peak matching (degrees)
            min_matches: Minimum number of peak matches required
            intensity_weight: Weight for intensity similarity (0-1, 0=position only)
            max_results: Maximum number of results to return
            ambient_only: Exclude structures measured at high P/T, whose shifted
                cells put lines at shifted 2θ
            
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
        
        results = []
        processed = 0
        
        for row in self._iter_reference_patterns(cursor, ambient_only):
            (mineral_id, mineral_name, formula, space_group,
             cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma,
             two_theta_json, intensities_json, d_spacings_json, ref_wavelength) = row
            
            try:
                theo_two_theta = parse_series(two_theta_json)
                theo_intensity = parse_series(intensities_json)
                theo_d_spacings = parse_series(d_spacings_json)
                
                theo_two_theta, theo_intensity = self._at_wavelength(
                    theo_two_theta, theo_intensity, theo_d_spacings,
                    ref_wavelength, exp_wavelength,
                )
                
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
                            two_theta_range: Tuple[float, float] = None,
                            ambient_only: bool = True) -> List[Dict]:
        """
        Search for phases using correlation analysis of full diffraction patterns
        
        Args:
            experimental_pattern: Dict with 'two_theta' and 'intensity' arrays
            min_correlation: Minimum correlation coefficient (0-1)
            max_results: Maximum number of results to return
            two_theta_range: Optional (min, max) 2θ range for comparison
            ambient_only: Exclude structures measured at high P/T
            
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
        
        results = []
        processed = 0
        
        for row in self._iter_reference_patterns(cursor, ambient_only):
            (mineral_id, mineral_name, formula, space_group,
             cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma,
             two_theta_json, intensities_json, d_spacings_json, ref_wavelength) = row
            
            try:
                theo_two_theta = parse_series(two_theta_json)
                theo_intensity = parse_series(intensities_json)
                theo_d_spacings = parse_series(d_spacings_json)
                
                theo_two_theta, theo_intensity = self._at_wavelength(
                    theo_two_theta, theo_intensity, theo_d_spacings,
                    ref_wavelength, exp_wavelength,
                )
                
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
                       max_results: int = 30,
                       full_pattern: Optional[Dict] = None,
                       ambient_only: bool = True) -> List[Dict]:
        """
        Combined search using both peak-based and correlation-based methods
        
        Args:
            experimental_data: Peak list, with 'two_theta' and 'intensity'
            peak_tolerance: 2θ tolerance for peak matching
            min_correlation: Minimum correlation for inclusion
            peak_weight: Weight for peak-based score (0-1)
            correlation_weight: Weight for correlation score (0-1)
            max_results: Maximum results to return
            full_pattern: The measured profile for the correlation half.
                Correlating against a bare peak list compares a few dozen
                isolated points and means nothing, so pass the profile whenever
                there is one; without it the peak list is used as before.
            ambient_only: Exclude structures measured at high P/T
            
        Returns:
            Combined and weighted results
        """
        print(f"🔍 Starting combined search (peak + correlation)...")
        
        # Perform both searches
        peak_results = self.search_by_peaks(
            experimental_data, 
            tolerance=peak_tolerance,
            min_matches=2,  # Lower threshold for combined search
            max_results=100,  # Get more candidates
            ambient_only=ambient_only,
        )
        
        correlation_results = self.search_by_correlation(
            full_pattern if full_pattern is not None else experimental_data,
            min_correlation=min_correlation,
            max_results=100,  # Get more candidates
            ambient_only=ambient_only,
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

    def ensemble_search(self, experimental_data: Dict,
                        methods: Optional[List[str]] = None,
                        max_results: int = 50,
                        **kwargs) -> List[Dict]:
        """
        Run multiple search methods and fuse scores by mineral_id.

        methods: subset of 'peaks', 'correlation', 'combined', 'ultrafast'
        For ultrafast, pass fast_search_engine= in kwargs.

        The peak methods want the peak list in `experimental_data`; the
        correlation methods want the measured profile, passed as
        full_pattern= in kwargs. They fall back to each other when only one
        is available.
        """
        methods = methods or ['peaks', 'correlation', 'ultrafast']
        fused: Dict[int, Dict] = {}
        full_pattern = kwargs.get('full_pattern') or experimental_data
        ambient_only = kwargs.get('ambient_only', True)

        if 'peaks' in methods and experimental_data.get('two_theta') is not None:
            for r in self.search_by_peaks(
                experimental_data,
                tolerance=kwargs.get('peak_tolerance', 0.2),
                min_matches=kwargs.get('min_matches', 2),
                max_results=max_results * 2,
                ambient_only=ambient_only,
            ):
                mid = r['mineral_id']
                entry = fused.setdefault(mid, {**r, 'method_scores': {}})
                entry['method_scores']['peaks'] = r.get('match_score', 0.0)
                entry.update({k: v for k, v in r.items() if k != 'method_scores'})

        if 'correlation' in methods:
            for r in self.search_by_correlation(
                full_pattern,
                min_correlation=kwargs.get('min_correlation', 0.25),
                max_results=max_results * 2,
                ambient_only=ambient_only,
            ):
                mid = r['mineral_id']
                entry = fused.setdefault(mid, {**r, 'method_scores': {}})
                entry['method_scores']['correlation'] = r.get('correlation', 0.0)
                for k, v in r.items():
                    if k != 'method_scores':
                        entry[k] = v

        if 'combined' in methods:
            for r in self.combined_search(
                experimental_data,
                peak_tolerance=kwargs.get('peak_tolerance', 0.2),
                min_correlation=kwargs.get('min_correlation', 0.25),
                max_results=max_results * 2,
                full_pattern=full_pattern,
                ambient_only=ambient_only,
            ):
                mid = r['mineral_id']
                entry = fused.setdefault(mid, {**r, 'method_scores': {}})
                entry['method_scores']['combined'] = r.get('combined_score', 0.0)
                for k, v in r.items():
                    if k != 'method_scores':
                        entry[k] = v

        if 'ultrafast' in methods:
            fast_engine = kwargs.get('fast_search_engine')
            if fast_engine is not None:
                for r in fast_engine.ultra_fast_correlation_search(
                    full_pattern,
                    min_correlation=kwargs.get('min_correlation', 0.25),
                    max_results=max_results * 2,
                    ambient_only=ambient_only,
                ):
                    mid = r['mineral_id']
                    entry = fused.setdefault(mid, {**r, 'method_scores': {}})
                    entry['method_scores']['ultrafast'] = r.get('correlation', 0.0)
                    for k, v in r.items():
                        if k != 'method_scores':
                            entry[k] = v

        results = []
        for mid, entry in fused.items():
            scores = list(entry.get('method_scores', {}).values())
            if not scores:
                continue
            entry['ensemble_score'] = float(max(scores))
            entry['ensemble_mean'] = float(np.mean(scores))
            entry['combined_score'] = entry['ensemble_score']
            entry['search_method'] = 'ensemble'
            results.append(entry)

        results.sort(key=lambda x: x['ensemble_score'], reverse=True)
        print(f"✅ Ensemble search complete: {len(results)} fused results")
        return results[:max_results]
    
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
        """
        2θ for a set of d-spacings at the target wavelength, via Bragg's law.

        Reflections with no solution — d ≤ 0, or sin θ > 1 at this wavelength —
        come back as NaN rather than being dropped, so the result stays aligned
        with the intensities belonging to the same lines.
        """
        d = np.asarray(d_spacings, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            sin_theta = np.where(d > 0, target_wavelength / (2.0 * d), np.nan)
        sin_theta = np.where(sin_theta <= 1.0, sin_theta, np.nan)
        return 2.0 * np.degrees(np.arcsin(sin_theta))
