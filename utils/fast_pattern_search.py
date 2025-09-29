"""
Ultra-fast pattern search engine for XRD phase identification
Optimized for instant correlation searches on large databases using techniques from Raman spectroscopy
"""

import numpy as np
import sqlite3
import json
from typing import Dict, List, Tuple, Optional
from scipy.fft import fft, ifft
from scipy.signal import correlate
import time

class FastPatternSearchEngine:
    """
    High-performance pattern search engine using optimized correlation algorithms
    Inspired by techniques used in Raman spectroscopy database searching
    """
    
    def __init__(self, db_path: str = None):
        """Initialize with pre-computed search indices"""
        from utils.local_database import LocalCIFDatabase
        self.local_db = LocalCIFDatabase(db_path)
        
        # Pre-computed search indices for ultra-fast searching
        self.search_index = None
        self.pattern_matrix = None
        self.mineral_metadata = None
        self.common_grid = None
        
        # Performance tracking
        self.last_search_time = 0
        self.index_build_time = 0
        
    def build_search_index(self, grid_resolution: float = 0.02, 
                          two_theta_range: Tuple[float, float] = (5.0, 90.0),
                          force_rebuild: bool = False,
                          use_pkl_cache: bool = True) -> bool:
        """
        Build optimized search index for ultra-fast correlation searches
        
        This is the key to instant searching - pre-compute all patterns on a common grid
        
        Args:
            grid_resolution: 2θ resolution in degrees (smaller = more accurate but larger)
            two_theta_range: (min, max) 2θ range for indexing
            force_rebuild: Force rebuild even if index exists
            use_pkl_cache: Use PKL caching for Raman-like performance
            
        Returns:
            True if index built successfully
        """
        
        # Try to load from PKL cache first (Raman-style performance)
        if use_pkl_cache and not force_rebuild:
            cache_file = f"data/xrd_search_index_{grid_resolution:.3f}_{two_theta_range[0]:.0f}_{two_theta_range[1]:.0f}.pkl"
            if self._load_pkl_cache(cache_file):
                print("🚀 Search index loaded from PKL cache (instant!)")
                return True
        
        if self.search_index is not None and not force_rebuild:
            print("🚀 Search index already loaded")
            return True
            
        print(f"🔨 Building fast search index...")
        print(f"   Grid resolution: {grid_resolution}°")
        print(f"   2θ range: {two_theta_range[0]}° - {two_theta_range[1]}°")
        
        start_time = time.time()
        
        # Create common 2θ grid
        min_2theta, max_2theta = two_theta_range
        self.common_grid = np.arange(min_2theta, max_2theta + grid_resolution, grid_resolution)
        grid_size = len(self.common_grid)
        
        print(f"   Grid points: {grid_size}")
        
        # Get all minerals with diffraction patterns
        conn = sqlite3.connect(self.local_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.id, m.mineral_name, m.chemical_formula, m.space_group,
                   dp.two_theta, dp.intensities, dp.d_spacings
            FROM minerals m
            JOIN diffraction_patterns dp ON m.id = dp.mineral_id
            WHERE dp.wavelength = 1.5406
            ORDER BY m.id
        ''')
        
        rows = cursor.fetchall()
        num_minerals = len(rows)
        
        if num_minerals == 0:
            print("❌ No diffraction patterns found in database")
            conn.close()
            return False
            
        print(f"   Processing {num_minerals} minerals...")
        
        # Pre-allocate pattern matrix (minerals × grid_points)
        self.pattern_matrix = np.zeros((num_minerals, grid_size), dtype=np.float32)
        self.mineral_metadata = []
        
        # Process each mineral
        for i, row in enumerate(rows):
            mineral_id, mineral_name, formula, space_group, two_theta_json, intensities_json, d_spacings_json = row
            
            try:
                # Parse pattern data
                theo_two_theta = np.array(json.loads(two_theta_json))
                theo_intensity = np.array(json.loads(intensities_json))
                
                # Generate continuous pattern on common grid using fast binning
                continuous_pattern = self._fast_pattern_generation(
                    theo_two_theta, theo_intensity, self.common_grid
                )
                
                # Normalize pattern (L2 norm for better correlation properties)
                pattern_norm = np.linalg.norm(continuous_pattern)
                if pattern_norm > 0:
                    continuous_pattern = continuous_pattern / pattern_norm
                
                # Store in matrix
                self.pattern_matrix[i, :] = continuous_pattern
                
                # Store metadata
                self.mineral_metadata.append({
                    'id': mineral_id,
                    'name': mineral_name,
                    'formula': formula,
                    'space_group': space_group,
                    'pattern_norm': pattern_norm
                })
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{num_minerals} minerals...")
                    
            except Exception as e:
                print(f"   Error processing {mineral_name}: {e}")
                # Fill with zeros for failed patterns
                self.pattern_matrix[i, :] = 0
                self.mineral_metadata.append({
                    'id': mineral_id,
                    'name': mineral_name,
                    'formula': formula,
                    'space_group': space_group,
                    'pattern_norm': 0
                })
        
        conn.close()
        
        # Remove empty patterns
        valid_patterns = np.any(self.pattern_matrix != 0, axis=1)
        self.pattern_matrix = self.pattern_matrix[valid_patterns]
        self.mineral_metadata = [meta for i, meta in enumerate(self.mineral_metadata) if valid_patterns[i]]
        
        # Build search index flag
        self.search_index = True
        
        self.index_build_time = time.time() - start_time
        print(f"✅ Search index built in {self.index_build_time:.2f}s")
        print(f"   Valid patterns: {len(self.mineral_metadata)}")
        print(f"   Matrix size: {self.pattern_matrix.shape}")
        print(f"   Memory usage: ~{self.pattern_matrix.nbytes / 1024 / 1024:.1f} MB")
        
        # Save to PKL cache for Raman-like instant loading next time
        if use_pkl_cache:
            cache_file = f"data/xrd_search_index_{grid_resolution:.3f}_{two_theta_range[0]:.0f}_{two_theta_range[1]:.0f}.pkl"
            self._save_pkl_cache(cache_file)
        
        return True
    
    def ultra_fast_correlation_search(self, experimental_pattern: Dict,
                                    min_correlation: float = 0.3,
                                    max_results: int = 50,
                                    wavelength_convert: bool = True) -> List[Dict]:
        """
        Ultra-fast correlation search using pre-computed pattern matrix
        
        This should be nearly instant even for large databases
        
        Args:
            experimental_pattern: Dict with 'two_theta' and 'intensity'
            min_correlation: Minimum correlation threshold
            max_results: Maximum results to return
            wavelength_convert: Convert wavelengths if needed
            
        Returns:
            List of correlation results sorted by score
        """
        
        if self.search_index is None:
            print("⚠️  Search index not built. Building now...")
            if not self.build_search_index():
                return []
        
        start_time = time.time()
        
        # Prepare experimental pattern
        exp_two_theta = np.array(experimental_pattern['two_theta'])
        exp_intensity = np.array(experimental_pattern['intensity'])
        exp_wavelength = experimental_pattern.get('wavelength', 1.5406)
        
        # Convert wavelength if needed (this is fast with d-spacings)
        if wavelength_convert and abs(exp_wavelength - 1.5406) > 0.0001:
            # Convert to d-spacings then back to Cu Kα 2θ for comparison
            exp_d_spacings = exp_wavelength / (2 * np.sin(np.radians(exp_two_theta / 2)))
            exp_two_theta = 2 * np.degrees(np.arcsin(1.5406 / (2 * exp_d_spacings)))
        
        # Generate experimental pattern on common grid
        exp_pattern = self._fast_pattern_generation(
            exp_two_theta, exp_intensity, self.common_grid
        )
        
        # Normalize experimental pattern
        exp_norm = np.linalg.norm(exp_pattern)
        if exp_norm > 0:
            exp_pattern = exp_pattern / exp_norm
        else:
            print("❌ Experimental pattern has zero intensity")
            return []
        
        # ULTRA-FAST MATRIX CORRELATION
        # This is the magic - single matrix multiplication for all correlations!
        correlations = np.dot(self.pattern_matrix, exp_pattern)
        
        # Find results above threshold
        valid_indices = np.where(correlations >= min_correlation)[0]
        
        if len(valid_indices) == 0:
            print(f"⚠️  No correlations above {min_correlation}")
            return []
        
        # Sort by correlation (descending)
        sorted_indices = valid_indices[np.argsort(correlations[valid_indices])[::-1]]
        
        # Limit results
        if len(sorted_indices) > max_results:
            sorted_indices = sorted_indices[:max_results]
        
        # Build results
        results = []
        for idx in sorted_indices:
            correlation = correlations[idx]
            metadata = self.mineral_metadata[idx]
            
            results.append({
                'mineral_id': metadata['id'],
                'mineral_name': metadata['name'],
                'chemical_formula': metadata['formula'],
                'space_group': metadata['space_group'],
                'correlation': float(correlation),
                'r_squared': float(correlation ** 2),
                'search_method': 'ultra_fast_correlation'
            })
        
        self.last_search_time = time.time() - start_time
        
        print(f"🚀 Ultra-fast search complete in {self.last_search_time*1000:.1f}ms")
        print(f"   Found {len(results)} matches above {min_correlation}")
        print(f"   Top correlation: {results[0]['correlation']:.3f}" if results else "")
        
        return results
    
    def _fast_pattern_generation(self, peak_positions: np.ndarray, 
                               peak_intensities: np.ndarray,
                               grid: np.ndarray,
                               fwhm: float = 0.1) -> np.ndarray:
        """
        Ultra-fast pattern generation using optimized binning instead of peak profiles
        
        This is much faster than generating individual pseudo-Voigt peaks
        """
        
        if len(peak_positions) == 0:
            return np.zeros_like(grid)
        
        # Ensure arrays are the right type and shape
        peak_positions = np.asarray(peak_positions, dtype=np.float64)
        peak_intensities = np.asarray(peak_intensities, dtype=np.float64)
        
        # Method 1: Simple binning (fastest)
        pattern = np.zeros_like(grid, dtype=np.float32)
        
        # Find grid indices for each peak
        grid_indices = np.searchsorted(grid, peak_positions)
        
        # Clip to valid range - fix the boundary condition
        valid_mask = (grid_indices >= 0) & (grid_indices < len(grid))
        
        if not np.any(valid_mask):
            return pattern  # No peaks within grid range
            
        valid_indices = grid_indices[valid_mask]
        valid_intensities = peak_intensities[valid_mask]
        
        # Ensure indices are within bounds (extra safety)
        valid_indices = np.clip(valid_indices, 0, len(grid) - 1)
        
        # Add intensities to grid (handle multiple peaks per bin)
        try:
            np.add.at(pattern, valid_indices, valid_intensities)
        except Exception as e:
            print(f"   Warning: Binning failed for pattern, using fallback: {e}")
            # Fallback: manual assignment
            for idx, intensity in zip(valid_indices, valid_intensities):
                if 0 <= idx < len(pattern):
                    pattern[idx] += intensity
        
        # Optional: Apply fast Gaussian smoothing for better peak shapes
        if fwhm > 0:
            try:
                grid_spacing = np.mean(np.diff(grid)) if len(grid) > 1 else 0.02
                sigma_points = fwhm / grid_spacing / 2.355  # Convert FWHM to sigma in grid points
                if sigma_points > 0.5:  # Only smooth if meaningful
                    pattern = self._fast_gaussian_smooth(pattern, sigma_points)
            except Exception as e:
                print(f"   Warning: Smoothing failed, using raw pattern: {e}")
        
        return pattern
    
    def _fast_gaussian_smooth(self, pattern: np.ndarray, sigma: float) -> np.ndarray:
        """
        Fast Gaussian smoothing using FFT convolution
        Much faster than scipy.ndimage.gaussian_filter for our use case
        """
        
        if sigma <= 0 or len(pattern) == 0:
            return pattern
        
        # Create Gaussian kernel
        kernel_size = max(3, int(6 * sigma))  # At least 3 points
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Don't smooth if kernel is larger than pattern
        if kernel_size >= len(pattern):
            return pattern
        
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        
        # Use simple convolution (more reliable)
        try:
            smoothed = np.convolve(pattern, kernel, mode='same')
            return smoothed.astype(pattern.dtype)
        except Exception:
            # Fallback: return original pattern
            return pattern
    
    def benchmark_search_speed(self, experimental_pattern: Dict, 
                             num_iterations: int = 10) -> Dict:
        """
        Benchmark search speed for performance comparison
        """
        
        if self.search_index is None:
            self.build_search_index()
        
        print(f"🏃 Benchmarking search speed ({num_iterations} iterations)...")
        
        times = []
        for i in range(num_iterations):
            start = time.time()
            results = self.ultra_fast_correlation_search(
                experimental_pattern, 
                min_correlation=0.1,  # Low threshold to test full database
                max_results=100
            )
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        benchmark_results = {
            'average_time_ms': avg_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'database_size': len(self.mineral_metadata),
            'patterns_per_second': len(self.mineral_metadata) / avg_time,
            'index_build_time_s': self.index_build_time
        }
        
        print(f"📊 Benchmark Results:")
        print(f"   Average search time: {avg_time*1000:.1f}ms")
        print(f"   Min/Max time: {min_time*1000:.1f}/{max_time*1000:.1f}ms")
        print(f"   Database size: {len(self.mineral_metadata)} patterns")
        print(f"   Search rate: {benchmark_results['patterns_per_second']:.0f} patterns/second")
        print(f"   Index build time: {self.index_build_time:.2f}s")
        
        return benchmark_results
    
    def get_search_statistics(self) -> Dict:
        """Get search engine statistics"""
        
        if self.search_index is None:
            return {'status': 'Index not built'}
        
        return {
            'status': 'Ready',
            'database_size': len(self.mineral_metadata),
            'grid_points': len(self.common_grid),
            'grid_range': f"{self.common_grid[0]:.1f}° - {self.common_grid[-1]:.1f}°",
            'grid_resolution': f"{self.common_grid[1] - self.common_grid[0]:.3f}°",
            'matrix_size_mb': self.pattern_matrix.nbytes / 1024 / 1024,
            'index_build_time_s': self.index_build_time,
            'last_search_time_ms': self.last_search_time * 1000
        }
    
    def export_search_index(self, file_path: str) -> bool:
        """
        Export search index to file for faster startup
        """
        
        if self.search_index is None:
            print("❌ No search index to export")
            return False
        
        try:
            np.savez_compressed(
                file_path,
                pattern_matrix=self.pattern_matrix,
                common_grid=self.common_grid,
                mineral_metadata=self.mineral_metadata,
                index_build_time=self.index_build_time
            )
            print(f"💾 Search index exported to {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    def import_search_index(self, file_path: str) -> bool:
        """
        Import pre-built search index from file
        """
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            self.pattern_matrix = data['pattern_matrix']
            self.common_grid = data['common_grid']
            self.mineral_metadata = data['mineral_metadata'].tolist()
            self.index_build_time = float(data['index_build_time'])
            self.search_index = True
            
            print(f"📂 Search index imported from {file_path}")
            print(f"   Database size: {len(self.mineral_metadata)} patterns")
            print(f"   Grid points: {len(self.common_grid)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False
    
    def _save_pkl_cache(self, cache_file: str) -> bool:
        """
        Save search index to PKL file for Raman-like instant loading
        """
        try:
            import pickle
            import os
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Package all search index data
            cache_data = {
                'pattern_matrix': self.pattern_matrix,
                'common_grid': self.common_grid,
                'mineral_metadata': self.mineral_metadata,
                'index_build_time': self.index_build_time,
                'version': '1.0'  # For future compatibility
            }
            
            print(f"💾 Saving PKL cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
            print(f"✅ PKL cache saved: {file_size_mb:.1f} MB")
            return True
            
        except Exception as e:
            print(f"❌ PKL cache save failed: {e}")
            return False
    
    def _load_pkl_cache(self, cache_file: str) -> bool:
        """
        Load search index from PKL file for instant startup
        """
        try:
            import pickle
            import os
            
            if not os.path.exists(cache_file):
                return False
            
            print(f"📂 Loading PKL cache: {cache_file}")
            start_time = time.time()
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore search index data
            self.pattern_matrix = cache_data['pattern_matrix']
            self.common_grid = cache_data['common_grid']
            self.mineral_metadata = cache_data['mineral_metadata']
            self.index_build_time = cache_data.get('index_build_time', 0)
            self.search_index = True
            
            load_time = time.time() - start_time
            file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
            
            print(f"✅ PKL cache loaded in {load_time:.3f}s ({file_size_mb:.1f} MB)")
            print(f"   Database size: {len(self.mineral_metadata)} patterns")
            print(f"   Matrix shape: {self.pattern_matrix.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ PKL cache load failed: {e}")
            return False
