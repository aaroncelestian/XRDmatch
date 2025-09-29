#!/usr/bin/env python3
"""
Performance comparison: PKL vs SQLite for XRD pattern searching
Demonstrates why your Raman code is so fast and how we can match it
"""

import numpy as np
import time
import sys
import os
import sqlite3
import json
import pickle

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_sqlite_performance(num_queries=10):
    """Test SQLite database performance (current XRD approach)"""
    
    print("🐌 Testing SQLite Performance (Current XRD Approach)")
    print("=" * 55)
    
    try:
        from utils.local_database import LocalCIFDatabase
        db = LocalCIFDatabase()
        
        # Get database statistics
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM diffraction_patterns WHERE wavelength = 1.5406")
        pattern_count = cursor.fetchone()[0]
        
        if pattern_count == 0:
            print("❌ No diffraction patterns found. Run pattern calculation first.")
            return None
        
        print(f"📊 Database: {pattern_count} patterns")
        
        # Test query performance
        times = []
        
        for i in range(num_queries):
            start_time = time.time()
            
            # Simulate what happens during correlation search
            cursor.execute('''
                SELECT m.id, m.mineral_name, m.chemical_formula,
                       dp.two_theta, dp.intensities, dp.d_spacings
                FROM minerals m
                JOIN diffraction_patterns dp ON m.id = dp.mineral_id
                WHERE dp.wavelength = 1.5406
                LIMIT 100
            ''')
            
            patterns = []
            for row in cursor.fetchall():
                mineral_id, name, formula, two_theta_json, intensities_json, d_spacings_json = row
                
                # Parse JSON (expensive!)
                two_theta = np.array(json.loads(two_theta_json))
                intensities = np.array(json.loads(intensities_json))
                
                # Normalize (what we do for each pattern)
                if np.max(intensities) > 0:
                    intensities = intensities / np.max(intensities)
                
                patterns.append({
                    'id': mineral_id,
                    'name': name,
                    'two_theta': two_theta,
                    'intensities': intensities
                })
            
            query_time = time.time() - start_time
            times.append(query_time)
            
            if i == 0:
                print(f"   Loaded {len(patterns)} patterns on first query")
        
        conn.close()
        
        avg_time = np.mean(times)
        print(f"📈 SQLite Results:")
        print(f"   Average query time: {avg_time*1000:.1f}ms")
        print(f"   Per-pattern time: {avg_time*1000/len(patterns):.2f}ms")
        print(f"   Bottlenecks: JSON parsing, disk I/O, row-by-row processing")
        
        return {
            'method': 'SQLite',
            'avg_time_ms': avg_time * 1000,
            'patterns_loaded': len(patterns),
            'patterns_per_second': len(patterns) / avg_time
        }
        
    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
        return None

def test_pkl_performance(num_queries=10):
    """Test PKL performance (Raman approach)"""
    
    print("\n🚀 Testing PKL Performance (Raman Approach)")
    print("=" * 45)
    
    try:
        # Create a simulated PKL database like your Raman code
        pkl_file = "data/test_xrd_database.pkl"
        
        # Check if PKL exists, if not create it
        if not os.path.exists(pkl_file):
            print("📦 Creating PKL database (one-time setup)...")
            create_pkl_database(pkl_file)
        
        # Load entire database into memory (Raman style)
        print("📂 Loading PKL database into memory...")
        load_start = time.time()
        
        with open(pkl_file, 'rb') as f:
            database = pickle.load(f)
        
        load_time = time.time() - load_start
        file_size_mb = os.path.getsize(pkl_file) / 1024 / 1024
        
        print(f"✅ PKL loaded in {load_time:.3f}s ({file_size_mb:.1f} MB)")
        print(f"   Database: {len(database)} patterns in memory")
        
        # Test query performance (everything already in RAM!)
        times = []
        
        for i in range(num_queries):
            start_time = time.time()
            
            # Simulate correlation search - everything is already in memory!
            patterns = []
            for name, data in list(database.items())[:100]:  # First 100 patterns
                # Data is already parsed NumPy arrays!
                two_theta = data['two_theta']
                intensities = data['intensities']
                
                # Normalize (same operation as SQLite)
                if np.max(intensities) > 0:
                    intensities = intensities / np.max(intensities)
                
                patterns.append({
                    'name': name,
                    'two_theta': two_theta,
                    'intensities': intensities
                })
            
            query_time = time.time() - start_time
            times.append(query_time)
        
        avg_time = np.mean(times)
        print(f"📈 PKL Results:")
        print(f"   Average query time: {avg_time*1000:.1f}ms")
        print(f"   Per-pattern time: {avg_time*1000/len(patterns):.2f}ms")
        print(f"   Advantages: No JSON parsing, no disk I/O, pure NumPy operations")
        
        return {
            'method': 'PKL',
            'avg_time_ms': avg_time * 1000,
            'patterns_loaded': len(patterns),
            'patterns_per_second': len(patterns) / avg_time,
            'load_time_s': load_time,
            'file_size_mb': file_size_mb
        }
        
    except Exception as e:
        print(f"❌ PKL test failed: {e}")
        return None

def create_pkl_database(pkl_file):
    """Create a PKL database from SQLite data"""
    
    try:
        from utils.local_database import LocalCIFDatabase
        db = LocalCIFDatabase()
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.id, m.mineral_name, m.chemical_formula,
                   dp.two_theta, dp.intensities, dp.d_spacings
            FROM minerals m
            JOIN diffraction_patterns dp ON m.id = dp.mineral_id
            WHERE dp.wavelength = 1.5406
        ''')
        
        database = {}
        
        for row in cursor.fetchall():
            mineral_id, name, formula, two_theta_json, intensities_json, d_spacings_json = row
            
            # Parse once and store as NumPy arrays (like Raman)
            two_theta = np.array(json.loads(two_theta_json))
            intensities = np.array(json.loads(intensities_json))
            d_spacings = np.array(json.loads(d_spacings_json))
            
            database[name] = {
                'id': mineral_id,
                'formula': formula,
                'two_theta': two_theta,
                'intensities': intensities,
                'd_spacings': d_spacings
            }
        
        conn.close()
        
        # Save as PKL (like Raman database)
        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        with open(pkl_file, 'wb') as f:
            pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"   Created PKL database: {len(database)} patterns")
        
    except Exception as e:
        print(f"   ❌ PKL creation failed: {e}")

def compare_performance():
    """Compare SQLite vs PKL performance"""
    
    print("🔬 XRD Pattern Search: SQLite vs PKL Performance Comparison")
    print("=" * 65)
    
    # Test both approaches
    sqlite_results = test_sqlite_performance(num_queries=5)
    pkl_results = test_pkl_performance(num_queries=5)
    
    if sqlite_results and pkl_results:
        print(f"\n📊 Performance Comparison")
        print("=" * 30)
        
        speedup = sqlite_results['avg_time_ms'] / pkl_results['avg_time_ms']
        
        print(f"SQLite (Current):    {sqlite_results['avg_time_ms']:.1f}ms")
        print(f"PKL (Raman-style):   {pkl_results['avg_time_ms']:.1f}ms")
        print(f"Speedup:             {speedup:.1f}x faster! 🚀")
        
        print(f"\nThroughput:")
        print(f"SQLite:              {sqlite_results['patterns_per_second']:.0f} patterns/second")
        print(f"PKL:                 {pkl_results['patterns_per_second']:.0f} patterns/second")
        
        print(f"\n💡 Why PKL is Faster (Like Your Raman Code):")
        print("• Entire database loaded into RAM at startup")
        print("• No JSON parsing during searches")
        print("• No SQL query overhead")
        print("• No disk I/O during searches")
        print("• Pure NumPy array operations")
        print("• Data already in optimal format")
        
        print(f"\n🔧 Our Hybrid Solution:")
        print("• Build search index once (SQLite → optimized matrix)")
        print("• Save as PKL cache for instant loading")
        print("• Single matrix multiplication for all correlations")
        print("• Best of both worlds: SQLite flexibility + PKL speed")
        
    else:
        print("❌ Could not complete performance comparison")

if __name__ == "__main__":
    compare_performance()
