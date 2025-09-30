"""
IMA Mineral Database Utility
Cross-references mineral names with official IMA data for chemistry and space groups
"""

import csv
import os
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

class IMAMineralDatabase:
    """
    Interface to the IMA (International Mineralogical Association) mineral database
    Provides authoritative mineral names, chemistry, and space groups
    """
    
    def __init__(self, csv_path: str = None):
        """Initialize with path to IMA CSV export"""
        if csv_path is None:
            # Default path relative to utils directory
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'IMA_Export_2025924_163537.csv')
        
        self.csv_path = csv_path
        self.minerals = {}  # mineral_name -> mineral_data
        self.mineral_names_lower = {}  # lowercase name -> official name
        
        if os.path.exists(csv_path):
            self._load_database()
        else:
            print(f"⚠️  IMA database not found at {csv_path}")
    
    def _load_database(self):
        """Load IMA mineral database from CSV"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    mineral_name = row.get('Mineral Name', '').strip()
                    if not mineral_name:
                        continue
                    
                    # Store mineral data
                    self.minerals[mineral_name] = {
                        'name': mineral_name,
                        'plain_name': row.get('Mineral Name (plain)', mineral_name),
                        'chemistry': row.get('IMA Chemistry', row.get('Valence Chemistry', '')),
                        'chemistry_concise': row.get('IMA Chemistry (concise)', ''),
                        'space_group': row.get('Space Groups', ''),
                        'crystal_system': row.get('Crystal Systems', ''),
                        'elements': row.get('Chemistry Elements', ''),
                        'ima_number': row.get('IMA Number', ''),
                        'database_id': row.get('Database ID', ''),
                        'year': row.get('Year First Published', ''),
                        'ima_status': row.get('IMA Status', ''),
                        'structural_group': row.get('Structural Groupname', '')
                    }
                    
                    # Create lowercase lookup
                    self.mineral_names_lower[mineral_name.lower()] = mineral_name
            
            print(f"✅ Loaded {len(self.minerals)} minerals from IMA database")
            
        except Exception as e:
            print(f"❌ Error loading IMA database: {e}")
    
    def get_mineral_info(self, mineral_name: str) -> Optional[Dict]:
        """
        Get mineral information by name (case-insensitive)
        
        Args:
            mineral_name: Mineral name to look up
            
        Returns:
            Dictionary with mineral data or None if not found
        """
        if not mineral_name:
            return None
        
        # Try exact match first
        if mineral_name in self.minerals:
            return self.minerals[mineral_name]
        
        # Try case-insensitive match
        mineral_lower = mineral_name.lower()
        if mineral_lower in self.mineral_names_lower:
            official_name = self.mineral_names_lower[mineral_lower]
            return self.minerals[official_name]
        
        return None
    
    def fuzzy_match_mineral(self, mineral_name: str, threshold: float = 0.8) -> Optional[Tuple[str, float, Dict]]:
        """
        Find best fuzzy match for a mineral name
        
        Args:
            mineral_name: Mineral name to match
            threshold: Minimum similarity score (0-1)
            
        Returns:
            Tuple of (matched_name, similarity_score, mineral_data) or None
        """
        if not mineral_name:
            return None
        
        mineral_lower = mineral_name.lower().strip()
        
        best_match = None
        best_score = 0.0
        
        for official_name_lower, official_name in self.mineral_names_lower.items():
            # Calculate similarity
            similarity = SequenceMatcher(None, mineral_lower, official_name_lower).ratio()
            
            if similarity > best_score:
                best_score = similarity
                best_match = official_name
        
        if best_score >= threshold and best_match:
            return (best_match, best_score, self.minerals[best_match])
        
        return None
    
    def correct_mineral_name(self, possible_name: str, possible_author: str = None) -> Optional[Dict]:
        """
        Correct a mineral name that might be confused with an author name
        
        Args:
            possible_name: String that might be mineral or author name
            possible_author: String that might be author or mineral name
            
        Returns:
            Dictionary with corrected mineral info or None
        """
        # Try the first string as mineral name
        info = self.get_mineral_info(possible_name)
        if info:
            return info
        
        # Try fuzzy match on first string
        fuzzy = self.fuzzy_match_mineral(possible_name, threshold=0.85)
        if fuzzy:
            return fuzzy[2]
        
        # If we have a second string, try that
        if possible_author:
            info = self.get_mineral_info(possible_author)
            if info:
                return info
            
            fuzzy = self.fuzzy_match_mineral(possible_author, threshold=0.85)
            if fuzzy:
                return fuzzy[2]
        
        return None
    
    def search_by_chemistry(self, elements: List[str]) -> List[Dict]:
        """
        Search for minerals containing specific elements
        
        Args:
            elements: List of element symbols (e.g., ['Ca', 'C', 'O'])
            
        Returns:
            List of matching mineral data dictionaries
        """
        results = []
        
        for mineral_data in self.minerals.values():
            mineral_elements = mineral_data.get('elements', '').split()
            
            # Check if all requested elements are present
            if all(elem in mineral_elements for elem in elements):
                results.append(mineral_data)
        
        return results
    
    def search_by_space_group(self, space_group: str) -> List[Dict]:
        """
        Search for minerals with a specific space group
        
        Args:
            space_group: Space group symbol (e.g., 'R-3c', 'P6_3/mmc')
            
        Returns:
            List of matching mineral data dictionaries
        """
        results = []
        space_group_clean = space_group.strip().replace(' ', '')
        
        for mineral_data in self.minerals.values():
            mineral_sg = mineral_data.get('space_group', '')
            
            # Handle multiple space groups separated by |
            if '|' in mineral_sg:
                sgs = [sg.strip().replace(' ', '') for sg in mineral_sg.split('|')]
            else:
                sgs = [mineral_sg.strip().replace(' ', '')]
            
            if space_group_clean in sgs:
                results.append(mineral_data)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        if not self.minerals:
            return {'status': 'Not loaded'}
        
        crystal_systems = {}
        for mineral_data in self.minerals.values():
            system = mineral_data.get('crystal_system', 'Unknown')
            crystal_systems[system] = crystal_systems.get(system, 0) + 1
        
        return {
            'total_minerals': len(self.minerals),
            'crystal_systems': crystal_systems,
            'database_path': self.csv_path
        }


# Global instance for easy access
_ima_db_instance = None

def get_ima_database() -> IMAMineralDatabase:
    """Get or create global IMA database instance"""
    global _ima_db_instance
    if _ima_db_instance is None:
        _ima_db_instance = IMAMineralDatabase()
    return _ima_db_instance
