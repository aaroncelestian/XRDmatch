"""
Shared analysis session — single source of truth for the guided workspace.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from utils import emphasis


class AnalysisSession(QObject):
    """Holds pattern, peaks, candidates, matches, and refinement state."""

    pattern_changed = pyqtSignal()
    peaks_changed = pyqtSignal()
    candidates_changed = pyqtSignal()
    matches_changed = pyqtSignal()
    refinement_changed = pyqtSignal()
    emphasis_changed = pyqtSignal()
    stage_status_changed = pyqtSignal()  # rail enable/complete refresh

    def __init__(self, parent=None):
        super().__init__(parent)
        self.wavelength: float = 1.5406
        self.raw_pattern: Optional[Dict[str, Any]] = None
        self.processed_pattern: Optional[Dict[str, Any]] = None
        self.background: Optional[Any] = None
        self.peaks: Optional[Dict[str, Any]] = None
        self.search_candidates: List[Dict[str, Any]] = []
        self.matched_phases: List[Dict[str, Any]] = []
        self.selected_phases: List[Dict[str, Any]] = []
        self.lebail_results: Optional[Dict[str, Any]] = None
        self.rir_results: Optional[Dict[str, Any]] = None
        self.file_path: Optional[str] = None
        # 2theta spans the user wants search/match to prioritise
        self.emphasis_regions: List[Dict[str, float]] = []

    # --- stage completion helpers ---

    def has_pattern(self) -> bool:
        return self.raw_pattern is not None

    def has_peaks(self) -> bool:
        return self.peaks is not None and len(self.peaks.get("two_theta", [])) > 0

    def has_candidates(self) -> bool:
        return len(self.search_candidates) > 0 or len(self.matched_phases) > 0

    def has_matches(self) -> bool:
        return len(self.matched_phases) > 0

    def active_pattern(self) -> Optional[Dict[str, Any]]:
        """Prefer processed pattern when available."""
        return self.processed_pattern or self.raw_pattern

    def set_raw_pattern(self, pattern: Dict[str, Any]) -> None:
        self.raw_pattern = pattern
        self.processed_pattern = None
        self.background = None
        self.peaks = None
        self.search_candidates = []
        self.matched_phases = []
        self.selected_phases = []
        self.lebail_results = None
        self.rir_results = None
        self.emphasis_regions = []
        self.file_path = pattern.get("file_path")
        if "wavelength" in pattern:
            self.wavelength = float(pattern["wavelength"])
        self.emphasis_changed.emit()
        self.pattern_changed.emit()
        self.peaks_changed.emit()
        self.candidates_changed.emit()
        self.matches_changed.emit()
        self.refinement_changed.emit()
        self.stage_status_changed.emit()

    def set_processed_pattern(self, pattern: Dict[str, Any], background=None) -> None:
        self.processed_pattern = pattern
        self.background = background
        self.pattern_changed.emit()
        self.stage_status_changed.emit()

    def set_peaks(self, peaks: Optional[Dict[str, Any]]) -> None:
        self.peaks = peaks
        self.peaks_changed.emit()
        self.stage_status_changed.emit()

    def set_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        self.search_candidates = list(candidates)
        self.candidates_changed.emit()
        self.stage_status_changed.emit()

    def add_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        existing_ids = {
            c.get("amcsd_id") or c.get("mineral") for c in self.search_candidates
        }
        for c in candidates:
            key = c.get("amcsd_id") or c.get("mineral")
            if key not in existing_ids:
                self.search_candidates.append(c)
                existing_ids.add(key)
        self.candidates_changed.emit()
        self.stage_status_changed.emit()

    def set_matched_phases(self, results: List[Dict[str, Any]]) -> None:
        self.matched_phases = list(results)
        self.matches_changed.emit()
        self.stage_status_changed.emit()

    def set_selected_phases(self, phases: List[Dict[str, Any]]) -> None:
        self.selected_phases = list(phases)
        self.matches_changed.emit()

    def set_lebail_results(self, results: Optional[Dict[str, Any]]) -> None:
        self.lebail_results = results
        self.refinement_changed.emit()
        self.stage_status_changed.emit()

    def set_rir_results(self, results: Optional[Dict[str, Any]]) -> None:
        self.rir_results = results
        self.refinement_changed.emit()

    # --- emphasised regions ---

    def set_emphasis_regions(self, regions: List[Dict[str, float]]) -> None:
        self.emphasis_regions = emphasis.normalize(regions)
        self.emphasis_changed.emit()

    def add_emphasis_region(self, lo: float, hi: float, weight: float) -> None:
        self.set_emphasis_regions(
            self.emphasis_regions + [emphasis.make_region(lo, hi, weight)]
        )

    def remove_emphasis_at(self, two_theta: float) -> bool:
        """Drop the region under a 2theta. True when one was removed."""
        hit = emphasis.region_at(self.emphasis_regions, two_theta)
        if hit is None:
            return False
        self.set_emphasis_regions([r for r in self.emphasis_regions if r is not hit])
        return True

    def clear_emphasis_regions(self) -> None:
        if self.emphasis_regions:
            self.set_emphasis_regions([])

    def set_wavelength(self, wavelength: float) -> None:
        self.wavelength = float(wavelength)
        if self.raw_pattern is not None:
            self.raw_pattern["wavelength"] = self.wavelength
        if self.processed_pattern is not None:
            self.processed_pattern["wavelength"] = self.wavelength
        self.pattern_changed.emit()
