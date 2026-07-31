"""Identify stage — pattern search + phase matching."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox, QToolBox,
    QVBoxLayout, QWidget,
)

from utils.fast_pattern_search import FastPatternSearchEngine
from utils.pattern_search import PatternSearchEngine
from utils.multi_phase_analyzer import MultiPhaseAnalyzer
from gui.matching_tab import PhaseMatchingThread


class IdentifyStage(QWidget):
    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self.fast_engine = FastPatternSearchEngine()
        self.search_engine = PatternSearchEngine()
        self.multi_phase_analyzer = MultiPhaseAnalyzer()
        self._match_thread = None
        self._search_results = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Ultra-Fast Correlation",
            "Peak Match",
            "Pearson Correlation",
            "Combined",
            "Ensemble",
        ])
        form.addRow("Search method:", self.method_combo)

        self.min_corr = QDoubleSpinBox()
        self.min_corr.setRange(0.1, 1.0)
        self.min_corr.setDecimals(2)
        self.min_corr.setValue(0.3)
        form.addRow("Min. correlation:", self.min_corr)

        self.max_results = QSpinBox()
        self.max_results.setRange(10, 200)
        self.max_results.setValue(50)
        form.addRow("Max results:", self.max_results)

        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.01, 2.0)
        self.tolerance.setDecimals(2)
        self.tolerance.setValue(0.20)
        form.addRow("Match 2θ tol (°):", self.tolerance)

        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setDecimals(2)
        self.min_score.setValue(0.01)
        form.addRow("Min. match score:", self.min_score)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.search_btn = QPushButton("Start Search")
        self.search_btn.setObjectName("primaryButton")
        self.search_btn.clicked.connect(self.start_search)
        btn_row.addWidget(self.search_btn)

        self.match_btn = QPushButton("Start Matching")
        self.match_btn.setObjectName("primaryButton")
        self.match_btn.clicked.connect(self.start_matching)
        self.match_btn.setEnabled(False)
        btn_row.addWidget(self.match_btn)
        layout.addLayout(btn_row)

        self.multi_btn = QPushButton("Multi-Phase Analysis")
        self.multi_btn.clicked.connect(self.start_multi_phase)
        self.multi_btn.setEnabled(False)
        layout.addWidget(self.multi_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Search the database, then run matching on candidates.")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        toolbox = QToolBox()
        adv = QWidget()
        adv_form = QFormLayout(adv)

        self.peak_tol = QDoubleSpinBox()
        self.peak_tol.setRange(0.05, 1.0)
        self.peak_tol.setDecimals(2)
        self.peak_tol.setValue(0.2)
        self.peak_tol.setSuffix("°")
        adv_form.addRow("Peak tolerance:", self.peak_tol)

        self.grid_res = QDoubleSpinBox()
        self.grid_res.setRange(0.005, 0.1)
        self.grid_res.setDecimals(3)
        self.grid_res.setValue(0.02)
        self.grid_res.setSuffix("°")
        adv_form.addRow("Index grid:", self.grid_res)

        build_btn = QPushButton("Build Search Index")
        build_btn.clicked.connect(self.build_index)
        adv_form.addRow(build_btn)

        self.peak_weight = QDoubleSpinBox()
        self.peak_weight.setRange(0.0, 1.0)
        self.peak_weight.setDecimals(2)
        self.peak_weight.setValue(0.6)
        adv_form.addRow("Peak weight:", self.peak_weight)

        self.corr_weight = QDoubleSpinBox()
        self.corr_weight.setRange(0.0, 1.0)
        self.corr_weight.setDecimals(2)
        self.corr_weight.setValue(0.4)
        adv_form.addRow("Corr. weight:", self.corr_weight)

        self.mp_max = QSpinBox()
        self.mp_max.setRange(1, 10)
        self.mp_max.setValue(5)
        adv_form.addRow("Multi-phase max:", self.mp_max)

        self.mp_delta = QDoubleSpinBox()
        self.mp_delta.setRange(0.1, 50.0)
        self.mp_delta.setDecimals(1)
        self.mp_delta.setValue(2.0)
        self.mp_delta.setSuffix("%")
        adv_form.addRow("Min ΔRwp:", self.mp_delta)

        toolbox.addItem(adv, "Advanced")
        layout.addWidget(toolbox)
        layout.addStretch()

    def on_enter(self):
        can = self.session.has_pattern()
        self.search_btn.setEnabled(can)
        self.match_btn.setEnabled(len(self.session.search_candidates) > 0 and self.session.has_peaks())
        if not can:
            self.status.setText("Load and process a pattern first.")
        elif not self.session.has_peaks():
            self.status.setText("Find peaks in Process for best matching (search still works).")
        else:
            self.status.setText("Ready to search.")

    def build_index(self):
        try:
            self.status.setText("Building search index…")
            ok = self.fast_engine.build_search_index(
                grid_resolution=self.grid_res.value(),
                force_rebuild=True,
            )
            self.status.setText("Index ready." if ok else "Index build failed.")
        except Exception as e:
            QMessageBox.critical(self, "Index Error", str(e))

    def start_search(self):
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Pattern", "Load a pattern first.")
            return

        method_idx = self.method_combo.currentIndex()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.search_btn.setEnabled(False)
        self.status.setText("Searching…")

        try:
            if method_idx == 0:
                results = self._ultra_fast(pattern)
            else:
                results = self._legacy_search(pattern, method_idx)

            self._search_results = results or []
            candidates = [self._result_to_phase(r) for r in self._search_results]
            self.session.set_candidates(candidates)
            self.workspace.set_results_candidates(self._search_results)
            self.match_btn.setEnabled(len(candidates) > 0 and self.session.has_peaks())
            self.status.setText(f"Found {len(candidates)} candidates. Select and match.")
            self.workspace.set_status(f"Search: {len(candidates)} candidates")
            self.workspace.refresh_plot()
        except Exception as e:
            QMessageBox.critical(self, "Search Error", str(e))
            self.status.setText("Search failed.")
        finally:
            self.progress.setVisible(False)
            self.search_btn.setEnabled(True)

    def _ultra_fast(self, pattern):
        if self.fast_engine.search_index is None:
            reply = QMessageBox.question(
                self, "Index Required",
                "Search index not loaded. Build/load it now?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return []
            self.build_index()
            if self.fast_engine.search_index is None:
                return []
        return self.fast_engine.ultra_fast_correlation_search(
            pattern,
            min_correlation=self.min_corr.value(),
            max_results=self.max_results.value(),
        )

    def _legacy_search(self, pattern, method_idx):
        peaks = self.session.peaks
        peak_data = pattern
        if peaks is not None:
            peak_data = {
                "two_theta": np.asarray(peaks["two_theta"]),
                "intensity": np.asarray(peaks["intensity"]),
                "wavelength": pattern.get("wavelength", 1.5406),
            }
        min_c = self.min_corr.value()
        max_r = self.max_results.value()
        ptol = self.peak_tol.value()

        if method_idx == 1:
            return self.search_engine.search_by_peaks(
                peak_data, tolerance=ptol, max_results=max_r
            )
        if method_idx == 2:
            return self.search_engine.search_by_correlation(
                pattern, min_correlation=min_c, max_results=max_r
            )
        if method_idx == 3:
            return self.search_engine.combined_search(
                peak_data if peaks is not None else pattern,
                peak_tolerance=ptol,
                min_correlation=min_c,
                max_results=max_r,
            )
        return self.search_engine.ensemble_search(
            peak_data if peaks is not None else pattern,
            methods=["peaks", "correlation", "ultrafast"],
            max_results=max_r,
            peak_tolerance=ptol,
            min_correlation=min_c,
            fast_search_engine=self.fast_engine,
        )

    @staticmethod
    def _result_to_phase(result: dict) -> dict:
        return {
            "id": result.get("mineral_id"),
            "amcsd_id": result.get("mineral_id"),
            "mineral": result.get("mineral_name", "Unknown"),
            "formula": result.get("chemical_formula", "Unknown"),
            "space_group": result.get("space_group", "Unknown"),
            "cell_a": result.get("cell_a"),
            "cell_b": result.get("cell_b"),
            "cell_c": result.get("cell_c"),
            "cell_alpha": result.get("cell_alpha"),
            "cell_beta": result.get("cell_beta"),
            "cell_gamma": result.get("cell_gamma"),
            "rir": result.get("rir"),
            "local_db": True,
            "search_score": result.get(
                "ensemble_score",
                result.get("combined_score", result.get("correlation", result.get("match_score", 0))),
            ),
        }

    def start_matching(self):
        if not self.session.has_peaks():
            QMessageBox.warning(self, "No Peaks", "Find peaks in the Process stage first.")
            return

        # Prefer checkbox selection from results strip when present
        phases = self.workspace.get_selected_candidates()
        if not phases:
            phases = self.session.search_candidates
        if not phases:
            QMessageBox.warning(
                self, "No Candidates",
                "Run search first (or add phases from Database Manager).",
            )
            return

        # Normalize weights
        pw = self.peak_weight.value()
        cw = self.corr_weight.value()
        total = pw + cw
        if total > 0:
            pw, cw = pw / total, cw / total

        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.match_btn.setEnabled(False)
        self.status.setText("Matching…")

        self._match_thread = PhaseMatchingThread(
            self.session.peaks, phases, self.tolerance.value(), pw, cw
        )
        self._match_thread.matching_complete.connect(self._on_match_done)
        self._match_thread.progress_updated.connect(self.progress.setValue)
        self._match_thread.start()

    def _on_match_done(self, results):
        self.progress.setVisible(False)
        self.match_btn.setEnabled(True)
        min_score = self.min_score.value()
        filtered = [r for r in results if r.get("match_score", 0) >= min_score]
        filtered.sort(key=lambda x: x.get("combined_score", x.get("match_score", 0)), reverse=True)
        self.session.set_matched_phases(filtered)
        # Auto-select top 5 for refine
        selected = filtered[:5]
        self.session.set_selected_phases(selected)
        self.workspace.set_results_matches(filtered)
        self.multi_btn.setEnabled(len(filtered) > 1)
        self.status.setText(f"Matched {len(filtered)} phases (top {len(selected)} selected).")
        self.workspace.set_status(f"Matched {len(filtered)} phases")
        self.workspace.refresh_plot()

    def start_multi_phase(self):
        pattern = self.session.active_pattern()
        results = self.session.matched_phases
        if not pattern or len(results) < 2:
            QMessageBox.warning(self, "Need Matches", "Run matching with multiple phases first.")
            return
        try:
            self.status.setText("Running multi-phase analysis…")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            candidates = [r.get("phase", r) for r in results]
            out = self.multi_phase_analyzer.joint_lebail_phase_identification(
                {
                    "two_theta": pattern["two_theta"],
                    "intensity": pattern["intensity"],
                    "wavelength": self.session.wavelength,
                },
                candidates,
                max_phases=self.mp_max.value(),
                min_delta_rwp=self.mp_delta.value(),
                residual_research=False,
            )
            identified = out.get("identified_phases") or []
            if identified:
                wrapped = []
                for p in identified:
                    if isinstance(p, dict) and "phase" in p:
                        wrapped.append(p)
                    else:
                        wrapped.append({
                            "phase": p if isinstance(p, dict) else {"mineral": str(p)},
                            "match_score": 1.0,
                            "combined_score": 1.0,
                        })
                self.session.set_matched_phases(wrapped)
                self.session.set_selected_phases(wrapped)
                self.workspace.set_results_matches(wrapped)
                self.status.setText(f"Multi-phase kept {len(wrapped)} phase(s).")
            else:
                self.status.setText("Multi-phase analysis returned no phases.")
            self.workspace.refresh_plot()
        except Exception as e:
            QMessageBox.critical(self, "Multi-Phase Error", str(e))
        finally:
            self.progress.setVisible(False)

    def add_phases_from_database(self, phases: list):
        """Called when Database Manager exports phases."""
        self.session.add_candidates(phases)
        self.match_btn.setEnabled(len(self.session.search_candidates) > 0 and self.session.has_peaks())
        self.status.setText(f"Added {len(phases)} phase(s) from database.")
