"""Identify stage — pattern search + phase matching."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from utils.fast_pattern_search import FastPatternSearchEngine
from utils.pattern_search import PatternSearchEngine
from utils.multi_phase_analyzer import MultiPhaseAnalyzer
from utils.local_database import LocalCIFDatabase
from utils.fingerprint_search import fingerprint_score, rank_by_fingerprint
from utils.residual_search import (
    build_residual_pattern,
    build_residual_peaks,
    filter_new_hits,
    is_excluded_hit,
    exclusion_sets,
    mineral_ids,
    mineral_key,
)
from gui.matching_tab import PhaseMatchingThread
from gui.widgets.control_bar import ControlRow, OptionsDialog


SEARCH_METHODS = [
    ("Fingerprint (mixtures)", "fingerprint"),
    ("Ultra-Fast Correlation", "ultrafast"),
    ("Peak Match", "peaks"),
    ("Pearson Correlation", "correlation"),
    ("Combined", "combined"),
    ("Ensemble", "ensemble"),
]


class IdentifyStage(QWidget):
    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self.fast_engine = FastPatternSearchEngine()
        self.search_engine = PatternSearchEngine()
        self.multi_phase_analyzer = MultiPhaseAnalyzer()
        self.local_db = LocalCIFDatabase()
        self._match_thread = None
        self._search_results = []
        self._theo_cache = {}
        self._options = None

        self.control_panel = self._build_controls()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

    # --- UI ---

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        actions = ControlRow()
        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("primaryButton")
        self.search_btn.setToolTip("Search the database for candidate phases")
        self.search_btn.clicked.connect(self.start_search)
        actions.add_widget(self.search_btn)

        self.match_btn = QPushButton("Match Selected")
        self.match_btn.setObjectName("primaryButton")
        self.match_btn.setToolTip("Run peak matching on the checked candidates")
        self.match_btn.clicked.connect(self.start_matching)
        self.match_btn.setEnabled(False)
        actions.add_widget(self.match_btn)

        self.residual_btn = QPushButton("Search Residual")
        self.residual_btn.setToolTip(
            "Keep selected phases, soft-subtract their contribution, and search again. "
            "Unmatched peaks are boosted; overlapping peaks keep partial weight."
        )
        self.residual_btn.clicked.connect(self.search_residual)
        self.residual_btn.setEnabled(False)
        actions.add_widget(self.residual_btn)

        self.multi_btn = QPushButton("Multi-Phase")
        self.multi_btn.setToolTip("Joint Le Bail accept/reject over the selected phases")
        self.multi_btn.clicked.connect(self.start_multi_phase)
        self.multi_btn.setEnabled(False)
        actions.add_widget(self.multi_btn)
        actions.add_separator()

        self.mineral_search = QLineEdit()
        self.mineral_search.setPlaceholderText("Add known mineral — e.g. quartz")
        self.mineral_search.setMinimumWidth(180)
        self.mineral_search.returnPressed.connect(self.add_mineral_by_name)
        actions.add_widget(self.mineral_search)
        self.add_mineral_btn = QPushButton("Add")
        self.add_mineral_btn.setToolTip("Search the local database and add a mineral as a candidate")
        self.add_mineral_btn.clicked.connect(self.add_mineral_by_name)
        actions.add_widget(self.add_mineral_btn)
        actions.add_stretch()
        layout.addWidget(actions)

        params = ControlRow()
        self.method_combo = QComboBox()
        for label, key in SEARCH_METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.setToolTip(
            "Fingerprint scores each candidate on its own strong lines, so minor "
            "phases in a mixture are not penalized for unexplained peaks."
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        params.add_field("Method:", self.method_combo, 170)

        self.min_corr = QDoubleSpinBox()
        self.min_corr.setRange(0.01, 1.0)
        self.min_corr.setDecimals(2)
        self.min_corr.setSingleStep(0.05)
        self.min_corr.setValue(0.30)
        params.add_field("Min corr:", self.min_corr, 74)

        self.fp_min_score = QDoubleSpinBox()
        self.fp_min_score.setRange(0.0, 1.0)
        self.fp_min_score.setDecimals(2)
        self.fp_min_score.setSingleStep(0.05)
        self.fp_min_score.setValue(0.40)
        self.fp_min_score.setToolTip(
            "Minimum fraction of a candidate's strong lines that must be present"
        )
        params.add_field("Min fingerprint:", self.fp_min_score, 74)

        self.max_results = QSpinBox()
        self.max_results.setRange(10, 500)
        self.max_results.setValue(50)
        params.add_field("Max results:", self.max_results, 74)

        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.01, 2.0)
        self.tolerance.setDecimals(2)
        self.tolerance.setValue(0.20)
        self.tolerance.setSuffix("°")
        params.add_field("2θ tol:", self.tolerance, 78)

        options_btn = QPushButton("Options…")
        options_btn.setToolTip("Fingerprint, residual, weighting, and multi-phase settings")
        options_btn.clicked.connect(self._show_options)
        params.add_widget(options_btn)
        params.add_stretch()
        self.actions_row = params  # workspace appends table actions here
        layout.addWidget(params)

        status_row = ControlRow(margins=(8, 0, 8, 4))
        self.status = QLabel("Load a pattern, find peaks, then search.")
        self.status.setObjectName("mutedLabel")
        status_row.add_widget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(180)
        status_row.add_widget(self.progress)
        layout.addWidget(status_row)

        self._build_option_widgets()
        self._on_method_changed()
        return panel

    def _build_option_widgets(self):
        """Advanced parameters — shown in the Options popup, owned here."""
        self.fp_n_peaks = QSpinBox()
        self.fp_n_peaks.setRange(3, 30)
        self.fp_n_peaks.setValue(10)
        self.fp_n_peaks.setToolTip("How many of the candidate's strongest lines define its fingerprint")

        self.fp_min_rel = QDoubleSpinBox()
        self.fp_min_rel.setRange(0.5, 50.0)
        self.fp_min_rel.setDecimals(1)
        self.fp_min_rel.setValue(5.0)
        self.fp_min_rel.setSuffix("%")
        self.fp_min_rel.setToolTip("Ignore reference lines weaker than this fraction of the phase maximum")

        self.fp_min_found = QSpinBox()
        self.fp_min_found.setRange(1, 20)
        self.fp_min_found.setValue(3)
        self.fp_min_found.setToolTip("Minimum number of fingerprint lines that must be found")

        self.fp_require_top = QCheckBox("Require strongest line present")
        self.fp_require_top.setChecked(False)
        self.fp_require_top.setToolTip(
            "Reject a candidate outright when its most intense line is missing"
        )

        self.pool_min_corr = QDoubleSpinBox()
        self.pool_min_corr.setRange(0.01, 1.0)
        self.pool_min_corr.setDecimals(2)
        self.pool_min_corr.setSingleStep(0.05)
        self.pool_min_corr.setValue(0.10)
        self.pool_min_corr.setToolTip(
            "Correlation floor for the candidate pool that fingerprint scoring reranks. "
            "Keep it low so minor phases survive to the rescoring step."
        )

        self.pool_size = QSpinBox()
        self.pool_size.setRange(50, 3000)
        self.pool_size.setSingleStep(50)
        self.pool_size.setValue(400)
        self.pool_size.setToolTip("How many candidates to pull before fingerprint rescoring")

        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setDecimals(2)
        self.min_score.setValue(0.01)

        self.peak_tol = QDoubleSpinBox()
        self.peak_tol.setRange(0.05, 1.0)
        self.peak_tol.setDecimals(2)
        self.peak_tol.setValue(0.2)
        self.peak_tol.setSuffix("°")

        self.peak_weight = QDoubleSpinBox()
        self.peak_weight.setRange(0.0, 1.0)
        self.peak_weight.setDecimals(2)
        self.peak_weight.setValue(0.6)

        self.corr_weight = QDoubleSpinBox()
        self.corr_weight.setRange(0.0, 1.0)
        self.corr_weight.setDecimals(2)
        self.corr_weight.setValue(0.4)

        self.overlap_keep = QDoubleSpinBox()
        self.overlap_keep.setRange(0.0, 1.0)
        self.overlap_keep.setDecimals(2)
        self.overlap_keep.setSingleStep(0.05)
        self.overlap_keep.setValue(0.35)
        self.overlap_keep.setToolTip(
            "Fraction of explained/overlapping intensity kept in the residual "
            "(0 = hard subtract, 1 = no subtract). Prevents discarding shared peaks."
        )

        self.unmatched_boost = QDoubleSpinBox()
        self.unmatched_boost.setRange(1.0, 3.0)
        self.unmatched_boost.setDecimals(2)
        self.unmatched_boost.setSingleStep(0.1)
        self.unmatched_boost.setValue(1.50)
        self.unmatched_boost.setToolTip(
            "Intensity multiplier for peaks not explained by selected phases"
        )

        self.mp_max = QSpinBox()
        self.mp_max.setRange(1, 10)
        self.mp_max.setValue(5)

        self.mp_delta = QDoubleSpinBox()
        self.mp_delta.setRange(0.1, 50.0)
        self.mp_delta.setDecimals(1)
        self.mp_delta.setValue(2.0)
        self.mp_delta.setSuffix("%")

    def _show_options(self):
        if self._options is None:
            dlg = OptionsDialog("Phase Search Options", self.workspace.window())
            dlg.add_heading("Fingerprint")
            dlg.add_row("Fingerprint lines:", self.fp_n_peaks)
            dlg.add_row("Min line intensity:", self.fp_min_rel)
            dlg.add_row("Min lines found:", self.fp_min_found)
            dlg.add_row("", self.fp_require_top)
            dlg.add_row("Pool min corr:", self.pool_min_corr)
            dlg.add_row("Pool size:", self.pool_size)

            dlg.add_heading("Matching")
            dlg.add_row("Min match score:", self.min_score)
            dlg.add_row("Peak tolerance:", self.peak_tol)
            dlg.add_row("Peak weight:", self.peak_weight)
            dlg.add_row("Corr. weight:", self.corr_weight)

            dlg.add_heading("Residual & multi-phase")
            dlg.add_row("Overlap keep:", self.overlap_keep)
            dlg.add_row("Unmatched boost:", self.unmatched_boost)
            dlg.add_row("Multi-phase max:", self.mp_max)
            dlg.add_row("Min ΔRwp:", self.mp_delta)
            self._options = dlg
        self._options.show_centered()

    def _method_key(self) -> str:
        return self.method_combo.currentData() or "fingerprint"

    def _on_method_changed(self, *_args):
        is_fp = self._method_key() == "fingerprint"
        self.fp_min_score.setEnabled(is_fp)

    # --- state ---

    def reset_results(self):
        self._search_results = []

    def on_enter(self):
        can = self.session.has_pattern()
        self.search_btn.setEnabled(can)
        has_candidates = (
            len(self.workspace.get_selected_candidates()) > 0
            or len(self.session.search_candidates) > 0
        )
        self.match_btn.setEnabled(has_candidates and self.session.has_peaks())
        self.residual_btn.setEnabled(
            can and (len(self.session.selected_phases) > 0 or len(self.session.matched_phases) > 0)
        )
        self.multi_btn.setEnabled(len(self.session.matched_phases) > 1)
        if not can:
            self.status.setText("Load and process a pattern first.")
        elif not self.session.has_peaks():
            self.status.setText("Find peaks in the Peaks tab — fingerprint search needs them.")
        else:
            self.status.setText("Ready to search. Check the candidates you want, then match.")

    def _measured_range(self):
        """Measured 2θ span — reference lines outside it cannot be expected."""
        pattern = self.session.active_pattern()
        if not pattern:
            return None
        tt = np.asarray(pattern["two_theta"], dtype=float)
        if len(tt) == 0:
            return None
        return float(np.min(tt)), float(np.max(tt))

    # --- theoretical peak access (shared with preview / details) ---

    def theoretical_peaks_for(self, result: dict):
        """Reference peaks for a search hit, match result, or phase dict."""
        if not isinstance(result, dict):
            return None
        theo = result.get("theoretical_peaks")
        if theo and len(theo.get("two_theta", [])) > 0:
            return theo

        phase = result.get("phase", result)
        mineral_id = (
            result.get("mineral_id")
            or (phase.get("id") if isinstance(phase, dict) else None)
            or result.get("id")
        )
        if mineral_id is None:
            return None
        wl = round(float(self.session.wavelength), 4)
        key = (str(mineral_id), wl)
        if key in self._theo_cache:
            return self._theo_cache[key]
        try:
            pattern = self.local_db.get_diffraction_pattern(int(mineral_id), wl)
        except Exception:
            pattern = None
        self._theo_cache[key] = pattern
        return pattern

    # --- mineral quick-add ---

    def add_mineral_by_name(self):
        query = self.mineral_search.text().strip()
        if len(query) < 2:
            QMessageBox.information(self, "Add Mineral", "Type at least 2 characters.")
            return
        try:
            hits = self.local_db.search_by_mineral_name(query, limit=40)
        except Exception as e:
            QMessageBox.critical(self, "Database Error", str(e))
            return
        if not hits:
            QMessageBox.information(self, "No Matches", f"No minerals matching “{query}”.")
            return

        exact = [h for h in hits if str(h.get("mineral_name", "")).lower() == query.lower()]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(hits) == 1:
            chosen = hits[0]
        else:
            chosen = self._pick_mineral_dialog(hits, query)
            if chosen is None:
                return

        phase = self._db_row_to_phase(chosen)
        self.session.add_candidates([phase])
        row = {
            "mineral_id": chosen.get("id"),
            "mineral_name": chosen.get("mineral_name"),
            "chemical_formula": chosen.get("chemical_formula"),
            "space_group": chosen.get("space_group"),
            "cell_a": chosen.get("cell_a"),
            "cell_b": chosen.get("cell_b"),
            "cell_c": chosen.get("cell_c"),
            "cell_alpha": chosen.get("cell_alpha"),
            "cell_beta": chosen.get("cell_beta"),
            "cell_gamma": chosen.get("cell_gamma"),
            "rir": chosen.get("rir"),
            "match_score": 1.0,
            "manual_add": True,
        }
        # Report how well the known mineral actually fits the peaks
        if self.session.has_peaks():
            theo = self.theoretical_peaks_for(row)
            if theo:
                info = fingerprint_score(
                    self.session.peaks["two_theta"],
                    self.session.peaks["intensity"],
                    theo.get("two_theta", []),
                    theo.get("intensity", []),
                    tolerance=self.tolerance.value(),
                    n_peaks=self.fp_n_peaks.value(),
                    min_rel_intensity=self.fp_min_rel.value(),
                    exp_range=self._measured_range(),
                )
                row["fingerprint"] = info
                row["fingerprint_score"] = info["score"]

        self._append_candidate_result(row)
        self.match_btn.setEnabled(self.session.has_peaks())
        self.mineral_search.clear()
        name = chosen.get("mineral_name", "phase")
        fp = row.get("fingerprint")
        if fp:
            self.status.setText(
                f"Added {name} — {fp['n_found']}/{fp['n_expected']} strong lines present "
                f"(fingerprint {fp['score']:.2f})."
            )
        else:
            self.status.setText(f"Added {name}. Select it and run matching when ready.")
        self.workspace.set_status(f"Added mineral: {name}")

    def _pick_mineral_dialog(self, hits: list, query: str):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Choose mineral — “{query}”")
        dlg.resize(480, 360)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"{len(hits)} matches — select one to add:"))
        lst = QListWidget()
        for h in hits:
            text = (
                f"{h.get('mineral_name', '?')}  ·  "
                f"{h.get('chemical_formula', '')}  ·  "
                f"{h.get('space_group', '')}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, h)
            lst.addItem(item)
        lst.setCurrentRow(0)
        lst.itemDoubleClicked.connect(lambda *_: dlg.accept())
        lay.addWidget(lst)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return None
        item = lst.currentItem()
        return item.data(Qt.UserRole) if item else None

    @staticmethod
    def _db_row_to_phase(mineral: dict) -> dict:
        return {
            "id": mineral.get("id"),
            "amcsd_id": mineral.get("amcsd_id") or mineral.get("id"),
            "mineral": mineral.get("mineral_name", "Unknown"),
            "formula": mineral.get("chemical_formula", "Unknown"),
            "space_group": mineral.get("space_group", "Unknown"),
            "cell_a": mineral.get("cell_a"),
            "cell_b": mineral.get("cell_b"),
            "cell_c": mineral.get("cell_c"),
            "cell_alpha": mineral.get("cell_alpha"),
            "cell_beta": mineral.get("cell_beta"),
            "cell_gamma": mineral.get("cell_gamma"),
            "rir": mineral.get("rir"),
            "cif_content": mineral.get("cif_content"),
            "local_db": True,
            "manual_add": True,
            "search_score": 1.0,
        }

    def _append_candidate_result(self, result: dict):
        """Merge a manual hit into the candidates table and check it."""
        existing = list(getattr(self.workspace, "_candidate_results", []) or [])
        key = (result.get("mineral_name") or "").lower()
        if not any((r.get("mineral_name") or "").lower() == key for r in existing):
            existing.insert(0, result)
        self._search_results = existing
        self.workspace.set_results_candidates(existing)
        self.workspace.check_candidate_rows([key])

    # --- search ---

    def start_search(self):
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Pattern", "Load a pattern first.")
            return
        self._run_search(pattern, residual_mode=False)

    def search_residual(self):
        """Search on soft residual after locking selected matched phases."""
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Pattern", "Load a pattern first.")
            return

        selected = self.workspace.get_selected_matches() or list(self.session.selected_phases)
        if not selected:
            QMessageBox.warning(
                self, "No Phases Selected",
                "Select one or more matched phases to keep, then Search Residual.\n\n"
                "Tip: Match candidates first, check the keepers, optionally Clear Unselected.",
            )
            return

        self.session.set_selected_phases(selected)
        overlap_keep = self.overlap_keep.value()
        boost = self.unmatched_boost.value()
        tol = self.tolerance.value()

        residual_pattern, pinfo = build_residual_pattern(
            pattern, selected, overlap_keep=overlap_keep
        )

        residual_peaks = None
        peak_info = {}
        if self.session.has_peaks():
            residual_peaks, peak_info = build_residual_peaks(
                self.session.peaks,
                selected,
                tol,
                unmatched_boost=boost,
                overlap_keep=overlap_keep,
            )

        self.status.setText(
            f"Residual search… remaining intensity {pinfo['fraction_remaining']*100:.0f}%"
            + (f", unmatched peaks {peak_info.get('n_unmatched', '?')}" if peak_info else "")
        )
        self._run_search(
            residual_pattern,
            residual_mode=True,
            residual_peaks=residual_peaks,
            kept_phases=selected,
        )

    def _run_search(
        self,
        pattern,
        *,
        residual_mode: bool = False,
        residual_peaks=None,
        exclude_keys=None,
        kept_phases=None,
    ):
        method = self._method_key()
        if method == "fingerprint" and not self.session.has_peaks():
            QMessageBox.warning(
                self, "Peaks Required",
                "Fingerprint search compares peak positions.\n\n"
                "Find peaks in the Peaks tab first, or pick another search method.",
            )
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.search_btn.setEnabled(False)
        self.residual_btn.setEnabled(False)
        if not residual_mode:
            self.status.setText("Searching…")

        try:
            if method == "fingerprint":
                results = self._fingerprint_search(pattern, residual_peaks)
            elif method == "ultrafast":
                results = self._ultra_fast(pattern, self.min_corr.value(), self.max_results.value())
            else:
                results = self._legacy_search(pattern, method, peaks_override=residual_peaks)

            results = results or []
            kept = list(kept_phases or []) or list(self.session.selected_phases)
            dropped = 0
            if residual_mode or kept:
                before = len(results)
                results = filter_new_hits(results, kept)
                dropped = before - len(results)

            if exclude_keys:
                results = [
                    r for r in results
                    if mineral_key({"mineral_name": r.get("mineral_name")}) not in exclude_keys
                ]

            self._search_results = results
            candidates = [self._result_to_phase(r) for r in results]
            self.session.set_candidates(candidates)
            self.workspace.set_results_candidates(results)
            self.match_btn.setEnabled(len(candidates) > 0 and self.session.has_peaks())

            label = "Residual search" if residual_mode else "Search"
            extra = f" (excluded {dropped} already-found)" if dropped else ""
            hint = (
                " Click a row to preview its peaks; arrow keys step through."
                if candidates else " Try a lower Min fingerprint or a wider 2θ tolerance."
            )
            self.status.setText(f"{label}: {len(candidates)} candidates{extra}.{hint}")
            self.workspace.set_status(f"{label}: {len(candidates)} candidates")
            self.workspace.refresh_plot()
        except Exception as e:
            QMessageBox.critical(self, "Search Error", str(e))
            self.status.setText("Search failed.")
        finally:
            self.progress.setVisible(False)
            self.search_btn.setEnabled(True)
            self.residual_btn.setEnabled(
                self.session.has_pattern()
                and (len(self.session.selected_phases) > 0 or len(self.session.matched_phases) > 0)
            )

    def _fingerprint_search(self, pattern, residual_peaks=None):
        """Broad candidate pool, then rerank on each candidate's own strong lines."""
        exp_peaks = residual_peaks if residual_peaks is not None else self.session.peaks
        pool_size = self.pool_size.value()
        pool = []

        if self.fast_engine.search_index is not None:
            pool = self.fast_engine.ultra_fast_correlation_search(
                pattern,
                min_correlation=self.pool_min_corr.value(),
                max_results=pool_size,
            ) or []
        if not pool:
            peak_data = {
                "two_theta": np.asarray(exp_peaks["two_theta"]),
                "intensity": np.asarray(exp_peaks["intensity"]),
                "wavelength": pattern.get("wavelength", self.session.wavelength),
            }
            pool = self.search_engine.search_by_peaks(
                peak_data, tolerance=self.peak_tol.value(), max_results=pool_size
            ) or []

        ranked = rank_by_fingerprint(
            pool,
            exp_peaks,
            self.theoretical_peaks_for,
            tolerance=self.tolerance.value(),
            n_peaks=self.fp_n_peaks.value(),
            min_rel_intensity=self.fp_min_rel.value(),
            min_score=self.fp_min_score.value(),
            min_found=self.fp_min_found.value(),
            require_top_peak=self.fp_require_top.isChecked(),
            max_results=self.max_results.value(),
            exp_range=self._measured_range(),
        )
        print(
            f"Fingerprint search: pool={len(pool)} → {len(ranked)} candidates "
            f"(min score {self.fp_min_score.value():.2f})"
        )
        return ranked

    def _ultra_fast(self, pattern, min_correlation, max_results):
        if self.fast_engine.search_index is None:
            QMessageBox.warning(
                self,
                "Search Index Required",
                "No search index is loaded.\n\n"
                "Open Database and build/load the search index there, then try again.",
            )
            return []
        return self.fast_engine.ultra_fast_correlation_search(
            pattern,
            min_correlation=min_correlation,
            max_results=max_results,
        )

    def _legacy_search(self, pattern, method: str, peaks_override=None):
        peaks = peaks_override if peaks_override is not None else self.session.peaks
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

        if method == "peaks":
            return self.search_engine.search_by_peaks(
                peak_data, tolerance=ptol, max_results=max_r
            )
        if method == "correlation":
            return self.search_engine.search_by_correlation(
                pattern, min_correlation=min_c, max_results=max_r
            )
        if method == "combined":
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
                "fingerprint_score",
                result.get(
                    "ensemble_score",
                    result.get("combined_score", result.get("correlation", result.get("match_score", 0))),
                ),
            ),
        }

    # --- matching ---

    def start_matching(self):
        if not self.session.has_peaks():
            QMessageBox.warning(self, "No Peaks", "Find peaks in the Peaks tab first.")
            return

        phases = self.workspace.get_selected_candidates()
        if not phases:
            QMessageBox.warning(
                self, "No Selection",
                "Check one or more candidates in the table, then Match Selected.\n\n"
                "(Nothing is auto-selected.)",
            )
            return

        kept = list(self.session.selected_phases)
        if kept:
            before = len(phases)
            phases = filter_new_hits(phases, kept)
            if not phases:
                QMessageBox.information(
                    self, "Already Found",
                    "All checked candidates are already in your kept phases "
                    "(same mineral ID or name).",
                )
                return
            if len(phases) < before:
                self.status.setText(
                    f"Matching {len(phases)} new candidate(s) "
                    f"(skipped {before - len(phases)} already-found)."
                )

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
        for r in filtered:
            self._attach_fingerprint(r)
        filtered.sort(
            key=lambda x: x.get("fingerprint_score", x.get("combined_score", x.get("match_score", 0))),
            reverse=True,
        )

        # Keep previously locked selections; never re-add the same mineral twice
        previous = list(self.session.selected_phases)
        excl_ids, excl_names = exclusion_sets(previous)
        merged = list(previous)
        new_count = 0
        for r in filtered:
            if is_excluded_hit(r, excl_ids, excl_names):
                continue
            merged.append(r)
            excl_ids |= mineral_ids(r)
            name = mineral_key(r)
            if name:
                excl_names.add(name)
            new_count += 1

        self.session.set_matched_phases(merged)
        self.session.set_selected_phases(previous)
        self.workspace.set_results_matches(merged, preselect=previous)
        self.multi_btn.setEnabled(len(merged) > 1)
        self.residual_btn.setEnabled(True)
        self.status.setText(
            f"Matched {new_count} new phase(s); {len(previous)} previously kept. "
            "Check phases to keep, then Clear Unselected or Search Residual."
        )
        self.workspace.set_status(f"Matched {new_count} new phases")
        self.workspace.refresh_plot()

    def _attach_fingerprint(self, result: dict):
        """Add fingerprint stats so mixtures rank on their own lines."""
        if "fingerprint" in result or not self.session.has_peaks():
            return
        theo = self.theoretical_peaks_for(result)
        if not theo:
            return
        info = fingerprint_score(
            self.session.peaks["two_theta"],
            self.session.peaks["intensity"],
            theo.get("two_theta", []),
            theo.get("intensity", []),
            tolerance=self.tolerance.value(),
            n_peaks=self.fp_n_peaks.value(),
            min_rel_intensity=self.fp_min_rel.value(),
            exp_range=self._measured_range(),
        )
        result["fingerprint"] = info
        result["fingerprint_score"] = info["score"]

    def start_multi_phase(self):
        pattern = self.session.active_pattern()
        results = self.workspace.get_selected_matches() or self.session.matched_phases
        if not pattern or len(results) < 2:
            QMessageBox.warning(
                self, "Need Matches",
                "Select at least two matched phases (or run matching with multiple phases).",
            )
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
                self.session.set_selected_phases([])
                self.workspace.set_results_matches(wrapped, preselect=[])
                self.status.setText(
                    f"Multi-phase kept {len(wrapped)} phase(s). Select which to use."
                )
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
        existing = list(getattr(self.workspace, "_candidate_results", []) or [])
        existing_names = {(r.get("mineral_name") or "").lower() for r in existing}
        new_keys = []
        for p in phases:
            name = p.get("mineral", p.get("mineral_name", "Unknown"))
            if name.lower() in existing_names:
                continue
            existing.insert(0, {
                "mineral_id": p.get("id") or p.get("amcsd_id"),
                "mineral_name": name,
                "chemical_formula": p.get("formula", p.get("chemical_formula", "")),
                "space_group": p.get("space_group", ""),
                "match_score": 1.0,
                "manual_add": True,
            })
            existing_names.add(name.lower())
            new_keys.append(name.lower())
        self._search_results = existing
        self.workspace.set_results_candidates(existing)
        self.workspace.check_candidate_rows(new_keys)
        self.match_btn.setEnabled(
            len(self.session.search_candidates) > 0 and self.session.has_peaks()
        )
        self.status.setText(f"Added {len(phases)} phase(s) from database.")
