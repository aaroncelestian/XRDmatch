"""Identify stage — pattern search + phase matching."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QToolBox, QVBoxLayout, QWidget,
)

from utils.fast_pattern_search import FastPatternSearchEngine
from utils.pattern_search import PatternSearchEngine
from utils.multi_phase_analyzer import MultiPhaseAnalyzer
from utils.local_database import LocalCIFDatabase
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
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Quick-add known mineral ---
        add_box = QVBoxLayout()
        add_label = QLabel("Add known mineral")
        add_label.setStyleSheet("font-weight: 600;")
        add_box.addWidget(add_label)
        add_row = QHBoxLayout()
        self.mineral_search = QLineEdit()
        self.mineral_search.setPlaceholderText("e.g. quartz, calcite…")
        self.mineral_search.returnPressed.connect(self.add_mineral_by_name)
        add_row.addWidget(self.mineral_search, 1)
        self.add_mineral_btn = QPushButton("Add")
        self.add_mineral_btn.setToolTip("Search the local database and add a mineral as a candidate")
        self.add_mineral_btn.clicked.connect(self.add_mineral_by_name)
        add_row.addWidget(self.add_mineral_btn)
        add_box.addLayout(add_row)
        layout.addLayout(add_box)

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

        residual_row = QHBoxLayout()
        self.residual_btn = QPushButton("Search Residual")
        self.residual_btn.setToolTip(
            "Keep selected phases, soft-subtract their contribution, and search again. "
            "Unmatched peaks are boosted; overlapping peaks keep partial weight."
        )
        self.residual_btn.clicked.connect(self.search_residual)
        self.residual_btn.setEnabled(False)
        residual_row.addWidget(self.residual_btn)

        self.multi_btn = QPushButton("Multi-Phase Analysis")
        self.multi_btn.clicked.connect(self.start_multi_phase)
        self.multi_btn.setEnabled(False)
        residual_row.addWidget(self.multi_btn)
        layout.addLayout(residual_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel(
            "Search the database, select candidates, then match. "
            "Use Search Residual for additional phases."
        )
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

        self.overlap_keep = QDoubleSpinBox()
        self.overlap_keep.setRange(0.0, 1.0)
        self.overlap_keep.setDecimals(2)
        self.overlap_keep.setSingleStep(0.05)
        self.overlap_keep.setValue(0.35)
        self.overlap_keep.setToolTip(
            "Fraction of explained/overlapping intensity kept in the residual "
            "(0 = hard subtract, 1 = no subtract). Prevents discarding shared peaks."
        )
        adv_form.addRow("Overlap keep:", self.overlap_keep)

        self.unmatched_boost = QDoubleSpinBox()
        self.unmatched_boost.setRange(1.0, 3.0)
        self.unmatched_boost.setDecimals(2)
        self.unmatched_boost.setSingleStep(0.1)
        self.unmatched_boost.setValue(1.50)
        self.unmatched_boost.setToolTip(
            "Intensity multiplier for peaks not explained by selected phases"
        )
        adv_form.addRow("Unmatched boost:", self.unmatched_boost)

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
        has_sel = len(self.workspace.get_selected_candidates()) > 0 or len(self.session.search_candidates) > 0
        self.match_btn.setEnabled(has_sel and self.session.has_peaks())
        self.residual_btn.setEnabled(
            self.session.has_pattern()
            and (len(self.session.selected_phases) > 0 or len(self.session.matched_phases) > 0)
        )
        self.multi_btn.setEnabled(len(self.session.matched_phases) > 1)
        if not can:
            self.status.setText("Load and process a pattern first.")
        elif not self.session.has_peaks():
            self.status.setText("Find peaks in the Peaks tab for best matching (search still works).")
        else:
            self.status.setText("Ready to search. Select candidates manually after search.")

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

        # Prefer exact (case-insensitive) name match when unique
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
        # Show in candidates table (merge into current list view)
        self._append_candidate_result({
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
            "match_score": 1.0,
            "manual_add": True,
        })
        self.match_btn.setEnabled(self.session.has_peaks())
        self.mineral_search.clear()
        name = chosen.get("mineral_name", "phase")
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
        """Merge a manual/search hit into the candidates table and select it."""
        existing = list(getattr(self.workspace, "_candidate_results", []) or [])
        key = (result.get("mineral_name") or "").lower()
        if not any((r.get("mineral_name") or "").lower() == key for r in existing):
            existing.insert(0, result)
        self._search_results = existing
        self.workspace.set_results_candidates(existing)
        # Auto-check only the newly added row (index 0 after insert)
        if self.workspace.results_table.rowCount() > 0:
            cb = self.workspace.results_table.cellWidget(0, 4)
            if cb:
                cb.setChecked(True)

    # --- search / match ---

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

        selected = self.workspace.get_selected_matches()
        if not selected:
            # Fall back to session selection
            selected = list(self.session.selected_phases)
        if not selected:
            QMessageBox.warning(
                self, "No Phases Selected",
                "Select one or more matched phases to keep, then Search Residual.\n\n"
                "Tip: Match candidates first, check the keepers, optionally Clear Unselected.",
            )
            return

        # Persist selection as the kept matched set
        self.session.set_selected_phases(selected)
        if getattr(self.workspace, "_results_mode", None) == "matches":
            # Optionally trim unselected so the kept set is clear
            pass

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
            + (
                f", unmatched peaks {peak_info.get('n_unmatched', '?')}"
                if peak_info else ""
            )
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
        method_idx = self.method_combo.currentIndex()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.search_btn.setEnabled(False)
        self.residual_btn.setEnabled(False)
        if not residual_mode:
            self.status.setText("Searching…")

        try:
            if method_idx == 0:
                results = self._ultra_fast(pattern)
            else:
                results = self._legacy_search(
                    pattern, method_idx, peaks_override=residual_peaks
                )

            results = results or []
            # Always drop already-found minerals (by ID and by name, e.g. no more quartz)
            kept = list(kept_phases or []) or list(self.session.selected_phases)
            if residual_mode or kept:
                before = len(results)
                results = filter_new_hits(results, kept)
                dropped = before - len(results)
            else:
                dropped = 0

            # Legacy name-only exclude_keys still honored if passed
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
            self.residual_btn.setEnabled(len(self.session.selected_phases) > 0 or bool(kept_phases))

            label = "Residual search" if residual_mode else "Search"
            extra = f" (excluded {dropped} already-found)" if dropped else ""
            self.status.setText(
                f"{label}: {len(candidates)} new candidates{extra}. "
                "Select phases manually, then match (or Clear Unselected)."
            )
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

    def _ultra_fast(self, pattern):
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
            min_correlation=self.min_corr.value(),
            max_results=self.max_results.value(),
        )

    def _legacy_search(self, pattern, method_idx, peaks_override=None):
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
            QMessageBox.warning(self, "No Peaks", "Find peaks in the Peaks tab first.")
            return

        phases = self.workspace.get_selected_candidates()
        if not phases:
            QMessageBox.warning(
                self, "No Selection",
                "Select one or more candidates in the table, then Start Matching.\n\n"
                "(Nothing is auto-selected — check the boxes you want.)",
            )
            return

        # Skip candidates that duplicate already-kept minerals (ID or name)
        kept = list(self.session.selected_phases)
        if kept:
            before = len(phases)
            phases = filter_new_hits(phases, kept)
            if not phases:
                QMessageBox.information(
                    self, "Already Found",
                    "All selected candidates are already in your kept phases "
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
        filtered.sort(key=lambda x: x.get("combined_score", x.get("match_score", 0)), reverse=True)

        # Keep previously locked selections + new matches (no auto-check).
        # Never re-add the same mineral ID or mineral name (e.g. more quartz).
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
        # Do not auto-select new matches — preserve prior selections only
        self.session.set_selected_phases(previous)
        self.workspace.set_results_matches(merged, preselect=previous)
        self.multi_btn.setEnabled(len(merged) > 1)
        self.residual_btn.setEnabled(True)
        self.status.setText(
            f"Matched {new_count} new phase(s); {len(previous)} previously selected. "
            "Check phases to keep, then Clear Unselected or Search Residual."
        )
        self.workspace.set_status(f"Matched {new_count} new phases")
        self.workspace.refresh_plot()

    def start_multi_phase(self):
        pattern = self.session.active_pattern()
        # Prefer explicitly selected matches
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
                # Still no auto-select — user chooses
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
        new_rows = []
        for p in phases:
            name = p.get("mineral", p.get("mineral_name", "Unknown"))
            if name.lower() in existing_names:
                continue
            row = {
                "mineral_id": p.get("id") or p.get("amcsd_id"),
                "mineral_name": name,
                "chemical_formula": p.get("formula", p.get("chemical_formula", "")),
                "space_group": p.get("space_group", ""),
                "match_score": 1.0,
                "manual_add": True,
            }
            existing.insert(0, row)
            existing_names.add(name.lower())
            new_rows.append(name.lower())
        self._search_results = existing
        self.workspace.set_results_candidates(existing)
        # Check the newly added rows (they were inserted at the front)
        for i, r in enumerate(existing):
            if (r.get("mineral_name") or "").lower() in new_rows:
                cb = self.workspace.results_table.cellWidget(i, 4)
                if cb:
                    cb.setChecked(True)
        self.match_btn.setEnabled(len(self.session.search_candidates) > 0 and self.session.has_peaks())
        self.status.setText(f"Added {len(phases)} phase(s) from database.")
