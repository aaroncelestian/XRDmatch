"""Analysis workspace — file browser, plot, and Background / Peaks / Phases tabs."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QMenu,
    QMessageBox, QPushButton, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from matplotlib_config import apply_plot_style, get_plot_palette
from gui.theme import get_current_mode
from gui.widgets.file_browser import FileBrowser
from gui.widgets.plot_host import create_plot_host
from gui.stages import ProcessStage, IdentifyStage
from gui.pattern_io import load_pattern_file


LEFT_MIN_WIDTH = 220
LEFT_DEFAULT = 280
BOTTOM_DEFAULT = 280

# Checkbox column indices differ between the two result table layouts
CAND_SELECT_COL = 5
MATCH_SELECT_COL = 0


class AnalysisWorkspace(QWidget):
    """Main analysis layout: files | plot + bottom tool tabs."""

    TAB_BACKGROUND = 0
    TAB_PEAKS = 1
    TAB_PHASES = 2

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._status_callback = None
        self._candidate_results = []
        self._results_mode = None  # "candidates" | "matches"
        self._quant_dialog = None
        self._database_dialog = None
        self._details_dialog = None
        self._preview = None  # {"name", "two_theta", "intensity"}

        self.process_stage = ProcessStage(session, self)
        self.identify_stage = IdentifyStage(session, self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.file_browser = FileBrowser()
        self.file_browser.setMinimumWidth(LEFT_MIN_WIDTH)
        self.file_browser.file_activated.connect(self.open_pattern_file)
        self.file_browser.wavelength_changed.connect(self._on_wavelength_changed)
        self.main_splitter.addWidget(self.file_browser)

        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setChildrenCollapsible(False)

        plot_host, self.figure, self.canvas, self.toolbar = create_plot_host(
            self, figsize=(10, 7)
        )
        self.ax = self.figure.add_subplot(111)
        # Back-compat aliases (search plot == main plot)
        self.search_figure = self.figure
        self.search_canvas = self.canvas
        self.search_ax = self.ax
        self._add_view_toggles()
        self.right_splitter.addWidget(plot_host)

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self._build_background_tab(), "Background")
        self.bottom_tabs.addTab(self._build_peaks_tab(), "Peaks")
        self.bottom_tabs.addTab(self._build_phases_tab(), "Phases")
        self.right_splitter.addWidget(self.bottom_tabs)

        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        root.addWidget(self.main_splitter)

        session.pattern_changed.connect(self._on_session_changed)
        session.peaks_changed.connect(self._on_session_changed)
        session.candidates_changed.connect(self._on_session_changed)
        session.matches_changed.connect(self._on_session_changed)

        self.refresh_plot()
        QTimer.singleShot(0, self._apply_default_sizes)

    # --- plot view toggles ---

    def _add_view_toggles(self):
        """Show/hide curves straight from the plot toolbar."""
        self.toolbar.addSeparator()
        self.view_toggles = {}
        for key, label, tip, checked in (
            ("raw", "Raw", "Show the unprocessed pattern", True),
            ("processed", "Processed", "Show the background-subtracted pattern", True),
            ("background", "Background", "Show the fitted background curve", True),
            ("peaks", "Peaks", "Show detected peak markers", True),
        ):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.setToolTip(tip)
            cb.stateChanged.connect(self.refresh_plot)
            self.toolbar.addWidget(cb)
            self.view_toggles[key] = cb

    def _visible(self, key: str) -> bool:
        cb = self.view_toggles.get(key)
        return cb.isChecked() if cb is not None else True

    # --- tab builders ---

    def _build_background_tab(self) -> QWidget:
        return self.process_stage.background_panel

    def _build_peaks_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.process_stage.peaks_panel)

        self.peaks_label = QLabel("Peaks")
        self.peaks_label.setObjectName("mutedLabel")
        self.peaks_label.setContentsMargins(8, 0, 8, 0)
        layout.addWidget(self.peaks_label)

        self.peaks_table = QTableWidget()
        self.peaks_table.setAlternatingRowColors(True)
        self.peaks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peaks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peaks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.peaks_table, 1)
        return tab

    def _build_phases_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        layout.addWidget(self.identify_stage.control_panel)
        self._add_phase_table_actions()

        header = QHBoxLayout()
        header.setContentsMargins(8, 0, 8, 0)
        self.phases_label = QLabel("Phases")
        self.phases_label.setObjectName("mutedLabel")
        header.addWidget(self.phases_label, 1)
        layout.addLayout(header)

        # Back-compat: results_table / results_label used by older helpers
        self.results_label = self.phases_label
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_results_menu)
        self.results_table.currentCellChanged.connect(self._on_results_row_changed)
        layout.addWidget(self.results_table, 1)
        return tab

    def _add_phase_table_actions(self):
        """Right-hand buttons on the Phases parameter row."""
        row = self.identify_stage.actions_row

        self.details_btn = QPushButton("Details…")
        self.details_btn.setToolTip("Show details for the highlighted phase (or right-click a row)")
        self.details_btn.clicked.connect(self.show_selected_phase_details)
        row.add_widget(self.details_btn)

        self.clear_unselected_btn = QPushButton("Clear Unselected")
        self.clear_unselected_btn.setToolTip(
            "Remove unchecked phases from the list; keep only your selections"
        )
        self.clear_unselected_btn.clicked.connect(self.clear_unselected_phases)
        row.add_widget(self.clear_unselected_btn)

        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Remove every mineral from the list and start over")
        self.clear_all_btn.clicked.connect(self.clear_all_phases)
        row.add_widget(self.clear_all_btn)

        self.quant_btn = QPushButton("Open Quant…")
        self.quant_btn.setToolTip("Open Le Bail / quantitative analysis window")
        self.quant_btn.clicked.connect(self.open_quant)
        row.add_widget(self.quant_btn)

    def _apply_default_sizes(self):
        total_w = max(self.main_splitter.width(), 1200)
        left = min(max(LEFT_DEFAULT, LEFT_MIN_WIDTH), 360)
        self.main_splitter.setSizes([left, max(400, total_w - left)])
        total_h = max(self.right_splitter.height(), 700)
        bottom = min(BOTTOM_DEFAULT, total_h // 3)
        self.right_splitter.setSizes([max(300, total_h - bottom), bottom])

    # --- navigation ---

    def show_bottom_tab(self, key: str):
        mapping = {
            "background": self.TAB_BACKGROUND,
            "process": self.TAB_BACKGROUND,
            "load": self.TAB_BACKGROUND,
            "peaks": self.TAB_PEAKS,
            "identify": self.TAB_PHASES,
            "phases": self.TAB_PHASES,
        }
        idx = mapping.get(key, self.TAB_BACKGROUND)
        self.bottom_tabs.setCurrentIndex(idx)
        if idx == self.TAB_BACKGROUND:
            self.process_stage.on_enter()
        elif idx == self.TAB_PEAKS:
            self.process_stage.on_enter()
        elif idx == self.TAB_PHASES:
            self.identify_stage.on_enter()

    def set_stage(self, key: str):
        """Compatibility shim for older call sites."""
        if key == "refine":
            self.open_quant()
        else:
            self.show_bottom_tab(key)

    def show_search_tab(self, section: Optional[str] = None):
        self.show_bottom_tab(section or "phases")

    def show_quant_tab(self):
        self.open_quant()

    def show_database_tab(self):
        self.open_database()

    # --- dialogs ---

    def open_quant(self):
        from gui.dialogs.quant_dialog import QuantDialog

        if self._quant_dialog is None:
            parent = self.window()
            self._quant_dialog = QuantDialog(
                self.session, parent=parent, status_callback=self.set_status
            )
        self._quant_dialog.show()
        self._quant_dialog.raise_()
        self._quant_dialog.activateWindow()
        if hasattr(self._quant_dialog.refine_stage, "on_enter"):
            self._quant_dialog.refine_stage.on_enter()
        self._quant_dialog.refresh_plot()

    def open_database(self):
        from gui.dialogs.database_dialog import DatabaseManagerDialog

        if self._database_dialog is None:
            parent = self.window()
            self._database_dialog = DatabaseManagerDialog(parent)
            self._database_dialog.phases_selected.connect(self._on_db_phases)
        self._database_dialog.show()
        self._database_dialog.raise_()
        self._database_dialog.activateWindow()

    def _on_db_phases(self, phases: list):
        self.identify_stage.add_phases_from_database(phases)
        self.show_bottom_tab("phases")
        self.set_status(f"Added {len(phases)} phase(s) from database")

    # --- status / theme ---

    def set_status_callback(self, callback):
        self._status_callback = callback

    def set_status(self, message: str):
        if self._status_callback:
            self._status_callback(message)

    def _on_session_changed(self):
        self.refresh_plot()

    def on_theme_changed(self, mode: str):
        apply_plot_style(self.figure, mode)
        self.refresh_plot()
        if self._quant_dialog is not None:
            self._quant_dialog.on_theme_changed(mode)

    # --- file loading ---

    def _on_wavelength_changed(self, wl: float):
        self.session.set_wavelength(wl)
        self.refresh_plot()

    def open_folder(self, path: Optional[str] = None):
        if path:
            self.file_browser.set_folder(path)
        else:
            self.file_browser.choose_folder()

    def find_peaks(self):
        self.show_bottom_tab("peaks")
        self.process_stage.find_peaks()

    def open_pattern_file(self, path: str):
        try:
            wl = self.file_browser.current_wavelength()
            pattern = load_pattern_file(path, wl)
            if pattern.get("file_format") == "XML":
                self.file_browser.set_custom_wavelength(pattern["wavelength"])
            else:
                pattern["wavelength"] = wl

            self.session.set_raw_pattern(pattern)
            name = os.path.basename(path)
            n = len(pattern["two_theta"])
            t0, t1 = float(pattern["two_theta"][0]), float(pattern["two_theta"][-1])
            meta = (
                f"{pattern['file_format']} · {n} points · "
                f"2θ {t0:.2f}–{t1:.2f}° · λ {pattern['wavelength']:.4f} Å"
            )
            self.file_browser.set_file_info(name, meta)
            self.file_browser.reveal_file(path)
            self.process_stage.on_enter()
            self.identify_stage.on_enter()
            self.show_bottom_tab("background")
            self.refresh_plot()
            self.set_status(f"Loaded {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load pattern:\n{e}")

    # --- results helpers ---

    def set_results_peaks(self, peaks: dict):
        self.show_bottom_tab("peaks")
        n = len(peaks.get("two_theta", []))
        self.peaks_label.setText(f"Peaks ({n})")
        self.peaks_table.clear()
        self.peaks_table.setColumnCount(3)
        self.peaks_table.setHorizontalHeaderLabels(["2θ (°)", "Intensity", "d (Å)"])
        tt = peaks.get("two_theta", [])
        inten = peaks.get("intensity", [])
        d = peaks.get("d_spacing", [])
        self.peaks_table.setRowCount(len(tt))
        for i in range(len(tt)):
            self.peaks_table.setItem(i, 0, QTableWidgetItem(f"{tt[i]:.3f}"))
            self.peaks_table.setItem(i, 1, QTableWidgetItem(f"{inten[i]:.0f}"))
            self.peaks_table.setItem(i, 2, QTableWidgetItem(f"{d[i]:.4f}"))

    def clear_peaks_table(self):
        self.peaks_label.setText("Peaks")
        self.peaks_table.clearContents()
        self.peaks_table.setRowCount(0)

    @staticmethod
    def _fingerprint_cell(result: dict) -> str:
        fp = result.get("fingerprint")
        if not fp:
            return "—"
        flag = "" if fp.get("top_found", True) else "  ⚠ no top line"
        return f"{fp['score']:.2f}  ({fp['n_found']}/{fp['n_expected']}){flag}"

    def set_results_candidates(self, results: list):
        self._results_mode = "candidates"
        self._candidate_results = list(results)
        self._preview = None
        self.show_bottom_tab("phases")
        self.phases_label.setText(
            f"Search candidates ({len(results)}) — click a row to preview peaks, "
            "check the ones to match, right-click for details"
        )
        self.results_table.blockSignals(True)
        self.results_table.clear()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(
            ["Mineral", "Formula", "Space Group", "Score", "Fingerprint", "Select"]
        )
        self.results_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(r.get("mineral_name", ""))))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(r.get("chemical_formula", ""))))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(r.get("space_group", ""))))
            score = r.get(
                "fingerprint_score",
                r.get("ensemble_score",
                      r.get("combined_score", r.get("correlation", r.get("match_score", 0)))),
            )
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{float(score):.3f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(self._fingerprint_cell(r)))
            cb = QCheckBox()
            cb.setChecked(False)  # user must opt in
            self.results_table.setCellWidget(i, CAND_SELECT_COL, cb)
        self.results_table.blockSignals(False)
        self.results_table.resizeColumnsToContents()

    def check_candidate_rows(self, names: list):
        """Check rows whose mineral name matches (used for manually added phases)."""
        if self._results_mode != "candidates" or not names:
            return
        wanted = {str(n).lower() for n in names if n}
        for i in range(self.results_table.rowCount()):
            item = self.results_table.item(i, 0)
            if item and item.text().lower() in wanted:
                cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
                if cb:
                    cb.setChecked(True)

    def get_selected_candidates(self) -> list:
        """Return only explicitly checked candidates (no auto-fallback)."""
        results = getattr(self, "_candidate_results", None) or []
        if self._results_mode != "candidates" or not results:
            return []
        selected = []
        for i, r in enumerate(results):
            cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
            if cb is not None and cb.isChecked():
                selected.append(self.identify_stage._result_to_phase(r))
        return selected

    def set_results_matches(self, results: list, preselect: Optional[list] = None):
        """
        Show matched phases. Nothing is checked unless `preselect` provides
        keys/names (or match objects) that should stay selected.
        """
        self._results_mode = "matches"
        self._preview = None
        self.show_bottom_tab("phases")
        self.phases_label.setText(
            f"Matched phases ({len(results)}) — check to keep and plot, "
            "click a row to preview, right-click for details"
        )
        self.results_table.blockSignals(True)
        self.results_table.clear()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(
            ["Select", "Phase", "Score", "Fingerprint", "Coverage", "Matches"]
        )
        self.results_table.setRowCount(len(results))

        from utils.residual_search import mineral_key

        preselect_keys = set()
        if preselect:
            for p in preselect:
                preselect_keys.add(mineral_key(p))

        for i, r in enumerate(results):
            phase = r.get("phase", r)
            name = phase.get("mineral", phase.get("mineral_name", f"Phase {i+1}"))
            cb = QCheckBox()
            cb.setChecked(bool(preselect_keys) and mineral_key(r) in preselect_keys)
            cb.stateChanged.connect(self._sync_selected_from_table)
            self.results_table.setCellWidget(i, MATCH_SELECT_COL, cb)
            self.results_table.setItem(i, 1, QTableWidgetItem(str(name)))
            score = r.get("combined_score", r.get("match_score", 0))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{float(score):.3f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(self._fingerprint_cell(r)))
            cov = r.get("coverage", 0)
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{float(cov):.2f}" if cov else "—"))
            nmatch = len(r.get("matches", []))
            self.results_table.setItem(i, 5, QTableWidgetItem(str(nmatch)))
        self.results_table.blockSignals(False)
        self.results_table.resizeColumnsToContents()
        self._sync_selected_from_table()

    def get_selected_matches(self) -> list:
        if self._results_mode != "matches":
            return list(self.session.selected_phases)
        selected = []
        matches = self.session.matched_phases
        for i in range(self.results_table.rowCount()):
            cb = self.results_table.cellWidget(i, MATCH_SELECT_COL)
            if cb and cb.isChecked() and i < len(matches):
                selected.append(matches[i])
        return selected

    # --- row preview / details / context menu ---

    def _result_at_row(self, row: int) -> Optional[dict]:
        if row < 0:
            return None
        if self._results_mode == "candidates":
            results = self._candidate_results
        elif self._results_mode == "matches":
            results = self.session.matched_phases
        else:
            return None
        return results[row] if row < len(results) else None

    def _on_results_row_changed(self, row: int, _col=0, _prow=-1, _pcol=0):
        """Preview the highlighted phase; driven by clicks and arrow keys alike."""
        result = self._result_at_row(row)
        if result is None:
            return
        theo = self.identify_stage.theoretical_peaks_for(result)
        if not theo or len(theo.get("two_theta", [])) == 0:
            self._preview = None
            self.set_status("No reference peaks available for preview")
            self.refresh_plot()
            return
        phase = result.get("phase", result)
        name = (
            phase.get("mineral")
            or phase.get("mineral_name")
            or result.get("mineral_name")
            or "Preview"
        )
        self._preview = {
            "name": str(name),
            "two_theta": np.asarray(theo["two_theta"], dtype=float),
            "intensity": np.asarray(theo["intensity"], dtype=float),
        }
        self.refresh_plot()

    def clear_preview(self):
        self._preview = None
        self.refresh_plot()

    def show_selected_phase_details(self):
        row = self.results_table.currentRow()
        if row < 0:
            QMessageBox.information(
                self, "No Phase Selected", "Click a phase in the table first."
            )
            return
        self.show_phase_details(row)

    def show_phase_details(self, row: int):
        result = self._result_at_row(row)
        if result is None:
            return
        from gui.dialogs.phase_details_dialog import PhaseDetailsDialog

        if self._details_dialog is None:
            self._details_dialog = PhaseDetailsDialog(self.window())
        theo = self.identify_stage.theoretical_peaks_for(result)
        self._details_dialog.show_phase(result, theo)

    def _show_results_menu(self, pos):
        row = self.results_table.rowAt(pos.y())
        if row < 0 or self._result_at_row(row) is None:
            return
        self.results_table.selectRow(row)

        select_col = CAND_SELECT_COL if self._results_mode == "candidates" else MATCH_SELECT_COL
        cb = self.results_table.cellWidget(row, select_col)

        menu = QMenu(self)
        menu.addAction("Details…", lambda: self.show_phase_details(row))
        menu.addAction("Preview peaks", lambda: self._on_results_row_changed(row))
        menu.addAction("Clear preview", self.clear_preview)
        if cb is not None:
            label = "Uncheck" if cb.isChecked() else "Check"
            menu.addSeparator()
            menu.addAction(label, lambda: cb.setChecked(not cb.isChecked()))
        menu.addSeparator()
        menu.addAction("Remove from list", lambda: self.remove_phase_row(row))
        menu.exec_(self.results_table.viewport().mapToGlobal(pos))

    def remove_phase_row(self, row: int):
        if self._results_mode == "candidates":
            checked = [
                i for i in range(self.results_table.rowCount())
                if (w := self.results_table.cellWidget(i, CAND_SELECT_COL)) and w.isChecked()
            ]
            results = [r for i, r in enumerate(self._candidate_results) if i != row]
            keep_names = [
                (self._candidate_results[i].get("mineral_name") or "")
                for i in checked if i != row
            ]
            self.session.set_candidates(
                [self.identify_stage._result_to_phase(r) for r in results]
            )
            self.identify_stage._search_results = results
            self.set_results_candidates(results)
            self.check_candidate_rows(keep_names)
            self.set_status("Removed candidate")
        elif self._results_mode == "matches":
            kept_selected = self.get_selected_matches()
            matches = [r for i, r in enumerate(self.session.matched_phases) if i != row]
            removed = self._result_at_row(row)
            kept_selected = [r for r in kept_selected if r is not removed]
            self.session.set_matched_phases(matches)
            self.session.set_selected_phases(kept_selected)
            self.set_results_matches(matches, preselect=kept_selected)
            self.set_status("Removed phase")

    def clear_all_phases(self):
        """Wipe candidates, matches, and selections."""
        if self.results_table.rowCount() == 0 and not self.session.matched_phases:
            self.set_status("Phase list is already empty")
            return
        if (
            QMessageBox.question(
                self, "Clear All Minerals",
                "Remove all minerals from the list, including matched and selected phases?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        self._candidate_results = []
        self._results_mode = None
        self._preview = None
        self.identify_stage.reset_results()
        self.session.set_candidates([])
        self.session.set_matched_phases([])
        self.session.set_selected_phases([])
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self.phases_label.setText("Phases — run a search or add a known mineral")
        self.identify_stage.on_enter()
        self.set_status("Cleared all minerals")
        self.refresh_plot()

    def clear_unselected_phases(self):
        """Drop unchecked rows; keep only user-selected phases."""
        if self._results_mode == "candidates":
            kept_results = []
            kept_phases = []
            for i, r in enumerate(self._candidate_results):
                cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
                if cb is not None and cb.isChecked():
                    kept_results.append(r)
                    kept_phases.append(self.identify_stage._result_to_phase(r))
            if not kept_phases:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "Nothing Selected",
                    "Select one or more candidates first, then Clear Unselected.",
                )
                return
            self.session.set_candidates(kept_phases)
            self.identify_stage._search_results = kept_results
            self.set_results_candidates(kept_results)
            # Re-check the kept ones so the user doesn't lose selection
            for i in range(self.results_table.rowCount()):
                cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
                if cb:
                    cb.setChecked(True)
            self.set_status(f"Kept {len(kept_phases)} candidate(s)")
            return

        if self._results_mode == "matches":
            kept = self.get_selected_matches()
            if not kept:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "Nothing Selected",
                    "Select one or more matched phases first, then Clear Unselected.",
                )
                return
            self.session.set_matched_phases(kept)
            self.session.set_selected_phases(kept)
            self.set_results_matches(kept, preselect=kept)
            self.set_status(f"Kept {len(kept)} matched phase(s)")
            return

        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "No List", "Run a search or match first.")

    def _sync_selected_from_table(self):
        if self._results_mode != "matches":
            return
        selected = []
        matches = self.session.matched_phases
        for i in range(self.results_table.rowCount()):
            cb = self.results_table.cellWidget(i, MATCH_SELECT_COL)
            if cb and cb.isChecked() and i < len(matches):
                selected.append(matches[i])
        self.session.set_selected_phases(selected)
        self.refresh_plot()

    # --- plotting ---

    def refresh_plot(self):
        mode = get_current_mode()
        ax = self.ax
        ax.clear()
        palette = get_plot_palette(mode)

        if self.session.has_matches() or self.session.selected_phases:
            self._plot_identify(ax, palette)
        elif self.session.processed_pattern is not None or self.session.has_peaks():
            self._plot_process(ax, palette)
        else:
            self._plot_load(ax, palette)

        self._draw_preview(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        apply_plot_style(self.figure, mode)
        self.canvas.draw_idle()

    def _draw_preview(self, ax):
        """Overlay the highlighted candidate's reference lines."""
        if not self._preview:
            return
        tt = self._preview["two_theta"]
        inten = self._preview["intensity"]
        if len(tt) == 0:
            return
        top = ax.get_ylim()[1] or 1.0
        imax = float(np.max(inten)) if len(inten) and np.max(inten) > 0 else 1.0
        heights = inten / imax * top * 0.75
        ax.vlines(
            tt, 0, heights, colors="#e0a300", lw=1.4, alpha=0.9, ls="-",
            label=f"Preview: {self._preview['name']}", zorder=5,
        )

    def _plot_load(self, ax, palette):
        pattern = self.session.raw_pattern
        ax.set_title("XRD Pattern")
        if not pattern:
            ax.text(
                0.5, 0.5, "Choose a folder and click a pattern file",
                ha="center", va="center", transform=ax.transAxes,
                color=palette.get("muted", palette["tick"]),
            )
            ax.set_xlabel("2θ (degrees)")
            ax.set_ylabel("Intensity")
            return
        ax.plot(
            pattern["two_theta"], pattern["intensity"],
            color=palette["exp_line"], lw=1.2, label="Experimental",
        )
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity")
        fmt = pattern.get("file_format", "")
        ax.set_title(f"XRD Pattern ({fmt})" if fmt else "XRD Pattern")
        ax.legend(loc="upper right")

    def _plot_process(self, ax, palette):
        raw = self.session.raw_pattern
        processed = self.session.processed_pattern
        if raw is not None and self._visible("raw"):
            ax.plot(
                raw["two_theta"], raw["intensity"],
                color=palette["diff_line"], lw=0.8, alpha=0.5, label="Raw",
            )
        if processed is not None:
            if self._visible("processed"):
                ax.plot(
                    processed["two_theta"], processed["intensity"],
                    color=palette["exp_line"], lw=1.2, label="Processed",
                )
            bg = self.session.background
            if bg is not None and raw is not None and self._visible("background"):
                ax.plot(
                    raw["two_theta"], bg, color=palette["calc_line"],
                    lw=1.0, ls="--", label="Background",
                )
        peaks = self.session.peaks
        if peaks is not None and processed is not None and self._visible("peaks"):
            ax.plot(
                peaks["two_theta"], peaks["intensity"], "o",
                color=palette["calc_line"], ms=4, label="Peaks",
            )
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity")
        ax.set_title("Processing Preview")

    def _plot_identify(self, ax, palette):
        pattern = self.session.active_pattern()
        if pattern is None:
            ax.set_title("Identify")
            return
        inten = np.asarray(pattern["intensity"], dtype=float)
        max_i = np.max(inten) if len(inten) else 1.0
        norm = (inten / max_i * 100.0) if max_i > 0 else inten
        if self._visible("processed"):
            ax.plot(
                pattern["two_theta"], norm, color=palette["exp_line"],
                lw=1.2, label="Experimental",
            )

        raw = self.session.raw_pattern
        if raw is not None and raw is not pattern and self._visible("raw"):
            rint = np.asarray(raw["intensity"], dtype=float)
            rmax = np.max(rint) if len(rint) else 1.0
            ax.plot(
                raw["two_theta"], (rint / rmax * 100.0) if rmax > 0 else rint,
                color=palette["diff_line"], lw=0.8, alpha=0.45, label="Raw",
            )

        bg = self.session.background
        if bg is not None and raw is not None and self._visible("background"):
            bgn = np.asarray(bg, dtype=float)
            rint = np.asarray(raw["intensity"], dtype=float)
            rmax = np.max(rint) if len(rint) else 1.0
            ax.plot(
                raw["two_theta"], (bgn / rmax * 100.0) if rmax > 0 else bgn,
                color=palette["calc_line"], lw=1.0, ls="--", label="Background",
            )

        colors = ["#c45c26", "#7a5cff", "#2a7a4b", "#b33a3a", "#5a6a7a"]
        selected = self.session.selected_phases
        for i, result in enumerate(selected[:5]):
            theo = result.get("theoretical_peaks") or self.identify_stage.theoretical_peaks_for(result)
            if not theo or len(theo.get("two_theta", [])) == 0:
                continue
            tt = np.asarray(theo["two_theta"])
            ti = np.asarray(theo["intensity"], dtype=float)
            tmax = np.max(ti) if len(ti) else 1.0
            tnorm = (ti / tmax * 80.0) if tmax > 0 else ti
            phase = result.get("phase", {})
            name = phase.get("mineral", f"Phase {i+1}")
            ax.vlines(
                tt, 0, tnorm, colors=colors[i % len(colors)],
                alpha=0.7, lw=1.0, label=name,
            )

        peaks = self.session.peaks
        if peaks is not None and self._visible("peaks"):
            pi = np.asarray(peaks["intensity"], dtype=float)
            pmax = np.max(pi) if len(pi) else 1.0
            ax.plot(
                peaks["two_theta"], (pi / pmax * 100.0) if pmax > 0 else pi,
                "o", color=palette["calc_line"], ms=3, label="Peaks",
            )

        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Normalized Intensity")
        ax.set_title("Phase Identification")
        ax.set_ylim(0, 110)
