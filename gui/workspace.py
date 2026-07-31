"""Analysis workspace — file browser, plot, and Background / Peaks / Phases tabs."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from matplotlib_config import apply_plot_style, get_plot_palette
from gui.theme import get_current_mode
from gui.widgets.file_browser import FileBrowser
from gui.widgets.plot_host import create_plot_host
from gui.stages import ProcessStage, IdentifyStage
from gui.pattern_io import load_pattern_file


LEFT_MIN_WIDTH = 220
LEFT_DEFAULT = 280
BOTTOM_DEFAULT = 260


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

    # --- tab builders ---

    def _build_background_tab(self) -> QWidget:
        return self.process_stage.background_panel

    def _build_peaks_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.process_stage.peaks_panel)

        table_wrap = QWidget()
        tw = QVBoxLayout(table_wrap)
        tw.setContentsMargins(6, 6, 6, 6)
        tw.setSpacing(4)
        self.peaks_label = QLabel("Peaks")
        self.peaks_label.setObjectName("mutedLabel")
        tw.addWidget(self.peaks_label)
        self.peaks_table = QTableWidget()
        self.peaks_table.setAlternatingRowColors(True)
        self.peaks_table.horizontalHeader().setStretchLastSection(True)
        tw.addWidget(self.peaks_table, 1)
        splitter.addWidget(table_wrap)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 600])
        layout.addWidget(splitter)
        return tab

    def _build_phases_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Identify controls in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.identify_stage)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(420)
        splitter.addWidget(scroll)

        right = QWidget()
        rw = QVBoxLayout(right)
        rw.setContentsMargins(6, 6, 6, 6)
        rw.setSpacing(4)

        header = QHBoxLayout()
        self.phases_label = QLabel("Phases")
        self.phases_label.setObjectName("mutedLabel")
        header.addWidget(self.phases_label, 1)
        self.clear_unselected_btn = QPushButton("Clear Unselected")
        self.clear_unselected_btn.setToolTip(
            "Remove unchecked phases from the list; keep only your selections"
        )
        self.clear_unselected_btn.clicked.connect(self.clear_unselected_phases)
        header.addWidget(self.clear_unselected_btn)
        self.quant_btn = QPushButton("Open Quant…")
        self.quant_btn.setToolTip("Open Le Bail / quantitative analysis window")
        self.quant_btn.clicked.connect(self.open_quant)
        header.addWidget(self.quant_btn)
        rw.addLayout(header)

        # Back-compat: results_table / results_label used by older helpers
        self.results_label = self.phases_label
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        rw.addWidget(self.results_table, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 600])
        layout.addWidget(splitter)
        return tab

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

    def set_results_candidates(self, results: list):
        self._results_mode = "candidates"
        self._candidate_results = list(results)
        self.show_bottom_tab("phases")
        self.phases_label.setText(f"Search candidates ({len(results)}) — select to match")
        self.results_table.clear()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Mineral", "Formula", "Space Group", "Score", "Select"]
        )
        self.results_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(r.get("mineral_name", ""))))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(r.get("chemical_formula", ""))))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(r.get("space_group", ""))))
            score = r.get(
                "ensemble_score",
                r.get("combined_score", r.get("correlation", r.get("match_score", 0))),
            )
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{float(score):.3f}"))
            cb = QCheckBox()
            cb.setChecked(False)  # user must opt in
            self.results_table.setCellWidget(i, 4, cb)

    def get_selected_candidates(self) -> list:
        """Return only explicitly checked candidates (no auto-fallback)."""
        results = getattr(self, "_candidate_results", None) or []
        if self._results_mode != "candidates" or not results:
            return []
        selected = []
        for i, r in enumerate(results):
            cb = self.results_table.cellWidget(i, 4)
            if cb is not None and cb.isChecked():
                selected.append(self.identify_stage._result_to_phase(r))
        return selected

    def set_results_matches(self, results: list, preselect: Optional[list] = None):
        """
        Show matched phases. Nothing is checked unless `preselect` provides
        keys/names (or match objects) that should stay selected.
        """
        self._results_mode = "matches"
        self.show_bottom_tab("phases")
        self.phases_label.setText(f"Matched phases ({len(results)}) — select to keep / plot")
        self.results_table.clear()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Select", "Phase", "Score", "Coverage", "Matches"]
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
            self.results_table.setCellWidget(i, 0, cb)
            self.results_table.setItem(i, 1, QTableWidgetItem(str(name)))
            score = r.get("combined_score", r.get("match_score", 0))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{float(score):.3f}"))
            cov = r.get("coverage", 0)
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{float(cov):.2f}" if cov else "—"))
            nmatch = len(r.get("matches", []))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(nmatch)))
        self._sync_selected_from_table()

    def get_selected_matches(self) -> list:
        if self._results_mode != "matches":
            return list(self.session.selected_phases)
        selected = []
        matches = self.session.matched_phases
        for i in range(self.results_table.rowCount()):
            cb = self.results_table.cellWidget(i, 0)
            if cb and cb.isChecked() and i < len(matches):
                selected.append(matches[i])
        return selected

    def clear_unselected_phases(self):
        """Drop unchecked rows; keep only user-selected phases."""
        if self._results_mode == "candidates":
            kept_results = []
            kept_phases = []
            for i, r in enumerate(self._candidate_results):
                cb = self.results_table.cellWidget(i, 4)
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
            self.set_results_candidates(kept_results)
            # Re-check the kept ones so the user doesn't lose selection
            for i in range(self.results_table.rowCount()):
                cb = self.results_table.cellWidget(i, 4)
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
            cb = self.results_table.cellWidget(i, 0)
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

        apply_plot_style(self.figure, mode)
        self.canvas.draw_idle()

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
        if raw is not None:
            ax.plot(
                raw["two_theta"], raw["intensity"],
                color=palette["diff_line"], lw=0.8, alpha=0.5, label="Raw",
            )
        if processed is not None:
            ax.plot(
                processed["two_theta"], processed["intensity"],
                color=palette["exp_line"], lw=1.2, label="Processed",
            )
            bg = self.session.background
            if bg is not None and raw is not None:
                ax.plot(
                    raw["two_theta"], bg, color=palette["calc_line"],
                    lw=1.0, ls="--", label="Background",
                )
        peaks = self.session.peaks
        if peaks is not None and processed is not None:
            ax.plot(
                peaks["two_theta"], peaks["intensity"], "o",
                color=palette["calc_line"], ms=4, label="Peaks",
            )
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity")
        ax.set_title("Processing Preview")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")

    def _plot_identify(self, ax, palette):
        pattern = self.session.active_pattern()
        if pattern is None:
            ax.set_title("Identify")
            return
        inten = np.asarray(pattern["intensity"], dtype=float)
        max_i = np.max(inten) if len(inten) else 1.0
        norm = (inten / max_i * 100.0) if max_i > 0 else inten
        ax.plot(
            pattern["two_theta"], norm, color=palette["exp_line"],
            lw=1.2, label="Experimental",
        )

        colors = ["#c45c26", "#7a5cff", "#2a7a4b", "#b33a3a", "#5a6a7a"]
        selected = self.session.selected_phases
        for i, result in enumerate(selected[:5]):
            theo = result.get("theoretical_peaks")
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
        if peaks is not None:
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
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)
