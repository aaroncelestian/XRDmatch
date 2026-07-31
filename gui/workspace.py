"""Analysis workspace — Search/Match, Quant Analysis, and Database tabs."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QLabel, QScrollArea, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib_config import apply_plot_style, get_plot_palette
from gui.theme import get_current_mode
from gui.widgets.section import CollapsibleSection
from gui.stages import LoadStage, ProcessStage, IdentifyStage, RefineStage
from gui.local_database_tab import LocalDatabaseTab


# Narrow control column — plot gets most of the window
RATIO_DEFAULT = (0.28, 0.72)
RATIO_COMPRESSED = (0.18, 0.82)
LEFT_MIN_WIDTH = 240
LEFT_MAX_WIDTH = 360


class AnalysisWorkspace(QWidget):
    """Tabbed workspace with controls | data splitters."""

    TAB_SEARCH = 0
    TAB_QUANT = 1
    TAB_DATABASE = 2

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._status_callback = None
        self._compressed = False
        self._candidate_results = []
        self._sections = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.load_stage = LoadStage(session, self)
        self.process_stage = ProcessStage(session, self)
        self.identify_stage = IdentifyStage(session, self)
        self.refine_stage = RefineStage(session, self)

        self.search_splitter = self._build_search_tab()
        self.quant_splitter = self._build_quant_tab()
        self.db_tab = LocalDatabaseTab()
        self.db_tab.phases_selected.connect(self._on_db_phases)

        self.tabs.addTab(self.search_splitter, "Search / Match")
        self.tabs.addTab(self.quant_splitter, "Quant Analysis")
        self.tabs.addTab(self.db_tab, "Database")

        self.tabs.currentChanged.connect(self._on_tab_changed)

        session.pattern_changed.connect(self._on_session_changed)
        session.peaks_changed.connect(self._on_session_changed)
        session.candidates_changed.connect(self._on_session_changed)
        session.matches_changed.connect(self._on_session_changed)
        session.refinement_changed.connect(self.refresh_plot)

        # Back-compat aliases used by refine export / theme
        self.figure = self.search_figure
        self.canvas = self.search_canvas
        self.ax = self.search_ax

        self.refresh_plot()
        QTimer.singleShot(0, lambda: self._apply_split_ratio(RATIO_DEFAULT))

    # --- layout builders ---

    def _build_search_tab(self) -> QSplitter:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left.setMinimumWidth(LEFT_MIN_WIDTH)
        left.setMaximumWidth(LEFT_MAX_WIDTH)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)

        self._sections["load"] = CollapsibleSection("Load", self.load_stage, expanded=True)
        self._sections["process"] = CollapsibleSection("Process", self.process_stage, expanded=True)
        self._sections["identify"] = CollapsibleSection("Identify", self.identify_stage, expanded=True)
        for key in ("load", "process", "identify"):
            controls_layout.addWidget(self._sections[key])
        controls_layout.addStretch()

        scroll.setWidget(controls)
        left_layout.addWidget(scroll)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        mode = get_current_mode()
        palette = get_plot_palette(mode)
        self.search_figure = Figure(figsize=(10, 7), facecolor=palette["figure_facecolor"])
        self.search_canvas = FigureCanvas(self.search_figure)
        self.search_toolbar = NavigationToolbar(self.search_canvas, right)
        right_layout.addWidget(self.search_toolbar)
        right_layout.addWidget(self.search_canvas, 1)
        self.search_ax = self.search_figure.add_subplot(111)
        apply_plot_style(self.search_figure, mode)

        results_wrap = QWidget()
        results_layout = QVBoxLayout(results_wrap)
        results_layout.setContentsMargins(4, 0, 4, 4)
        results_layout.setSpacing(2)
        self.results_label = QLabel("Results")
        self.results_label.setObjectName("mutedLabel")
        results_layout.addWidget(self.results_label)

        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(160)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        right_layout.addWidget(results_wrap)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_quant_tab(self) -> QSplitter:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left.setMinimumWidth(LEFT_MIN_WIDTH)
        left.setMaximumWidth(LEFT_MAX_WIDTH)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)
        controls_layout.addWidget(self.refine_stage)
        controls_layout.addStretch()
        scroll.setWidget(controls)
        left_layout.addWidget(scroll)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        mode = get_current_mode()
        palette = get_plot_palette(mode)
        self.quant_figure = Figure(figsize=(10, 7), facecolor=palette["figure_facecolor"])
        self.quant_canvas = FigureCanvas(self.quant_figure)
        self.quant_toolbar = NavigationToolbar(self.quant_canvas, right)
        right_layout.addWidget(self.quant_toolbar)
        right_layout.addWidget(self.quant_canvas, 1)
        self.quant_ax = self.quant_figure.add_subplot(111)
        apply_plot_style(self.quant_figure, mode)

        quant_results = QWidget()
        qr_layout = QVBoxLayout(quant_results)
        qr_layout.setContentsMargins(4, 0, 4, 4)
        qr_layout.setSpacing(2)
        self.quant_results_label = QLabel("Refinement")
        self.quant_results_label.setObjectName("mutedLabel")
        qr_layout.addWidget(self.quant_results_label)
        self.quant_results_table = QTableWidget()
        self.quant_results_table.setMaximumHeight(120)
        self.quant_results_table.setAlternatingRowColors(True)
        self.quant_results_table.horizontalHeader().setStretchLastSection(True)
        qr_layout.addWidget(self.quant_results_table)
        right_layout.addWidget(quant_results)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    # --- splitter ratios ---

    def _active_splitter(self) -> Optional[QSplitter]:
        idx = self.tabs.currentIndex()
        if idx == self.TAB_SEARCH:
            return self.search_splitter
        if idx == self.TAB_QUANT:
            return self.quant_splitter
        return None

    def _apply_split_ratio(self, ratio: Tuple[float, float], splitter: Optional[QSplitter] = None):
        targets = [splitter] if splitter is not None else [self.search_splitter, self.quant_splitter]
        for sp in targets:
            if sp is None:
                continue
            total = max(sp.width(), 800)
            left = int(total * ratio[0])
            left = max(LEFT_MIN_WIDTH, min(LEFT_MAX_WIDTH, left))
            right = max(100, total - left)
            sp.setSizes([left, right])

    def set_controls_compressed(self, compressed: bool):
        self._compressed = compressed
        ratio = RATIO_COMPRESSED if compressed else RATIO_DEFAULT
        active = self._active_splitter()
        if active is not None:
            self._apply_split_ratio(ratio, active)
        else:
            # Database tab: still update both so they match when returning
            self._apply_split_ratio(ratio)

    def toggle_controls_compressed(self) -> bool:
        self.set_controls_compressed(not self._compressed)
        return self._compressed

    def is_controls_compressed(self) -> bool:
        return self._compressed

    # --- navigation helpers ---

    def show_search_tab(self, section: Optional[str] = None):
        self.tabs.setCurrentIndex(self.TAB_SEARCH)
        if section and section in self._sections:
            self._sections[section].set_expanded(True)
        self.refresh_plot()

    def show_quant_tab(self):
        self.tabs.setCurrentIndex(self.TAB_QUANT)
        if hasattr(self.refine_stage, "on_enter"):
            self.refine_stage.on_enter()
        self.refresh_plot()

    def show_database_tab(self):
        self.tabs.setCurrentIndex(self.TAB_DATABASE)

    def set_stage(self, key: str):
        """Compatibility shim for older call sites (load/process/identify/refine)."""
        if key == "refine":
            self.show_quant_tab()
        elif key in ("load", "process", "identify"):
            self.show_search_tab(section=key)
        else:
            self.show_search_tab()

    def _on_tab_changed(self, index: int):
        if index == self.TAB_SEARCH:
            for stage in (self.load_stage, self.process_stage, self.identify_stage):
                if hasattr(stage, "on_enter"):
                    stage.on_enter()
            self.figure = self.search_figure
            self.canvas = self.search_canvas
            self.ax = self.search_ax
        elif index == self.TAB_QUANT:
            if hasattr(self.refine_stage, "on_enter"):
                self.refine_stage.on_enter()
            self.figure = self.quant_figure
            self.canvas = self.quant_canvas
            self.ax = self.quant_ax
        self.refresh_plot()
        # Re-apply current compress ratio to the newly visible splitter
        ratio = RATIO_COMPRESSED if self._compressed else RATIO_DEFAULT
        active = self._active_splitter()
        if active is not None:
            self._apply_split_ratio(ratio, active)

    def _on_db_phases(self, phases: list):
        self.identify_stage.add_phases_from_database(phases)
        self.show_search_tab(section="identify")
        self.set_status(f"Added {len(phases)} phase(s) from database")

    def set_status_callback(self, callback):
        self._status_callback = callback

    def set_status(self, message: str):
        if self._status_callback:
            self._status_callback(message)

    def _on_session_changed(self):
        self.refresh_plot()

    def on_theme_changed(self, mode: str):
        apply_plot_style(self.search_figure, mode)
        apply_plot_style(self.quant_figure, mode)
        self.refresh_plot()

    def find_peaks(self):
        self.show_search_tab(section="process")
        self.process_stage.find_peaks()

    def open_pattern_file(self, path: str):
        self.show_search_tab(section="load")
        self.load_stage.load_file(path)

    # --- results strip helpers ---

    def set_results_peaks(self, peaks: dict):
        self.results_label.setText(f"Peaks ({len(peaks.get('two_theta', []))})")
        self.results_table.clear()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["2θ (°)", "Intensity", "d (Å)"])
        tt = peaks.get("two_theta", [])
        inten = peaks.get("intensity", [])
        d = peaks.get("d_spacing", [])
        self.results_table.setRowCount(len(tt))
        for i in range(len(tt)):
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{tt[i]:.3f}"))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{inten[i]:.0f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{d[i]:.4f}"))

    def set_results_candidates(self, results: list):
        self._candidate_results = list(results)
        self.results_label.setText(f"Search candidates ({len(results)})")
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
            score = r.get("ensemble_score", r.get("combined_score", r.get("correlation", r.get("match_score", 0))))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{float(score):.3f}"))
            cb = QCheckBox()
            cb.setChecked(i < 20)
            self.results_table.setCellWidget(i, 4, cb)

    def get_selected_candidates(self) -> list:
        """Return phase dicts for checked search candidates (or all session candidates)."""
        results = getattr(self, "_candidate_results", None)
        if not results:
            return list(self.session.search_candidates)
        selected = []
        for i, r in enumerate(results):
            cb = self.results_table.cellWidget(i, 4)
            if cb is None or cb.isChecked():
                selected.append(self.identify_stage._result_to_phase(r))
        return selected

    def set_results_matches(self, results: list):
        self.results_label.setText(f"Matched phases ({len(results)})")
        self.results_table.clear()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Select", "Phase", "Score", "Coverage", "Matches"]
        )
        self.results_table.setRowCount(len(results))
        for i, r in enumerate(results):
            phase = r.get("phase", r)
            name = phase.get("mineral", phase.get("mineral_name", f"Phase {i+1}"))
            cb = QCheckBox()
            cb.setChecked(i < 5)
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

    def _sync_selected_from_table(self):
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
        self._refresh_search_plot(mode)
        self._refresh_quant_plot(mode)

    def _refresh_search_plot(self, mode: str):
        ax = self.search_ax
        ax.clear()
        palette = get_plot_palette(mode)

        if self.session.has_matches() or self.session.selected_phases:
            self._plot_identify(ax, palette)
        elif self.session.processed_pattern is not None or self.session.has_peaks():
            self._plot_process(ax, palette)
        else:
            self._plot_load(ax, palette)

        apply_plot_style(self.search_figure, mode)
        self.search_canvas.draw_idle()

    def _refresh_quant_plot(self, mode: str):
        ax = self.quant_ax
        ax.clear()
        palette = get_plot_palette(mode)
        self._plot_refine(ax, palette)
        apply_plot_style(self.quant_figure, mode)
        self.quant_canvas.draw_idle()

        # Update quant results strip from lebail
        results = self.session.lebail_results
        if results and results.get("success"):
            refined = results.get("refined_phases") or []
            rr = results.get("refinement_results") or {}
            rwp = rr.get("rwp") or results.get("rwp")
            label = "Refinement"
            if rwp is not None:
                label = f"Refinement (Rwp={float(rwp):.2f}%)"
            self.quant_results_label.setText(label)
            self.quant_results_table.clear()
            self.quant_results_table.setColumnCount(2)
            self.quant_results_table.setHorizontalHeaderLabels(["Phase", "Scale / info"])
            if refined:
                self.quant_results_table.setRowCount(len(refined))
                for i, p in enumerate(refined):
                    name = p.get("mineral", p.get("mineral_name", f"Phase {i+1}")) if isinstance(p, dict) else str(p)
                    info = ""
                    if isinstance(p, dict):
                        scale = p.get("scale", p.get("scale_factor"))
                        if scale is not None:
                            info = f"scale={float(scale):.4g}"
                    self.quant_results_table.setItem(i, 0, QTableWidgetItem(str(name)))
                    self.quant_results_table.setItem(i, 1, QTableWidgetItem(info))
            else:
                self.quant_results_table.setRowCount(0)
        else:
            self.quant_results_label.setText("Refinement")
            self.quant_results_table.setRowCount(0)

    def _plot_load(self, ax, palette):
        pattern = self.session.raw_pattern
        ax.set_title("XRD Pattern")
        if not pattern:
            ax.text(
                0.5, 0.5, "Load a pattern to begin", ha="center", va="center",
                transform=ax.transAxes,
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
        ax.plot(pattern["two_theta"], norm, color=palette["exp_line"],
                lw=1.2, label="Experimental")

        colors = ["#c45c26", "#7a5cff", "#2a7a4b", "#b33a3a", "#5a6a7a"]
        selected = self.session.selected_phases or self.session.matched_phases[:3]
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

    def _plot_refine(self, ax, palette):
        results = self.session.lebail_results
        pattern = self.session.active_pattern()
        if results and results.get("success"):
            rr = results.get("refinement_results") or {}
            tt = rr.get("two_theta", pattern["two_theta"] if pattern else None)
            exp = rr.get("experimental_intensity")
            calc = rr.get("calculated_pattern")
            if tt is not None and exp is not None:
                ax.plot(tt, exp, color=palette["exp_line"], lw=1.2, label="Experimental")
            if tt is not None and calc is not None:
                ax.plot(tt, calc, color=palette["calc_line"], lw=1.2, label="Calculated")
                if exp is not None:
                    diff = np.asarray(exp) - np.asarray(calc)
                    offset = -0.15 * (np.max(exp) if len(exp) else 0)
                    ax.plot(tt, diff + offset, color=palette["diff_line"],
                            lw=0.8, label="Difference")
            ax.set_title("Le Bail Refinement")
        elif pattern is not None:
            ax.plot(
                pattern["two_theta"], pattern["intensity"],
                color=palette["exp_line"], lw=1.2, label="Experimental",
            )
            ax.set_title("Quant — run Le Bail to see fit")
        else:
            ax.text(
                0.5, 0.5, "Match phases, then run Le Bail",
                ha="center", va="center", transform=ax.transAxes,
                color=palette.get("muted", palette["tick"]),
            )
            ax.set_title("Quant Analysis")
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")
