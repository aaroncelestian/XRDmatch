"""Guided analysis workspace — rail, stage stack, persistent plot, results strip."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView, QCheckBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib_config import apply_plot_style, get_plot_palette
from gui.theme import get_current_mode
from gui.stage_rail import StageRail
from gui.stages import LoadStage, ProcessStage, IdentifyStage, RefineStage


class AnalysisWorkspace(QWidget):
    """Single-screen guided workflow."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_stage = "load"
        self._status_callback = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        body = QSplitter(Qt.Horizontal)
        root.addWidget(body, 1)

        # Left: rail + stage controls
        left = QWidget()
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.rail = StageRail()
        self.rail.setFixedWidth(120)
        self.rail.stage_selected.connect(self.set_stage)
        left_layout.addWidget(self.rail)

        self.stage_stack = QStackedWidget()
        self.load_stage = LoadStage(session, self)
        self.process_stage = ProcessStage(session, self)
        self.identify_stage = IdentifyStage(session, self)
        self.refine_stage = RefineStage(session, self)

        self._stage_widgets = {
            "load": self.load_stage,
            "process": self.process_stage,
            "identify": self.identify_stage,
            "refine": self.refine_stage,
        }
        for key in ("load", "process", "identify", "refine"):
            self.stage_stack.addWidget(self._stage_widgets[key])

        left_layout.addWidget(self.stage_stack, 1)
        left.setMinimumWidth(360)
        left.setMaximumWidth(480)
        body.addWidget(left)

        # Right: plot
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)

        mode = get_current_mode()
        palette = get_plot_palette(mode)
        self.figure = Figure(figsize=(10, 7), facecolor=palette["figure_facecolor"])
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, plot_panel)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        apply_plot_style(self.figure, mode)
        body.addWidget(plot_panel)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([400, 900])

        # Bottom results strip
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
        root.addWidget(results_wrap)

        session.pattern_changed.connect(self._on_session_changed)
        session.peaks_changed.connect(self._on_session_changed)
        session.candidates_changed.connect(self._on_session_changed)
        session.matches_changed.connect(self._on_session_changed)
        session.refinement_changed.connect(self.refresh_plot)
        session.stage_status_changed.connect(self._update_rail)

        self.set_stage("load")
        self.refresh_plot()

    def set_status_callback(self, callback):
        self._status_callback = callback

    def set_status(self, message: str):
        if self._status_callback:
            self._status_callback(message)

    def set_stage(self, key: str):
        if key not in self._stage_widgets:
            return
        # Gate: don't jump to disabled stages
        btn = self.rail._buttons.get(key)
        if btn is not None and not btn.isEnabled() and key != "load":
            return
        self._current_stage = key
        self.stage_stack.setCurrentWidget(self._stage_widgets[key])
        self.rail.set_current(key)
        stage = self._stage_widgets[key]
        if hasattr(stage, "on_enter"):
            stage.on_enter()
        self.refresh_plot()

    def _update_rail(self):
        self.rail.update_availability(self.session)

    def _on_session_changed(self):
        self._update_rail()
        self.refresh_plot()

    def on_theme_changed(self, mode: str):
        apply_plot_style(self.figure, mode)
        self.refresh_plot()

    def find_peaks(self):
        self.set_stage("process")
        self.process_stage.find_peaks()

    def open_pattern_file(self, path: str):
        self.set_stage("load")
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
        self.ax.clear()
        mode = get_current_mode()
        palette = get_plot_palette(mode)
        stage = self._current_stage

        if stage == "refine" and self.session.lebail_results:
            self._plot_refine(palette)
        elif stage == "identify":
            self._plot_identify(palette)
        elif stage == "process":
            self._plot_process(palette)
        else:
            self._plot_load(palette)

        apply_plot_style(self.figure, mode)
        self.canvas.draw_idle()

    def _plot_load(self, palette):
        pattern = self.session.raw_pattern
        self.ax.set_title("XRD Pattern")
        if not pattern:
            self.ax.text(0.5, 0.5, "Load a pattern to begin", ha="center", va="center",
                         transform=self.ax.transAxes, color=palette["muted"] if "muted" in palette else palette["tick"])
            self.ax.set_xlabel("2θ (degrees)")
            self.ax.set_ylabel("Intensity")
            return
        self.ax.plot(pattern["two_theta"], pattern["intensity"],
                     color=palette["exp_line"], lw=1.2, label="Experimental")
        self.ax.set_xlabel("2θ (degrees)")
        self.ax.set_ylabel("Intensity")
        fmt = pattern.get("file_format", "")
        self.ax.set_title(f"XRD Pattern ({fmt})" if fmt else "XRD Pattern")
        self.ax.legend(loc="upper right")

    def _plot_process(self, palette):
        raw = self.session.raw_pattern
        processed = self.session.processed_pattern
        if raw is not None:
            self.ax.plot(raw["two_theta"], raw["intensity"],
                         color=palette["diff_line"], lw=0.8, alpha=0.5, label="Raw")
        if processed is not None:
            self.ax.plot(processed["two_theta"], processed["intensity"],
                         color=palette["exp_line"], lw=1.2, label="Processed")
            bg = self.session.background
            if bg is not None and raw is not None:
                self.ax.plot(raw["two_theta"], bg, color=palette["calc_line"],
                             lw=1.0, ls="--", label="Background")
        peaks = self.session.peaks
        if peaks is not None and processed is not None:
            self.ax.plot(peaks["two_theta"], peaks["intensity"], "o",
                         color=palette["calc_line"], ms=4, label="Peaks")
        self.ax.set_xlabel("2θ (degrees)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Processing Preview")
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc="upper right")

    def _plot_identify(self, palette):
        pattern = self.session.active_pattern()
        if pattern is None:
            self.ax.set_title("Identify")
            return
        inten = np.asarray(pattern["intensity"], dtype=float)
        max_i = np.max(inten) if len(inten) else 1.0
        norm = (inten / max_i * 100.0) if max_i > 0 else inten
        self.ax.plot(pattern["two_theta"], norm, color=palette["exp_line"],
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
            self.ax.vlines(tt, 0, tnorm, colors=colors[i % len(colors)],
                           alpha=0.7, lw=1.0, label=name)

        peaks = self.session.peaks
        if peaks is not None:
            pi = np.asarray(peaks["intensity"], dtype=float)
            pmax = np.max(pi) if len(pi) else 1.0
            self.ax.plot(peaks["two_theta"], (pi / pmax * 100.0) if pmax > 0 else pi,
                         "o", color=palette["calc_line"], ms=3, label="Peaks")

        self.ax.set_xlabel("2θ (degrees)")
        self.ax.set_ylabel("Normalized Intensity")
        self.ax.set_title("Phase Identification")
        self.ax.set_ylim(0, 110)
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc="upper right", fontsize=8)

    def _plot_refine(self, palette):
        results = self.session.lebail_results
        pattern = self.session.active_pattern()
        if results and results.get("success"):
            rr = results.get("refinement_results") or {}
            tt = rr.get("two_theta", pattern["two_theta"] if pattern else None)
            exp = rr.get("experimental_intensity")
            calc = rr.get("calculated_pattern")
            if tt is not None and exp is not None:
                self.ax.plot(tt, exp, color=palette["exp_line"], lw=1.2, label="Experimental")
            if tt is not None and calc is not None:
                self.ax.plot(tt, calc, color=palette["calc_line"], lw=1.2, label="Calculated")
                if exp is not None:
                    diff = np.asarray(exp) - np.asarray(calc)
                    offset = -0.15 * (np.max(exp) if len(exp) else 0)
                    self.ax.plot(tt, diff + offset, color=palette["diff_line"],
                                 lw=0.8, label="Difference")
            self.ax.set_title("Le Bail Refinement")
        elif pattern is not None:
            self.ax.plot(pattern["two_theta"], pattern["intensity"],
                         color=palette["exp_line"], lw=1.2, label="Experimental")
            self.ax.set_title("Refine — run Le Bail to see fit")
        self.ax.set_xlabel("2θ (degrees)")
        self.ax.set_ylabel("Intensity")
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc="upper right")
