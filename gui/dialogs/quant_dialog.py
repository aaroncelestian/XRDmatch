"""Quant Analysis dialog — Le Bail refinement in its own window."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from matplotlib_config import apply_plot_style, get_plot_palette
from gui.theme import get_current_mode
from gui.widgets.plot_host import create_plot_host
from gui.stages.refine_stage import RefineStage


class QuantDialog(QDialog):
    """Non-modal tool window for quantitative / Le Bail analysis."""

    def __init__(self, session, parent=None, status_callback=None):
        super().__init__(parent)
        self.session = session
        self._status_callback = status_callback
        self.setWindowTitle("Quant Analysis")
        self.setWindowModality(Qt.NonModal)
        self.resize(1100, 720)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # RefineStage expects a "workspace" with plot + status helpers — use self
        self.refine_stage = RefineStage(session, self)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.refine_stage)
        left.setMinimumWidth(280)
        left.setMaximumWidth(420)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        host, self.quant_figure, self.quant_canvas, self.quant_toolbar = create_plot_host(
            right, figsize=(9, 6)
        )
        self.quant_ax = self.quant_figure.add_subplot(111)
        self.figure = self.quant_figure
        self.canvas = self.quant_canvas
        self.ax = self.quant_ax
        right_layout.addWidget(host, 1)

        results_wrap = QWidget()
        qr = QVBoxLayout(results_wrap)
        qr.setContentsMargins(4, 0, 4, 4)
        qr.setSpacing(2)
        self.quant_results_label = QLabel("Refinement")
        self.quant_results_label.setObjectName("mutedLabel")
        qr.addWidget(self.quant_results_label)
        self.quant_results_table = QTableWidget()
        self.quant_results_table.setMaximumHeight(140)
        self.quant_results_table.setAlternatingRowColors(True)
        self.quant_results_table.horizontalHeader().setStretchLastSection(True)
        qr.addWidget(self.quant_results_table)
        right_layout.addWidget(results_wrap)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 780])
        root.addWidget(splitter)

        session.refinement_changed.connect(self.refresh_plot)
        session.matches_changed.connect(self._on_phases_changed)
        session.pattern_changed.connect(self._on_phases_changed)

    def set_status(self, message: str):
        if self._status_callback:
            self._status_callback(message)

    def _on_phases_changed(self):
        if hasattr(self.refine_stage, "on_enter"):
            self.refine_stage.on_enter()
        self.refresh_plot()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self.refine_stage, "on_enter"):
            self.refine_stage.on_enter()
        self.refresh_plot()

    def on_theme_changed(self, mode: str):
        apply_plot_style(self.quant_figure, mode)
        self.refresh_plot()

    def refresh_plot(self):
        mode = get_current_mode()
        ax = self.quant_ax
        ax.clear()
        palette = get_plot_palette(mode)
        self._plot_refine(ax, palette)
        apply_plot_style(self.quant_figure, mode)
        self.quant_canvas.draw_idle()
        self._update_results_table()

    def _update_results_table(self):
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
                    name = (
                        p.get("mineral", p.get("mineral_name", f"Phase {i+1}"))
                        if isinstance(p, dict) else str(p)
                    )
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
                    ax.plot(
                        tt, diff + offset, color=palette["diff_line"],
                        lw=0.8, label="Difference",
                    )
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
