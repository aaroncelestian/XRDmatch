"""Quant Analysis dialog — Le Bail refinement in its own window."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from matplotlib_config import apply_plot_style, draw_error_bars, get_plot_palette
from gui import display_settings
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
        self.quant_results_table.setMaximumHeight(190)
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
        apply_plot_style(self.quant_figure, mode, show_grid=display_settings.show_grid())
        self.refresh_plot()

    def on_display_settings_changed(self, prefs: dict):
        dpi = prefs.get("plot_dpi")
        if dpi is not None:
            self.refine_stage.dpi.setValue(int(dpi))
        self.refresh_plot()

    def refresh_plot(self):
        mode = get_current_mode()
        ax = self.quant_ax
        ax.clear()
        palette = get_plot_palette(mode)
        self._plot_refine(ax, palette)
        apply_plot_style(self.quant_figure, mode, show_grid=display_settings.show_grid())
        self.quant_canvas.draw_idle()
        self._update_results_table()

    REFINE_COLUMNS = [
        "Phase", "wt%", "Scale", "a (Å)", "c (Å)", "V (Å³)",
        "Lattice Δ%", "Absorb.", "Harmonics", "Contrib.%",
    ]
    RIR_COLUMNS = ["Phase", "wt%", "Scale", "Fitted I", "RIR", "Pattern share %"]

    def _update_results_table(self):
        """Refinement details: Le Bail when it has run, otherwise RIR quant."""
        results = self.session.lebail_results
        if results and results.get("success"):
            self._show_lebail_details(results)
        elif self.session.rir_results:
            self._show_rir_details(self.session.rir_results)
        else:
            self.quant_results_label.setText(
                "Refinement — run Le Bail, or RIR Quant in the Phases tab"
            )
            self.quant_results_table.clear()
            self.quant_results_table.setRowCount(0)
            self.quant_results_table.setColumnCount(0)

    def _set_columns(self, columns):
        self.quant_results_table.clear()
        self.quant_results_table.setColumnCount(len(columns))
        self.quant_results_table.setHorizontalHeaderLabels(columns)

    @staticmethod
    def _cell(text: str, tooltip: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        if tooltip:
            item.setToolTip(tooltip)
        return item

    def _show_lebail_details(self, results: dict):
        inner = results.get("refinement_results") or {}
        factors = inner.get("final_r_factors") or results.get("r_factors") or {}
        summary = inner.get("phase_summary") or []
        globals_ = inner.get("global_parameters") or {}

        header = []
        for key, label in (("Rwp", "Rwp"), ("Rp", "Rp"), ("GoF", "GoF")):
            value = factors.get(key)
            if value is not None:
                header.append(f"{label}={float(value):.2f}" + ("%" if key != "GoF" else ""))
        header.append(f"zero={globals_.get('zero_shift', 0.0):+.4f}°")
        header.append(f"disp={globals_.get('displacement', 0.0):+.4f}°")
        if inner.get("iterations"):
            header.append(f"{inner['iterations']} cycles")
        if inner.get("intensity_model") == "extract":
            header.append("Le Bail extraction — wt% unavailable")
        self.quant_results_label.setText("Refinement — " + "  ·  ".join(header))

        self._set_columns(self.REFINE_COLUMNS)
        self.quant_results_table.setRowCount(len(summary))
        for row, phase in enumerate(summary):
            cell = phase.get("unit_cell") or {}
            base = phase.get("base_unit_cell") or {}
            lattice = phase.get("lattice_scale", 1.0)
            coeffs = phase.get("harmonic_coeffs") or []
            wt = phase.get("weight_percent")
            contribution = phase.get("contribution_percent")

            values = [
                (str(phase.get("name", f"Phase {row + 1}")), phase.get("formula", "")),
                (f"{wt:.1f}" if wt is not None else "—",
                 "No RIR value in the database" if phase.get("rir") is None
                 else "Chung RIR weight percent"),
                (f"{phase.get('scale', 0.0):.4g}", ""),
                (f"{cell.get('a', 0.0):.4f}", f"start {base.get('a', 0.0):.4f} Å"),
                (f"{cell.get('c', 0.0):.4f}", f"start {base.get('c', 0.0):.4f} Å"),
                (f"{cell.get('volume', 0.0):.2f}", ""),
                (f"{(lattice - 1.0) * 100:+.3f}", "Isotropic lattice dilation"),
                (f"{phase.get('absorption', 0.0):+.4f}", ""),
                (", ".join(f"{c:+.3f}" for c in coeffs) if any(coeffs) else "—",
                 "Even-order harmonic coefficients c2, c4, c6"),
                (f"{contribution:.1f}" if contribution is not None else "—",
                 "Share of the calculated pattern intensity"),
            ]
            for column, (text, tooltip) in enumerate(values):
                self.quant_results_table.setItem(row, column, self._cell(text, tooltip))
        self.quant_results_table.resizeColumnsToContents()

    def _show_rir_details(self, result: dict):
        phases = result.get("phases") or []
        header = [
            f"fit Rwp={result.get('rwp', float('nan')):.1f}%",
            f"{result.get('explained_fraction', 0.0) * 100:.0f}% of intensity explained",
            f"FWHM={result.get('fwhm', 0.0):.3f}°",
        ]
        if result.get("missing_rir"):
            header.append(f"no RIR: {', '.join(result['missing_rir'][:3])}")
        self.quant_results_label.setText("RIR quantification — " + "  ·  ".join(header))

        self._set_columns(self.RIR_COLUMNS)
        self.quant_results_table.setRowCount(len(phases))
        total_pattern = sum(p.get("pattern_intensity", 0.0) for p in phases) or 1.0
        for row, phase in enumerate(phases):
            wt = phase.get("weight_percent")
            rir = phase.get("rir")
            values = [
                (str(phase.get("name", f"Phase {row + 1}")), ""),
                (f"{wt:.1f}" if wt is not None else "—",
                 "No RIR value in the database" if rir is None else "Chung RIR weight percent"),
                (f"{phase.get('scale', 0.0):.4g}", ""),
                (f"{phase.get('line_intensity', 0.0):.4g}", "Strongest-line intensity from the fit"),
                (f"{rir:.3f}" if rir else "—", "I/I_corundum from AMCSD"),
                (f"{phase.get('pattern_intensity', 0.0) / total_pattern * 100:.1f}",
                 "Share of the fitted pattern intensity, before the RIR conversion"),
            ]
            for column, (text, tooltip) in enumerate(values):
                self.quant_results_table.setItem(row, column, self._cell(text, tooltip))
        self.quant_results_table.resizeColumnsToContents()

    def _plot_refine(self, ax, palette):
        results = self.session.lebail_results
        pattern = self.session.active_pattern()
        lw = display_settings.line_width()
        diff_lw = display_settings.line_width(0.7)
        if results and results.get("success"):
            rr = results.get("refinement_results") or {}
            tt = rr.get("two_theta", pattern["two_theta"] if pattern else None)
            exp = rr.get("experimental_intensity")
            calc = rr.get("calculated_pattern")
            if tt is not None and exp is not None:
                ax.plot(tt, exp, color=palette["exp_line"], lw=lw, label="Experimental")
            if tt is not None and calc is not None:
                ax.plot(tt, calc, color=palette["calc_line"], lw=lw, label="Calculated")
                if exp is not None:
                    diff = np.asarray(exp) - np.asarray(calc)
                    offset = -0.15 * (np.max(exp) if len(exp) else 0)
                    ax.plot(
                        tt, diff + offset, color=palette["diff_line"],
                        lw=diff_lw, label="Difference",
                    )
            ax.set_title("Le Bail Refinement")
        elif self.session.rir_results:
            rir = self.session.rir_results
            tt = rir["two_theta"]
            exp = np.asarray(rir["observed"])
            calc = np.asarray(rir["calculated"])
            ax.plot(tt, exp, color=palette["exp_line"], lw=lw, label="Experimental")
            ax.plot(tt, calc, color=palette["calc_line"], lw=lw, label="RIR fit")
            offset = -0.15 * (np.max(exp) if len(exp) else 0)
            ax.plot(
                tt, exp - calc + offset, color=palette["diff_line"],
                lw=diff_lw, label="Difference",
            )
            ax.set_title("RIR Quantification — fixed reference patterns, scale only")
        elif pattern is not None:
            ax.plot(
                pattern["two_theta"], pattern["intensity"],
                color=palette["exp_line"], lw=lw, label="Experimental",
            )
            if display_settings.show_error_bars():
                draw_error_bars(
                    ax, pattern["two_theta"], pattern["intensity"],
                    pattern.get("intensity_error"), palette["exp_line"],
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
        if display_settings.show_legend() and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")
