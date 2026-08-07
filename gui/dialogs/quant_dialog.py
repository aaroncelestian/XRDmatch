"""Quant Analysis dialog — Le Bail plot and results; controls live in Parameters."""

from __future__ import annotations

import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from matplotlib_config import apply_plot_style, draw_error_bars, get_plot_palette
from gui import display_settings, refinement_table
from gui.theme import get_current_mode
from gui.dialogs.refinement_details_dialog import RefinementDetailsDialog
from gui.focus import hold_focus
from gui.widgets.copyable_table import CopyableTable
from gui.widgets.plot_host import create_plot_host


class QuantDialog(QDialog):
    """Non-modal plot window for quantitative / Le Bail analysis."""

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
        root.setSpacing(2)

        host, self.quant_figure, self.quant_canvas, self.quant_toolbar = create_plot_host(
            self, figsize=(9, 6)
        )
        self.quant_ax = self.quant_figure.add_subplot(111)
        self.figure = self.quant_figure
        self.canvas = self.quant_canvas
        self.ax = self.quant_ax
        root.addWidget(host, 1)

        results_wrap = QWidget()
        qr = QVBoxLayout(results_wrap)
        qr.setContentsMargins(4, 0, 4, 4)
        qr.setSpacing(2)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self.quant_results_label = QLabel("Refinement")
        self.quant_results_label.setObjectName("mutedLabel")
        # The line ends with caveats about how the numbers were arrived at, so
        # it has to wrap rather than clip the very part that qualifies them
        self.quant_results_label.setWordWrap(True)
        header_row.addWidget(self.quant_results_label, 1)
        self.details_btn = QPushButton("Parameters…")
        self.details_btn.setToolTip(
            "Open the refinement controls: global and per-phase parameters, "
            "Run Le Bail, and export."
        )
        self.details_btn.clicked.connect(self.show_details)
        header_row.addWidget(self.details_btn)
        qr.addLayout(header_row)

        self.quant_results_table = CopyableTable()
        self.quant_results_table.setMaximumHeight(190)
        self.quant_results_table.horizontalHeader().setStretchLastSection(True)
        self.quant_results_table.setToolTip(
            "Ctrl-C copies the selection, or the whole table when nothing is selected"
        )
        qr.addWidget(self.quant_results_table)
        root.addWidget(results_wrap)

        self._details_dialog = None

        session.refinement_changed.connect(self.refresh_plot)
        session.matches_changed.connect(self._on_phases_changed)
        session.pattern_changed.connect(self._on_phases_changed)

    @property
    def refine_stage(self):
        """Controls live in the Parameters window; exposed for callers/tests."""
        self._ensure_details()
        return self._details_dialog.refine_stage

    def _ensure_details(self):
        if self._details_dialog is None:
            self._details_dialog = RefinementDetailsDialog(self.session, self)
            # Closing the parameter window should come back here, not to the
            # main window behind it
            self._details_dialog.finished.connect(lambda _result: hold_focus(self))

    def show_details(self):
        self._ensure_details()
        self._details_dialog.show()
        self._details_dialog.raise_()
        self._details_dialog.activateWindow()

    def set_status(self, message: str):
        if self._status_callback:
            self._status_callback(message)

    def _on_phases_changed(self):
        if self._details_dialog is not None and hasattr(
            self._details_dialog.refine_stage, "on_enter"
        ):
            self._details_dialog.refine_stage.on_enter()
        if self._details_dialog is not None and self._details_dialog.isVisible():
            self._details_dialog.refresh()
        self.refresh_plot()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_plot()
        # Parameters owns Run / export / all refinement controls
        self.show_details()

    def closeEvent(self, event):
        if self._details_dialog is not None:
            self._details_dialog.close()
        super().closeEvent(event)

    def on_theme_changed(self, mode: str):
        apply_plot_style(self.quant_figure, mode, show_grid=display_settings.show_grid())
        self.refresh_plot()

    def on_display_settings_changed(self, prefs: dict):
        dpi = prefs.get("plot_dpi")
        if dpi is not None and self._details_dialog is not None:
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

    def _update_results_table(self):
        """Refinement details: Le Bail when it has run, otherwise RIR quant."""
        results = self.session.lebail_results
        if results and results.get("success"):
            self._show_lebail_details(results)
        elif self.session.rir_results:
            self._show_rir_details(self.session.rir_results)
        else:
            self.quant_results_label.setText(
                "Refinement — run Le Bail from Parameters, or RIR Quant in Phases"
            )
            self.quant_results_table.set_content([], [])
        self.details_btn.setEnabled(True)

    def _show_lebail_details(self, results: dict):
        parts = refinement_table.summary_headline(results)
        self.quant_results_label.setText("Refinement — " + "  ·  ".join(parts))
        self.quant_results_label.setToolTip(
            refinement_table.weight_basis_note(results) or ""
        )

        labels = [label for label, _ in refinement_table.SUMMARY_COLUMNS]
        tips = [tip for _, tip in refinement_table.SUMMARY_COLUMNS]
        self.quant_results_table.set_content(
            labels,
            refinement_table.summary_rows(results),
            tooltips=refinement_table.summary_tooltips(results),
            header_tooltips=tips,
        )

    def _show_rir_details(self, result: dict):
        header = [
            f"fit Rwp={result.get('rwp', float('nan')):.1f}%",
            f"{result.get('explained_fraction', 0.0) * 100:.0f}% of intensity explained",
            f"FWHM={result.get('fwhm', 0.0):.3f}°",
        ]
        if result.get("missing_rir"):
            header.append(f"no RIR: {', '.join(result['missing_rir'][:3])}")
        self.quant_results_label.setText("RIR quantification — " + "  ·  ".join(header))

        self.quant_results_table.set_content(
            [label for label, _ in refinement_table.RIR_COLUMNS],
            refinement_table.rir_rows(result),
            header_tooltips=[tip for _, tip in refinement_table.RIR_COLUMNS],
        )

    def _plot_title(self, text: str) -> str:
        """Name the data file so an exported figure identifies itself."""
        name = os.path.basename(self.session.file_path or "")
        return f"{text} ({name})" if name else text

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
            ax.set_title(self._plot_title("Le Bail Refinement"))
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
            ax.set_title(
                self._plot_title(
                    "RIR Quantification — fixed reference patterns, scale only"
                )
            )
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
            ax.set_title(self._plot_title("Quant — run Le Bail from Parameters"))
        else:
            ax.text(
                0.5, 0.5, "Match phases, then run Le Bail from Parameters",
                ha="center", va="center", transform=ax.transAxes,
                color=palette.get("muted", palette["tick"]),
            )
            ax.set_title("Quant Analysis")
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity")
        if display_settings.show_legend() and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")
