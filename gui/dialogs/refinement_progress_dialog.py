"""Live view of a refinement while it runs."""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton,
    QVBoxLayout,
)

from matplotlib_config import apply_plot_style, get_plot_palette
from gui import display_settings
from gui.theme import get_current_mode
from gui.widgets.plot_host import create_plot_host


class RefinementWorker(QThread):
    """
    Runs the refinement off the GUI thread.

    The refinement is a long stretch of numpy work with no natural yield points,
    so on the GUI thread it would block every repaint and a progress window
    would sit there frozen. Here the engine's callbacks fire on this thread and
    the signals carry the results back, which Qt queues onto the GUI thread.
    """

    progressed = pyqtSignal(dict)
    logged = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, analyzer, call_kwargs, parent=None):
        super().__init__(parent)
        self._analyzer = analyzer
        self._kwargs = call_kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            results = self._analyzer.perform_lebail_refinement(
                progress_callback=self.progressed.emit,
                log_callback=self.logged.emit,
                cancel_check=lambda: self._cancel,
                **self._kwargs,
            )
        except Exception as e:  # noqa: BLE001 - reported to the user verbatim
            self.failed.emit(str(e))
            return
        self.finished_ok.emit(results or {})


class RefinementProgressDialog(QDialog):
    """
    Shows the fit taking shape, cycle by cycle, and offers a way to stop.

    Watching the calculated curve is how you tell early that a refinement is
    going wrong -- peaks walking away from the data, a phase collapsing -- which
    is worth knowing before sitting through the remaining cycles.
    """

    def __init__(self, worker: RefinementWorker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.results = None
        self.error = None
        self.cancelled = False

        self._rwp_trace = []
        self._observed = None
        self._two_theta = None

        self.setWindowTitle("Le Bail Refinement")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(720, 560)
        self.resize(880, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.status = QLabel("Starting…")
        self.status.setStyleSheet("font-weight: 600;")
        root.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate until the cycle count is known
        root.addWidget(self.progress)

        host, self.figure, self.canvas, _toolbar = create_plot_host(
            self, figsize=(8, 5), with_toolbar=False
        )
        self.fit_ax = self.figure.add_subplot(2, 1, 1)
        self.trace_ax = self.figure.add_subplot(2, 1, 2)
        root.addWidget(host, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFixedHeight(120)
        root.addWidget(self.log)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip(
            "Stop after the current phase and keep the fit reached so far"
        )
        self.stop_btn.clicked.connect(self.request_stop)
        buttons.addWidget(self.stop_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        buttons.addWidget(self.close_btn)
        root.addLayout(buttons)

        worker.progressed.connect(self.on_progress)
        worker.logged.connect(self.on_log)
        worker.finished_ok.connect(self.on_finished)
        worker.failed.connect(self.on_failed)

        self._draw()
        # The worker has to start once the dialog's event loop is running,
        # otherwise a fast refinement could finish before exec_ begins and the
        # completion signal would arrive with nothing listening for it.
        QTimer.singleShot(0, worker.start)

    def set_observed(self, two_theta, intensity):
        """The measured pattern, drawn once and left as the backdrop."""
        self._two_theta = np.asarray(two_theta, dtype=float)
        self._observed = np.asarray(intensity, dtype=float)

    # --- worker signals ---

    def on_log(self, message: str):
        text = message.strip()
        if text:
            self.log.appendPlainText(text)

    def on_progress(self, payload: dict):
        total = payload.get("total_iterations") or 0
        iteration = payload.get("iteration") or 0
        if total:
            self.progress.setRange(0, int(total))
            self.progress.setValue(int(iteration))

        parts = []
        message = payload.get("message")
        if message:
            parts.append(str(message))
        if total:
            parts.append(f"cycle {iteration} of {total}")
        factors = payload.get("r_factors") or {}
        if factors.get("Rwp") is not None:
            parts.append(f"Rwp = {float(factors['Rwp']):.2f}%")
        self.status.setText("  ·  ".join(parts) or "Refining…")

        if payload.get("phase_of_work") != "cycle":
            return

        if factors.get("Rwp") is not None:
            self._rwp_trace.append((iteration, float(factors["Rwp"])))
        calculated = payload.get("calculated_pattern")
        self._draw(calculated)

    def on_finished(self, results: dict):
        self.results = results
        self.cancelled = bool(results.get("cancelled"))
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.close_btn.setDefault(True)
        if self.cancelled:
            self.status.setText("Stopped — the fit reached so far has been kept")
            return
        self.accept()

    def on_failed(self, message: str):
        self.error = message
        self.status.setText(f"Refinement failed: {message}")
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.close_btn.setDefault(True)

    def request_stop(self):
        self.worker.cancel()
        self.stop_btn.setEnabled(False)
        self.status.setText("Stopping after the current phase…")

    # --- drawing ---

    def _draw(self, calculated=None):
        mode = get_current_mode()
        palette = get_plot_palette(mode)
        lw = display_settings.line_width()

        self.fit_ax.clear()
        if self._observed is not None:
            self.fit_ax.plot(
                self._two_theta, self._observed, color=palette["exp_line"],
                lw=lw, label="Observed",
            )
        if calculated is not None and self._observed is not None:
            calculated = np.asarray(calculated, dtype=float)
            if len(calculated) == len(self._observed):
                self.fit_ax.plot(
                    self._two_theta, calculated, color=palette["calc_line"],
                    lw=lw, label="Calculated",
                )
                difference = self._observed - calculated
                offset = -0.15 * float(np.max(self._observed)) if len(self._observed) else 0.0
                self.fit_ax.plot(
                    self._two_theta, difference + offset,
                    color=palette["diff_line"], lw=display_settings.line_width(0.7),
                    label="Difference",
                )
        self.fit_ax.set_xlabel("2θ (degrees)")
        self.fit_ax.set_ylabel("Intensity")
        if self.fit_ax.get_legend_handles_labels()[0]:
            self.fit_ax.legend(loc="upper right", fontsize=8)

        self.trace_ax.clear()
        if self._rwp_trace:
            cycles = [point[0] for point in self._rwp_trace]
            values = [point[1] for point in self._rwp_trace]
            self.trace_ax.plot(
                cycles, values, "o-", color=palette["calc_line"],
                lw=lw, ms=display_settings.marker_size(3.0),
            )
            # Once a refinement settles, Rwp moves by hundredths. Autoscaling
            # magnifies that into a dramatic-looking slope, so hold a floor on
            # the span and let a converged run read as the flat line it is.
            low, high = min(values), max(values)
            floor = max(0.05, 0.02 * abs(high))
            if high - low < floor:
                middle = 0.5 * (low + high)
                self.trace_ax.set_ylim(middle - floor / 2, middle + floor / 2)
        self.trace_ax.set_xlabel("Cycle")
        self.trace_ax.set_ylabel("Rwp (%)")
        self.trace_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Late cycles move Rwp by hundredths, and the default offset notation
        # turns that into an unreadable "1e-6+4.09e1" header
        self.trace_ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        self.trace_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        self.figure.tight_layout()
        apply_plot_style(self.figure, mode, show_grid=display_settings.show_grid())
        self.canvas.draw_idle()

    # --- lifetime ---

    def reject(self):
        """Escape and the window close button mean stop, not abandon."""
        if self.worker.isRunning():
            self.request_stop()
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.request_stop()
            event.ignore()
            return
        super().closeEvent(event)
