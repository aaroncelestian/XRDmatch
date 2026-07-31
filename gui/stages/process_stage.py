"""Process stage — background subtraction and peak finding (wide control bars)."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from gui.widgets.control_bar import ControlRow, OptionsDialog


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    try:
        L = len(y)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = diags(w, 0, shape=(L, L))
        for _ in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
    except Exception:
        return np.zeros_like(y)


class ProcessStage(QWidget):
    """Coordinator for background + peak finding; UI lives in two wide panels."""

    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self._background = None
        self._bg_options = None
        self._peak_options = None

        self.background_panel = self._build_background_panel()
        self.peaks_panel = self._build_peaks_panel()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

    # --- background panel ---

    def _build_background_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        row = ControlRow()
        self.apply_btn = QPushButton("Apply Processing")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.setToolTip("Apply background subtraction and smoothing")
        self.apply_btn.clicked.connect(self.apply_processing)
        row.add_widget(self.apply_btn)
        row.add_separator()

        self.enable_bg = QCheckBox("ALS background")
        self.enable_bg.setChecked(True)
        row.add_widget(self.enable_bg)

        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setRange(2, 8)
        self.lambda_slider.setValue(5)
        self.lambda_label = QLabel("1e5")
        self.lambda_slider.valueChanged.connect(
            lambda v: self.lambda_label.setText(f"1e{v}")
        )
        row.add_field("Smoothness λ:", self.lambda_slider, 120)
        row.add_widget(self.lambda_label)

        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(0.001, 0.1)
        self.p_spin.setDecimals(3)
        self.p_spin.setSingleStep(0.001)
        self.p_spin.setValue(0.01)
        row.add_field("Asymmetry p:", self.p_spin, 84)

        self.displacement = QDoubleSpinBox()
        self.displacement.setRange(-2.0, 2.0)
        self.displacement.setDecimals(4)
        self.displacement.setSingleStep(0.001)
        self.displacement.setValue(0.0)
        self.displacement.setSuffix("°")
        row.add_field("2θ offset:", self.displacement, 100)

        row.add_separator()
        options_btn = QPushButton("Options…")
        options_btn.setToolTip("Smoothing, noise reduction, and ALS iterations")
        options_btn.clicked.connect(self._show_bg_options)
        row.add_widget(options_btn)
        row.add_stretch()
        layout.addWidget(row)

        self.bg_status = QLabel("Load a pattern first.")
        self.bg_status.setObjectName("mutedLabel")
        self.bg_status.setWordWrap(True)
        self.bg_status.setContentsMargins(8, 0, 8, 4)
        layout.addWidget(self.bg_status)
        layout.addStretch()

        # Advanced widgets live in the popup but are owned by this stage
        self.iterations = QSpinBox()
        self.iterations.setRange(5, 50)
        self.iterations.setValue(10)

        self.enable_smooth = QCheckBox("Smoothing")
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(3, 21)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setValue(5)

        self.enable_noise = QCheckBox("Median noise reduction")
        return panel

    def _show_bg_options(self):
        if self._bg_options is None:
            dlg = OptionsDialog(
                "Background Options",
                self.workspace.window(),
                "Applied the next time you press Apply Processing.",
            )
            dlg.add_row("ALS iterations:", self.iterations)
            dlg.add_row("", self.enable_smooth)
            dlg.add_row("Smooth window:", self.smooth_window)
            dlg.add_row("", self.enable_noise)
            self._bg_options = dlg
        self._bg_options.show_centered()

    # --- peaks panel ---

    def _build_peaks_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        row = ControlRow()
        self.peaks_btn = QPushButton("Find Peaks")
        self.peaks_btn.setObjectName("primaryButton")
        self.peaks_btn.setToolTip("Detect peaks on the processed pattern")
        self.peaks_btn.clicked.connect(self.find_peaks)
        row.add_widget(self.peaks_btn)

        self.clear_peaks_btn = QPushButton("Clear Peaks")
        self.clear_peaks_btn.setToolTip("Discard the current peak list")
        self.clear_peaks_btn.clicked.connect(self.clear_peaks)
        row.add_widget(self.clear_peaks_btn)
        row.add_separator()

        self.sensitivity = QComboBox()
        self.sensitivity.addItems(["High", "Medium", "Low"])
        self.sensitivity.setCurrentIndex(1)
        self.sensitivity.setToolTip(
            "High/Medium/Low sets relative prominence to ~1% / 2.5% / 5% of max intensity"
        )
        row.add_field("Sensitivity:", self.sensitivity, 92)

        self.min_height = QSpinBox()
        self.min_height.setRange(1, 100000)
        self.min_height.setValue(50)
        row.add_field("Min height:", self.min_height, 88)

        self.min_prominence = QSpinBox()
        self.min_prominence.setRange(1, 100000)
        self.min_prominence.setValue(20)
        row.add_field("Prominence:", self.min_prominence, 88)

        self.min_sep = QDoubleSpinBox()
        self.min_sep.setRange(0.02, 2.0)
        self.min_sep.setDecimals(2)
        self.min_sep.setSingleStep(0.02)
        self.min_sep.setValue(0.12)
        self.min_sep.setSuffix("°")
        self.min_sep.setToolTip(
            "Minimum 2θ separation between peaks (reduces duplicate picks on one peak)"
        )
        row.add_field("Min 2θ sep:", self.min_sep, 88)

        row.add_separator()
        options_btn = QPushButton("Options…")
        options_btn.setToolTip("Peak width and detection smoothing")
        options_btn.clicked.connect(self._show_peak_options)
        row.add_widget(options_btn)
        row.add_stretch()
        layout.addWidget(row)

        self.peak_status = QLabel("Apply background, then find peaks.")
        self.peak_status.setObjectName("mutedLabel")
        self.peak_status.setWordWrap(True)
        self.peak_status.setContentsMargins(8, 0, 8, 4)
        layout.addWidget(self.peak_status)

        self.min_width = QSpinBox()
        self.min_width.setRange(1, 50)
        self.min_width.setValue(2)
        self.min_width.setToolTip("Minimum peak width in data points")

        self.detect_smooth = QCheckBox("Smooth for detection only")
        self.detect_smooth.setChecked(True)
        self.detect_smooth.setToolTip(
            "Light smoothing before find_peaks to suppress shoulder false positives; "
            "reported intensities still use the unsmoothed pattern"
        )
        return panel

    def _show_peak_options(self):
        if self._peak_options is None:
            dlg = OptionsDialog(
                "Peak Detection Options",
                self.workspace.window(),
                "Applied the next time you press Find Peaks.",
            )
            dlg.add_row("Peak min width (pts):", self.min_width)
            dlg.add_row("", self.detect_smooth)
            self._peak_options = dlg
        self._peak_options.show_centered()

    @property
    def status(self):
        """Back-compat alias used by older call sites."""
        return self.peak_status

    def on_enter(self):
        if self.session.has_pattern():
            self.bg_status.setText("Adjust background, then Apply Processing.")
            self.peak_status.setText("Apply background/smoothing, then Find Peaks.")
            if self.session.processed_pattern is None:
                self.apply_processing(silent=True)
        else:
            self.bg_status.setText("Load a pattern first.")
            self.peak_status.setText("Load a pattern first.")

    def apply_processing(self, silent=False):
        raw = self.session.raw_pattern
        if raw is None:
            if not silent:
                QMessageBox.warning(self, "No Pattern", "Load a pattern first.")
            return

        intensity = np.asarray(raw["intensity"], dtype=float).copy()
        two_theta = np.asarray(raw["two_theta"], dtype=float).copy()

        offset = self.displacement.value()
        if abs(offset) > 1e-12:
            two_theta = two_theta + offset

        background = None
        if self.enable_bg.isChecked():
            lam = 10 ** self.lambda_slider.value()
            background = als_baseline(
                intensity,
                lam=lam,
                p=self.p_spin.value(),
                niter=self.iterations.value(),
            )
            intensity = np.maximum(intensity - background, 0)

        if self.enable_smooth.isChecked():
            intensity = uniform_filter1d(intensity, size=self.smooth_window.value())
        if self.enable_noise.isChecked():
            intensity = median_filter(intensity, size=3)

        processed = {
            "two_theta": two_theta,
            "intensity": intensity,
            "intensity_error": raw.get("intensity_error"),
            "file_path": raw.get("file_path"),
            "file_format": raw.get("file_format"),
            "wavelength": self.session.wavelength,
            "processed": True,
        }
        self._background = background
        self.session.set_processed_pattern(processed, background=background)
        self.workspace.refresh_plot()
        if not silent:
            self.bg_status.setText("Processing applied.")
            self.workspace.set_status("Processing applied")

    def clear_peaks(self):
        if not self.session.has_peaks():
            self.peak_status.setText("No peaks to clear.")
            return
        self.session.set_peaks(None)
        self.workspace.clear_peaks_table()
        self.peak_status.setText("Peak list cleared.")
        self.workspace.set_status("Peaks cleared")
        self.workspace.refresh_plot()

    def find_peaks(self):
        if self.session.processed_pattern is None:
            self.apply_processing(silent=True)
        pattern = self.session.processed_pattern
        if pattern is None:
            QMessageBox.warning(self, "No Pattern", "Load and process a pattern first.")
            return

        intensity = np.asarray(pattern["intensity"], dtype=float)
        two_theta = np.asarray(pattern["two_theta"], dtype=float)
        if len(intensity) < 5:
            QMessageBox.warning(self, "No Peaks", "Pattern is too short for peak finding.")
            return

        min_height = self.min_height.value()
        min_prominence = self.min_prominence.value()
        min_width = self.min_width.value()
        min_sep = self.min_sep.value()
        sensitivity = self.sensitivity.currentIndex()

        detect = intensity.copy()
        if self.detect_smooth.isChecked():
            win = max(3, min(11, int(round(0.08 / max(self._median_step(two_theta), 1e-6)))))
            if win % 2 == 0:
                win += 1
            detect = uniform_filter1d(detect, size=win)

        imax = float(np.max(detect)) if len(detect) else 1.0
        noise = self._estimate_noise(detect)

        rel_frac = (0.01, 0.025, 0.05)[sensitivity]
        prominence_threshold = max(float(min_prominence), rel_frac * imax, 3.0 * noise)
        height_threshold = max(float(min_height), 2.0 * noise)

        step = self._median_step(two_theta)
        dist_pts = max(2, int(np.ceil(min_sep / max(step, 1e-6))))

        peaks_idx, _ = find_peaks(
            detect,
            height=height_threshold,
            distance=dist_pts,
            prominence=prominence_threshold,
            width=max(1, min_width),
        )
        peaks_idx = [int(i) for i in peaks_idx if two_theta[int(i)] >= 3.0]

        peaks_idx = [
            self._refine_to_local_max(intensity, i, half_window=max(2, dist_pts // 2))
            for i in peaks_idx
        ]
        peaks_idx = sorted(set(peaks_idx))
        peaks_idx = self._merge_nearby_peaks(peaks_idx, two_theta, intensity, min_sep)

        if not peaks_idx:
            QMessageBox.warning(
                self, "No Peaks",
                "No peaks found. Try High sensitivity, lower prominence, or smaller 2θ separation.",
            )
            return

        peak_tt = two_theta[peaks_idx]
        peak_int = intensity[peaks_idx]
        wl = self.session.wavelength
        with np.errstate(divide="ignore", invalid="ignore"):
            d_spacing = wl / (2.0 * np.sin(np.radians(peak_tt / 2.0)))
            d_spacing = np.where(np.isfinite(d_spacing), d_spacing, np.nan)

        peaks = {
            "indices": np.asarray(peaks_idx, dtype=int),
            "two_theta": peak_tt,
            "intensity": peak_int,
            "d_spacing": d_spacing,
            "wavelength": self.session.wavelength,
        }
        self.session.set_peaks(peaks)
        self.peak_status.setText(
            f"Found {len(peaks_idx)} peaks "
            f"(sep≥{min_sep:.2f}°, prom≥{prominence_threshold:.0f})."
        )
        self.workspace.refresh_plot()
        self.workspace.set_results_peaks(peaks)
        self.workspace.set_status(f"Found {len(peaks_idx)} peaks")

    @staticmethod
    def _median_step(two_theta: np.ndarray) -> float:
        d = np.diff(np.asarray(two_theta, dtype=float))
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if len(d) else 0.01

    @staticmethod
    def _estimate_noise(intensity: np.ndarray) -> float:
        y = np.asarray(intensity, dtype=float)
        n = len(y)
        if n < 10:
            return float(np.std(y)) if n else 1.0
        chunk = y[: max(20, min(200, n // 8))]
        med = np.median(chunk)
        mad = np.median(np.abs(chunk - med))
        return float(max(1.4826 * mad, np.std(chunk) * 0.5, 1.0))

    @staticmethod
    def _refine_to_local_max(intensity: np.ndarray, idx: int, half_window: int = 3) -> int:
        lo = max(0, idx - half_window)
        hi = min(len(intensity), idx + half_window + 1)
        return int(lo + np.argmax(intensity[lo:hi]))

    @staticmethod
    def _merge_nearby_peaks(idxs, two_theta, intensity, min_sep_deg: float):
        if not idxs:
            return []
        order = sorted(idxs, key=lambda i: float(intensity[i]), reverse=True)
        kept = []
        for i in order:
            tt = float(two_theta[i])
            if all(abs(tt - float(two_theta[j])) >= min_sep_deg for j in kept):
                kept.append(i)
        return sorted(kept)
