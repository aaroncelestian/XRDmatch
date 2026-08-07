"""Process stage — background subtraction and peak finding (wide control bars)."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from gui.widgets.control_bar import ControlRow, OptionsDialog
from utils.kalpha_filter import strip_alpha2_peaks


def als_baseline(y, lam=1e5, p=0.05, niter=10, center_noise=True):
    """
    Asymmetric least-squares baseline (Eilers & Boelens).

    Small *p* down-weights points above the curve so peaks do not pull the
    fit up — but left alone that also parks the curve under the noise floor.
    When *center_noise* is set, the finished curve is shifted so that the
    non-peak residuals sit about zero: the background runs through the middle
    of the continuum noise rather than along its lower envelope.
    """
    try:
        y = np.asarray(y, dtype=float)
        L = len(y)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = diags(w, 0, shape=(L, L))
        z = y.copy()
        for _ in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        if center_noise:
            z = _center_baseline_on_noise(y, z)
        return z
    except Exception:
        return np.zeros_like(y, dtype=float)


def _center_baseline_on_noise(y, z):
    """Shift z so continuum residuals (noise, not peaks) average near zero."""
    residual = np.asarray(y, dtype=float) - np.asarray(z, dtype=float)
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(residual)) or 1.0
    # Keep points that are not clear positive peaks
    mask = residual < med + 3.0 * scale
    if np.count_nonzero(mask) < max(10, len(residual) // 20):
        mask = np.ones(len(residual), dtype=bool)
    return z + float(np.median(residual[mask]))


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

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._live_preview)
        self._wire_live_preview()

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
        self.apply_btn.setToolTip(
            "Commit background subtraction and smoothing. The plot also updates "
            "live as you change the settings below."
        )
        self.apply_btn.clicked.connect(lambda: self.apply_processing(silent=False))
        row.add_widget(self.apply_btn)
        row.add_separator()

        self.enable_bg = QCheckBox("ALS background")
        self.enable_bg.setChecked(True)
        self.enable_bg.setToolTip(
            "Fit a smooth continuum under the pattern and subtract it. The curve "
            "is placed through the middle of the background noise, not along its "
            "lower edge."
        )
        row.add_widget(self.enable_bg)

        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setRange(2, 8)
        self.lambda_slider.setValue(5)
        self.lambda_label = QLabel("1e5")
        self.lambda_slider.valueChanged.connect(
            lambda v: self.lambda_label.setText(f"1e{v}")
        )
        self.lambda_slider.setToolTip(
            "Smoothness of the background. Higher λ → stiffer curve that ignores "
            "broad humps less; lower λ → more flexible and can follow peak feet."
        )
        row.add_field("Smoothness λ:", self.lambda_slider, 120)
        row.add_widget(self.lambda_label)

        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(0.001, 0.5)
        self.p_spin.setDecimals(3)
        self.p_spin.setSingleStep(0.005)
        self.p_spin.setValue(0.05)
        self.p_spin.setToolTip(
            "ALS asymmetry. Lower values ignore points above the curve more "
            "strongly (peaks cannot lift the fit, but the curve sits lower). "
            "Higher values (~0.1–0.2) track the continuum more closely. The fit "
            "is then recentered so residuals in the noise average near zero."
        )
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

    def _wire_live_preview(self):
        for signal in (
            self.enable_bg.stateChanged,
            self.lambda_slider.valueChanged,
            self.p_spin.valueChanged,
            self.displacement.valueChanged,
            self.iterations.valueChanged,
            self.enable_smooth.stateChanged,
            self.smooth_window.valueChanged,
            self.enable_noise.stateChanged,
        ):
            signal.connect(self._schedule_preview)

    def _schedule_preview(self, *_args):
        if not self.session.has_pattern():
            return
        self._preview_timer.start()

    def _live_preview(self):
        self.apply_processing(silent=True)

    def _show_bg_options(self):
        if self._bg_options is None:
            dlg = OptionsDialog(
                "Background Options",
                self.workspace.window(),
                "Changes update the plot live.",
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

        self.edit_peaks_btn = QPushButton("Edit on Plot")
        self.edit_peaks_btn.setCheckable(True)
        self.edit_peaks_btn.setToolTip(
            "Click the plot to add a peak, right-click a peak to remove it. "
            "Turn off to pan and zoom normally."
        )
        self.edit_peaks_btn.toggled.connect(self.workspace.set_peak_edit_mode)
        row.add_widget(self.edit_peaks_btn)
        row.add_separator()

        # Noise is the honest detection limit: a percent of the strongest peak
        # means very different things at 100 counts and at 100,000 counts
        self.snr_min = QDoubleSpinBox()
        self.snr_min.setRange(0.0, 20.0)
        self.snr_min.setDecimals(1)
        self.snr_min.setSingleStep(0.5)
        self.snr_min.setValue(3.0)
        self.snr_min.setSuffix("σ")
        self.snr_min.setToolTip(
            "How far a peak must stand above the noise measured next to it. "
            "3σ is a safe default; drop to 2σ to chase very weak lines, raise to "
            "5σ to keep only the obvious ones. Set 0 to disable."
        )
        row.add_field("Min S/N:", self.snr_min, 84)

        # Percent of max keeps detection scale-free across normalized and raw counts
        self.min_height = QDoubleSpinBox()
        self.min_height.setRange(0.0, 100.0)
        self.min_height.setDecimals(2)
        self.min_height.setSingleStep(0.10)
        self.min_height.setValue(10.0)
        self.min_height.setSuffix("%")
        self.min_height.setToolTip(
            "Extra height floor as a percent of the strongest peak. Left low, the "
            "S/N test decides; raise it to ignore weak lines you do not care about."
        )
        row.add_field("Min height:", self.min_height, 92)

        self.min_prominence = QDoubleSpinBox()
        self.min_prominence.setRange(0.0, 100.0)
        self.min_prominence.setDecimals(2)
        self.min_prominence.setSingleStep(0.10)
        self.min_prominence.setValue(5.0)
        self.min_prominence.setSuffix("%")
        self.min_prominence.setToolTip(
            "Extra prominence floor above the local baseline, as a percent of the "
            "strongest peak. Nothing else raises this."
        )
        row.add_field("Prominence:", self.min_prominence, 92)

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

        self.reject_alpha2 = QCheckBox("Reject Kα2 satellites")
        self.reject_alpha2.setChecked(True)
        self.reject_alpha2.setToolTip(
            "Drop peaks that sit at the Kα2 position of a stronger peak with a "
            "consistent intensity ratio"
        )

        self.alpha2_ratio_max = QDoubleSpinBox()
        self.alpha2_ratio_max.setRange(0.2, 1.0)
        self.alpha2_ratio_max.setDecimals(2)
        self.alpha2_ratio_max.setSingleStep(0.05)
        self.alpha2_ratio_max.setValue(0.75)
        self.alpha2_ratio_max.setToolTip(
            "Maximum satellite/parent intensity ratio treated as Kα2 (nominal 0.5)"
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
            dlg.add_row("", self.reject_alpha2)
            dlg.add_row("Max Kα2/Kα1 ratio:", self.alpha2_ratio_max)
            self._peak_options = dlg
        self._peak_options.show_centered()

    @property
    def status(self):
        """Back-compat alias used by older call sites."""
        return self.peak_status

    def on_enter(self):
        if not self.session.has_pattern():
            self.bg_status.setText("Load a pattern first.")
            self.peak_status.setText("Load a pattern first.")
            return

        # Do not auto-run ALS here — it blocks the UI on every file-tree click.
        # Processing starts when the user changes a control (live preview) or
        # presses Apply / Find Peaks.
        if self.session.processed_pattern is None:
            self.bg_status.setText(
                "Raw pattern loaded. Adjust λ / p for a live preview, "
                "or press Apply Processing."
            )
        else:
            self.bg_status.setText(
                "Background updates live as you change λ and p. "
                "The fit runs through the middle of the continuum noise."
            )
        # Entering the tab must not overwrite the result of a completed run
        if not self.session.has_peaks():
            self.peak_status.setText("Apply background/smoothing, then Find Peaks.")

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
                center_noise=True,
            )
            # Leave negative continuum noise; clipping would recreate a floor bias
            intensity = intensity - background

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
        elif self.enable_bg.isChecked():
            self.bg_status.setText(
                f"Live preview — λ=1e{self.lambda_slider.value()}, "
                f"p={self.p_spin.value():.3f}"
            )

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
        snr_min = self.snr_min.value()

        detect = intensity.copy()
        if self.detect_smooth.isChecked():
            win = max(3, min(11, int(round(0.08 / max(self._median_step(two_theta), 1e-6)))))
            if win % 2 == 0:
                win += 1
            detect = uniform_filter1d(detect, size=win)

        imax = float(np.max(detect)) if len(detect) else 1.0
        noise = self._estimate_noise(detect)

        prominence_threshold = min_prominence / 100.0 * imax
        height_threshold = min_height / 100.0 * imax

        step = self._median_step(two_theta)
        dist_pts = max(2, int(np.ceil(min_sep / max(step, 1e-6))))

        # Detect permissively, then apply the user's limits and a local noise
        # test explicitly, so nothing silently overrides the typed thresholds
        peaks_idx, props = find_peaks(
            detect,
            distance=dist_pts,
            prominence=max(0.5 * noise, 1e-12),
            width=max(1, min_width),
        )
        prominences = np.asarray(props.get("prominences", []), dtype=float)
        local_noise = self._local_noise(detect, dist_pts)
        keep = []
        for k, i in enumerate(peaks_idx):
            i = int(i)
            if two_theta[i] < 3.0:
                continue
            if detect[i] < height_threshold:
                continue
            prom = float(prominences[k]) if k < len(prominences) else 0.0
            if prom < prominence_threshold:
                continue
            if snr_min > 0 and prom < snr_min * float(local_noise[i]):
                continue
            keep.append(i)
        peaks_idx = keep

        peaks_idx = [
            self._refine_to_local_max(intensity, i, half_window=max(2, dist_pts // 2))
            for i in peaks_idx
        ]
        peaks_idx = sorted(set(peaks_idx))
        peaks_idx = self._merge_nearby_peaks(peaks_idx, two_theta, intensity, min_sep)

        if not peaks_idx:
            QMessageBox.warning(
                self, "No Peaks",
                "No peaks found. Lower Min height and Prominence, reduce Min local S/N "
                "in Options, or use a smaller 2θ separation.",
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

        alpha2_note = ""
        if self.reject_alpha2.isChecked():
            peaks, satellites = strip_alpha2_peaks(
                peaks, wl, max_intensity_ratio=self.alpha2_ratio_max.value()
            )
            if satellites:
                alpha2_note = f", removed {len(satellites)} Kα2"
                for s in satellites:
                    print(
                        f"   Kα2 satellite at {s['two_theta']:.3f}° "
                        f"(parent {s['parent_two_theta']:.3f}°, "
                        f"Δ{s['separation']:.3f}°, I/I₁={s['intensity_ratio']:.2f})"
                    )

        n_kept = len(peaks["two_theta"])
        self.session.set_peaks(peaks)
        limit = f"≥{snr_min:.1f}σ local noise" if snr_min > 0 else "no S/N limit"
        weakest = float(np.min(peaks["intensity"])) / max(imax, 1e-9) * 100 if n_kept else 0.0
        self.peak_status.setText(
            f"Found {n_kept} peaks ({limit}, height≥{min_height:.2f}%, "
            f"sep≥{min_sep:.2f}°{alpha2_note}). Weakest kept: {weakest:.2f}% of max."
        )
        self.workspace.refresh_plot()
        self.workspace.set_results_peaks(peaks)
        self.workspace.set_status(f"Found {n_kept} peaks")

    @staticmethod
    def _median_step(two_theta: np.ndarray) -> float:
        d = np.diff(np.asarray(two_theta, dtype=float))
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if len(d) else 0.01

    @staticmethod
    def _estimate_noise(intensity: np.ndarray) -> float:
        """
        Robust point-to-point noise over the whole pattern.

        Differencing removes the baseline and the median shrugs off peaks, so a
        strong line near the start no longer inflates the estimate and bury the
        weak lines of a minor phase.
        """
        y = np.asarray(intensity, dtype=float)
        y = y[np.isfinite(y)]
        n = len(y)
        if n < 10:
            return float(np.std(y)) if n else 1.0

        d = np.diff(y)
        mad = float(np.median(np.abs(d - np.median(d))))
        sigma = 1.4826 * mad / np.sqrt(2.0)
        if sigma <= 0:
            sigma = float(np.std(d)) / np.sqrt(2.0)
        span = float(np.max(y)) if len(y) else 1.0
        return float(max(sigma, 1e-4 * max(span, 1.0)))

    @staticmethod
    def _local_noise(intensity: np.ndarray, peak_pts: int) -> np.ndarray:
        """
        Point-to-point noise measured in a window around every channel.

        A single global sigma is set by the noisiest part of the pattern, which
        is usually the low-angle end; measuring locally lets a weak line in a
        quiet high-angle region pass on its own merits.
        """
        y = np.asarray(intensity, dtype=float)
        if len(y) < 8:
            return np.full(len(y), 1e-9)

        # Window a good deal wider than a peak, so peaks stay a minority
        window = int(np.clip(max(peak_pts * 12, 51), 21, max(21, len(y) // 4)))
        if window % 2 == 0:
            window += 1

        steps = np.abs(np.diff(y, prepend=y[0]))
        med = median_filter(steps, size=window, mode="nearest")
        # median(|Δ|) = 0.6745 σ_Δ for Gaussian noise, and σ_Δ = √2 σ_point
        sigma = 1.4826 * med / np.sqrt(2.0)
        floor = 1e-6 * max(float(np.max(y)), 1.0)
        return np.maximum(sigma, floor)

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
