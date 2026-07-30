"""Process stage — background subtraction and peak finding."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QToolBox,
    QVBoxLayout, QWidget,
)
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


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
    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self._background = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Process")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(title)

        # Primary controls
        self.enable_bg = QCheckBox("Background subtraction (ALS)")
        self.enable_bg.setChecked(True)
        layout.addWidget(self.enable_bg)

        form = QFormLayout()
        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setRange(2, 8)
        self.lambda_slider.setValue(5)
        self.lambda_label = QLabel("1e5")
        self.lambda_slider.valueChanged.connect(
            lambda v: self.lambda_label.setText(f"1e{v}")
        )
        lam_row = QHBoxLayout()
        lam_row.addWidget(self.lambda_slider)
        lam_row.addWidget(self.lambda_label)
        form.addRow("Smoothness λ:", lam_row)

        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(0.001, 0.1)
        self.p_spin.setDecimals(3)
        self.p_spin.setSingleStep(0.001)
        self.p_spin.setValue(0.01)
        form.addRow("Asymmetry p:", self.p_spin)

        self.min_height = QSpinBox()
        self.min_height.setRange(1, 100000)
        self.min_height.setValue(50)
        form.addRow("Peak min height:", self.min_height)

        self.min_prominence = QSpinBox()
        self.min_prominence.setRange(1, 10000)
        self.min_prominence.setValue(10)
        form.addRow("Peak prominence:", self.min_prominence)

        self.sensitivity = QComboBox()
        self.sensitivity.addItems(["High", "Medium", "Low"])
        form.addRow("Sensitivity:", self.sensitivity)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Processing")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.clicked.connect(self.apply_processing)
        btn_row.addWidget(self.apply_btn)

        self.peaks_btn = QPushButton("Find Peaks")
        self.peaks_btn.setObjectName("primaryButton")
        self.peaks_btn.clicked.connect(self.find_peaks)
        btn_row.addWidget(self.peaks_btn)
        layout.addLayout(btn_row)

        self.status = QLabel("Load a pattern first.")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # Advanced
        toolbox = QToolBox()
        adv = QWidget()
        adv_layout = QFormLayout(adv)

        self.iterations = QSpinBox()
        self.iterations.setRange(5, 50)
        self.iterations.setValue(10)
        adv_layout.addRow("ALS iterations:", self.iterations)

        self.min_width = QSpinBox()
        self.min_width.setRange(1, 20)
        self.min_width.setValue(1)
        adv_layout.addRow("Peak min width:", self.min_width)

        self.min_distance = QSpinBox()
        self.min_distance.setRange(1, 50)
        self.min_distance.setValue(3)
        adv_layout.addRow("Peak min distance:", self.min_distance)

        self.enable_smooth = QCheckBox("Smoothing")
        adv_layout.addRow(self.enable_smooth)
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(3, 21)
        self.smooth_window.setSingleStep(2)
        self.smooth_window.setValue(5)
        adv_layout.addRow("Smooth window:", self.smooth_window)

        self.enable_noise = QCheckBox("Median noise reduction")
        adv_layout.addRow(self.enable_noise)

        self.displacement = QDoubleSpinBox()
        self.displacement.setRange(-2.0, 2.0)
        self.displacement.setDecimals(4)
        self.displacement.setSingleStep(0.001)
        self.displacement.setValue(0.0)
        adv_layout.addRow("2θ offset (°):", self.displacement)

        toolbox.addItem(adv, "Advanced")
        layout.addWidget(toolbox)

        nav = QHBoxLayout()
        back = QPushButton("← Load")
        back.clicked.connect(lambda: self.workspace.set_stage("load"))
        nav.addWidget(back)
        nav.addStretch()
        self.next_btn = QPushButton("Continue to Identify →")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self.workspace.set_stage("identify"))
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)
        layout.addStretch()

    def on_enter(self):
        if self.session.has_pattern():
            self.status.setText("Adjust background, then Find Peaks.")
            if self.session.processed_pattern is None:
                # Seed processed = raw for plotting
                self.apply_processing(silent=True)
        else:
            self.status.setText("Load a pattern first.")

    def apply_processing(self, silent=False):
        raw = self.session.raw_pattern
        if raw is None:
            if not silent:
                QMessageBox.warning(self, "No Pattern", "Load a pattern first.")
            return

        intensity = np.asarray(raw["intensity"], dtype=float).copy()
        two_theta = np.asarray(raw["two_theta"], dtype=float).copy()

        # Displacement
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
            self.status.setText("Processing applied.")
            self.workspace.set_status("Processing applied")

    def find_peaks(self):
        if self.session.processed_pattern is None:
            self.apply_processing(silent=True)
        pattern = self.session.processed_pattern
        if pattern is None:
            QMessageBox.warning(self, "No Pattern", "Load and process a pattern first.")
            return

        intensity = np.asarray(pattern["intensity"], dtype=float)
        two_theta = np.asarray(pattern["two_theta"], dtype=float)
        min_height = self.min_height.value()
        min_prominence = self.min_prominence.value()
        min_width = self.min_width.value()
        min_distance = self.min_distance.value()
        sensitivity = self.sensitivity.currentIndex()

        if sensitivity == 0:
            height_threshold = min_height
            prominence_threshold = min_prominence
        elif sensitivity == 1:
            noise = np.std(intensity[: max(1, min(100, len(intensity) // 10))])
            height_threshold = max(min_height, noise * 2)
            prominence_threshold = max(min_prominence, noise * 0.5)
        else:
            noise = np.std(intensity[: max(1, min(100, len(intensity) // 10))])
            height_threshold = max(min_height, noise * 5)
            prominence_threshold = max(min_prominence, noise * 2)

        peaks_idx, _ = find_peaks(
            intensity,
            height=height_threshold,
            distance=min_distance,
            prominence=prominence_threshold,
            width=min_width,
        )
        peaks_idx = [int(i) for i in peaks_idx if two_theta[int(i)] >= 3.0]

        if not peaks_idx:
            QMessageBox.warning(
                self, "No Peaks",
                "No peaks found. Try lowering height/prominence or raising sensitivity.",
            )
            return

        peak_tt = two_theta[peaks_idx]
        peak_int = intensity[peaks_idx]
        wl = self.session.wavelength
        d_spacing = wl / (2.0 * np.sin(np.radians(peak_tt / 2.0)))

        peaks = {
            "indices": np.asarray(peaks_idx, dtype=int),
            "two_theta": peak_tt,
            "intensity": peak_int,
            "d_spacing": d_spacing,
            "wavelength": self.session.wavelength,
        }
        self.session.set_peaks(peaks)
        self.next_btn.setEnabled(True)
        self.status.setText(f"Found {len(peaks_idx)} peaks.")
        self.workspace.refresh_plot()
        self.workspace.set_results_peaks(peaks)
        self.workspace.set_status(f"Found {len(peaks_idx)} peaks")
