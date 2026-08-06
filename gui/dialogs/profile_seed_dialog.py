"""
Fit one peak, and start the instrument profile from what it says.

The instrument resolution curve is the one part of a refinement that is not a
property of the specimen, and it is the part most often left at a default. This
window fits a single isolated peak with the same profile function the refinement
uses, splits the width it finds into the instrument's Gaussian and the sample's
Lorentzian, and offers the Gaussian part as a starting width.

It also fits the Kα2 satellite ratio, which answers a question no other view in
the program does: whether the satellites are in the data at all. A ratio near
the nominal 0.5 says they are, and that a model without them has been fitting
every peak too wide and skewed towards high 2θ.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QVBoxLayout,
)

from matplotlib_config import apply_plot_style, get_plot_palette
from gui import display_settings
from gui.theme import get_current_mode
from gui.widgets.control_bar import compact, no_wheel
from gui.widgets.plot_host import create_plot_host
from utils import kalpha_filter as kalpha
from utils.profile_functions import skew_description
from utils.profile_seed import candidate_peaks, fit_peak


class ProfileSeedDialog(QDialog):
    """Fits a chosen peak and hands back the profile terms it implies."""

    def __init__(self, two_theta, intensity, wavelength: float, parent=None):
        super().__init__(parent)
        self._x = np.asarray(two_theta, dtype=float)
        self._y = np.asarray(intensity, dtype=float)
        self._wavelength = float(wavelength)
        self._fit: Optional[Dict] = None
        self.applied: Dict = {}

        self.setWindowTitle("Seed Instrument Profile From a Peak")
        self.setMinimumSize(720, 600)
        self.resize(820, 660)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addWidget(self._intro())
        root.addLayout(self._peak_row())

        host, self.figure, self.canvas, _toolbar = create_plot_host(
            self, figsize=(7, 3.6), with_toolbar=False
        )
        self.ax = self.figure.add_subplot(1, 1, 1)
        root.addWidget(host, 1)

        root.addWidget(self._results_panel())
        root.addLayout(self._button_row())

        self._candidates: List[Dict] = candidate_peaks(self._x, self._y)
        self._fill_candidates()
        if self._candidates:
            self.refit()

    # --- layout -------------------------------------------------------------

    def _intro(self) -> QLabel:
        label = QLabel(
            "Pick a strong, isolated peak. Its width is split into the "
            "instrument's Gaussian part and the sample's Lorentzian part, and "
            "the Gaussian part is what the instrument profile should start from."
        )
        label.setObjectName("mutedLabel")
        label.setWordWrap(True)
        return label

    def _peak_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.peak_choice = no_wheel(QComboBox())
        self.peak_choice.setToolTip(
            "Peaks ranked by height and by how far they are from anything taller"
        )
        self.peak_choice.currentIndexChanged.connect(self._on_candidate_chosen)
        row.addWidget(QLabel("Peak:"))
        row.addWidget(compact(self.peak_choice, 260))

        self.centre = no_wheel(QDoubleSpinBox())
        self.centre.setDecimals(3)
        self.centre.setSuffix("°")
        span = (float(self._x[0]), float(self._x[-1])) if len(self._x) else (0.0, 180.0)
        self.centre.setRange(*span)
        self.centre.setSingleStep(0.05)
        row.addWidget(QLabel("2θ:"))
        row.addWidget(compact(self.centre, 100))

        self.fit_alpha2 = QCheckBox("Fit Kα2 ratio")
        self.fit_alpha2.setChecked(True)
        self.fit_alpha2.setToolTip(
            "Fit the satellite as part of the peak, so its intensity is not "
            "mistaken for width. Untick for synchrotron or monochromated data."
        )
        row.addWidget(self.fit_alpha2)

        refit = QPushButton("Fit")
        refit.clicked.connect(self.refit)
        row.addWidget(refit)
        row.addStretch()
        return row

    def _results_panel(self) -> QLabel:
        self.results = QLabel("No fit yet.")
        self.results.setTextFormat(Qt.RichText)
        self.results.setWordWrap(True)
        return self.results

    def _button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.apply_width = QCheckBox("Set the instrument width")
        self.apply_width.setChecked(True)
        self.apply_width.setToolTip(
            "Sets Initial FWHM to the Gaussian part of the fitted width, which "
            "is what seeds W. U and V stay at zero: one peak says nothing about "
            "how the resolution varies with angle."
        )
        row.addWidget(self.apply_width)

        self.apply_alpha2 = QCheckBox("Model Kα2 satellites")
        self.apply_alpha2.setChecked(False)
        self.apply_alpha2.setToolTip(
            "Turns on the doublet in the refinement at the ratio fitted here"
        )
        row.addWidget(self.apply_alpha2)

        row.addStretch()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        row.addWidget(self.apply_btn)

        close = QPushButton("Cancel")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        return row

    # --- fitting ------------------------------------------------------------

    def _fill_candidates(self):
        self.peak_choice.blockSignals(True)
        self.peak_choice.clear()
        for peak in self._candidates:
            clearance = peak['isolation']
            note = "alone" if not np.isfinite(clearance) else f"{clearance:.2f}° clear"
            self.peak_choice.addItem(
                f"{peak['two_theta']:.3f}°   ({note})", peak['two_theta']
            )
        self.peak_choice.blockSignals(False)
        if self._candidates:
            self.centre.setValue(self._candidates[0]['two_theta'])
        else:
            self.results.setText("No peaks found in this pattern.")

    def _on_candidate_chosen(self, index: int):
        value = self.peak_choice.itemData(index)
        if value is not None:
            self.centre.setValue(float(value))
            self.refit()

    def refit(self):
        try:
            self._fit = fit_peak(
                self._x, self._y, self.centre.value(), self._wavelength,
                fit_alpha2=self.fit_alpha2.isChecked(),
            )
        except Exception as e:  # noqa: BLE001 - shown to the user verbatim
            self._fit = None
            self.apply_btn.setEnabled(False)
            self.results.setText(f"Could not fit a peak here: {e}")
            self._draw()
            return

        self.apply_btn.setEnabled(True)
        self.apply_alpha2.setChecked(
            self._fit['alpha2_ratio'] > 0.1
        )
        self._show_numbers()
        self._draw()

    def _show_numbers(self):
        fit = self._fit
        strain = fit['microstrain']
        size = fit['crystallite_size']
        ratio = fit['alpha2_ratio']

        satellite = "not fitted"
        if self.fit_alpha2.isChecked():
            separation = float(kalpha.alpha2_separation(
                fit['centre'], kalpha.alpha2_ratio(self._wavelength)
            ))
            verdict = ("the doublet is in the data"
                       if ratio > 0.25 else
                       "little or none of it is left" if ratio < 0.1 else
                       "partly stripped, or the fit is unsure")
            satellite = (f"{ratio:.2f} of the parent, {separation:+.3f}° away "
                         f"— {verdict}")

        skew = fit['skew']
        lean = (f"{skew:+.3f} ({skew_description(skew)}), "
                f"the same as an axial term of {fit['axial_asymmetry']:+.3f}")

        self.results.setText(
            "<table cellspacing='2'>"
            f"<tr><td><b>Centre</b></td><td>{fit['centre']:.4f}°</td>"
            f"<td style='padding-left:18px'><b>Measured FWHM</b></td>"
            f"<td>{fit['fwhm']:.4f}°, η = {fit['eta']:.3f}</td></tr>"
            f"<tr><td><b>Instrument</b></td>"
            f"<td>{fit['gauss_fwhm']:.4f}° Gaussian</td>"
            f"<td style='padding-left:18px'><b>W</b></td>"
            f"<td>{fit['w_param']:.6f} (U = V = 0)</td></tr>"
            f"<tr><td><b>Sample</b></td>"
            f"<td>{fit['lorentz_fwhm']:.4f}° Lorentzian</td>"
            f"<td style='padding-left:18px'><b>which is</b></td>"
            f"<td>{strain:.0f}×10⁻⁶ strain, or {size:.2f} µm crystallites</td></tr>"
            f"<tr><td><b>Kα2</b></td><td colspan='3'>{satellite}</td></tr>"
            f"<tr><td><b>Skew</b></td><td colspan='3'>{lean}</td></tr>"
            f"<tr><td><b>Misfit</b></td><td colspan='3'>{fit['misfit']:.2f}% "
            f"over the fitted window</td></tr>"
            "</table>"
        )
        self.apply_width.setText(
            f"Set the instrument width to {fit['gauss_fwhm']:.3f}°"
        )
        self.apply_alpha2.setText(f"Model Kα2 satellites at {ratio:.2f}")

    # --- plot ---------------------------------------------------------------

    def _draw(self):
        mode = get_current_mode()
        palette = get_plot_palette(mode)
        self.ax.clear()

        if self._fit is None:
            centre = self.centre.value()
            window = np.abs(self._x - centre) <= 1.0
            if np.any(window):
                self.ax.plot(self._x[window], self._y[window], '-',
                             color=palette['exp_line'],
                             lw=display_settings.line_width())
        else:
            fit = self._fit
            x, observed, fitted = fit['two_theta'], fit['observed'], fit['fitted']
            self.ax.plot(x, observed, 'o', color=palette['exp_line'],
                         ms=display_settings.marker_size(2.6), label='Observed')
            self.ax.plot(x, fitted, '-', color=palette['calc_line'],
                         lw=display_settings.line_width(1.4), label='Fitted')

            # The residual sits below the peak so that what is left over can be
            # seen against what it was left over from
            offset = -0.18 * float(np.max(observed) - np.min(observed) or 1.0)
            self.ax.plot(x, observed - fitted + offset, '-',
                         color=palette['diff_line'],
                         lw=display_settings.line_width(0.8), label='Difference')
            self.ax.axhline(offset, color=palette['grid'], lw=0.6)

            if fit['alpha2_ratio'] > 0.01:
                separation = float(kalpha.alpha2_separation(
                    fit['centre'], kalpha.alpha2_ratio(self._wavelength)
                ))
                self.ax.axvline(fit['centre'] + separation, ls='--', lw=0.8,
                                color=palette['diff_line'], alpha=0.7,
                                label='Kα2')
            self.ax.set_title(
                f"{fit['centre']:.3f}°   FWHM {fit['fwhm']:.3f}°   η {fit['eta']:.2f}"
            )
            if display_settings.show_legend():
                self.ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        self.ax.set_xlabel('2θ (degrees)')
        self.ax.set_ylabel('Intensity')
        apply_plot_style(self.figure, mode, show_grid=display_settings.show_grid())
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # --- result -------------------------------------------------------------

    def _apply(self):
        if self._fit is None:
            return
        if not (self.apply_width.isChecked() or self.apply_alpha2.isChecked()):
            QMessageBox.information(
                self, "Nothing to Apply",
                "Tick what you want carried back into the refinement settings."
            )
            return

        self.applied = {}
        if self.apply_width.isChecked():
            self.applied['fwhm'] = self._fit['gauss_fwhm']
        if self.apply_alpha2.isChecked():
            self.applied['alpha2_ratio'] = self._fit['alpha2_ratio']
        self.applied['fit'] = self._fit
        self.accept()
