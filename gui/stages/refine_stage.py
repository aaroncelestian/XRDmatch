"""Refine stage — Le Bail refinement and export."""

from __future__ import annotations

import csv

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox, QToolBox,
    QVBoxLayout, QWidget,
)

from utils.multi_phase_analyzer import MultiPhaseAnalyzer
from utils.lebail_refinement import LeBailRefinement
from matplotlib_config import get_plot_palette
from gui.theme import get_current_mode


class RefineStage(QWidget):
    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self.analyzer = MultiPhaseAnalyzer()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Refine & Export")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(title)

        form = QFormLayout()
        self.max_iter = QSpinBox()
        self.max_iter.setRange(3, 50)
        self.max_iter.setValue(10)
        form.addRow("Max iterations:", self.max_iter)

        self.fwhm = QDoubleSpinBox()
        self.fwhm.setRange(0.005, 1.0)
        self.fwhm.setDecimals(3)
        self.fwhm.setSingleStep(0.005)
        self.fwhm.setValue(0.1)
        self.fwhm.setSuffix("°")
        form.addRow("Initial FWHM:", self.fwhm)

        self.intensity_model = QComboBox()
        self.intensity_model.addItem("Reference intensities (quantitative)", "fixed")
        self.intensity_model.addItem("Le Bail extraction (profile only)", "extract")
        self.intensity_model.setToolTip(
            "Reference intensities: calculated intensities stay tied to the reference "
            "pattern and one scale per phase is refined. Weight percents, absorption, "
            "and texture are only determinable this way.\n\n"
            "Le Bail extraction: intensities are partitioned out of the observed "
            "pattern. It gives the best profile and cell fit, but it absorbs the scale "
            "factor, so nothing is left to quantify with."
        )
        form.addRow("Intensity model:", self.intensity_model)
        layout.addLayout(form)

        self.refine_btn = QPushButton("Run Le Bail Refinement")
        self.refine_btn.setObjectName("primaryButton")
        self.refine_btn.clicked.connect(self.run_lebail)
        layout.addWidget(self.refine_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Select matched phases in the Phases tab, then refine.")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        export_row = QHBoxLayout()
        self.export_png = QPushButton("Export PNG")
        self.export_png.clicked.connect(lambda: self.export_plot("png"))
        export_row.addWidget(self.export_png)
        self.export_pdf = QPushButton("Export PDF")
        self.export_pdf.clicked.connect(lambda: self.export_plot("pdf"))
        export_row.addWidget(self.export_pdf)
        self.export_csv = QPushButton("Export CSV")
        self.export_csv.clicked.connect(self.export_csv_data)
        export_row.addWidget(self.export_csv)
        layout.addLayout(export_row)

        toolbox = QToolBox()

        # --- Global parameters: one value for the whole pattern ---
        glob = QWidget()
        glob_form = QFormLayout(glob)

        self.refine_zero_shift = QCheckBox("Refine zero shift")
        self.refine_zero_shift.setChecked(True)
        self.refine_zero_shift.setToolTip(
            "Constant 2θ offset from detector zero-point error. Refined once "
            "against the whole pattern, so no single phase can absorb it into "
            "its own lattice."
        )
        glob_form.addRow(self.refine_zero_shift)

        self.refine_displacement = QCheckBox("Refine sample displacement")
        self.refine_displacement.setChecked(True)
        self.refine_displacement.setToolTip(
            "Specimen height error in Bragg-Brentano geometry, which shifts peaks "
            "by a term proportional to cos θ. It is the usual cause of a shift "
            "that grows towards low angle and cannot be fixed by zero shift alone."
        )
        glob_form.addRow(self.refine_displacement)

        self.refine_profile = QCheckBox("Refine peak profile (U, V, W, η)")
        self.refine_profile.setChecked(True)
        glob_form.addRow(self.refine_profile)

        self.eta = QDoubleSpinBox()
        self.eta.setRange(0.0, 1.0)
        self.eta.setSingleStep(0.1)
        self.eta.setValue(0.5)
        glob_form.addRow("Peak shape η:", self.eta)

        toolbox.addItem(glob, "Global parameters")

        # --- Phase-specific parameters ---
        per_phase = QWidget()
        phase_form = QFormLayout(per_phase)

        self.refine_cell = QCheckBox("Refine unit cell")
        self.refine_cell.setChecked(True)
        self.refine_cell.setToolTip(
            "Refines an isotropic lattice dilation per phase, reported as scaled "
            "cell edges and volume.\n\n"
            "Anisotropic a/b/c refinement needs Miller indices per reflection, "
            "which the stored reference patterns do not carry."
        )
        phase_form.addRow(self.refine_cell)

        self.refine_absorption = QCheckBox("Refine absorption")
        self.refine_absorption.setChecked(False)
        self.refine_absorption.setToolTip(
            "Angle-dependent intensity loss for each phase, of the form "
            "exp(-a / sin θ), anchored at the pattern midpoint so it is not "
            "degenerate with the scale factor. Absorbs microabsorption contrast "
            "between phases of differing particle size or density.\n\n"
            "Needs the reference-intensity model."
        )
        phase_form.addRow(self.refine_absorption)

        self.refine_harmonics = QCheckBox("Refine spherical harmonics")
        self.refine_harmonics.setChecked(False)
        self.refine_harmonics.setToolTip(
            "Axially symmetric preferred-orientation correction: an even-order "
            "harmonic expansion in cos θ, which averages to zero over the pattern "
            "and so stays separable from the scale factor.\n\n"
            "A full orientation distribution needs Miller indices per reflection. "
            "Needs the reference-intensity model."
        )
        phase_form.addRow(self.refine_harmonics)

        self.harmonic_order = QComboBox()
        for label, value in (("2 (1 term)", 2), ("4 (2 terms)", 4), ("6 (3 terms)", 6)):
            self.harmonic_order.addItem(label, value)
        self.harmonic_order.setCurrentIndex(1)
        self.harmonic_order.setToolTip(
            "Highest harmonic order. Higher orders describe sharper texture but "
            "add parameters that can trade against the peak profile."
        )
        phase_form.addRow("Harmonic order:", self.harmonic_order)

        self.refine_intensities = QCheckBox("Refine intensities (Pawley)")
        self.refine_intensities.setChecked(False)
        self.refine_intensities.setToolTip(
            "Frees every reflection intensity. Fits almost anything, so it "
            "invalidates the weight percents — use it only to check the profile."
        )
        phase_form.addRow(self.refine_intensities)

        self.max_scale = QDoubleSpinBox()
        self.max_scale.setRange(1.0, 1000.0)
        self.max_scale.setValue(100.0)
        phase_form.addRow("Max scale:", self.max_scale)

        toolbox.addItem(per_phase, "Phase parameters")

        # --- Range and export ---
        adv = QWidget()
        adv_form = QFormLayout(adv)

        self.use_range = QCheckBox("Limit 2θ range")
        adv_form.addRow(self.use_range)
        self.min_2th = QDoubleSpinBox()
        self.min_2th.setRange(0, 180)
        self.min_2th.setValue(10)
        self.min_2th.setSuffix("°")
        adv_form.addRow("Min 2θ:", self.min_2th)
        self.max_2th = QDoubleSpinBox()
        self.max_2th.setRange(0, 180)
        self.max_2th.setValue(90)
        self.max_2th.setSuffix("°")
        adv_form.addRow("Max 2θ:", self.max_2th)

        self.dpi = QSpinBox()
        self.dpi.setRange(72, 600)
        self.dpi.setValue(300)
        adv_form.addRow("Export DPI:", self.dpi)

        toolbox.addItem(adv, "Range & export")
        layout.addWidget(toolbox)
        layout.addStretch()

        self.intensity_model.currentIndexChanged.connect(self._on_intensity_model_changed)
        self.refine_harmonics.toggled.connect(self._on_intensity_model_changed)
        self._on_intensity_model_changed()

    def _on_intensity_model_changed(self, *_args):
        """Absorption and texture are only determinable with fixed intensities."""
        quantitative = self.intensity_model.currentData() == "fixed"
        self.refine_absorption.setEnabled(quantitative)
        self.refine_harmonics.setEnabled(quantitative)
        self.harmonic_order.setEnabled(quantitative and self.refine_harmonics.isChecked())
        if not quantitative:
            hint = (
                "Le Bail extraction absorbs the scale factor, so weight percents, "
                "absorption, and texture are unavailable in this mode."
            )
            self.refine_absorption.setToolTip(hint)
            self.refine_harmonics.setToolTip(hint)

    def on_enter(self):
        n = len(self.session.selected_phases) or len(self.session.matched_phases)
        self.refine_btn.setEnabled(n > 0 and self.session.has_pattern())
        if n == 0:
            self.status.setText("No phases selected. Match phases in the Phases tab first.")
        else:
            self.status.setText(f"{n} phase(s) ready for refinement.")

    def _phases_for_refine(self):
        return self.session.selected_phases or self.session.matched_phases

    def run_lebail(self):
        pattern = self.session.active_pattern()
        phases = self._phases_for_refine()
        if not pattern or not phases:
            QMessageBox.warning(self, "No Data", "Need a pattern and matched phases.")
            return

        # Unwrap match results to phase dicts expected by analyzer
        phase_list = []
        for item in phases:
            if isinstance(item, dict) and "phase" in item:
                phase_list.append(item)
            else:
                phase_list.append({"phase": item, "match_score": 1.0})

        try:
            self.refine_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.status.setText("Running Le Bail refinement…")

            experimental_data = {
                "two_theta": pattern["two_theta"],
                "intensity": pattern["intensity"],
                "wavelength": self.session.wavelength,
                "errors": pattern.get("intensity_error"),
            }

            two_theta_range = None
            if self.use_range.isChecked():
                two_theta_range = (self.min_2th.value(), self.max_2th.value())

            fwhm = self.fwhm.value()
            if abs(fwhm - 0.1) < 0.001 and self.session.wavelength < 0.5:
                fwhm = 0.015
                self.fwhm.setValue(fwhm)

            initial_w = fwhm ** 2
            refinement_params = {
                "initial_u": initial_w * 0.05,
                "initial_v": 0.0,
                "initial_w": initial_w,
                "initial_eta": self.eta.value(),
                "max_scale": self.max_scale.value(),
                "refine_cell": self.refine_cell.isChecked(),
                "refine_profile": self.refine_profile.isChecked(),
                "refine_intensities": self.refine_intensities.isChecked(),
                "intensity_model": self.intensity_model.currentData() or "fixed",
                "refine_zero_shift": self.refine_zero_shift.isChecked(),
                "refine_displacement": self.refine_displacement.isChecked(),
                "refine_absorption": self.refine_absorption.isChecked(),
                "refine_harmonics": self.refine_harmonics.isChecked(),
                "harmonic_order": (
                    self.harmonic_order.currentData()
                    if self.refine_harmonics.isChecked() else 0
                ),
            }

            LeBailRefinement.plot_callback = None
            results = self.analyzer.perform_lebail_refinement(
                experimental_data,
                phase_list,
                max_iterations=self.max_iter.value(),
                two_theta_range=two_theta_range,
                refinement_params=refinement_params,
            )
            self.session.set_lebail_results(results)
            self.status.setText(self._completion_message(results))
            self.workspace.refresh_plot()
            self.workspace.set_status("Le Bail refinement complete")
        except Exception as e:
            QMessageBox.critical(self, "Refinement Error", str(e))
            self.status.setText("Refinement failed.")
        finally:
            self.progress.setVisible(False)
            self.refine_btn.setEnabled(True)

    @staticmethod
    def _completion_message(results) -> str:
        """Rwp plus the headline weight percents, when they are meaningful."""
        if not results or not results.get("success"):
            return "Refinement finished without a result."
        inner = results.get("refinement_results") or {}
        factors = inner.get("final_r_factors") or results.get("r_factors") or {}
        rwp = factors.get("Rwp")
        message = f"Refinement complete — Rwp = {rwp:.2f}%" if rwp is not None else "Refinement complete."

        quantified = [
            row for row in (inner.get("phase_summary") or [])
            if row.get("weight_percent") is not None
        ]
        if quantified:
            quantified.sort(key=lambda r: -r["weight_percent"])
            headline = ", ".join(
                f"{row['name']} {row['weight_percent']:.1f}%" for row in quantified[:4]
            )
            message += f". {headline}"
        elif inner.get("intensity_model") == "extract":
            message += ". Switch to the reference-intensity model for weight percents."
        return message

    def export_plot(self, fmt: str):
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Plot as {fmt.upper()}", f"xrd_plot.{fmt}",
            f"{fmt.upper()} (*.{fmt});;All files (*.*)",
        )
        if not path:
            return
        try:
            fig = getattr(self.workspace, "quant_figure", None) or self.workspace.figure
            fig.savefig(path, dpi=self.dpi.value(), bbox_inches="tight")
            self.workspace.set_status(f"Exported {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_csv_data(self):
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Data", "No pattern to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "xrd_data.csv", "CSV (*.csv);;All files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["two_theta", "intensity"])
                for x, y in zip(pattern["two_theta"], pattern["intensity"]):
                    writer.writerow([float(x), float(y)])
            self.workspace.set_status(f"Exported {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
