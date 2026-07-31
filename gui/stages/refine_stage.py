"""Refine stage — Le Bail refinement and export."""

from __future__ import annotations

import csv

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
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
        layout.addLayout(form)

        self.refine_btn = QPushButton("Run Le Bail Refinement")
        self.refine_btn.setObjectName("primaryButton")
        self.refine_btn.clicked.connect(self.run_lebail)
        layout.addWidget(self.refine_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Select matched phases in Search / Match, then refine.")
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
        adv = QWidget()
        adv_form = QFormLayout(adv)

        self.max_scale = QDoubleSpinBox()
        self.max_scale.setRange(1.0, 1000.0)
        self.max_scale.setValue(100.0)
        adv_form.addRow("Max scale:", self.max_scale)

        self.eta = QDoubleSpinBox()
        self.eta.setRange(0.0, 1.0)
        self.eta.setSingleStep(0.1)
        self.eta.setValue(0.5)
        adv_form.addRow("Peak shape η:", self.eta)

        self.refine_cell = QCheckBox("Refine unit cell")
        self.refine_cell.setChecked(True)
        adv_form.addRow(self.refine_cell)

        self.refine_profile = QCheckBox("Refine peak profile")
        self.refine_profile.setChecked(True)
        adv_form.addRow(self.refine_profile)

        self.refine_intensities = QCheckBox("Refine intensities (Pawley)")
        self.refine_intensities.setChecked(False)
        adv_form.addRow(self.refine_intensities)

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

        toolbox.addItem(adv, "Advanced")
        layout.addWidget(toolbox)
        layout.addStretch()

    def on_enter(self):
        n = len(self.session.selected_phases) or len(self.session.matched_phases)
        self.refine_btn.setEnabled(n > 0 and self.session.has_pattern())
        if n == 0:
            self.status.setText("No phases selected. Match phases in Search / Match first.")
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
            rwp = None
            if results and results.get("success"):
                rr = results.get("refinement_results") or {}
                rwp = rr.get("rwp") or results.get("rwp")
            if rwp is not None:
                self.status.setText(f"Refinement complete — Rwp = {rwp:.2f}%")
            else:
                self.status.setText("Refinement finished.")
            self.workspace.refresh_plot()
            self.workspace.set_status("Le Bail refinement complete")
        except Exception as e:
            QMessageBox.critical(self, "Refinement Error", str(e))
            self.status.setText("Refinement failed.")
        finally:
            self.progress.setVisible(False)
            self.refine_btn.setEnabled(True)

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
