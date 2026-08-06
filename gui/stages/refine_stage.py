"""Refine stage — Le Bail refinement and export.

The controls live in the Refinement Parameters window, beside the per-phase
grid, so that setting up a run and reading what the last one found are the same
place rather than two.
"""

from __future__ import annotations

import csv

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from utils import kalpha_filter as kalpha
from utils.multi_phase_analyzer import MultiPhaseAnalyzer
from utils.lebail_refinement import LeBailRefinement
from gui import display_settings, refinement_table
from gui.focus import hold_focus, restores_focus
from gui.widgets.control_bar import compact, no_wheel
from gui.widgets.section import SectionFrame
from gui.dialogs.profile_seed_dialog import ProfileSeedDialog
from gui.dialogs.refinement_progress_dialog import (
    RefinementProgressDialog, RefinementWorker,
)


class RefineStage(QWidget):
    _MAX_SCALE_HINT = (
        "Upper bound on the overall scale factor of a phase. The lower bound is "
        "always zero, so a phase that is not really present can refine away.\n\n"
        "Raise it if a phase pins at the bound. The effective limit is never less "
        "than twenty times the starting scale, so a phase that begins large keeps "
        "headroom regardless of this value."
    )
    _MAX_SCALE_PAWLEY_HINT = (
        "The Pawley solve frees every reflection intensity, which absorbs the "
        "overall scale factor entirely. No scale is refined, so this bound is "
        "unused."
    )

    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self.analyzer = MultiPhaseAnalyzer()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        layout.addLayout(self._build_run_row())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status = QLabel("Select matched phases in the Phases tab, then refine.")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        columns = QHBoxLayout()
        columns.setSpacing(8)
        for column in (
            self._build_global_group(),
            self._build_phase_group(),
            self._build_side_column(),
        ):
            columns.addWidget(column, 1, Qt.AlignTop)
        layout.addLayout(columns, 1)

        self.intensity_model.currentIndexChanged.connect(self._on_intensity_model_changed)
        self.refine_harmonics.toggled.connect(self._on_intensity_model_changed)
        self._on_intensity_model_changed()

        self.refine_intensities.toggled.connect(self._on_pawley_toggled)
        self._on_pawley_toggled()

    def _build_run_row(self):
        """The run button and the settings that qualify a whole run."""
        row = QHBoxLayout()
        row.setSpacing(8)

        self.refine_btn = QPushButton("Run Le Bail Refinement")
        self.refine_btn.setObjectName("primaryButton")
        self.refine_btn.clicked.connect(self.run_lebail)
        row.addWidget(self.refine_btn)

        self.max_iter = no_wheel(QSpinBox())
        self.max_iter.setRange(3, 50)
        self.max_iter.setValue(10)
        row.addWidget(self._muted("Max iterations:"))
        row.addWidget(compact(self.max_iter, 70))

        self.fwhm = no_wheel(QDoubleSpinBox())
        self.fwhm.setRange(0.005, 1.0)
        self.fwhm.setDecimals(3)
        self.fwhm.setSingleStep(0.005)
        self.fwhm.setValue(0.1)
        self.fwhm.setSuffix("°")
        row.addWidget(self._muted("Initial FWHM:"))
        row.addWidget(compact(self.fwhm, 90))

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
        row.addWidget(self._muted("Intensity model:"))
        row.addWidget(compact(self.intensity_model, 300))
        row.addStretch()
        return row

    @staticmethod
    def _muted(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("mutedLabel")
        return label

    @staticmethod
    def _section(title: str):
        """A titled group whose rows sit at their natural spacing."""
        group = SectionFrame(title)
        form = QFormLayout()
        form.setSpacing(6)
        group.body.addLayout(form)
        return group, form

    def _build_global_group(self):
        """Global parameters: one value for the whole pattern."""
        group, glob_form = self._section("Global parameters")

        self.continue_previous = QCheckBox("Start from previous refinement")
        self.continue_previous.setChecked(True)
        self.continue_previous.setToolTip(
            "Begin each run at the values the last one reached, so a parameter "
            "whose box you have since unticked stays at its refined value "
            "instead of returning to a default.\n\n"
            "This is what lets you refine in stages: free the sample "
            "displacement, let it settle, untick it, then refine the unit cell "
            "against the displacement you just found.\n\n"
            "Untick to start every run from the defaults instead."
        )
        glob_form.addRow(self.continue_previous)

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

        self.refine_instrument = QCheckBox("Refine instrument profile (U, V, W)")
        self.refine_instrument.setChecked(False)
        self.refine_instrument.setToolTip(
            "The instrument resolution curve is shared by every phase and is "
            "held fixed by default. Turning this on is rarely useful: U, V and "
            "W are strongly correlated and can be driven into a non-physical "
            "region that freezes the refinement. Prefer calibrating them from a "
            "standard, or leave the defaults alone."
        )
        glob_form.addRow(self.refine_instrument)

        self.refine_axial = QCheckBox("Refine axial asymmetry")
        self.refine_axial.setChecked(False)
        self.refine_axial.setToolTip(
            "Skew from beam divergence out of the diffraction plane, shared by "
            "every phase. Its effect goes as cot 2θ, so it drags a low-angle "
            "tail out of the peaks below about 20° and fades to nothing by 90°. "
            "Turn this on when every phase in the pattern leans the same way at "
            "low angle; if only one phase is skewed, use the per-phase term "
            "instead."
        )
        glob_form.addRow(self.refine_axial)
        glob_form.addRow(self._build_alpha2_row())

        self.seed_btn = QPushButton("Seed profile from a peak…")
        self.seed_btn.setToolTip(
            "Fit one isolated peak and start the instrument width from what it "
            "measures, instead of from a default that has never seen this "
            "diffractometer.\n\n"
            "The fit splits the peak's width into a Gaussian part, which belongs "
            "to the instrument, and a Lorentzian part, which belongs to the "
            "specimen. Left uncalibrated, the instrument width is too narrow and "
            "microstrain grows to make up the difference, which gets the total "
            "width right and the peak shape wrong.\n\n"
            "It also fits the Kα2 ratio, which is how to find out whether the "
            "satellites are in your data."
        )
        self.seed_btn.clicked.connect(self.seed_profile_from_peak)
        glob_form.addRow(self.seed_btn)

        self.fit_peaks_only = QCheckBox("Fit only near modelled peaks")
        self.fit_peaks_only.setChecked(False)
        self.fit_peaks_only.setToolTip(
            "A background-subtracted pattern is mostly empty, and those empty "
            "points carry the largest weight because the error model is "
            "smallest where the intensity is smallest. The refinement can end "
            "up steered by counting noise between the peaks. Tick this to fit "
            "only within a few widths of each reflection. The region is fixed "
            "from the starting positions so the peaks cannot slide out of it. "
            "This changes the fit, not just the reported numbers — Rwp(peaks) "
            "in the results lets you compare a run with it on against one "
            "with it off."
        )
        glob_form.addRow(self.fit_peaks_only)
        return group

    def _build_alpha2_row(self):
        """Whether each reflection is one line or a doublet."""
        row = QHBoxLayout()
        row.setSpacing(6)

        self.model_alpha2 = QCheckBox("Model Kα2 satellites")
        self.model_alpha2.setChecked(False)
        self.model_alpha2.setToolTip(
            "A lab source emits two Kα lines, so every reflection is a doublet: "
            "a Kα1 line and a satellite at higher 2θ with about half the "
            "intensity, a tenth of a degree away at 36° and growing as tan θ.\n\n"
            "Satellites that are in the data but not in the model cannot be "
            "fitted, only compensated: the missing intensity is taken up as extra "
            "width and a lean towards high 2θ, on every peak of every phase. If "
            "your phases refine a large negative asymmetry, this is the first "
            "thing to suspect.\n\n"
            "Leave off for synchrotron or monochromated data, or if the "
            "satellites have already been stripped. 'Seed profile from a peak' "
            "will tell you which you have.\n\n"
            "With satellites modelled the pattern wavelength should be Kα1, not "
            "the doublet average."
        )
        row.addWidget(self.model_alpha2)

        self.alpha2_ratio = no_wheel(QDoubleSpinBox())
        self.alpha2_ratio.setRange(0.0, 0.6)
        self.alpha2_ratio.setDecimals(2)
        self.alpha2_ratio.setSingleStep(0.05)
        self.alpha2_ratio.setValue(kalpha.NOMINAL_INTENSITY_RATIO)
        self.alpha2_ratio.setToolTip(
            "Satellite intensity as a fraction of its parent. The transition "
            "probabilities put it at 0.50; a lower value fits data whose "
            "satellites were partly removed."
        )
        row.addWidget(self._muted("Kα2/Kα1:"))
        row.addWidget(compact(self.alpha2_ratio, 70))

        self.refine_alpha2 = QCheckBox("refine")
        self.refine_alpha2.setToolTip(
            "Refine the ratio against the whole pattern. Worth one run as a "
            "diagnostic: a value that settles near 0.5 confirms the doublet is "
            "there, and one that goes to zero says it is not."
        )
        row.addWidget(self.refine_alpha2)
        row.addStretch()

        self.model_alpha2.toggled.connect(self.alpha2_ratio.setEnabled)
        self.model_alpha2.toggled.connect(self.refine_alpha2.setEnabled)
        self.alpha2_ratio.setEnabled(False)
        self.refine_alpha2.setEnabled(False)
        return row

    @restores_focus
    def seed_profile_from_peak(self):
        """Fit one peak and take the instrument terms it implies."""
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Pattern",
                                "Load and process a pattern first.")
            return

        dialog = ProfileSeedDialog(
            pattern["two_theta"], pattern["intensity"],
            self.session.wavelength, self,
        )
        if dialog.exec_() != QDialog.Accepted or not dialog.applied:
            return

        taken = []
        if "fwhm" in dialog.applied:
            self.fwhm.setValue(round(float(dialog.applied["fwhm"]), 3))
            taken.append(f"initial FWHM {self.fwhm.value():.3f}°")
        if "alpha2_ratio" in dialog.applied:
            ratio = float(dialog.applied["alpha2_ratio"])
            self.model_alpha2.setChecked(ratio > 0.0)
            self.alpha2_ratio.setValue(round(ratio, 2))
            taken.append(f"Kα2/Kα1 = {ratio:.2f}")
        if taken:
            self.status.setText(
                "Seeded from the fitted peak: " + ", ".join(taken)
                + ". The instrument now carries the width it measured, so start "
                "the sample terms low."
            )

    def _build_phase_group(self):
        """Terms carried by each phase in its own right."""
        group, phase_form = self._section("Phase parameters")

        self.refine_strain = QCheckBox("Refine microstrain")
        self.refine_strain.setChecked(True)
        self.refine_strain.setToolTip(
            "Per-phase Lorentzian broadening proportional to tan θ, the "
            "signature of a distribution of lattice constants. Strictly "
            "positive and well separated from crystallite-size broadening, "
            "so it is far more stable under refinement than the instrument "
            "U, V, W polynomial."
        )
        phase_form.addRow(self.refine_strain)

        self.refine_size = QCheckBox("Refine crystallite size")
        self.refine_size.setChecked(False)
        self.refine_size.setToolTip(
            "Per-phase Lorentzian broadening proportional to 1/cos θ "
            "(Scherrer). Negligible for most hand-ground minerals; turn on "
            "for nanomaterials or precipitates. Refining size and strain "
            "together needs peaks over a wide 2θ range."
        )
        phase_form.addRow(self.refine_size)

        self.refine_asymmetry = QCheckBox("Refine peak asymmetry")
        self.refine_asymmetry.setChecked(False)
        self.refine_asymmetry.setToolTip(
            "Lets one phase's peaks be skewed independently of the others, for "
            "sample effects rather than instrument ones. Stacking disorder in a "
            "layered structure is the usual cause, so this is what the "
            "phyllosilicates need — chlorite, the micas, the clays — while the "
            "framework and chain silicates beside them stay symmetric.\n\n"
            "The two flanks are widened and narrowed about the same mean width, "
            "so this changes the peak shape without competing with crystallite "
            "size or microstrain for the width."
        )
        phase_form.addRow(self.refine_asymmetry)

        self.refine_cell = QCheckBox("Refine unit cell")
        self.refine_cell.setChecked(True)
        self.refine_cell.setToolTip(
            "Refines a, b, c and the cell angles separately for each phase, so "
            "one axis can expand while another contracts.\n\n"
            "What the symmetry of the starting cell fixes is held: equal axes "
            "move together and a right angle stays a right angle. Miller "
            "indices are recovered from the reference d-spacings to do this; a "
            "phase whose reflections will not index is dilated as a whole "
            "instead, and the parameter window says which happened.\n\n"
            "Hold one parameter while the rest refine by unticking it there."
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

        self.harmonic_order = no_wheel(QComboBox())
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

        self.max_scale = no_wheel(QDoubleSpinBox())
        self.max_scale.setRange(1.0, 1000.0)
        self.max_scale.setValue(100.0)
        self.max_scale_label = QLabel("Max scale:")
        phase_form.addRow(self.max_scale_label, compact(self.max_scale, 90))
        return group

    def _build_side_column(self):
        """The window a run fits over, and what leaves the program after it."""
        column = QWidget()
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(8)
        column_layout.addWidget(self._build_range_group())
        column_layout.addWidget(self._build_export_group())
        return column

    def _build_range_group(self):
        group, adv_form = self._section("Fitted range")

        self.use_range = QCheckBox("Limit 2θ range")
        adv_form.addRow(self.use_range)
        self.min_2th = no_wheel(QDoubleSpinBox())
        self.min_2th.setRange(0, 180)
        self.min_2th.setValue(10)
        self.min_2th.setSuffix("°")
        adv_form.addRow("Min 2θ:", compact(self.min_2th, 90))
        self.max_2th = no_wheel(QDoubleSpinBox())
        self.max_2th.setRange(0, 180)
        self.max_2th.setValue(90)
        self.max_2th.setSuffix("°")
        adv_form.addRow("Max 2θ:", compact(self.max_2th, 90))
        return group

    def _build_export_group(self):
        group, export_form = self._section("Export")

        plot_row = QHBoxLayout()
        self.export_png = QPushButton("PNG")
        self.export_png.setToolTip("Save the plot as a raster image.")
        self.export_png.clicked.connect(lambda: self.export_plot("png"))
        plot_row.addWidget(self.export_png)
        self.export_pdf = QPushButton("PDF")
        self.export_pdf.setToolTip("Save the plot as a vector figure.")
        self.export_pdf.clicked.connect(lambda: self.export_plot("pdf"))
        plot_row.addWidget(self.export_pdf)
        export_form.addRow("Plot:", plot_row)

        data_row = QHBoxLayout()
        self.export_csv = QPushButton("Results table")
        self.export_csv.setToolTip(
            "Write the per-phase results as CSV, under the statistics that "
            "qualify them."
        )
        self.export_csv.clicked.connect(self.export_csv_data)
        data_row.addWidget(self.export_csv)
        self.export_pattern_btn = QPushButton("Pattern data")
        self.export_pattern_btn.setToolTip(
            "Write the pattern point by point as CSV: observed, and the "
            "calculated and difference curves once a refinement has run."
        )
        self.export_pattern_btn.clicked.connect(self.export_pattern_csv)
        data_row.addWidget(self.export_pattern_btn)
        export_form.addRow("Data:", data_row)

        self.dpi = no_wheel(QSpinBox())
        self.dpi.setRange(72, 600)
        self.dpi.setValue(display_settings.export_dpi())
        self.dpi.setToolTip("Resolution of the PNG export.")
        self.dpi.valueChanged.connect(
            lambda value: display_settings.update({"plot_dpi": value})
        )
        export_form.addRow("Image DPI:", compact(self.dpi, 90))
        return group

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

    def _on_pawley_toggled(self, *_args):
        """Pawley intensities absorb the scale factor, so its bound does nothing."""
        enabled = not self.refine_intensities.isChecked()
        hint = self._MAX_SCALE_HINT if enabled else self._MAX_SCALE_PAWLEY_HINT
        self.max_scale.setEnabled(enabled)
        self.max_scale.setToolTip(hint)
        self.max_scale_label.setEnabled(enabled)
        self.max_scale_label.setToolTip(hint)

    def on_enter(self):
        n = len(self.session.selected_phases) or len(self.session.matched_phases)
        self.refine_btn.setEnabled(n > 0 and self.session.has_pattern())
        if n == 0:
            self.status.setText("No phases selected. Match phases in the Phases tab first.")
        else:
            self.status.setText(f"{n} phase(s) ready for refinement.")

    def _phases_for_refine(self):
        return self.session.selected_phases or self.session.matched_phases

    # Values a run hands to the next one. Everything here is a quantity the
    # refinement itself determines, so restarting it from a default would throw
    # away the result of the previous stage.
    _CARRIED_PHASE_KEYS = (
        ("scale_factor", "scale"),
        ("crystallite_size", "crystallite_size"),
        ("microstrain", "microstrain"),
        ("asymmetry", "asymmetry"),
        ("lattice_scale", "lattice_scale"),
        ("unit_cell", "unit_cell"),
        ("absorption", "absorption"),
        ("harmonic_coeffs", "harmonic_coeffs"),
    )

    def _carried_values(self):
        """Starting values from the last refinement: (per phase, global)."""
        results = self.session.lebail_results
        if not self.continue_previous.isChecked():
            return {}, {}
        if not (results and results.get("success")):
            return {}, {}

        inner = results.get("refinement_results") or {}
        per_phase = {}
        for row in inner.get("phase_summary") or []:
            name = row.get("name")
            if not name:
                continue
            per_phase[name] = {
                key: row.get(source) for key, source in self._CARRIED_PHASE_KEYS
            }

        previous = inner.get("global_parameters") or {}
        carried_globals = {
            key: previous.get(key)
            for key in ("zero_shift", "displacement", "axial_asymmetry")
        }
        # The instrument widths are seeded from the Initial FWHM box, so only
        # carry them when the previous run actually refined them away from it.
        if previous.get("refine_instrument_profile"):
            for key in ("u_param", "v_param", "w_param"):
                carried_globals[key] = previous.get(key)
        # Likewise the satellite ratio: the box holds it unless a run refined it,
        # and unticking the doublet has to turn it off rather than carry it on
        if previous.get("refine_alpha2_ratio") and self.model_alpha2.isChecked():
            carried_globals["alpha2_ratio"] = previous.get("alpha2_ratio")
        return per_phase, carried_globals

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

            # Seed the instrument Gaussian with the user's FWHM estimate; the
            # sample Lorentzian terms start from GSAS-II defaults and refine
            # from there. Keeping the instrument fixed is what keeps the
            # size/strain fit stable.
            initial_w = fwhm ** 2
            refine_size = self.refine_size.isChecked()
            refine_strain = self.refine_strain.isChecked()
            refine_asymmetry = self.refine_asymmetry.isChecked()
            carry_over, carry_globals = self._carried_values()
            refinement_params = {
                "carry_over": carry_over,
                "carry_globals": carry_globals,
                "initial_u": initial_w * 0.05,
                "initial_v": 0.0,
                "initial_w": initial_w,
                "max_scale": self.max_scale.value(),
                "refine_cell": self.refine_cell.isChecked(),
                "refine_profile": refine_size or refine_strain or refine_asymmetry,
                "refine_size": refine_size,
                "refine_strain": refine_strain,
                "refine_asymmetry": refine_asymmetry,
                "refine_instrument_profile": self.refine_instrument.isChecked(),
                "refine_axial_asymmetry": self.refine_axial.isChecked(),
                "alpha2_ratio": (self.alpha2_ratio.value()
                                 if self.model_alpha2.isChecked() else 0.0),
                "refine_alpha2_ratio": (self.model_alpha2.isChecked()
                                        and self.refine_alpha2.isChecked()),
                "refine_intensities": self.refine_intensities.isChecked(),
                "intensity_model": self.intensity_model.currentData() or "fixed",
                "fit_peak_regions_only": self.fit_peaks_only.isChecked(),
                # Anything set by hand in the parameter grid overrides the
                # run-wide defaults above, for that phase only
                "phase_overrides": getattr(self.session, "phase_overrides", {}) or {},
                "refine_zero_shift": self.refine_zero_shift.isChecked(),
                "refine_displacement": self.refine_displacement.isChecked(),
                "refine_absorption": self.refine_absorption.isChecked(),
                "refine_harmonics": self.refine_harmonics.isChecked(),
                # The order stays set even when the coefficients are not being
                # refined, so that a texture correction already found is still
                # applied. All-zero coefficients make the correction a no-op.
                "harmonic_order": self.harmonic_order.currentData(),
            }

            LeBailRefinement.plot_callback = None
            results, error = self._run_watched(
                experimental_data, phase_list, two_theta_range, refinement_params
            )
            if error:
                raise RuntimeError(error)
            if results is None:
                self.status.setText("Refinement cancelled.")
                return

            self.session.set_lebail_results(results)
            self.status.setText(self._completion_message(results))
            self.workspace.refresh_plot()
            self.workspace.set_status(
                "Le Bail refinement stopped early" if results.get("cancelled")
                else "Le Bail refinement complete"
            )
        except Exception as e:
            QMessageBox.critical(self, "Refinement Error", str(e))
            self.status.setText("Refinement failed.")
        finally:
            self.progress.setVisible(False)
            self.refine_btn.setEnabled(True)

    def _run_watched(self, experimental_data, phase_list, two_theta_range,
                     refinement_params):
        """
        Run the refinement behind a progress window, and wait for it.

        The window is modal and runs its own event loop, so this call still
        returns the finished results to its caller while the refinement itself
        happens on a worker thread and the window stays live.
        """
        worker = RefinementWorker(self.analyzer, {
            "experimental_data": experimental_data,
            "identified_phases": phase_list,
            "max_iterations": self.max_iter.value(),
            "two_theta_range": two_theta_range,
            "refinement_params": refinement_params,
        })
        dialog = RefinementProgressDialog(worker, self)
        dialog.set_observed(
            experimental_data["two_theta"], experimental_data["intensity"]
        )
        try:
            dialog.exec_()
        finally:
            worker.cancel()
            worker.wait(5000)
            hold_focus(self)
        return dialog.results, dialog.error

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

    @restores_focus
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

    @restores_focus
    def export_csv_data(self):
        """Write the results table, under the statistics that qualify it."""
        results = self.session.lebail_results
        if not (results and results.get("success")):
            QMessageBox.warning(
                self, "No Results",
                "Run a Le Bail refinement before exporting the results table.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results Table", "refinement_results.csv",
            "CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for name, value in refinement_table.global_rows(results):
                    writer.writerow([name, value])
                writer.writerow([])
                writer.writerow(
                    [label for label, _ in refinement_table.SUMMARY_COLUMNS]
                )
                writer.writerows(refinement_table.summary_rows(results))
            self.workspace.set_status(f"Exported {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    @restores_focus
    def export_pattern_csv(self):
        """Write the pattern itself: observed, calculated and difference."""
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Data", "No pattern to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Pattern CSV", "xrd_pattern.csv",
            "CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return

        two_theta = np.asarray(pattern["two_theta"], dtype=float)
        observed = np.asarray(pattern["intensity"], dtype=float)
        calculated = None
        results = self.session.lebail_results
        if results and results.get("success"):
            inner = results.get("refinement_results") or {}
            candidate = inner.get("calculated_pattern")
            # The refinement may have run over a narrowed 2theta range, in which
            # case its grid no longer lines up with the pattern
            if candidate is not None and len(candidate) == len(observed):
                calculated = np.asarray(candidate, dtype=float)

        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if calculated is None:
                    writer.writerow(["two_theta", "intensity"])
                    writer.writerows(zip(two_theta, observed))
                else:
                    writer.writerow(
                        ["two_theta", "observed", "calculated", "difference"]
                    )
                    writer.writerows(
                        zip(two_theta, observed, calculated, observed - calculated)
                    )
            self.workspace.set_status(f"Exported {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
