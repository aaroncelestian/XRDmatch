"""Identify stage — pattern search + phase matching."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from utils.fast_pattern_search import FastPatternSearchEngine
from utils.pattern_search import PatternSearchEngine
from utils.local_database import LocalCIFDatabase
from utils.rir_quant import quantify as rir_quantify, summary_lines as rir_summary_lines
from utils.fingerprint_search import (
    coincidence_fraction,
    fingerprint_score,
    rank_by_fingerprint,
    select_fingerprint_peaks,
)
from utils.conditions import (AMBIENT_MAX_PRESSURE_GPA, AMBIENT_MAX_TEMPERATURE_K,
                             AMBIENT_MIN_TEMPERATURE_K)
from utils.residual_search import (
    build_residual_pattern,
    build_residual_peaks,
    filter_new_hits,
    is_excluded_hit,
    exclusion_sets,
    mineral_ids,
    mineral_key,
)
from utils.two_theta_shift import (
    DISPLACEMENT,
    MIN_LINES_TO_FIT,
    SHIFT_MODELS,
    describe as describe_shift,
    fit_shift,
    remove_shift,
    shift_pattern,
    unshift_pattern,
)
from gui.matching_tab import PhaseMatchingThread
from gui.widgets.control_bar import OptionsDialog, compact


SEARCH_METHODS = [
    ("Fingerprint (mixtures)", "fingerprint"),
    ("Ultra-Fast Correlation", "ultrafast"),
    ("Peak Match", "peaks"),
    ("Pearson Correlation", "correlation"),
    ("Combined", "combined"),
    ("Ensemble", "ensemble"),
]


class IdentifyStage(QWidget):
    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self.fast_engine = FastPatternSearchEngine()
        self.search_engine = PatternSearchEngine()
        self.local_db = LocalCIFDatabase()
        self._match_thread = None
        self._search_results = []
        self._kept_phases = []  # accepted across residual rounds
        self._theo_cache = {}
        self._options = None

        self.control_panel = self._build_controls()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

    # --- UI ---

    def _build_controls(self) -> QWidget:
        """Controls laid out as a grid so everything stays visible at once."""
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(8, 4, 8, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # Row 0 — primary actions
        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("primaryButton")
        self.search_btn.setToolTip("Search the database for candidate phases")
        self.search_btn.clicked.connect(self.start_search)
        run_row.addWidget(self.search_btn)

        self.match_btn = QPushButton("Match Selected")
        self.match_btn.setObjectName("primaryButton")
        self.match_btn.setToolTip("Run peak matching on the checked candidates")
        self.match_btn.clicked.connect(self.start_matching)
        self.match_btn.setEnabled(False)
        run_row.addWidget(self.match_btn)

        self.residual_btn = QPushButton("Search Residual")
        self.residual_btn.setToolTip(
            "Keep selected phases, soft-subtract their contribution, and search again. "
            "Unmatched peaks are boosted; overlapping peaks keep partial weight."
        )
        self.residual_btn.clicked.connect(self.search_residual)
        self.residual_btn.setEnabled(False)
        run_row.addWidget(self.residual_btn)

        self.rir_btn = QPushButton("RIR Quant")
        self.rir_btn.setToolTip(
            "Weight percents from reference intensity ratios (Chung) for the checked phases"
        )
        self.rir_btn.clicked.connect(self.run_rir_quant)
        self.rir_btn.setEnabled(False)
        run_row.addWidget(self.rir_btn)
        run_row.addStretch()
        grid.addLayout(run_row, 0, 0, 1, 7)

        # Row 1 — quick add
        self.mineral_search = QLineEdit()
        self.mineral_search.setPlaceholderText("Add known mineral — e.g. quartz")
        self.mineral_search.returnPressed.connect(self.add_mineral_by_name)
        self.add_mineral_btn = QPushButton("Add")
        self.add_mineral_btn.setToolTip("Search the local database and add a mineral as a candidate")
        self.add_mineral_btn.clicked.connect(self.add_mineral_by_name)
        grid.addWidget(self._label("Add mineral:"), 1, 0)
        grid.addWidget(self.mineral_search, 1, 1, 1, 4)
        grid.addWidget(self.add_mineral_btn, 1, 5)

        # Rows 2-3 — search parameters, two label/field pairs per row
        self.method_combo = QComboBox()
        for label, key in SEARCH_METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.setToolTip(
            "Fingerprint scores each candidate on its own strong lines, so minor "
            "phases in a mixture are not penalized for unexplained peaks."
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

        self.min_corr = QDoubleSpinBox()
        self.min_corr.setRange(0.01, 1.0)
        self.min_corr.setDecimals(2)
        self.min_corr.setSingleStep(0.05)
        self.min_corr.setValue(0.30)

        self.fp_min_score = QDoubleSpinBox()
        self.fp_min_score.setRange(0.0, 1.0)
        self.fp_min_score.setDecimals(2)
        self.fp_min_score.setSingleStep(0.05)
        self.fp_min_score.setValue(0.25)
        self.fp_min_score.setToolTip(
            "Minimum fingerprint score to list a candidate. The score credits a "
            "candidate only for matches beyond what its line count would hit by "
            "chance, so it runs lower than a plain 'fraction of lines present'."
        )

        self.max_results = QSpinBox()
        self.max_results.setRange(10, 500)
        self.max_results.setValue(50)

        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.01, 2.0)
        self.tolerance.setDecimals(2)
        self.tolerance.setValue(0.20)
        self.tolerance.setSuffix("°")

        self.ambient_only = QCheckBox("Ambient only")
        self.ambient_only.setChecked(True)
        self.ambient_only.setToolTip(
            f"Search only structures measured at or below {AMBIENT_MAX_PRESSURE_GPA:g} GPa\n"
            f"and within {AMBIENT_MIN_TEMPERATURE_K:.0f}–{AMBIENT_MAX_TEMPERATURE_K:.0f} K.\n\n"
            "About a quarter of the AMCSD archive is high-pressure or high-temperature\n"
            "work. Those cells are compressed or expanded, so their lines sit at shifted\n"
            "2θ and match the wrong phases. Uncheck only if your sample really was\n"
            "measured off-ambient."
        )

        options_btn = QPushButton("Options…")
        options_btn.setToolTip("Fingerprint, residual, weighting, and multi-phase settings")
        options_btn.clicked.connect(self._show_options)

        self._build_shift_widgets()

        grid.addWidget(self._label("Method:"), 2, 0)
        grid.addWidget(compact(self.method_combo, 170), 2, 1)
        grid.addWidget(self._label("2θ tol:"), 2, 2)
        grid.addWidget(compact(self.tolerance, 80), 2, 3)
        grid.addWidget(self._label("Min fingerprint:"), 2, 4)
        grid.addWidget(compact(self.fp_min_score, 80), 2, 5)

        grid.addWidget(self._label("Max results:"), 3, 0)
        grid.addWidget(compact(self.max_results, 80), 3, 1)
        grid.addWidget(self._label("Min corr:"), 3, 2)
        grid.addWidget(compact(self.min_corr, 80), 3, 3)
        grid.addWidget(self.ambient_only, 3, 4)
        grid.addWidget(options_btn, 3, 5)

        # Row 4 — sample displacement correction
        grid.addWidget(self._label("2θ shift:"), 4, 0)
        grid.addWidget(compact(self.shift, 80), 4, 1)
        grid.addWidget(self._label("Auto fit ±:"), 4, 2)
        grid.addWidget(compact(self.shift_span, 80), 4, 3)
        grid.addWidget(self.fit_shift_btn, 4, 4)
        grid.addWidget(self.clear_shift_btn, 4, 5)

        # Row 5 — list actions, filled in by the workspace
        self.table_actions = QHBoxLayout()
        self.table_actions.setSpacing(6)
        grid.addLayout(self.table_actions, 5, 0, 1, 7)

        # Row 6 — status and progress
        self.status = QLabel("Load a pattern, find peaks, then search.")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        grid.addWidget(self.status, 6, 0, 1, 5)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(140)
        grid.addWidget(self.progress, 6, 5, 1, 2)

        grid.setColumnStretch(6, 1)
        grid.setRowStretch(7, 1)

        self._build_option_widgets()
        self._on_method_changed()
        return panel

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("mutedLabel")
        return label

    def add_action_widget(self, widget: QWidget):
        """Let the workspace place list-related buttons in the controls grid."""
        self.table_actions.addWidget(widget)
        return widget

    def finish_action_row(self):
        self.table_actions.addStretch()

    def _build_shift_widgets(self):
        """
        Sample-displacement correction for the reference lines.

        A displaced mount moves every observed line, so a database pattern sat
        at its nominal 2θ misses peaks it should hit and the phase never makes
        the candidate list. These controls move the reference lines instead of
        the data, which leaves the measured pattern untouched.
        """
        self.shift = QDoubleSpinBox()
        self.shift.setRange(-3.0, 3.0)
        self.shift.setDecimals(3)
        self.shift.setSingleStep(0.01)
        self.shift.setValue(0.0)
        self.shift.setSuffix("°")
        self.shift.setToolTip(
            "Move reference lines to where a displaced sample puts them.\n\n"
            "Positive shifts them to higher 2θ. With the displacement model the "
            "value is the shift extrapolated to 2θ = 0 and the actual shift is "
            "value × cos θ, so it fades out towards high angle.\n\n"
            "This corrects the reference positions only — the measured pattern "
            "is untouched. To correct the data itself use the 2θ offset in the "
            "Background tab."
        )

        self.shift_span = QDoubleSpinBox()
        self.shift_span.setRange(0.0, 2.0)
        self.shift_span.setDecimals(2)
        self.shift_span.setSingleStep(0.05)
        self.shift_span.setValue(0.0)
        self.shift_span.setSuffix("°")
        self.shift_span.setToolTip(
            "Fit each candidate its own shift within this much of the value on "
            "the left (0 = off).\n\n"
            "Use this when the displacement is unknown: a phase that is present "
            "but sitting 0.3° off will not be found at any sensible tolerance, "
            "and widening the tolerance instead matches everything.\n\n"
            "One shift has to satisfy all of a phase's strong lines at once, so "
            "it stays real evidence — but keep the range only as wide as the "
            "displacement you actually expect."
        )

        self.shift_model = QComboBox()
        for label, key in SHIFT_MODELS:
            self.shift_model.addItem(label, key)
        self.shift_model.setToolTip(
            "Displacement is the usual cause in Bragg-Brentano: the shift goes "
            "as cos θ, largest at low angle. Zero shift is a flat detector "
            "offset, the same at every angle."
        )

        self.fit_shift_btn = QPushButton("Fit to Row")
        self.fit_shift_btn.setToolTip(
            "Fit the shift to the highlighted candidate's lines and put the "
            "result in the box. Use it once you recognize a phase, then turn "
            "auto fit off and search again with the displacement pinned down."
        )
        self.fit_shift_btn.clicked.connect(self.fit_shift_to_row)

        self.clear_shift_btn = QPushButton("No Shift")
        self.clear_shift_btn.setToolTip("Reset the shift and the auto-fit range to zero")
        self.clear_shift_btn.clicked.connect(self.clear_shift)

        for spin in (self.shift, self.shift_span):
            spin.valueChanged.connect(self._on_shift_changed)

    def _on_shift_changed(self, *_args):
        self.workspace.refresh_plot()

    def clear_shift(self):
        self.shift.setValue(0.0)
        self.shift_span.setValue(0.0)

    def shift_model_key(self) -> str:
        return self.shift_model.currentData() or DISPLACEMENT

    def shift_for(self, result=None) -> float:
        """
        The 2θ shift that applies to one phase.

        Phase dicts carry an already-resolved value. Search hits carry the
        shift fitted while they were scored, which only counts while auto fit
        is on; otherwise every phase uses the single manual setting.
        """
        if isinstance(result, dict):
            phase = result.get("phase")
            for src in (result, phase if isinstance(phase, dict) else {}):
                resolved = src.get("two_theta_shift")
                if resolved is not None:
                    return float(resolved)
            if self.shift_span.value() > 0:
                fitted = (result.get("fingerprint") or {}).get("shift")
                if fitted is not None:
                    return float(fitted)
        return float(self.shift.value())

    def fit_shift_to_row(self):
        """Pin the shift down from a phase the user has recognized."""
        result = self.workspace.current_result()
        if result is None:
            QMessageBox.information(
                self, "No Phase Selected",
                "Click the phase you recognize in the list, then Fit to Row.",
            )
            return
        if not self.session.has_peaks():
            QMessageBox.warning(self, "No Peaks", "Find peaks in the Peaks tab first.")
            return

        theo = self.theoretical_peaks_for(result)
        if not theo or len(theo.get("two_theta", [])) == 0:
            QMessageBox.information(
                self, "No Reference Lines",
                "This phase has no reference pattern to fit against.",
            )
            return

        fp = select_fingerprint_peaks(
            theo.get("two_theta", []),
            theo.get("intensity", []),
            n_peaks=self.fp_n_peaks.value(),
            min_rel_intensity=self.fp_min_rel.value(),
            two_theta_range=self._measured_range(),
        )
        # A generous window even when auto fit is off, since this is the step
        # that tells the user how far off the mount actually is
        span = self.shift_span.value() or 1.0
        fitted, n_found = fit_shift(
            self.session.peaks["two_theta"],
            fp["two_theta"],
            fp["intensity"],
            tolerance=self.tolerance.value(),
            center=0.0,
            span=span,
            model=self.shift_model_key(),
        )
        name = result.get("mineral_name") or result.get("phase", {}).get("mineral", "phase")
        if n_found < MIN_LINES_TO_FIT:
            QMessageBox.information(
                self, "Not Enough Lines",
                f"Only {n_found} of {name}'s strong lines land on a peak within "
                f"±{span:.2f}°, which is too few to pin a shift down.\n\n"
                "Widen the auto fit range or pick a phase you are surer of.",
            )
            return

        self.shift.setValue(fitted)
        self.status.setText(
            f"Fitted {describe_shift(fitted, self.shift_model_key())} from {name} "
            f"({n_found} lines). Set Auto fit to 0 and search again to apply it "
            "to every candidate."
        )
        self.workspace.set_status(f"2θ shift {fitted:+.3f}° from {name}")

    def _build_option_widgets(self):
        """Advanced parameters — shown in the Options popup, owned here."""
        self.fp_n_peaks = QSpinBox()
        self.fp_n_peaks.setRange(3, 30)
        self.fp_n_peaks.setValue(10)
        self.fp_n_peaks.setToolTip("How many of the candidate's strongest lines define its fingerprint")

        self.fp_min_rel = QDoubleSpinBox()
        self.fp_min_rel.setRange(0.5, 50.0)
        self.fp_min_rel.setDecimals(1)
        self.fp_min_rel.setValue(5.0)
        self.fp_min_rel.setSuffix("%")
        self.fp_min_rel.setToolTip("Ignore reference lines weaker than this fraction of the phase maximum")

        self.fp_min_found = QSpinBox()
        self.fp_min_found.setRange(1, 20)
        self.fp_min_found.setValue(3)
        self.fp_min_found.setToolTip("Minimum number of fingerprint lines that must be found")

        self.fp_require_top = QCheckBox("Require strongest line present")
        self.fp_require_top.setChecked(False)
        self.fp_require_top.setToolTip(
            "Reject a candidate outright when its most intense line is missing"
        )

        self.fp_dedupe = QCheckBox("One entry per mineral name")
        self.fp_dedupe.setChecked(True)
        self.fp_dedupe.setToolTip(
            "Keep only the best-scoring database record for each mineral, so duplicate "
            "entries do not push minor phases out of the list"
        )

        self.pool_min_coverage = QDoubleSpinBox()
        self.pool_min_coverage.setRange(0.01, 1.0)
        self.pool_min_coverage.setDecimals(2)
        self.pool_min_coverage.setSingleStep(0.05)
        self.pool_min_coverage.setValue(0.10)
        self.pool_min_coverage.setToolTip(
            "Screening floor: fraction of a phase's own intensity that must fall on "
            "observed peaks to enter the candidate pool. Keep it low for minor phases."
        )

        self.search_max_peaks = QSpinBox()
        self.search_max_peaks.setRange(0, 500)
        self.search_max_peaks.setSingleStep(5)
        self.search_max_peaks.setValue(35)
        self.search_max_peaks.setToolTip(
            "Match on this many of the strongest peaks (0 = all). Every extra weak "
            "peak widens the net of positions a phase can match by chance, so a "
            "long list makes real phases harder to pick out, not easier. Residual "
            "search boosts unexplained peaks, so they compete for these slots."
        )

        self.search_min_int = QDoubleSpinBox()
        self.search_min_int.setRange(0.0, 20.0)
        self.search_min_int.setDecimals(2)
        self.search_min_int.setSingleStep(0.25)
        self.search_min_int.setValue(0.50)
        self.search_min_int.setSuffix("%")
        self.search_min_int.setToolTip(
            "Searching ignores peaks weaker than this percent of the strongest peak. "
            "Noise-level peaks match almost any phase by chance, which buries the "
            "real ones. They stay in the Peaks tab and still count as unexplained "
            "intensity for residual search."
        )

        self.pool_size = QSpinBox()
        self.pool_size.setRange(50, 10000)
        self.pool_size.setSingleStep(250)
        self.pool_size.setValue(3000)
        self.pool_size.setToolTip(
            "How many screened candidates to score in detail. Minor phases in a "
            "mixture often screen around rank 1000, so keep this generous."
        )

        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(0.0, 1.0)
        self.min_score.setDecimals(2)
        self.min_score.setValue(0.01)

        self.peak_tol = QDoubleSpinBox()
        self.peak_tol.setRange(0.05, 1.0)
        self.peak_tol.setDecimals(2)
        self.peak_tol.setValue(0.2)
        self.peak_tol.setSuffix("°")

        self.peak_weight = QDoubleSpinBox()
        self.peak_weight.setRange(0.0, 1.0)
        self.peak_weight.setDecimals(2)
        self.peak_weight.setValue(0.6)

        self.corr_weight = QDoubleSpinBox()
        self.corr_weight.setRange(0.0, 1.0)
        self.corr_weight.setDecimals(2)
        self.corr_weight.setValue(0.4)

        self.overlap_keep = QDoubleSpinBox()
        self.overlap_keep.setRange(0.0, 1.0)
        self.overlap_keep.setDecimals(2)
        self.overlap_keep.setSingleStep(0.05)
        self.overlap_keep.setValue(0.35)
        self.overlap_keep.setToolTip(
            "Fraction of explained/overlapping intensity kept in the residual "
            "(0 = hard subtract, 1 = no subtract). Prevents discarding shared peaks."
        )

        self.unmatched_boost = QDoubleSpinBox()
        self.unmatched_boost.setRange(1.0, 3.0)
        self.unmatched_boost.setDecimals(2)
        self.unmatched_boost.setSingleStep(0.1)
        self.unmatched_boost.setValue(1.50)
        self.unmatched_boost.setToolTip(
            "Intensity multiplier for peaks not explained by selected phases"
        )

        self.rir_fwhm = QDoubleSpinBox()
        self.rir_fwhm.setRange(0.01, 2.0)
        self.rir_fwhm.setDecimals(3)
        self.rir_fwhm.setSingleStep(0.01)
        self.rir_fwhm.setValue(0.12)
        self.rir_fwhm.setSuffix("°")
        self.rir_fwhm.setToolTip(
            "Peak width used to fit reference patterns for RIR quantification.\n\n"
            "Set it near your measured FWHM. Too narrow and the fit misses "
            "intensity in the peak flanks; too wide and it absorbs neighbours."
        )

    def _show_options(self):
        if self._options is None:
            dlg = OptionsDialog("Phase Search Options", self.workspace.window())
            dlg.add_heading("Fingerprint")
            dlg.add_row("Fingerprint lines:", self.fp_n_peaks)
            dlg.add_row("Min line intensity:", self.fp_min_rel)
            dlg.add_row("Min lines found:", self.fp_min_found)
            dlg.add_row("", self.fp_require_top)
            dlg.add_row("", self.fp_dedupe)
            dlg.add_row("Search peak count:", self.search_max_peaks)
            dlg.add_row("Search peak floor:", self.search_min_int)
            dlg.add_row("Screen min coverage:", self.pool_min_coverage)
            dlg.add_row("Pool size:", self.pool_size)

            dlg.add_heading("2θ shift")
            dlg.add_row("Shift model:", self.shift_model)

            dlg.add_heading("Matching")
            dlg.add_row("Min match score:", self.min_score)
            dlg.add_row("Peak tolerance:", self.peak_tol)
            dlg.add_row("Peak weight:", self.peak_weight)
            dlg.add_row("Corr. weight:", self.corr_weight)

            dlg.add_heading("Residual & quantification")
            dlg.add_row("Overlap keep:", self.overlap_keep)
            dlg.add_row("Unmatched boost:", self.unmatched_boost)
            dlg.add_row("RIR fit FWHM:", self.rir_fwhm)
            self._options = dlg
        self._options.show_centered()

    def _method_key(self) -> str:
        return self.method_combo.currentData() or "fingerprint"

    def _on_method_changed(self, *_args):
        is_fp = self._method_key() == "fingerprint"
        self.fp_min_score.setEnabled(is_fp)

    # --- state ---

    def reset_results(self):
        self._search_results = []
        self._kept_phases = []

    def on_enter(self):
        self.update_action_states()
        can = self.session.has_pattern()
        if not can:
            self.status.setText("Load and process a pattern first.")
        elif not self.session.has_peaks():
            self.status.setText("Find peaks in the Peaks tab — fingerprint search needs them.")
        else:
            self.status.setText("Ready to search. Check the candidates you want, then match.")

    def accepted_phases(self) -> list:
        """
        Phases the user has committed to, from whichever list is showing.

        Checked candidates count as accepted, so residual and multi-phase work
        without having to run peak matching first. Phases kept from earlier
        residual rounds stay in the list even though the table now shows the
        newest candidates.
        """
        if getattr(self.workspace, "_results_mode", None) == "matches":
            return self.workspace.get_selected_matches()

        accepted = list(self._kept_phases)
        seen_ids, seen_names = exclusion_sets(accepted)
        for phase in self.workspace.get_selected_candidates():
            entry = {
                "phase": phase,
                "mineral_id": phase.get("id"),
                "mineral_name": phase.get("mineral"),
                "match_score": phase.get("search_score", 1.0),
            }
            if is_excluded_hit(entry, seen_ids, seen_names):
                continue
            theo = self.reference_peaks_for(phase)
            if theo:
                entry["theoretical_peaks"] = theo
            accepted.append(entry)
            seen_ids |= mineral_ids(entry)
            name = mineral_key(entry)
            if name:
                seen_names.add(name)
        return accepted

    def update_action_states(self):
        """Keep buttons in step with what is checked in the list."""
        can = self.session.has_pattern()
        has_peaks = self.session.has_peaks()
        self.search_btn.setEnabled(can)

        checked_candidates = self.workspace.get_selected_candidates()
        accepted = self.accepted_phases()

        self.match_btn.setEnabled(bool(checked_candidates) and has_peaks)
        self.match_btn.setToolTip(
            "Run peak matching on the checked candidates"
            if checked_candidates else
            "Check one or more candidates in the list first"
        )

        self.residual_btn.setEnabled(can and has_peaks and len(accepted) > 0)
        self.residual_btn.setToolTip(
            "Keep the checked phases, down-weight the peaks they explain, and search "
            "again for what is left"
            if accepted else
            "Check the phases you have already identified, then search the residual"
        )

        self.rir_btn.setEnabled(can and len(accepted) > 0)
        self.rir_btn.setToolTip(
            f"RIR weight percents for the {len(accepted)} checked phase(s)"
            if accepted else
            "Check the phases you want quantified"
        )

    def _measured_range(self):
        """Measured 2θ span — reference lines outside it cannot be expected."""
        pattern = self.session.active_pattern()
        if not pattern:
            return None
        tt = np.asarray(pattern["two_theta"], dtype=float)
        if len(tt) == 0:
            return None
        return float(np.min(tt)), float(np.max(tt))

    # --- theoretical peak access (shared with preview / details) ---

    def theoretical_peaks_for(self, result: dict):
        """
        Reference peaks for a search hit, match result, or phase dict.

        Positions are the unshifted database ones, which is what scoring needs
        — it fits or applies the shift itself. Use `reference_peaks_for` for
        anything that has to line up with the measured pattern.
        """
        if not isinstance(result, dict):
            return None
        theo = result.get("theoretical_peaks")
        if theo and len(theo.get("two_theta", [])) > 0:
            return unshift_pattern(theo)

        phase = result.get("phase", result)
        mineral_id = (
            result.get("mineral_id")
            or (phase.get("id") if isinstance(phase, dict) else None)
            or result.get("id")
        )
        if mineral_id is None:
            return None
        wl = round(float(self.session.wavelength), 4)
        key = (str(mineral_id), wl)
        if key in self._theo_cache:
            return self._theo_cache[key]
        try:
            pattern = self.local_db.get_diffraction_pattern(int(mineral_id), wl)
        except Exception:
            pattern = None
        self._theo_cache[key] = pattern
        return pattern

    def reference_peaks_for(self, result: dict):
        """Reference peaks moved to where this phase's shift puts them."""
        theo = self.theoretical_peaks_for(result)
        if not theo:
            return theo
        return shift_pattern(theo, self.shift_for(result), self.shift_model_key())

    # --- mineral quick-add ---

    def add_mineral_by_name(self):
        query = self.mineral_search.text().strip()
        if len(query) < 2:
            QMessageBox.information(self, "Add Mineral", "Type at least 2 characters.")
            return
        try:
            hits = self.local_db.search_by_mineral_name(query, limit=40)
        except Exception as e:
            QMessageBox.critical(self, "Database Error", str(e))
            return
        if not hits:
            QMessageBox.information(self, "No Matches", f"No minerals matching “{query}”.")
            return

        # Partial matches come back mixed in, so put the closest names first
        needle = query.lower()

        def relevance(hit):
            name = str(hit.get("mineral_name", "")).lower()
            if name == needle:
                return (0, name)
            if name.startswith(needle):
                return (1, name)
            return (2, name)

        hits.sort(key=relevance)
        exact = [h for h in hits if str(h.get("mineral_name", "")).lower() == needle]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(hits) == 1:
            chosen = hits[0]
        else:
            chosen = self._pick_mineral_dialog(hits, query)
            if chosen is None:
                return

        phase = self._db_row_to_phase(chosen)
        self.session.add_candidates([phase])
        row = {
            "mineral_id": chosen.get("id"),
            "mineral_name": chosen.get("mineral_name"),
            "chemical_formula": chosen.get("chemical_formula"),
            "space_group": chosen.get("space_group"),
            "cell_a": chosen.get("cell_a"),
            "cell_b": chosen.get("cell_b"),
            "cell_c": chosen.get("cell_c"),
            "cell_alpha": chosen.get("cell_alpha"),
            "cell_beta": chosen.get("cell_beta"),
            "cell_gamma": chosen.get("cell_gamma"),
            "rir": chosen.get("rir"),
            "match_score": 1.0,
            "manual_add": True,
        }
        # Report how well the known mineral actually fits the peaks
        if self.session.has_peaks():
            theo = self.theoretical_peaks_for(row)
            if theo:
                info = fingerprint_score(
                    self.session.peaks["two_theta"],
                    self.session.peaks["intensity"],
                    theo.get("two_theta", []),
                    theo.get("intensity", []),
                    tolerance=self.tolerance.value(),
                    n_peaks=self.fp_n_peaks.value(),
                    min_rel_intensity=self.fp_min_rel.value(),
                    exp_range=self._measured_range(),
                    shift=self.shift.value(),
                    shift_span=self.shift_span.value(),
                    shift_model=self.shift_model_key(),
                )
                row["fingerprint"] = info
                row["fingerprint_score"] = info["score"]

        self._append_candidate_result(row)
        self.update_action_states()
        self.mineral_search.clear()
        name = chosen.get("mineral_name", "phase")
        fp = row.get("fingerprint")
        if fp:
            self.status.setText(
                f"Added {name} — {fp['n_found']}/{fp['n_expected']} strong lines present "
                f"(fingerprint {fp['score']:.2f})."
            )
        else:
            self.status.setText(f"Added {name}. Select it and run matching when ready.")
        self.workspace.set_status(f"Added mineral: {name}")

    def _pick_mineral_dialog(self, hits: list, query: str):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Choose mineral — “{query}”")
        dlg.resize(480, 360)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"{len(hits)} matches — select one to add:"))
        lst = QListWidget()
        for h in hits:
            text = (
                f"{h.get('mineral_name', '?')}  ·  "
                f"{h.get('chemical_formula', '')}  ·  "
                f"{h.get('space_group', '')}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, h)
            lst.addItem(item)
        lst.setCurrentRow(0)
        lst.itemDoubleClicked.connect(lambda *_: dlg.accept())
        lay.addWidget(lst)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return None
        item = lst.currentItem()
        return item.data(Qt.UserRole) if item else None

    @staticmethod
    def _db_row_to_phase(mineral: dict) -> dict:
        return {
            "id": mineral.get("id"),
            "amcsd_id": mineral.get("amcsd_id") or mineral.get("id"),
            "mineral": mineral.get("mineral_name", "Unknown"),
            "formula": mineral.get("chemical_formula", "Unknown"),
            "space_group": mineral.get("space_group", "Unknown"),
            "cell_a": mineral.get("cell_a"),
            "cell_b": mineral.get("cell_b"),
            "cell_c": mineral.get("cell_c"),
            "cell_alpha": mineral.get("cell_alpha"),
            "cell_beta": mineral.get("cell_beta"),
            "cell_gamma": mineral.get("cell_gamma"),
            "rir": mineral.get("rir"),
            "cif_content": mineral.get("cif_content"),
            "local_db": True,
            "manual_add": True,
            "search_score": 1.0,
        }

    def _append_candidate_result(self, result: dict):
        """Merge a manual hit into the candidates table and check it."""
        existing = list(getattr(self.workspace, "_candidate_results", []) or [])
        key = (result.get("mineral_name") or "").lower()
        if not any((r.get("mineral_name") or "").lower() == key for r in existing):
            existing.insert(0, result)
        self._search_results = existing
        self.workspace.set_results_candidates(existing)
        self.workspace.check_candidate_rows([key])

    # --- search ---

    def start_search(self):
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Pattern", "Load a pattern first.")
            return
        self._run_search(pattern, residual_mode=False)

    def search_residual(self):
        """Search on soft residual after locking selected matched phases."""
        pattern = self.session.active_pattern()
        if not pattern:
            QMessageBox.warning(self, "No Pattern", "Load a pattern first.")
            return

        if not self.session.has_peaks():
            QMessageBox.warning(
                self, "Peaks Required",
                "Residual search works from the peak list. Find peaks in the Peaks tab first.",
            )
            return

        selected = self.accepted_phases() or list(self.session.selected_phases)
        if not selected:
            QMessageBox.warning(
                self, "No Phases Selected",
                "Check the phases you have already identified, then Search Residual.\n\n"
                "This works from the candidate list too — no need to run matching first.",
            )
            return

        self._kept_phases = selected
        self.session.set_selected_phases(selected)
        overlap_keep = self.overlap_keep.value()
        boost = self.unmatched_boost.value()
        tol = self.tolerance.value()

        residual_pattern, pinfo = build_residual_pattern(
            pattern, selected, overlap_keep=overlap_keep
        )

        residual_peaks = None
        peak_info = {}
        if self.session.has_peaks():
            residual_peaks, peak_info = build_residual_peaks(
                self.session.peaks,
                selected,
                tol,
                unmatched_boost=boost,
                overlap_keep=overlap_keep,
            )

        self.status.setText(
            f"Residual search… remaining intensity {pinfo['fraction_remaining']*100:.0f}%"
            + (f", unmatched peaks {peak_info.get('n_unmatched', '?')}" if peak_info else "")
        )
        self._run_search(
            residual_pattern,
            residual_mode=True,
            residual_peaks=residual_peaks,
            kept_phases=selected,
        )

    def _run_search(
        self,
        pattern,
        *,
        residual_mode: bool = False,
        residual_peaks=None,
        exclude_keys=None,
        kept_phases=None,
    ):
        method = self._method_key()
        if method == "fingerprint" and not self.session.has_peaks():
            QMessageBox.warning(
                self, "Peaks Required",
                "Fingerprint search compares peak positions.\n\n"
                "Find peaks in the Peaks tab first, or pick another search method.",
            )
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.search_btn.setEnabled(False)
        self.residual_btn.setEnabled(False)
        if not residual_mode:
            self.status.setText("Searching…")

        try:
            if method == "fingerprint":
                results = self._fingerprint_search(pattern, residual_peaks)
            elif method == "ultrafast":
                results = self._ultra_fast(pattern, self.min_corr.value(), self.max_results.value())
            else:
                results = self._legacy_search(pattern, method, peaks_override=residual_peaks)

            results = results or []
            kept = list(kept_phases or []) or list(self.session.selected_phases)
            dropped = 0
            if residual_mode or kept:
                before = len(results)
                results = filter_new_hits(results, kept)
                dropped = before - len(results)

            if exclude_keys:
                results = [
                    r for r in results
                    if mineral_key({"mineral_name": r.get("mineral_name")}) not in exclude_keys
                ]

            self._search_results = results
            candidates = [self._result_to_phase(r) for r in results]
            self.session.set_candidates(candidates)
            self.workspace.set_results_candidates(results)
            self.update_action_states()

            label = "Residual search" if residual_mode else "Search"
            extra = f" (excluded {dropped} already-found)" if dropped else ""
            if candidates:
                hint = " Click a row to preview its peaks; arrow keys step through."
            elif method == "fingerprint" and self.shift_span.value() <= 0:
                hint = (
                    " Try a lower Min fingerprint, or set Auto fit ± to 0.30° — a "
                    "displaced sample puts every line off position and no phase matches."
                )
            else:
                hint = " Try a lower Min fingerprint or a wider 2θ tolerance."
            if method == "fingerprint":
                hint += self._shift_summary(results)
            chance = getattr(self, "_last_chance", 0.0)
            if method == "fingerprint" and chance > 0.5:
                hint += (
                    f" Note: match windows cover {chance*100:.0f}% of the pattern, so "
                    "positions barely constrain the answer — lower Search peak count "
                    "or 2θ tol in Options."
                )
            self.status.setText(f"{label}: {len(candidates)} candidates{extra}.{hint}")
            self.workspace.set_status(f"{label}: {len(candidates)} candidates")
            self.workspace.refresh_plot()
        except Exception as e:
            QMessageBox.critical(self, "Search Error", str(e))
            self.status.setText("Search failed.")
        finally:
            self.progress.setVisible(False)
            self.update_action_states()

    def _significant_peaks(self, peaks: dict) -> dict:
        """
        Peaks strong enough to be worth matching on.

        Positions alone stop discriminating once the list includes noise-level
        peaks: their windows cover so much of the pattern that most phases look
        fully present. Weak peaks stay in the session for the user and for
        residual bookkeeping; they just do not drive the search.
        """
        inten = np.asarray(peaks.get("intensity", []), dtype=float)
        if len(inten) == 0:
            return peaks
        imax = float(np.max(inten))
        if imax <= 0:
            return peaks

        keep = np.ones(len(inten), dtype=bool)
        floor_pct = self.search_min_int.value()
        if floor_pct > 0:
            keep &= inten >= floor_pct / 100.0 * imax

        limit = self.search_max_peaks.value()
        if 0 < limit < int(np.sum(keep)):
            eligible = np.flatnonzero(keep)
            strongest = eligible[np.argsort(inten[eligible])[::-1][:limit]]
            keep = np.zeros(len(inten), dtype=bool)
            keep[strongest] = True

        if np.all(keep) or np.sum(keep) < 4:  # never strip the list bare
            return peaks

        out = {}
        for key, value in peaks.items():
            arr = np.asarray(value) if isinstance(value, (list, np.ndarray)) else None
            out[key] = arr[keep] if arr is not None and arr.shape[:1] == keep.shape else value
        print(f"Search uses the strongest {int(np.sum(keep))} of {len(inten)} peaks")
        return out

    def _fingerprint_search(self, pattern, residual_peaks=None):
        """Screen the database on peak positions, then score each candidate's own lines."""
        exp_peaks = residual_peaks if residual_peaks is not None else self.session.peaks
        exp_peaks = self._significant_peaks(exp_peaks)
        exp_weights = None
        if residual_peaks is not None:
            w = residual_peaks.get("residual_weights")
            if w is not None and len(w) == len(residual_peaks.get("two_theta", [])):
                exp_weights = np.asarray(w, dtype=float)

        pool_size = self.pool_size.value()
        tol = self.tolerance.value()
        wl = pattern.get("wavelength", self.session.wavelength)
        shift = self.shift.value()
        span = self.shift_span.value()
        model = self.shift_model_key()
        pool = []

        # How much of the pattern a reference line can hit by luck; warned about
        # in the status line because it decides whether matching means anything
        self._last_chance = coincidence_fraction(
            exp_peaks.get("two_theta", []), tol, self._measured_range(),
        )

        if self.fast_engine.search_index is not None:
            pool = self.fast_engine.screen_by_peak_coverage(
                exp_peaks["two_theta"],
                weights=exp_weights,
                tolerance=tol,
                top_n=pool_size,
                min_coverage=self.pool_min_coverage.value(),
                wavelength=wl,
                ambient_only=self.ambient_only.isChecked(),
                shift=shift,
                shift_span=span,
                shift_model=model,
            ) or []
        if not pool:
            peak_data = {
                # The legacy engine has no shift of its own, so hand it peaks
                # already pulled back onto the reference scale
                "two_theta": remove_shift(exp_peaks["two_theta"], shift, model),
                "intensity": np.asarray(exp_peaks["intensity"]),
                "wavelength": wl,
            }
            pool = self.search_engine.search_by_peaks(
                peak_data, tolerance=self.peak_tol.value() + span,
                max_results=pool_size,
                ambient_only=self.ambient_only.isChecked(),
            ) or []

        ranked = rank_by_fingerprint(
            pool,
            exp_peaks,
            self.theoretical_peaks_for,
            tolerance=tol,
            n_peaks=self.fp_n_peaks.value(),
            min_rel_intensity=self.fp_min_rel.value(),
            min_score=self.fp_min_score.value(),
            min_found=self.fp_min_found.value(),
            require_top_peak=self.fp_require_top.isChecked(),
            max_results=self.max_results.value(),
            exp_range=self._measured_range(),
            dedupe_by_name=self.fp_dedupe.isChecked(),
            exp_weights=exp_weights,
            shift=shift,
            shift_span=span,
            shift_model=model,
        )
        mode = "residual-weighted" if exp_weights is not None else "presence"
        print(
            f"Fingerprint search ({mode}): pool={len(pool)} → {len(ranked)} candidates "
            f"(min score {self.fp_min_score.value():.2f})"
        )
        return ranked

    def _shift_summary(self, results: list) -> str:
        """
        What the fitted shifts say about the mount, for the status line.

        Every candidate gets its own shift, and a wrong phase will happily
        invent one, so an average across the list means nothing. What does mean
        something is how many of the top hits land on the *same* shift: that is
        the pattern telling you the sample really is displaced.
        """
        model = self.shift_model_key()
        if self.shift_span.value() <= 0:
            shift = self.shift.value()
            return f" Reference lines held at {describe_shift(shift, model)}." if shift else ""

        fitted = [
            float(r["fingerprint"]["shift"])
            for r in results[:10]
            if (r.get("fingerprint") or {}).get("shift") is not None
        ]
        if not fitted:
            return ""
        best = fitted[0]
        agree = sum(1 for s in fitted if abs(s - best) <= self.tolerance.value())
        note = (
            f" Auto fit: the top hit sits at {describe_shift(best, model)}, "
            f"and {agree} of the top {len(fitted)} agree."
        )
        if agree >= 3:
            note += " Fit to Row on a phase you trust, then set Auto fit to 0 to lock it in."
        else:
            note += " Little agreement — treat the fitted shifts as guesses for now."
        return note

    def _ultra_fast(self, pattern, min_correlation, max_results):
        if self.fast_engine.search_index is None:
            QMessageBox.warning(
                self,
                "Search Index Required",
                "No search index is loaded.\n\n"
                "Open Database and build/load the search index there, then try again.",
            )
            return []
        return self.fast_engine.ultra_fast_correlation_search(
            pattern,
            min_correlation=min_correlation,
            max_results=max_results,
            ambient_only=self.ambient_only.isChecked(),
        )

    def _legacy_search(self, pattern, method: str, peaks_override=None):
        peaks = peaks_override if peaks_override is not None else self.session.peaks
        peak_data = pattern
        if peaks is not None:
            peak_data = {
                "two_theta": np.asarray(peaks["two_theta"]),
                "intensity": np.asarray(peaks["intensity"]),
                "wavelength": pattern.get("wavelength", 1.5406),
            }
        min_c = self.min_corr.value()
        max_r = self.max_results.value()
        ptol = self.peak_tol.value()

        ambient_only = self.ambient_only.isChecked()

        if method == "peaks":
            return self.search_engine.search_by_peaks(
                peak_data, tolerance=ptol, max_results=max_r,
                ambient_only=ambient_only,
            )
        if method == "correlation":
            return self.search_engine.search_by_correlation(
                pattern, min_correlation=min_c, max_results=max_r,
                ambient_only=ambient_only,
            )
        if method == "combined":
            return self.search_engine.combined_search(
                peak_data if peaks is not None else pattern,
                peak_tolerance=ptol,
                min_correlation=min_c,
                max_results=max_r,
            )
        return self.search_engine.ensemble_search(
            peak_data if peaks is not None else pattern,
            methods=["peaks", "correlation", "ultrafast"],
            max_results=max_r,
            peak_tolerance=ptol,
            min_correlation=min_c,
            fast_search_engine=self.fast_engine,
        )

    def _result_to_phase(self, result: dict) -> dict:
        return {
            "id": result.get("mineral_id"),
            "amcsd_id": result.get("mineral_id"),
            "mineral": result.get("mineral_name", "Unknown"),
            "formula": result.get("chemical_formula", "Unknown"),
            "space_group": result.get("space_group", "Unknown"),
            "cell_a": result.get("cell_a"),
            "cell_b": result.get("cell_b"),
            "cell_c": result.get("cell_c"),
            "cell_alpha": result.get("cell_alpha"),
            "cell_beta": result.get("cell_beta"),
            "cell_gamma": result.get("cell_gamma"),
            "rir": result.get("rir"),
            "local_db": True,
            # Resolved here so matching, plotting, and quantification all place
            # this phase's lines the same way the search scored them
            "two_theta_shift": self.shift_for(result),
            "search_score": result.get(
                "fingerprint_score",
                result.get(
                    "ensemble_score",
                    result.get("combined_score", result.get("correlation", result.get("match_score", 0))),
                ),
            ),
        }

    # --- matching ---

    def start_matching(self):
        if not self.session.has_peaks():
            QMessageBox.warning(self, "No Peaks", "Find peaks in the Peaks tab first.")
            return

        phases = self.workspace.get_selected_candidates()
        if not phases:
            QMessageBox.warning(
                self, "No Selection",
                "Check one or more candidates in the table, then Match Selected.\n\n"
                "(Nothing is auto-selected.)",
            )
            return

        # Only phases carried over from earlier rounds are "already found";
        # the checked candidates are exactly what the user wants matched now.
        kept = list(self._kept_phases)
        if kept:
            before = len(phases)
            phases = filter_new_hits(phases, kept)
            if not phases:
                QMessageBox.information(
                    self, "Already Found",
                    "All checked candidates are already in your kept phases "
                    "(same mineral ID or name).",
                )
                return
            if len(phases) < before:
                self.status.setText(
                    f"Matching {len(phases)} new candidate(s) "
                    f"(skipped {before - len(phases)} already-found)."
                )

        pw = self.peak_weight.value()
        cw = self.corr_weight.value()
        total = pw + cw
        if total > 0:
            pw, cw = pw / total, cw / total

        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.match_btn.setEnabled(False)
        self.status.setText("Matching…")

        self._match_thread = PhaseMatchingThread(
            self.session.peaks, phases, self.tolerance.value(), pw, cw,
            shift=self.shift.value(), shift_model=self.shift_model_key(),
        )
        self._match_thread.matching_complete.connect(self._on_match_done)
        self._match_thread.progress_updated.connect(self.progress.setValue)
        self._match_thread.start()

    def _on_match_done(self, results):
        self.progress.setVisible(False)
        self.match_btn.setEnabled(True)
        min_score = self.min_score.value()
        filtered = [r for r in results if r.get("match_score", 0) >= min_score]
        for r in filtered:
            self._attach_fingerprint(r)
        filtered.sort(
            key=lambda x: x.get("fingerprint_score", x.get("combined_score", x.get("match_score", 0))),
            reverse=True,
        )

        # Keep previously locked selections; never re-add the same mineral twice
        previous = list(self._kept_phases)
        excl_ids, excl_names = exclusion_sets(previous)
        merged = list(previous)
        new_count = 0
        for r in filtered:
            if is_excluded_hit(r, excl_ids, excl_names):
                continue
            merged.append(r)
            excl_ids |= mineral_ids(r)
            name = mineral_key(r)
            if name:
                excl_names.add(name)
            new_count += 1

        self.session.set_matched_phases(merged)
        self.session.set_selected_phases(previous)
        self.workspace.set_results_matches(merged, preselect=previous)
        self.update_action_states()
        self.status.setText(
            f"Matched {new_count} new phase(s); {len(previous)} previously kept. "
            "Check phases to keep, then Clear Unselected or Search Residual."
        )
        self.workspace.set_status(f"Matched {new_count} new phases")
        self.workspace.refresh_plot()

    def _attach_fingerprint(self, result: dict):
        """Add fingerprint stats so mixtures rank on their own lines."""
        if "fingerprint" in result or not self.session.has_peaks():
            return
        theo = self.theoretical_peaks_for(result)
        if not theo:
            return
        info = fingerprint_score(
            self.session.peaks["two_theta"],
            self.session.peaks["intensity"],
            theo.get("two_theta", []),
            theo.get("intensity", []),
            tolerance=self.tolerance.value(),
            n_peaks=self.fp_n_peaks.value(),
            min_rel_intensity=self.fp_min_rel.value(),
            exp_range=self._measured_range(),
            shift=self.shift_for(result),
            shift_span=self.shift_span.value(),
            shift_model=self.shift_model_key(),
        )
        result["fingerprint"] = info
        result["fingerprint_score"] = info["score"]

    def run_rir_quant(self):
        """
        Weight percents for the checked phases from reference intensity ratios.

        Fast enough to run on the UI thread: one non-negative least squares fit
        of fixed reference patterns, no refinement. Le Bail lives in the Quant
        window, where the cell and correction terms can move.
        """
        pattern = self.session.active_pattern()
        phases = self.accepted_phases() or list(self.session.selected_phases)
        if not pattern or not phases:
            QMessageBox.warning(
                self, "No Phases Selected",
                "Check the phases you want quantified, then RIR Quant.",
            )
            return

        try:
            result = rir_quantify(
                pattern,
                phases,
                fwhm=self.rir_fwhm.value(),
                theoretical_for=self.reference_peaks_for,
            )
        except Exception as exc:
            QMessageBox.critical(self, "RIR Error", str(exc))
            self.status.setText("RIR quantification failed.")
            return

        if result is None:
            QMessageBox.warning(
                self, "No Reference Patterns",
                "None of the checked phases has reference lines inside the "
                "measured 2θ range, so there is nothing to fit.",
            )
            return

        # The fitted profile is the phase's share of the observed intensity, which
        # is what the plot needs to draw its reference lines at the right height
        for fitted in result["phases"]:
            entry = fitted.get("entry")
            if isinstance(entry, dict):
                entry["contribution"] = fitted["profile"]

        self.session.set_rir_results(result)
        for line in rir_summary_lines(result):
            print(line)

        quantified = [p for p in result["phases"] if p.get("weight_percent") is not None]
        if quantified:
            headline = ", ".join(
                f"{p['name']} {p['weight_percent']:.1f}%" for p in quantified[:4]
            )
            extra = "" if len(quantified) <= 4 else f" +{len(quantified) - 4} more"
            note = ""
            if result["missing_rir"]:
                note = (
                    f" No RIR for {', '.join(result['missing_rir'][:3])} — "
                    "excluded from the normalization."
                )
            self.status.setText(
                f"RIR wt%: {headline}{extra} (fit Rwp {result['rwp']:.1f}%, "
                f"{result['explained_fraction'] * 100:.0f}% of intensity explained).{note}"
            )
            self.workspace.set_status(f"RIR quantification: {headline}{extra}")
        else:
            self.status.setText(
                "No phase could be quantified — none of the checked phases has a "
                "RIR value in the database, or all fitted to zero."
            )
            self.workspace.set_status("RIR quantification: nothing to report")
        self.update_action_states()
        self.workspace.refresh_plot()

    def add_phases_from_database(self, phases: list):
        """Called when Database Manager exports phases."""
        self.session.add_candidates(phases)
        existing = list(getattr(self.workspace, "_candidate_results", []) or [])
        existing_names = {(r.get("mineral_name") or "").lower() for r in existing}
        new_keys = []
        for p in phases:
            name = p.get("mineral", p.get("mineral_name", "Unknown"))
            if name.lower() in existing_names:
                continue
            existing.insert(0, {
                "mineral_id": p.get("id") or p.get("amcsd_id"),
                "mineral_name": name,
                "chemical_formula": p.get("formula", p.get("chemical_formula", "")),
                "space_group": p.get("space_group", ""),
                "match_score": 1.0,
                "manual_add": True,
            })
            existing_names.add(name.lower())
            new_keys.append(name.lower())
        self._search_results = existing
        self.workspace.set_results_candidates(existing)
        self.workspace.check_candidate_rows(new_keys)
        self.update_action_states()
        self.status.setText(f"Added {len(phases)} phase(s) from database.")
