"""Identify stage — pattern search + phase matching."""

from __future__ import annotations

import textwrap

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from utils import emphasis, search_debug
from utils.emphasis import DEFAULT_WEIGHT as EMPHASIS_DEFAULT_WEIGHT
from utils.fast_pattern_search import FastPatternSearchEngine
from utils.pattern_search import PatternSearchEngine
from utils.local_database import get_local_database
from utils.rir_quant import quantify as rir_quantify, summary_lines as rir_summary_lines
from utils.fingerprint_search import (
    coincidence_fraction,
    fingerprint_score,
    rank_by_fingerprint,
    select_fingerprint_peaks,
)
from utils.conditions import (AMBIENT_MAX_PRESSURE_GPA, AMBIENT_MAX_TEMPERATURE_K,
                             AMBIENT_MIN_TEMPERATURE_K, is_ambient)
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
from gui.dialogs.mineral_picker_dialog import MineralPickerDialog
from gui.dialogs.search_report_dialog import SearchReportDialog
from gui.matching_tab import PhaseMatchingThread
from gui.widgets.control_bar import OptionsDialog, compact


REPORT_WIDTH = 74


def _prose(text: str, indent: str = "  ") -> list:
    """Wrap explanatory text by hand — the report box does not wrap lines."""
    return textwrap.wrap(text, width=REPORT_WIDTH,
                         initial_indent=indent, subsequent_indent=indent)


LABEL_WIDTH = 25


def _defined(term: str, text: str) -> list:
    """One glossary entry, its continuation lines hanging under the text."""
    head = f"  {term:<{LABEL_WIDTH}s}"
    return textwrap.wrap(text, width=REPORT_WIDTH, initial_indent=head,
                         subsequent_indent=" " * len(head))


# Fit to Row scans at least this far either side of zero, so a mount that is
# badly displaced can still be measured from a phase the user recognizes
FIT_TO_ROW_MIN_SPAN = 1.0

# How many records the mineral picker loads. Common minerals run to several
# hundred entries — 594 spinels — and every one of them gets scored against the
# measured peaks when the picker opens, so the list is capped rather than
# unbounded. Anything past this is reachable by typing more of the name.
MINERAL_PICKER_LIMIT = 300


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
        self.local_db = get_local_database()
        self._match_thread = None
        self._search_results = []
        self._kept_phases = []  # accepted across residual rounds
        self._options = None
        self._report_dialog = None
        self._picker = None

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
        self.match_btn.setToolTip(
            "Score the checked candidates in full and turn them into your "
            "phase list.\n\n"
            "Search ranks the whole database quickly on each candidate's "
            "strongest lines. This compares the complete reference pattern of "
            "the few you believe against the measured peaks, then replaces the "
            "list with those matches — which is what Search Residual, RIR "
            "Quant, and the Quant window work from."
        )
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
        self.add_mineral_btn.setToolTip(
            "Search the local database and add a mineral as a candidate.\n\n"
            "Most minerals have many records — different compositions, and "
            "different pressures and temperatures — whose lines sit in "
            "different places. When there is more than one, a chooser opens "
            "with them ranked by how well they fit your peaks, and draws the "
            "highlighted record on the pattern."
        )
        self.add_mineral_btn.clicked.connect(self.add_mineral_by_name)
        self.explain_btn = QPushButton("Why not?")
        self.explain_btn.setToolTip(
            "Explain why a mineral you expected is missing from the results.\n\n"
            "Type its name in the box on the left, then press this. The search "
            "runs again while following that one mineral, and the report says "
            "which setting dropped it, how well it actually fits the pattern, "
            "and which changes would bring it back."
        )
        self.explain_btn.clicked.connect(self.explain_missing_phase)
        grid.addWidget(self._label("Add mineral:"), 1, 0)
        grid.addWidget(self.mineral_search, 1, 1, 1, 3)
        grid.addWidget(self.add_mineral_btn, 1, 4)
        grid.addWidget(self.explain_btn, 1, 5)

        # Rows 2-3 — search parameters, two label/field pairs per row
        self.method_combo = QComboBox()
        for label, key in SEARCH_METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.setToolTip(
            "Fingerprint scores each candidate on its own strong lines, so minor "
            "phases in a mixture are not penalized for unexplained peaks.\n\n"
            "Fingerprint and Ultra-Fast Correlation return in seconds. Peak "
            "Match, Pearson Correlation, Combined, and Ensemble compare every "
            "stored pattern one at a time and take about a minute, during which "
            "the window will not respond."
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
            "Minimum fingerprint score to list a candidate. The score is driven by "
            "how unlikely the match would be by chance (binomial evidence), so a "
            "perfect line count in a dense peak list no longer saturates at 1.0."
        )

        self.max_results = QSpinBox()
        self.max_results.setRange(10, 500)
        self.max_results.setValue(100)
        self.max_results.setToolTip(
            "How many candidates to list. Minor phases in a mixture often land "
            "between ranks 50 and 100 once the dominant phase's coincidences "
            "fill the top of the list."
        )

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
        self._build_emphasis_widgets()

        grid.addWidget(self._label("Method:"), 2, 0)
        grid.addWidget(compact(self.method_combo, 170), 2, 1)
        grid.addWidget(self._label("2θ tol:"), 2, 2)
        grid.addWidget(compact(self.tolerance, 80), 2, 3)
        self.fp_min_score_label = self._label("Min fingerprint:")
        grid.addWidget(self.fp_min_score_label, 2, 4)
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

        # Row 5 — emphasised regions
        grid.addWidget(self._label("Emphasis:"), 5, 0)
        grid.addWidget(self.emphasis_btn, 5, 1)
        grid.addWidget(self._label("Weight:"), 5, 2)
        grid.addWidget(compact(self.emphasis_weight, 80), 5, 3)
        grid.addWidget(self.clear_emphasis_btn, 5, 4)
        grid.addWidget(self.emphasis_label, 5, 5, 1, 2)

        # Row 6 — list actions, filled in by the workspace
        self.table_actions = QHBoxLayout()
        self.table_actions.setSpacing(6)
        grid.addLayout(self.table_actions, 6, 0, 1, 7)

        # Row 7 — status and progress
        self.status = QLabel("Load a pattern, find peaks, then search.")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        grid.addWidget(self.status, 7, 0, 1, 5)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(140)
        grid.addWidget(self.progress, 7, 5, 1, 2)

        grid.setColumnStretch(6, 1)
        grid.setRowStretch(8, 1)

        self._build_option_widgets()
        self._sync_method_controls()
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
            "With Auto fit above zero this is only the centre of each "
            "candidate's own fitted shift; set Auto fit to 0 and every phase "
            "follows this box.\n\n"
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
            "Fit the shift to the highlighted phase's lines, put it in the box, "
            "and move every listed phase onto it — the displacement belongs to "
            "the mount, not to one mineral.\n\n"
            "Use it once you recognize a phase, then search again so the "
            "ranking is done with the displacement pinned down."
        )
        self.fit_shift_btn.clicked.connect(self.fit_shift_to_row)

        self.clear_shift_btn = QPushButton("No Shift")
        self.clear_shift_btn.setToolTip("Reset the shift and the auto-fit range to zero")
        self.clear_shift_btn.clicked.connect(self.clear_shift)

        for spin in (self.shift, self.shift_span):
            spin.valueChanged.connect(self._on_shift_changed)

    def _build_emphasis_widgets(self):
        """
        Priority 2θ ranges, drawn straight on the pattern.

        The ranking is a whole-pattern statistic, so a phase whose evidence
        lives in one narrow window competes against phases scattered over the
        full range and loses. Emphasis lets the user say which window carries
        the question, and the peaks in it are weighted up for both the coverage
        screen and the final score.
        """
        self.emphasis_btn = QPushButton("Highlight on Plot")
        self.emphasis_btn.setCheckable(True)
        self.emphasis_btn.setToolTip(
            "Drag across the plot to mark a 2θ range that search should treat as "
            "the priority; right-click a shaded band to drop it.\n\n"
            "Peaks inside a region are weighted up, so candidates that explain "
            "them outrank candidates that explain the same number of lines "
            "elsewhere. Emphasised peaks also survive the search peak cull.\n\n"
            "Turn off to pan and zoom normally."
        )
        self.emphasis_btn.toggled.connect(self.workspace.set_emphasis_mode)

        self.emphasis_weight = QDoubleSpinBox()
        self.emphasis_weight.setRange(1.0, 50.0)
        self.emphasis_weight.setDecimals(1)
        self.emphasis_weight.setSingleStep(1.0)
        self.emphasis_weight.setValue(EMPHASIS_DEFAULT_WEIGHT)
        self.emphasis_weight.setPrefix("×")
        self.emphasis_weight.setToolTip(
            "How much more a peak inside a region counts than one outside it. "
            "Applies to regions drawn from now on.\n\n"
            "×1 is no emphasis. Around ×5 makes the region decisive without "
            "discarding the rest of the pattern; very large values effectively "
            "search the region alone."
        )

        self.clear_emphasis_btn = QPushButton("Clear")
        self.clear_emphasis_btn.setToolTip("Remove every emphasised region")
        self.clear_emphasis_btn.clicked.connect(self.clear_emphasis)

        self.emphasis_label = QLabel("none")
        self.emphasis_label.setObjectName("mutedLabel")
        self.session.emphasis_changed.connect(self._on_emphasis_changed)
        self.session.peaks_changed.connect(self._on_emphasis_changed)
        self.method_combo.currentIndexChanged.connect(self._on_emphasis_changed)

    def clear_emphasis(self):
        self.session.clear_emphasis_regions()

    def _on_emphasis_changed(self, *_args):
        regions = self.session.emphasis_regions
        text = emphasis.describe(regions)
        if regions:
            peaks = (self.session.peaks or {}).get("two_theta", [])
            n = int(np.sum(emphasis.inside(peaks, regions)))
            text += f", {n} peak{'' if n == 1 else 's'}"
            if n < 3:
                # Two or three lines coincide with hundreds of phases, so a
                # region this thin ranks by luck rather than by evidence
                text += " — too few to identify a phase, widen it"
            if self._method_key() != "fingerprint":
                text += " — only the Fingerprint method uses emphasis"
        self.emphasis_label.setText(text)
        self.emphasis_label.setToolTip(text)

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

        While auto fit is on each phase keeps the shift it was scored with: a
        value already resolved onto the phase dict, or the one fitted as it
        ranked. With auto fit off there is a single displacement for the whole
        mount, so the manual setting wins — otherwise a phase frozen during an
        earlier auto-fit search would ignore the box for the rest of the session.
        """
        if isinstance(result, dict) and self.shift_span.value() > 0:
            phase = result.get("phase")
            for src in (result, phase if isinstance(phase, dict) else {}):
                resolved = src.get("two_theta_shift")
                if resolved is not None:
                    return float(resolved)
            fitted = (result.get("fingerprint") or {}).get("shift")
            if fitted is not None:
                return float(fitted)
        return float(self.shift.value())

    def _pin_shift_on_results(self, value: float) -> int:
        """
        Overwrite the shift frozen onto every phase currently in play.

        Search and matching each store the shift they scored a phase with, so
        a shift fitted afterwards has to replace those copies or the lines stay
        where the search happened to put them.
        """
        groups = [
            getattr(self.workspace, "_candidate_results", None) or [],
            self.session.matched_phases or [],
            self.session.selected_phases or [],
            self._kept_phases,
        ]
        seen, count = set(), 0
        for group in groups:
            for result in group:
                if not isinstance(result, dict):
                    continue
                phase = result.get("phase")
                for target in (result, phase if isinstance(phase, dict) else None):
                    if target is None or id(target) in seen:
                        continue
                    seen.add(id(target))
                    target["two_theta_shift"] = float(value)
                count += 1
        return count

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
        # A generous window whatever auto fit is set to, since this is the step
        # that tells the user how far off the mount actually is — a narrow auto
        # fit range must not stop a large displacement from being found
        span = max(self.shift_span.value(), FIT_TO_ROW_MIN_SPAN)
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
                "Widen the auto fit range, loosen the 2θ tolerance, or pick a "
                "phase you are surer of.",
            )
            return

        # The displacement belongs to the mount, not to one phase, so the fit
        # replaces the shift on every listed phase as well as the manual box
        n_phases = self._pin_shift_on_results(fitted)
        self.shift.blockSignals(True)
        self.shift.setValue(fitted)
        self.shift.blockSignals(False)
        self.workspace.refresh_plot()

        applied = f" Applied to {n_phases} listed phase(s)." if n_phases else ""
        self.status.setText(
            f"Fitted {describe_shift(fitted, self.shift_model_key())} from {name} "
            f"({n_found} lines).{applied} Search again to rank every candidate "
            "at this shift."
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
        self.search_max_peaks.setValue(0)
        self.search_max_peaks.setToolTip(
            "Match on this many of the strongest peaks (0 = all). The coincidence "
            "baseline is intensity-weighted, so weak peaks stay available for "
            "minor phases without making every reference look present by luck. "
            "Set a positive limit only if noise-level peaks are flooding the list."
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
        self._sync_method_controls(explain=True)

    def _sync_method_controls(self, explain: bool = False):
        """
        Grey out Min fingerprint for the methods that do not read it.

        A greyed threshold with no stated reason reads as a broken control, so
        a change of method says in the status line which threshold the new
        method uses instead. Only fingerprint scoring consults Min fingerprint;
        Peak Match ranks on tolerance alone and the rest use Min corr.
        """
        is_fp = self._method_key() == "fingerprint"
        self.fp_min_score.setEnabled(is_fp)
        self.fp_min_score_label.setEnabled(is_fp)
        if not explain:
            return
        if is_fp:
            self.status.setText(
                "Fingerprint method: 'Min fingerprint' is the score a candidate "
                "needs to be listed."
            )
        else:
            instead = ("'Peak tolerance' in Options" if self._method_key() == "peaks"
                       else "'Min corr'")
            self.status.setText(
                f"{self.method_combo.currentText()} does not use 'Min fingerprint', "
                f"so it is greyed out — this method is controlled by {instead}."
            )

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
        newest candidates. Minerals checked on the Shortlist tab are added on
        top, since that list is deliberately independent of the current search.
        """
        if getattr(self.workspace, "_results_mode", None) == "matches":
            return self._with_shortlist(self.workspace.get_selected_matches())

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
        return self._with_shortlist(accepted)

    def _with_shortlist(self, accepted: list) -> list:
        """Append the checked shortlist minerals that are not already here."""
        panel = getattr(self.workspace, "shortlist_panel", None)
        if panel is None:
            return accepted

        merged = list(accepted)
        seen_ids, seen_names = exclusion_sets(merged)
        for entry in panel.checked_entries():
            if is_excluded_hit(entry, seen_ids, seen_names):
                continue
            merged.append(entry)
            seen_ids |= mineral_ids(entry)
            name = mineral_key(entry)
            if name:
                seen_names.add(name)
        return merged

    def update_action_states(self):
        """Keep buttons in step with what is checked in the list."""
        can = self.session.has_pattern()
        has_peaks = self.session.has_peaks()
        self.search_btn.setEnabled(can)

        checked_candidates = self.workspace.get_selected_candidates()
        accepted = self.accepted_phases()

        self.match_btn.setEnabled(bool(checked_candidates) and has_peaks)
        if not checked_candidates:
            self.match_btn.setToolTip("Check one or more candidates in the list first")
        else:
            self.match_btn.setToolTip(
                f"Score the {len(checked_candidates)} checked candidate(s) in full "
                "and turn them into your phase list.\n\n"
                "Search ranks the whole database quickly on each candidate's "
                "strongest lines. This compares their complete reference patterns "
                "against the measured peaks, then replaces the list with those "
                "matches."
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
            f"RIR weight percents for the {len(accepted)} checked phase(s), "
            "counting anything checked on the Shortlist tab"
            if accepted else
            "Check the phases you want quantified, here or on the Shortlist tab"
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
        # get_diffraction_pattern memoizes, and does it process-wide, so a
        # second cache here would only add a way to go stale after an import
        try:
            return self.local_db.get_diffraction_pattern(
                int(mineral_id), round(float(self.session.wavelength), 4)
            )
        except Exception:
            return None

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
            # Non-ambient records are fetched too: the picker filters them out
            # by default but lets the user bring them back without a re-query
            hits = self.local_db.search_by_mineral_name(query, limit=MINERAL_PICKER_LIMIT)
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
        # One record needs no choosing. Everything else does: several records of
        # the same mineral differ in composition and in the conditions they were
        # measured at, and those cells put the lines in different places.
        exact = [h for h in hits if str(h.get("mineral_name", "")).lower() == needle]
        if len(hits) == 1 or len(exact) == 1:
            self._add_mineral_record(exact[0] if exact else hits[0])
            return
        self._open_mineral_picker(hits, query)

    def _open_mineral_picker(self, hits: list, query: str):
        """Let the user compare the records against the pattern before adding one."""
        if getattr(self, "_picker", None) is not None:
            self._picker.close()

        restore = self.workspace.current_preview()
        dialog = MineralPickerDialog(
            hits, query,
            score_fn=self._score_hit if self.session.has_peaks() else None,
            preview_fn=self.workspace.preview_phase,
            tolerance=self.tolerance.value(),
            ambient_only=self.ambient_only.isChecked(),
            truncated=len(hits) >= MINERAL_PICKER_LIMIT,
            parent=self.workspace.window(),
        )

        def finished(_code):
            self._picker = None
            chosen = dialog.chosen()
            if chosen is None:
                self.workspace.restore_preview(restore)
            else:
                self._add_mineral_record(chosen)

        dialog.finished.connect(finished)
        self._picker = dialog
        dialog.show()
        dialog.raise_()
        ambient = sum(1 for h in hits if self._hit_is_ambient(h))
        self.status.setText(
            f"{len(hits)} records for “{query}” ({ambient} at ambient conditions) — "
            "click one to see its lines on the pattern."
        )

    def _score_hit(self, hit: dict):
        """How well one database record's strong lines fit the measured peaks."""
        if not self.session.has_peaks():
            return None
        theo = self.theoretical_peaks_for(hit)
        if not theo:
            return None
        return fingerprint_score(
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

    @staticmethod
    def _hit_is_ambient(hit: dict) -> bool:
        return is_ambient(hit.get("pressure_gpa"), hit.get("temperature_k"))

    def _add_mineral_record(self, chosen: dict):
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
        # Report how well the known mineral actually fits the peaks. The picker
        # has usually scored it already, in which case that result stands.
        info = chosen.get("fingerprint") or self._score_hit(chosen)
        if info:
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
        # Two records of the same mineral are a pair worth comparing — different
        # composition, different cell, different line positions — so only the
        # same record counts as already present
        record = result.get("mineral_id")
        name = (result.get("mineral_name") or "").lower()

        def duplicate(other: dict) -> bool:
            if record is not None and other.get("mineral_id") is not None:
                return other.get("mineral_id") == record
            return (other.get("mineral_name") or "").lower() == name

        existing_match = next((r for r in existing if duplicate(r)), None)
        if existing_match is None:
            existing.insert(0, result)
        self._search_results = existing
        self.workspace.set_results_candidates(existing)
        self.workspace.check_candidate_result(existing_match or result)

    # --- diagnosis ---

    def explain_missing_phase(self):
        """
        Follow one mineral through the search and report what rejected it.

        A search silently discards candidates at a dozen thresholds, so "my
        phase is not in the list" is otherwise unanswerable. This re-runs the
        search with tracing on, names the gate that dropped the phase and the
        setting that controls it, and scores the phase under a few alternative
        settings so the fix is visible rather than guessed at.
        """
        query = self.mineral_search.text().strip()
        if len(query) < 2:
            QMessageBox.information(
                self, "Which mineral?",
                "Type the name of the mineral you expected to see in the "
                "'Add mineral' box, then press 'Why not?' again.\n\n"
                "The search will run once more while following that mineral, "
                "and the report will say what dropped it.",
            )
            return
        if not self.session.has_peaks():
            QMessageBox.warning(
                self, "Peaks Required",
                "This report compares your peak positions against the "
                "database, so it needs a peak list. Find peaks in the Peaks "
                "tab first, then try again.",
            )
            return

        with search_debug.tracing([query]) as trace:
            results = self._fingerprint_search(self.session.active_pattern())

        report = self._build_diagnosis(query, trace, results)
        print("\n" + report["body"])
        if self._report_dialog is None:
            self._report_dialog = SearchReportDialog(self.workspace.window())
        self._report_dialog.show_report(report["title"], report["verdict"], report["body"])

    def _records_named(self, query: str) -> list:
        """
        Index records for this mineral.

        Exact name matches win outright when they exist: asked about enstatite,
        a report about clinoenstatite answers a different question.
        """
        needle = query.strip().lower()
        exact, partial = [], []
        for meta in self.fast_engine.mineral_metadata or []:
            name = str(meta.get("name", "")).strip().lower()
            if name == needle:
                exact.append(meta)
            elif needle in name:
                partial.append(meta)
        return exact or partial

    def _build_diagnosis(self, query: str, trace, results: list) -> dict:
        """
        Assemble the report, plainest and most actionable part first.

        The order matters more than the content: the answer to "why is my phase
        missing" comes first, then the settings that would change it, and only
        then the numbers and the raw trace that back both up. Reading the trace
        to reach the verdict, as an unordered dump would demand, is the whole
        difficulty this report is meant to remove.
        """
        peaks = self.session.peaks
        rng = self._measured_range()
        tol = self.tolerance.value()
        listed = [
            r for r in results
            if query.lower() in str(r.get("mineral_name", "")).lower()
        ]

        records = self._records_named(query)
        if not records:
            return {
                "title": f"“{query}” is not a name in the database",
                "verdict": f"No database record has a name containing “{query}”, "
                           "so the search never had a record of it to accept or "
                           "reject, and no setting can bring it back.",
                "body": "\n".join(
                    ["WHAT HAPPENED", "-" * 60]
                    + _prose(
                        f"“{query}” matches no mineral name in the search index. "
                        "Check the spelling first. Failing that, the database may "
                        "index the phase under a different name — a structural or "
                        "polymorph name rather than the common one, or a solid "
                        "solution end member. Try a shorter fragment of the name "
                        "to see what is there."
                    )
                    + ["", trace.report(query)]
                ),
            }

        # Score every matching record on the peaks the search actually used and
        # on the full list, because the gap between them is itself a diagnosis
        culled = self._significant_peaks(peaks)
        scored = self._score_records(records, culled, rng)
        scored_full = self._score_records(records, peaks, rng)
        best = scored[0] if scored else None
        best_full = scored_full[0] if scored_full else None

        verdict = self._verdict_text(query, trace, listed, best, best_full)

        lines = ["1. WHAT HAPPENED", "-" * 60]
        lines += _prose(verdict)
        lines.append("")
        lines += self._sensitivity(records, peaks, culled, rng)
        lines.append("")
        lines += self._fit_summary(query, records, best, best_full, culled, peaks)
        if best_full:
            lines.append("")
            lines += self._line_by_line(best_full, culled, peaks, tol)
        lines.append("")
        lines.append(trace.report(query, heading="5. SEARCH INTERNALS"))

        if listed:
            title = f"“{query}” is in the results"
        else:
            title = f"Why “{query}” is not in the results"
        return {
            "title": f"{title} — {len(records)} database record(s) named it",
            "verdict": verdict,
            "body": "\n".join(lines),
        }

    def _fit_summary(self, query, records, best, best_full, culled, full) -> list:
        """
        The winning record's numbers, side by side on both peak lists.

        Two columns rather than one because the difference between them is a
        diagnosis on its own: a phase that fits the full peak list but not the
        searched one was never absent from the pattern, it just needed peaks
        that 'Search peak count' discarded before scoring.
        """
        head = ["3. HOW WELL IT ACTUALLY FITS", "-" * 60]
        columns = [
            (label, entry["info"], entry["meta"], len(peaks.get("two_theta", [])))
            for label, entry, peaks in (("search list", best, culled),
                                        ("all peaks", best_full, full))
            if entry
        ]
        if not columns:
            return head + _prose(
                f"None of the {len(records)} record(s) named “{query}” has a "
                "calculated pattern, so there is nothing to score against your "
                "peaks."
            )

        n_culled = len(culled.get("two_theta", []))
        n_full = len(full.get("two_theta", []))
        if len(columns) == 2 and n_culled == n_full:
            # Nothing was culled, so the two columns would be identical
            columns = [("your peaks", *columns[0][1:])]
            out = head + _prose(
                f"The best fitting of {len(records)} record(s) named “{query}”, "
                f"scored against all {n_full} of your peaks."
            )
        else:
            out = head + _prose(
                f"The best fitting of {len(records)} record(s) named “{query}”, "
                "scored twice: against the peaks the search actually used, and "
                "against your full peak list. A large gap between the two "
                "columns means 'Search peak count' threw away peaks this phase "
                "needed."
            )
        out.append("")

        def row(label, *cells):
            body = "".join(f"{c:<24s}" for c in cells)
            out.append(f"  {label:<{LABEL_WIDTH}s}{body}".rstrip())

        row("", *[f"{label} ({n} peaks)" for label, _, _, n in columns])
        row("database record", *[f"id {meta.get('id')}" for _, _, meta, _ in columns])
        row("fingerprint score", *[f"{i['score']:.3f}" for _, i, _, _ in columns])
        row("needed to be listed", f"{self.fp_min_score.value():.2f}")
        row("reference lines found",
            *[f"{i['n_found']} of {i['n_expected']}" for _, i, _, _ in columns])
        row("strongest line",
            *["found" if i["top_found"] else "MISSING" for _, i, _, _ in columns])
        row("line intensity explained",
            *[f"{i['presence'] * 100:.0f}%" for _, i, _, _ in columns])
        row("odds of a lucky hit",
            *[f"{i['chance_match'] * 100:.0f}% per line" for _, i, _, _ in columns])
        row("evidence",
            *[f"1 in {10 ** i.get('evidence', 0.0):,.0f}" for _, i, _, _ in columns])
        row("2θ agreement",
            *[f"{i['position_quality'] * 100:.0f}%" for _, i, _, _ in columns])
        row("intensity pattern",
            *[f"{i['intensity_consistency'] * 100:.0f}%" for _, i, _, _ in columns])

        out.append("")
        out += _defined("fingerprint score",
                        "the overall 0–1 match, combining everything below. "
                        "Compared against 'Min fingerprint' to decide listing.")
        out += _defined("reference lines found",
                        "how many of this phase's fingerprint lines landed on "
                        "one of your peaks, within the 2θ tolerance.")
        out += _defined("line intensity explained",
                        "the same thing weighted by line strength: 100% means "
                        "every line is accounted for, and missing one strong "
                        "line costs far more than missing a weak one.")
        out += _defined("odds of a lucky hit",
                        "how much of the pattern the match windows cover. At "
                        "50% a reference line has a coin-flip chance of hitting "
                        "a peak by accident, so matches prove little.")
        out += _defined("evidence",
                        "the chance of getting this many matches from a phase "
                        "that is not there. 1 in 10 is nothing; 1 in 10,000 is "
                        "hard to explain away.")
        out += _defined("2θ agreement",
                        "how centred the matched lines are in the tolerance "
                        "window. Low values with many matches mean a shifted "
                        "pattern — try 'Auto fit ±'.")
        out += _defined("intensity pattern",
                        "whether your peak heights rise and fall like the "
                        "reference. Low is normal for preferred orientation or "
                        "a heavily overlapped mixture.")
        return out

    def _score_records(self, records: list, peaks: dict, rng) -> list:
        """Fingerprint every record against one peak list, best score first."""
        chance = coincidence_fraction(peaks.get("two_theta", []),
                                      self.tolerance.value(), rng)
        scored = []
        for meta in records:
            theo = self.theoretical_peaks_for({"mineral_id": meta.get("id")})
            if not theo:
                continue
            info = fingerprint_score(
                peaks.get("two_theta", []),
                peaks.get("intensity", []),
                theo.get("two_theta", []),
                theo.get("intensity", []),
                tolerance=self.tolerance.value(),
                n_peaks=self.fp_n_peaks.value(),
                min_rel_intensity=self.fp_min_rel.value(),
                exp_range=rng,
                chance=chance,
                shift=self.shift.value(),
                shift_span=self.shift_span.value(),
                shift_model=self.shift_model_key(),
            )
            scored.append({"info": info, "meta": meta, "theo": theo})
        scored.sort(key=lambda e: -e["info"]["score"])
        return scored

    def _line_by_line(self, entry: dict, culled: dict, full: dict, tol: float) -> list:
        """
        Each fingerprint line against both peak lists.

        A line that misses on the search list but hits on the full one was not
        absent from the pattern — the peak it needed was culled before scoring.
        """
        theo = entry["theo"]
        fp = select_fingerprint_peaks(
            theo.get("two_theta", []),
            theo.get("intensity", []),
            n_peaks=self.fp_n_peaks.value(),
            min_rel_intensity=self.fp_min_rel.value(),
            two_theta_range=self._measured_range(),
        )
        head = ["4. LINE BY LINE", "-" * 60]
        ct = np.asarray(culled.get("two_theta", []), dtype=float)
        ft = np.asarray(full.get("two_theta", []), dtype=float)
        fi = np.asarray(full.get("intensity", []), dtype=float)
        if len(fp["two_theta"]) == 0 or len(ft) == 0:
            return head + _prose(
                "None of this phase's reference lines fall inside the 2θ range "
                "you measured, so there was nothing to match."
            )
        imax = float(np.max(fi)) if np.max(fi) > 0 else 1.0

        out = head + _prose(
            f"Every fingerprint line of record id {entry['meta'].get('id')} and "
            f"the nearest peak to it. A line counts as found when that peak is "
            f"within ±{tol:.2f}° ('2θ tol'). Lines marked MISS on the search "
            "list but found in your full peak list were not missing from the "
            "pattern — the peak they needed was culled before scoring. "
            "Offsets are signed: a column of them with the same sign is a "
            "displaced sample, which 'Auto fit ±' can absorb."
        )
        out += ["",
                "  reference line     on search list   nearest of all peaks"
                "          its height",
                "  " + "-" * 76]
        for t, rel in zip(fp["two_theta"], fp["intensity"]):
            # Signed, so the offset says which side of the reference line the
            # peak sits on — a column of same-sign offsets is a shifted pattern
            dc = (ct - t) if len(ct) else np.array([np.inf])
            jc = int(np.argmin(np.abs(dc)))
            df = ft - t
            jf = int(np.argmin(np.abs(df)))
            cull_hit = "found" if abs(dc[jc]) <= tol else "MISS "
            full_hit = "found" if abs(df[jf]) <= tol else "MISS "
            out.append(
                f"  {t:7.3f}° I={rel:5.1f}   {cull_hit} {dc[jc]:+.3f}°    "
                f"{full_hit} at {ft[jf]:7.3f}° ({df[jf]:+.3f}°)   "
                f"{fi[jf] / imax * 100:5.1f}% of max"
            )
        return out

    def _sensitivity(self, records: list, full: dict, culled: dict, rng) -> list:
        """
        Score the phase under the settings most likely to be holding it back.

        Naming the gate that rejected a phase still leaves the user guessing at
        how far off it was. Rescoring under a handful of loosened settings turns
        that guess into a decision: either one of these rows lists the phase, or
        none does and the phase genuinely is not a good match for the pattern.
        """
        gate = self.fp_min_score.value()

        def best_score(peaks, **overrides):
            chance = coincidence_fraction(peaks.get("two_theta", []),
                                          self.tolerance.value(), rng)
            # Weighted runs are ranked on the weighted score, so compare that one
            key = "residual_score" if overrides.get("exp_weights") is not None else "score"
            best = 0.0
            found = 0
            for meta in records:
                theo = self.theoretical_peaks_for({"mineral_id": meta.get("id")})
                if not theo:
                    continue
                kwargs = dict(
                    tolerance=self.tolerance.value(),
                    n_peaks=self.fp_n_peaks.value(),
                    min_rel_intensity=self.fp_min_rel.value(),
                    exp_range=rng, chance=chance,
                    shift=self.shift.value(),
                    shift_span=self.shift_span.value(),
                    shift_model=self.shift_model_key(),
                )
                kwargs.update(overrides)
                info = fingerprint_score(
                    peaks.get("two_theta", []), peaks.get("intensity", []),
                    theo.get("two_theta", []), theo.get("intensity", []),
                    **kwargs,
                )
                if info[key] > best:
                    best, found = info[key], info["n_found"]
            return best, found

        n_culled = len(culled.get("two_theta", []))
        n_full = len(full.get("two_theta", []))
        variants = [
            (f"leave everything as it is ({n_culled} peaks)", None, culled, {}),
            (f"score on all {n_full} of your peaks",
             "set 'Search peak count' to 0 in Options",
             full, {}),
            (f"widen the 2θ window to 0.30° (now {self.tolerance.value():.2f}°)",
             "set '2θ tol' to 0.30°",
             culled, {"tolerance": 0.30}),
            ("let the reference lines shift by up to 0.30°",
             "set 'Auto fit ±' to 0.30°",
             culled, {"shift_span": 0.30}),
            ("all your peaks and a 0.30° shift together",
             "set 'Search peak count' to 0 and 'Auto fit ±' to 0.30°",
             full, {"shift_span": 0.30}),
        ]
        regions = self.session.emphasis_regions
        if regions:
            weights = emphasis.peak_weights(culled.get("two_theta", []), regions)
            if weights is not None:
                variants.append(
                    (f"weight your emphasis on {emphasis.describe(regions)}",
                     None, culled, {"exp_weights": weights})
                )

        out = ["2. WHAT WOULD CHANGE THE ANSWER", "-" * 60]
        out += _prose(
            "The same phase rescored under settings you can change. 'Min "
            f"fingerprint' is {gate:.2f}, so any row reaching that score would "
            "put the phase in the results list."
        )
        out.append("")
        remedy = None
        passes_as_is = False
        for label, action, peaks, overrides in variants:
            if peaks is full and n_culled == n_full:
                continue  # nothing was culled, so this repeats the row above
            score, found = best_score(peaks, **overrides)
            listed = score >= gate
            out.append(f"  {label:45s} {score:.3f} ({found} lines)  "
                       f"{'WOULD BE LISTED' if listed else 'still dropped'}")
            if not listed:
                continue
            if action is None:
                passes_as_is = True
            elif remedy is None:
                remedy = (action, score)
        out.append("")
        if passes_as_is:
            out += _prose(
                "It already scores well enough on the settings in use, so the "
                "score is not what kept it out — section 1 names the gate that "
                "did, and section 5 has the counts behind it."
            )
        elif remedy:
            out += _prose(f"Try this first: {remedy[0]}, then search again. That "
                          f"change alone takes the score to {remedy[1]:.3f}.")
        else:
            out += _prose(
                "No single change above is enough. Either the peaks this phase "
                "needs are genuinely absent from the pattern, or this record is "
                "the wrong polymorph or cell for your sample. Section 4 shows "
                "which individual lines are missing."
            )
        return out

    def _verdict_text(self, query, trace, listed, best, best_full) -> str:
        """
        The outcome in plain sentences: what happened, and what to do about it.

        This is the only part of the report most users will read, so it has to
        stand on its own — the gate that decided the outcome, how close the
        phase came to passing it, and the control that would change it.
        """
        if listed:
            top = max(listed, key=lambda r: r.get("fingerprint_score", 0))
            return (f"{top['mineral_name']} IS in the results, at fingerprint "
                    f"score {top['fingerprint_score']:.3f}. Nothing rejected it — "
                    "if you cannot see it, sort the list by score or scroll "
                    "further down it.")

        record = trace.verdict(query)
        gate = record["gate"] if record else None
        reason = search_debug.GATE_REASONS.get(gate)
        named = query[:1].upper() + query[1:]
        parts = []
        if reason:
            parts.append(reason.format(phase=named))
        else:
            parts.append(f"No record of {named} ever reached scoring: nothing "
                         "named it entered the candidate pool, so none of the "
                         "match thresholds is responsible.")

        if gate == "min_score":
            score = record.get("score")
            if score is None and best:
                score = best["info"]["score"]
            if score is not None:
                parts.append(f"Its best record scored {score:.3f} where "
                             f"{self.fp_min_score.value():.2f} was needed.")

        if best and best_full:
            delta = best_full["info"]["score"] - best["info"]["score"]
            if delta > 0.05 and best_full["info"]["score"] >= self.fp_min_score.value():
                parts.append(
                    f"Against your full peak list it scores "
                    f"{best_full['info']['score']:.3f} "
                    f"({best_full['info']['n_found']} of "
                    f"{best_full['info']['n_expected']} lines found), so the "
                    "peaks it needs are in the pattern — 'Search peak count' "
                    "discarded them before scoring."
                )

        fix = search_debug.GATE_FIXES.get(gate)
        if fix:
            parts.append(fix)
        return " ".join(parts)

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

        # Everything carried into the residual round is part of this pattern's
        # answer, including phases locked in during earlier rounds that the
        # table no longer shows. Shortlisting the lot here is what makes moving
        # on safe: the round is about to replace the list they were checked in.
        self.workspace.shortlist_phases(selected)

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
            if method == "fingerprint" and self.session.emphasis_regions:
                label += f" emphasising {emphasis.describe(self.session.emphasis_regions)}"
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

        # An emphasised region is the user saying these peaks are the question,
        # so intensity cuts must not be what removes them
        protect = emphasis.inside(peaks.get("two_theta", []), self.session.emphasis_regions)
        if len(protect) == len(keep):
            keep |= protect

        if np.all(keep) or np.sum(keep) < 4:  # never strip the list bare
            search_debug.stage(f"Search peak list: all {len(inten)} peaks kept")
            return peaks

        out = {}
        for key, value in peaks.items():
            arr = np.asarray(value) if isinstance(value, (list, np.ndarray)) else None
            out[key] = arr[keep] if arr is not None and arr.shape[:1] == keep.shape else value
        print(f"Search uses the strongest {int(np.sum(keep))} of {len(inten)} peaks")
        search_debug.stage(
            f"Search peak list: strongest {int(np.sum(keep))} of {len(inten)} peaks "
            f"(weakest kept {float(np.min(inten[keep])) / imax * 100:.2f}% of max; "
            f"'Search peak count' {limit}, floor {floor_pct:.2f}%)"
        )
        return out

    def _fingerprint_search(self, pattern, residual_peaks=None):
        """Screen the database on peak positions, then score each candidate's own lines."""
        exp_peaks = residual_peaks if residual_peaks is not None else self.session.peaks
        exp_peaks = self._significant_peaks(exp_peaks)
        exp_tt = exp_peaks.get("two_theta", [])
        exp_weights = None
        if residual_peaks is not None:
            # Read the culled copy: _significant_peaks slices every per-peak
            # array, and the weights have to line up with the peaks that survived
            w = exp_peaks.get("residual_weights")
            if w is not None and len(w) == len(exp_tt):
                exp_weights = np.asarray(w, dtype=float)

        boost = emphasis.peak_weights(exp_tt, self.session.emphasis_regions)
        if boost is not None:
            exp_weights = boost if exp_weights is None else exp_weights * boost
            # Only the ratio between weights carries meaning, and the coverage
            # screen multiplies them in raw: left above 1.0 they push coverage
            # past 1.0 and 'Screen min coverage' stops being a fraction.
            exp_weights = exp_weights / float(np.max(exp_weights))
            search_debug.stage(
                f"Emphasis: {emphasis.describe(self.session.emphasis_regions)} — "
                f"{int(np.sum(boost > 1.0))} of {len(boost)} search peaks weighted up"
            )

        pool_size = self.pool_size.value()
        tol = self.tolerance.value()
        wl = pattern.get("wavelength", self.session.wavelength)
        shift = self.shift.value()
        span = self.shift_span.value()
        model = self.shift_model_key()
        pool = []

        # How much of the pattern a reference line can hit by luck; warned about
        # in the status line because it decides whether matching means anything.
        # Intensity-weighted to match fingerprint_score's baseline.
        exp_int = np.asarray(exp_peaks.get("intensity", []), dtype=float)
        self._last_chance = coincidence_fraction(
            exp_peaks.get("two_theta", []), tol, self._measured_range(),
            np.sqrt(np.maximum(exp_int, 0.0)),
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
            weights_rank_only=residual_peaks is None,
            shift=shift,
            shift_span=span,
            shift_model=model,
        )
        parts = [
            name for name, on in (
                ("residual", residual_peaks is not None),
                ("emphasis", boost is not None),
            ) if on
        ]
        mode = "-".join(parts) + "-weighted" if parts else "presence"
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
            note += " Fit to Row on a phase you trust to lock that shift in for every phase."
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
        """
        The pre-fingerprint search methods, run straight on the UI thread.

        Correlation compares every stored pattern point by point, so these
        take on the order of a minute over the full database and the window
        stops responding while they do. The wait cursor is there to say the
        application is working rather than wedged.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return self._legacy_search_run(pattern, method, peaks_override)
        finally:
            QApplication.restoreOverrideCursor()

    def _legacy_search_run(self, pattern, method: str, peaks_override=None):
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
        # The peak half wants the peak list, the correlation half wants the
        # measured profile — handing either one both makes its score meaningless
        if method == "combined":
            return self.search_engine.combined_search(
                peak_data,
                peak_tolerance=ptol,
                min_correlation=min_c,
                max_results=max_r,
                full_pattern=pattern,
                ambient_only=ambient_only,
            )
        return self.search_engine.ensemble_search(
            peak_data,
            methods=["peaks", "correlation", "ultrafast"],
            max_results=max_r,
            peak_tolerance=ptol,
            min_correlation=min_c,
            full_pattern=pattern,
            ambient_only=ambient_only,
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

        # Matching replaces the candidate list with its results, so anything
        # checked in it has to be recorded before it disappears
        self.workspace.shortlist_phases(phases)

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
        weak = len(results) - len(filtered)
        for r in filtered:
            self._attach_fingerprint(r)
        filtered.sort(
            key=lambda x: x.get("fingerprint_score", x.get("combined_score", x.get("match_score", 0))),
            reverse=True,
        )

        # Every checked candidate earns its own row: the database holds many
        # records per mineral, and picking three forsterites is a deliberate
        # comparison, not a duplicate. Only the very same record is dropped.
        previous = list(self._kept_phases)
        seen_ids = set()
        for p in previous:
            seen_ids |= mineral_ids(p)
        merged = list(previous)
        new_count = 0
        duplicates = 0
        for r in filtered:
            ids = mineral_ids(r)
            if ids and ids & seen_ids:
                duplicates += 1
                continue
            merged.append(r)
            seen_ids |= ids
            new_count += 1

        self.session.set_matched_phases(merged)
        self.session.set_selected_phases(previous)
        self.workspace.set_results_matches(merged, preselect=previous)
        self.update_action_states()
        skipped = []
        if weak:
            skipped.append(f"{weak} below the {min_score:.2f} match score")
        if duplicates:
            skipped.append(f"{duplicates} already in the list")
        note = f" Skipped {', '.join(skipped)}." if skipped else ""
        self.status.setText(
            f"Matched {new_count} new phase(s); {len(previous)} previously kept.{note} "
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
