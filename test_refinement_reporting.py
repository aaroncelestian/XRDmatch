#!/usr/bin/env python3
"""
Tests for carrying refined values between runs, and for reporting them.

Staged refinement depends on a parameter surviving the run after the one that
determined it: free the sample displacement, let it settle, untick it, then
refine the cell against the displacement just found. If unticking a box reset
the value, the second stage would be fitting against a zero.
"""

import csv
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui import refinement_table  # noqa: E402
from gui.session import AnalysisSession  # noqa: E402
from gui.dialogs.quant_dialog import QuantDialog  # noqa: E402
from gui.dialogs.refinement_details_dialog import RefinementDetailsDialog  # noqa: E402
from gui.widgets.copyable_table import CopyableTable  # noqa: E402
from utils.multi_phase_analyzer import MultiPhaseAnalyzer  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


# --- a small but real two-phase refinement ---------------------------------

PEAKS_A = np.array([20.9, 26.6, 36.5, 50.1, 59.9])
PEAKS_B = np.array([23.5, 28.1, 40.3, 47.2, 55.4])
INTEN_A = np.array([100.0, 85.0, 30.0, 35.0, 18.0])
INTEN_B = np.array([60.0, 90.0, 25.0, 40.0, 20.0])


def _phase(name, peaks, intensities, cell_a, rir=3.0):
    return {
        "phase": {
            "mineral": name,
            "formula": "SiO2",
            "rir": rir,
            "cell_a": cell_a, "cell_b": cell_a, "cell_c": cell_a * 1.1,
            "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 120.0,
        },
        "theoretical_peaks": {
            "two_theta": peaks,
            "intensity": intensities,
            "d_spacing": np.full(len(peaks), 2.0),
        },
        "optimized_scaling": 1.0,
    }


def _observed(shift=0.0):
    two_theta = np.linspace(15.0, 65.0, 2500)
    pattern = np.zeros_like(two_theta)
    for peaks, inten, scale in ((PEAKS_A, INTEN_A, 1.0), (PEAKS_B, INTEN_B, 0.6)):
        for centre, height in zip(peaks, inten):
            pattern += scale * height * np.exp(
                -0.5 * ((two_theta - (centre + shift)) / 0.09) ** 2
            )
    return {"two_theta": two_theta, "intensity": pattern, "wavelength": 1.5406}


def _refine(refinement_params, phases=None):
    analyzer = MultiPhaseAnalyzer()
    return analyzer.perform_lebail_refinement(
        _observed(shift=0.05),
        phases or [_phase("Quartz", PEAKS_A, INTEN_A, 4.913),
                   _phase("Albite", PEAKS_B, INTEN_B, 8.144)],
        max_iterations=4,
        refinement_params=refinement_params,
    )


BASE_PARAMS = {
    "initial_u": 0.0005, "initial_v": 0.0, "initial_w": 0.01,
    "intensity_model": "fixed", "refine_cell": True, "refine_profile": True,
    "refine_strain": True, "max_scale": 100.0,
}


# --- what the weight percents are based on ---------------------------------

_MIXED_RIR = [
    _phase("Quartz", PEAKS_A, INTEN_A, 4.913, rir=1.546),
    _phase("Albite", PEAKS_B, INTEN_B, 8.144, rir=None),
]


def test_complete_rir_set_gives_chung_weight_percents():
    summary = _refine(BASE_PARAMS)["refinement_results"]["phase_summary"]
    assert all(row["weight_percent_basis"] == "rir" for row in summary)
    assert sum(row["weight_percent"] for row in summary) == pytest.approx(100.0)
    # With equal RIRs the split follows the integrated intensity of each
    # strongest line, not its height: the two phases refine to different widths,
    # and a narrower peak is not more material. See test_weight_percent.py.
    ratio = summary[0]["line_area"] / summary[1]["line_area"]
    assert summary[0]["weight_percent"] / summary[1]["weight_percent"] == pytest.approx(ratio)


def test_missing_rir_falls_back_to_pattern_contribution():
    """
    The reported case: one phase has an I/Ic, the rest show a dash.

    Chung normalises over every phase, so quantifying only the ones with an RIR
    reports them as the whole sample. Falling back to pattern contribution keeps
    all the phases on one basis and visible.
    """
    summary = _refine(BASE_PARAMS, phases=_MIXED_RIR)["refinement_results"]["phase_summary"]

    assert all(row["weight_percent_basis"] == "contribution" for row in summary)
    for row in summary:
        assert row["weight_percent"] == pytest.approx(row["contribution_percent"])
    assert sum(row["weight_percent"] for row in summary) == pytest.approx(100.0, abs=0.5)


def test_the_only_rir_phase_is_not_reported_as_the_whole_sample():
    """An RIR on one phase of two must not make that phase 100 wt%."""
    summary = _refine(BASE_PARAMS, phases=_MIXED_RIR)["refinement_results"]["phase_summary"]
    quartz = next(row for row in summary if row["name"] == "Quartz")
    assert quartz["rir"] == pytest.approx(1.546)
    assert quartz["weight_percent"] < 99.0


def test_a_phase_without_rir_still_gets_a_weight_percent():
    summary = _refine(BASE_PARAMS, phases=_MIXED_RIR)["refinement_results"]["phase_summary"]
    albite = next(row for row in summary if row["name"] == "Albite")
    assert albite["rir"] is None
    assert albite["weight_percent"] is not None and albite["weight_percent"] > 0


def test_extraction_still_reports_no_weight_percent():
    """Free intensities absorb the scale, so there is nothing to quantify."""
    summary = _refine(
        {**BASE_PARAMS, "intensity_model": "extract"}
    )["refinement_results"]["phase_summary"]
    assert all(row["weight_percent"] is None for row in summary)
    assert all(row["weight_percent_basis"] is None for row in summary)


def test_the_fallback_is_declared_in_the_reporting():
    results = _refine(BASE_PARAMS, phases=_MIXED_RIR)

    assert refinement_table.weight_basis(results) == "contribution"
    assert refinement_table.phases_missing_rir(results) == ["Albite"]

    headline = " ".join(refinement_table.summary_headline(results))
    assert "pattern contribution" in headline and "Albite" in headline

    labels = [row[0] for row in refinement_table.detail_rows(results)]
    assert "Weight percent from" in labels
    basis_row = next(
        row for row in refinement_table.detail_rows(results)
        if row[0] == "Weight percent from"
    )
    assert all("Pattern contribution" in cell for cell in basis_row[1:])

    wt_tooltips = [row[1] for row in refinement_table.summary_tooltips(results)]
    assert all("I/Ic" in note for note in wt_tooltips)


def test_a_chung_result_is_labelled_as_chung():
    results = _refine(BASE_PARAMS)
    assert refinement_table.weight_basis(results) == "rir"
    assert "pattern contribution" not in " ".join(
        refinement_table.summary_headline(results)
    )
    basis_row = next(
        row for row in refinement_table.detail_rows(results)
        if row[0] == "Weight percent from"
    )
    assert all(cell == "Chung RIR" for cell in basis_row[1:])


# --- carrying values between runs ------------------------------------------

def test_unrefined_parameter_keeps_its_carried_value():
    """
    Untick displacement after refining it and the value must stay put.

    This is the reported bug: the engine is rebuilt for every run, so without
    the values being handed forward, a parameter that is no longer refined
    reverts to the default it started from.
    """
    first = _refine({**BASE_PARAMS, "refine_displacement": True,
                     "refine_zero_shift": True})
    globals_first = first["refinement_results"]["global_parameters"]
    displacement = globals_first["displacement"]
    assert displacement != 0.0, "the test needs a displacement that actually moved"

    second = _refine({
        **BASE_PARAMS,
        "refine_displacement": False,
        "refine_zero_shift": False,
        "carry_globals": {
            "displacement": displacement,
            "zero_shift": globals_first["zero_shift"],
        },
    })
    globals_second = second["refinement_results"]["global_parameters"]
    assert globals_second["displacement"] == pytest.approx(displacement)
    assert globals_second["zero_shift"] == pytest.approx(globals_first["zero_shift"])


def test_without_carry_over_the_value_is_lost():
    """The old behaviour, kept as a test so the fix cannot silently regress."""
    result = _refine({**BASE_PARAMS, "refine_displacement": False})
    assert result["refinement_results"]["global_parameters"]["displacement"] == 0.0


def test_phase_values_are_carried_by_name():
    first = _refine({**BASE_PARAMS, "refine_strain": True})
    summary = first["refinement_results"]["phase_summary"]
    carried = {
        row["name"]: {
            "microstrain": row["microstrain"],
            "lattice_scale": row["lattice_scale"],
            "scale_factor": row["scale"],
        }
        for row in summary
    }

    second = _refine({
        **BASE_PARAMS, "refine_strain": False, "refine_cell": False,
        "refine_profile": False, "carry_over": carried,
    })
    for row in second["refinement_results"]["phase_summary"]:
        expected = carried[row["name"]]
        assert row["microstrain"] == pytest.approx(expected["microstrain"])
        assert row["lattice_scale"] == pytest.approx(expected["lattice_scale"])


def test_carried_lattice_scale_is_reflected_in_the_reported_cell():
    """
    A dilation held fixed must still show up as scaled cell edges.

    The cell is otherwise only recomputed when lattice_scale is among the
    parameters being refined, so a carried-over dilation would shift the peaks
    while the table went on reporting the starting cell.
    """
    carried = {"Quartz": {"lattice_scale": 1.004}}
    result = _refine({**BASE_PARAMS, "refine_cell": False, "carry_over": carried})

    row = next(r for r in result["refinement_results"]["phase_summary"]
               if r["name"] == "Quartz")
    assert row["unit_cell"]["a"] == pytest.approx(4.913 * 1.004, rel=1e-6)
    assert row["base_unit_cell"]["a"] == pytest.approx(4.913, rel=1e-6)
    # An isotropic dilation leaves the angles alone
    assert row["unit_cell"]["gamma"] == pytest.approx(120.0)


# --- the reported table ----------------------------------------------------

@pytest.fixture(scope="module")
def results():
    return _refine({**BASE_PARAMS, "refine_displacement": True})


def test_summary_table_reports_the_whole_cell(results):
    labels = [label for label, _ in refinement_table.SUMMARY_COLUMNS]
    for expected in ("a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)",
                     "V (Å³)", "Δlattice %", "wt%", "Scale", "Strain",
                     "Contrib.%"):
        assert expected in labels

    rows = refinement_table.summary_rows(results)
    assert len(rows) == 2
    for row in rows:
        assert len(row) == len(labels)
        assert all(cell != "" for cell in row)

    named = {row[0]: row for row in rows}
    assert set(named) == {"Quartz", "Albite"}
    # b is reported, and for these cells differs from c
    quartz = named["Quartz"]
    assert float(quartz[labels.index("b (Å)")]) == pytest.approx(4.913, abs=0.2)
    assert float(quartz[labels.index("γ (°)")]) == pytest.approx(120.0)


def test_missing_values_render_as_a_dash():
    empty = {"refinement_results": {"phase_summary": [{"name": "X"}]}}
    row = refinement_table.summary_rows(empty)[0]
    assert row[0] == "X"
    assert row[1:] == ["—"] * (len(refinement_table.SUMMARY_COLUMNS) - 1)


def test_detail_rows_cover_every_phase(results):
    names = refinement_table.phase_names(results)
    assert names == ["Quartz", "Albite"]

    rows = refinement_table.detail_rows(results)
    labels = [row[0] for row in rows]
    for expected in ("Absorption", "Harmonic coefficients", "Crystallite size (µm)",
                     "Microstrain (×10⁻⁶)", "Starting a (Å)", "RIR (I/Ic)"):
        assert expected in labels
    for row in rows:
        assert len(row) == 1 + len(names)

    global_labels = [name for name, _ in refinement_table.global_rows(results)]
    assert "Sample displacement (°)" in global_labels
    assert "Rwp (%)" in global_labels


# --- copying and exporting -------------------------------------------------

def test_table_copies_as_tab_separated_text(qt_app):
    table = CopyableTable()
    table.set_content(["A", "B"], [["1", "2"], ["3", "4"]])

    table.clearSelection()
    table.copy_selection()
    assert QApplication.clipboard().text() == "A\tB\n1\t2\n3\t4"

    table.setRangeSelected(
        __import__("PyQt5.QtWidgets", fromlist=["QTableWidgetSelectionRange"])
        .QTableWidgetSelectionRange(1, 0, 1, 1), True
    )
    table.copy_selection()
    assert QApplication.clipboard().text() == "3\t4"


def test_quant_table_is_populated_and_copyable(qt_app, results):
    session = AnalysisSession()
    session.set_lebail_results(results)
    dialog = QuantDialog(session)
    dialog.refresh_plot()

    table = dialog.quant_results_table
    assert table.rowCount() == 2
    assert "a (Å)" in table.headers()
    assert table.headers() == [
        label for label, _ in refinement_table.SUMMARY_COLUMNS
    ]

    table.copy_all()
    copied = QApplication.clipboard().text().splitlines()
    assert copied[0].split("\t")[0] == "Phase"
    assert len(copied) == 3
    assert dialog.details_btn.isEnabled()


def test_details_dialog_lists_globals_and_phases(qt_app, results):
    session = AnalysisSession()
    session.set_lebail_results(results)
    dialog = RefinementDetailsDialog(session)

    assert dialog.phase_table.headers() == ["Parameter", "Quartz", "Albite"]
    assert dialog.global_table.rowCount() > 5
    assert dialog.phase_table.rowCount() > 15

    dialog.copy_all()
    text = QApplication.clipboard().text()
    assert "Global" in text and "Per phase" in text and "Quartz" in text


def test_csv_export_writes_the_table(qt_app, results, tmp_path, monkeypatch):
    session = AnalysisSession()
    session.set_lebail_results(results)
    dialog = QuantDialog(session)
    stage = dialog.refine_stage

    target = str(tmp_path / "results.csv")
    monkeypatch.setattr(
        "gui.stages.refine_stage.QFileDialog.getSaveFileName",
        lambda *a, **k: (target, "CSV (*.csv)"),
    )
    stage.export_csv_data()

    with open(target, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    flat = [r[0] for r in rows if r]
    assert "Rwp (%)" in flat
    assert "Sample displacement (°)" in flat

    header_index = next(i for i, r in enumerate(rows) if r and r[0] == "Phase")
    assert rows[header_index] == [
        label for label, _ in refinement_table.SUMMARY_COLUMNS
    ]
    assert {rows[header_index + 1][0], rows[header_index + 2][0]} == {"Quartz", "Albite"}


def test_csv_export_refuses_without_a_refinement(qt_app, monkeypatch):
    session = AnalysisSession()
    dialog = QuantDialog(session)
    warned = []
    monkeypatch.setattr(
        "gui.stages.refine_stage.QMessageBox.warning",
        lambda *args, **kwargs: warned.append(args[2]),
    )
    dialog.refine_stage.export_csv_data()
    assert warned and "Le Bail" in warned[0]


def test_staged_refinement_through_the_gui(qt_app):
    """
    The reported workflow, driven through the widgets.

    Refine the sample displacement, untick it, then refine the unit cell. The
    displacement found in the first pass has to still be there in the second.
    """
    session = AnalysisSession()
    session.set_raw_pattern(_observed(shift=0.05))
    session.matched_phases = [
        _phase("Quartz", PEAKS_A, INTEN_A, 4.913),
        _phase("Albite", PEAKS_B, INTEN_B, 8.144),
    ]

    dialog = QuantDialog(session)
    stage = dialog.refine_stage
    stage.max_iter.setValue(4)
    stage.intensity_model.setCurrentIndex(0)  # reference intensities

    stage.refine_displacement.setChecked(True)
    stage.refine_cell.setChecked(False)
    stage.run_lebail()

    first = session.lebail_results["refinement_results"]["global_parameters"]
    displacement = first["displacement"]
    assert displacement != 0.0

    # Fix the displacement, now refine the cell against it
    stage.refine_displacement.setChecked(False)
    stage.refine_zero_shift.setChecked(False)
    stage.refine_cell.setChecked(True)
    stage.run_lebail()

    second = session.lebail_results["refinement_results"]["global_parameters"]
    assert second["displacement"] == pytest.approx(displacement)
    assert second["zero_shift"] == pytest.approx(first["zero_shift"])

    # And unticking "start from previous" goes back to the defaults
    stage.continue_previous.setChecked(False)
    stage.run_lebail()
    third = session.lebail_results["refinement_results"]["global_parameters"]
    assert third["displacement"] == 0.0


def test_pattern_export_includes_the_fit(qt_app, results, tmp_path, monkeypatch):
    session = AnalysisSession()
    session.set_raw_pattern(_observed(shift=0.05))
    session.set_lebail_results(results)
    dialog = QuantDialog(session)

    target = str(tmp_path / "pattern.csv")
    monkeypatch.setattr(
        "gui.stages.refine_stage.QFileDialog.getSaveFileName",
        lambda *a, **k: (target, "CSV (*.csv)"),
    )
    dialog.refine_stage.export_pattern_csv()

    with open(target, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["two_theta", "observed", "calculated", "difference"]
    assert len(rows) == len(session.active_pattern()["two_theta"]) + 1
