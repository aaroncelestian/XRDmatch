#!/usr/bin/env python3
"""
Tests for the editable parameter grid and the path from a typed value to the
refinement that uses it.

The grid replaced a read-only window. Its whole purpose is that a value can be
read and changed in the same place, so the tests that matter are the ones that
follow an edit all the way through: tick states become refine flags, an
unticked cell becomes a value held against the optimiser, and neither is lost
when a run finishes and the window repopulates itself.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMainWindow  # noqa: E402

from gui import refinement_table  # noqa: E402
from gui.dialogs.refinement_details_dialog import RefinementDetailsDialog  # noqa: E402
from gui.session import AnalysisSession  # noqa: E402
from gui.widgets.parameter_matrix import PHASE_ROWS, ParameterMatrix  # noqa: E402
from test_refinement_reporting import (  # noqa: E402
    BASE_PARAMS, INTEN_A, INTEN_B, PEAKS_A, PEAKS_B, _observed, _phase,
)
from utils.multi_phase_analyzer import MultiPhaseAnalyzer  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


VALUES = {
    "Quartz": {"scale_factor": 1.25, "microstrain": 800.0, "crystallite_size": 0.5,
               "asymmetry": 0.0, "lattice_scale": 1.0, "absorption": 0.0,
               "refine_scale": True, "refine_strain": True, "refine_size": False},
    "Albite": {"scale_factor": 0.75, "microstrain": 150.0, "crystallite_size": 0.2,
               "asymmetry": -0.3, "lattice_scale": 0.99, "absorption": 0.0,
               "refine_scale": True, "refine_strain": False, "refine_size": False},
}


def _row_index(label):
    return [r.label for r in PHASE_ROWS].index(label)


def _filled(qt_app, quantitative=True):
    matrix = ParameterMatrix()
    matrix.set_phases(list(VALUES), VALUES, quantitative=quantitative)
    return matrix


# --- what the grid shows ---------------------------------------------------

def test_the_grid_opens_where_the_last_run_finished(qt_app):
    """A parameter is nudged from where it stands, not guessed from nothing."""
    matrix = _filled(qt_app)
    column = 1 + list(VALUES).index("Albite")
    assert float(matrix.item(_row_index("Microstrain (×10⁻⁶)"), column).text()) == 150.0


def test_tick_state_follows_what_was_refined(qt_app):
    matrix = _filled(qt_app)
    row = _row_index("Microstrain (×10⁻⁶)")
    quartz = matrix.item(row, 1 + list(VALUES).index("Quartz"))
    albite = matrix.item(row, 1 + list(VALUES).index("Albite"))
    assert quartz.checkState() == Qt.Checked
    assert albite.checkState() == Qt.Unchecked


def test_terms_that_le_bail_absorbs_are_not_offered(qt_app):
    """
    Scale and absorption cannot be determined when the intensities are
    extracted from the data, so they must not look adjustable.
    """
    matrix = _filled(qt_app, quantitative=False)
    item = matrix.item(_row_index("Scale factor"), 1)
    assert not (item.flags() & Qt.ItemIsUserCheckable)
    strain = matrix.item(_row_index("Microstrain (×10⁻⁶)"), 1)
    assert strain.flags() & Qt.ItemIsUserCheckable


# --- what the grid returns -------------------------------------------------

def test_an_unticked_cell_becomes_a_held_value(qt_app):
    matrix = _filled(qt_app)
    overrides = matrix.overrides()
    assert "microstrain" in overrides["Albite"]["_locked"]
    assert overrides["Albite"]["microstrain"] == pytest.approx(150.0)
    assert "microstrain" not in overrides["Quartz"].get("_locked", [])


def test_a_ticked_cell_is_refined_not_held(qt_app):
    matrix = _filled(qt_app)
    overrides = matrix.overrides()
    assert overrides["Quartz"]["refine_strain"] is True
    assert "microstrain" not in overrides["Quartz"].get("_locked", [])


def test_typing_a_value_is_carried_through(qt_app):
    matrix = _filled(qt_app)
    row = _row_index("Crystallite size (µm)")
    matrix.item(row, 1).setText("0.042")
    overrides = matrix.overrides()
    assert overrides["Quartz"]["crystallite_size"] == pytest.approx(0.042)
    assert "crystallite_size" in overrides["Quartz"]["_locked"]


def test_an_emptied_cell_holds_nothing(qt_app):
    """Clearing a cell should fall back to the default, not pin a blank."""
    matrix = _filled(qt_app)
    row = _row_index("Crystallite size (µm)")
    matrix.item(row, 1).setText("")
    overrides = matrix.overrides()
    assert "crystallite_size" not in overrides["Quartz"]
    assert "crystallite_size" not in overrides["Quartz"].get("_locked", [])


def test_nonsense_in_a_cell_is_ignored_rather_than_crashing(qt_app):
    matrix = _filled(qt_app)
    matrix.item(_row_index("Microstrain (×10⁻⁶)"), 1).setText("about eight hundred")
    assert "microstrain" not in matrix.overrides()["Quartz"]


def test_one_parameter_can_be_set_across_every_phase(qt_app):
    matrix = _filled(qt_app)
    matrix.set_all("Microstrain (×10⁻⁶)", False)
    overrides = matrix.overrides()
    assert all("microstrain" in overrides[n]["_locked"] for n in VALUES)


# --- the whole path --------------------------------------------------------

def _results():
    analyzer = MultiPhaseAnalyzer()
    return analyzer.perform_lebail_refinement(
        _observed(shift=0.05),
        [_phase("Quartz", PEAKS_A, INTEN_A, 4.913),
         _phase("Albite", PEAKS_B, INTEN_B, 8.144)],
        max_iterations=4, refinement_params=BASE_PARAMS,
    )


def test_a_value_typed_in_the_grid_reaches_the_refinement(qt_app):
    """
    The point of the whole feature: type a number, run, and the refinement
    holds that number instead of solving for one.
    """
    session = AnalysisSession()
    session.set_lebail_results(_results())
    main = QMainWindow()
    dialog = RefinementDetailsDialog(session, parent=main)

    row = _row_index("Microstrain (×10⁻⁶)")
    item = dialog.matrix.item(row, 1 + refinement_table.phase_names(
        session.lebail_results).index("Albite"))
    item.setCheckState(Qt.Unchecked)
    item.setText("742")

    assert session.phase_overrides["Albite"]["microstrain"] == pytest.approx(742.0)

    analyzer = MultiPhaseAnalyzer()
    analyzer.perform_lebail_refinement(
        _observed(shift=0.05),
        [_phase("Quartz", PEAKS_A, INTEN_A, 4.913),
         _phase("Albite", PEAKS_B, INTEN_B, 8.144)],
        max_iterations=4,
        refinement_params={**BASE_PARAMS,
                           "phase_overrides": session.phase_overrides},
    )
    held = [p for p in analyzer.lebail_engine.phases
            if p["data"]["phase"]["mineral"] == "Albite"][0]
    assert held["parameters"]["microstrain"] == pytest.approx(742.0)
    dialog.close(); main.close()


def test_a_completed_run_does_not_wipe_hand_set_values(qt_app):
    """
    The window repopulates itself whenever a refinement finishes. An edit made
    before that run has to survive it, or the feature would undo itself.
    """
    session = AnalysisSession()
    session.set_lebail_results(_results())
    main = QMainWindow()
    dialog = RefinementDetailsDialog(session, parent=main)

    row = _row_index("Crystallite size (µm)")
    column = 1 + refinement_table.phase_names(session.lebail_results).index("Quartz")
    dialog.matrix.item(row, column).setCheckState(Qt.Unchecked)
    dialog.matrix.item(row, column).setText("0.333")

    session.set_lebail_results(_results())  # a new run lands

    assert dialog.matrix.item(row, column).text() == "0.333"
    assert session.phase_overrides["Quartz"]["crystallite_size"] == pytest.approx(0.333)
    dialog.close(); main.close()


def test_reset_clears_every_hand_set_value(qt_app):
    session = AnalysisSession()
    session.set_lebail_results(_results())
    main = QMainWindow()
    dialog = RefinementDetailsDialog(session, parent=main)

    dialog.matrix.item(_row_index("Microstrain (×10⁻⁶)"), 1).setText("999")
    assert session.phase_overrides
    dialog._reset_overrides()
    assert session.phase_overrides == {}
    dialog.close(); main.close()


def test_the_window_opens_before_any_refinement(qt_app):
    session = AnalysisSession()
    session.set_raw_pattern(_observed())
    main = QMainWindow()
    dialog = RefinementDetailsDialog(session, parent=main)
    assert dialog.matrix.rowCount() >= 0  # must not raise
    dialog.close(); main.close()


def test_loading_a_new_pattern_forgets_the_overrides(qt_app):
    session = AnalysisSession()
    session.phase_overrides = {"Quartz": {"microstrain": 500.0}}
    session.set_raw_pattern(_observed())
    assert session.phase_overrides == {}
