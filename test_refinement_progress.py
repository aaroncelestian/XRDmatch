#!/usr/bin/env python3
"""
Tests for watching a refinement while it runs, and stopping it early.

A refinement is minutes of numpy work with nothing to look at, so the progress
window has to report cycle by cycle and has to be able to call the whole thing
off when the fit is visibly going nowhere -- keeping whatever it reached rather
than throwing the run away.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui.dialogs.refinement_progress_dialog import (  # noqa: E402
    RefinementProgressDialog, RefinementWorker,
)
from utils.multi_phase_analyzer import MultiPhaseAnalyzer  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


PEAKS = np.array([20.9, 26.6, 36.5, 50.1, 59.9])
INTENSITIES = np.array([100.0, 85.0, 30.0, 35.0, 18.0])


def _phase():
    return {
        "phase": {
            "mineral": "Quartz", "formula": "SiO2", "rir": 3.0,
            "cell_a": 4.913, "cell_b": 4.913, "cell_c": 5.405,
            "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 120.0,
        },
        "theoretical_peaks": {
            "two_theta": PEAKS,
            "intensity": INTENSITIES,
            "d_spacing": np.full(len(PEAKS), 2.0),
        },
        "optimized_scaling": 1.0,
    }


def _observed():
    two_theta = np.linspace(15.0, 65.0, 2000)
    pattern = np.zeros_like(two_theta)
    for centre, height in zip(PEAKS, INTENSITIES):
        pattern += height * np.exp(-0.5 * ((two_theta - (centre + 0.05)) / 0.09) ** 2)
    return {"two_theta": two_theta, "intensity": pattern, "wavelength": 1.5406}


PARAMS = {
    "initial_u": 0.0005, "initial_v": 0.0, "initial_w": 0.01,
    "intensity_model": "fixed", "refine_cell": True, "refine_profile": True,
    "max_scale": 100.0,
}


def _refine(max_iterations=4, **hooks):
    return MultiPhaseAnalyzer().perform_lebail_refinement(
        _observed(), [_phase()], max_iterations=max_iterations,
        refinement_params=dict(PARAMS), **hooks,
    )


# --- the progress stream ---------------------------------------------------

def test_progress_reports_every_cycle_with_a_drawable_fit():
    updates = []
    results = _refine(progress_callback=updates.append)
    assert results["success"]

    cycles = [u for u in updates if u.get("phase_of_work") == "cycle"]
    assert len(cycles) == results["refinement_results"]["iterations"]

    observed = _observed()["intensity"]
    for update in cycles:
        assert update["iteration"] >= 1
        assert update["total_iterations"] >= update["iteration"]
        assert update["r_factors"]["Rwp"] >= 0
        # The window plots this against the observed pattern, so it has to
        # arrive complete rather than as a reference the engine keeps mutating
        assert len(update["calculated_pattern"]) == len(observed)
        assert update["message"]


def test_progress_names_the_phase_being_worked_on():
    updates = []
    _refine(progress_callback=updates.append)
    phase_steps = [u for u in updates if u.get("phase_of_work") == "phase"]
    assert phase_steps
    assert any("Quartz" in u["message"] for u in phase_steps)


def test_log_messages_reach_the_watcher():
    lines = []
    _refine(log_callback=lines.append)
    assert any("Le Bail" in line for line in lines)


def test_a_failing_watcher_cannot_stop_the_refinement():
    def explode(_payload):
        raise RuntimeError("watcher fell over")

    results = _refine(progress_callback=explode)
    assert results["success"]


# --- stopping early --------------------------------------------------------

def test_cancelling_keeps_the_fit_reached_so_far():
    seen = []

    def stop_after_first_cycle(payload):
        if payload.get("phase_of_work") == "cycle":
            seen.append(payload)

    results = _refine(
        max_iterations=12,
        progress_callback=stop_after_first_cycle,
        cancel_check=lambda: len(seen) >= 1,
    )

    assert results["success"]
    assert results["cancelled"]
    inner = results["refinement_results"]
    assert inner["cancelled"] and not inner["converged"]
    # Stopping early must still hand back a usable fit, not an empty shell
    assert inner["iterations"] >= 1
    assert inner["final_r_factors"]["Rwp"] >= 0
    assert len(inner["calculated_pattern"]) == len(inner["two_theta"])
    # And it must actually cut the run short
    assert inner["iterations"] < 12


def test_cancelling_before_any_cycle_still_returns_a_pattern():
    results = _refine(max_iterations=6, cancel_check=lambda: True)
    inner = results["refinement_results"]
    assert results["success"] and inner["cancelled"]
    assert len(inner["calculated_pattern"]) == len(inner["two_theta"])


def test_an_uncancelled_run_is_not_marked_cancelled():
    results = _refine()
    assert not results["cancelled"]
    assert not results["refinement_results"]["cancelled"]


# --- the window ------------------------------------------------------------

def _worker():
    return RefinementWorker(MultiPhaseAnalyzer(), {
        "experimental_data": _observed(),
        "identified_phases": [_phase()],
        "max_iterations": 3,
        "refinement_params": dict(PARAMS),
    })


def test_window_tracks_the_run_and_closes_when_it_finishes(qt_app):
    worker = _worker()
    dialog = RefinementProgressDialog(worker)
    observed = _observed()
    dialog.set_observed(observed["two_theta"], observed["intensity"])

    dialog.exec_()
    worker.wait(30000)

    assert dialog.error is None
    assert dialog.results is not None and dialog.results["success"]
    assert dialog._rwp_trace, "the Rwp trace should have been plotted"
    assert "Rwp" in dialog.status.text()
    assert dialog.log.toPlainText().strip()
    assert not dialog.isVisible()


def test_window_stops_the_run_and_stays_open_to_say_so(qt_app):
    worker = RefinementWorker(MultiPhaseAnalyzer(), {
        "experimental_data": _observed(),
        "identified_phases": [_phase()],
        "max_iterations": 40,
        "refinement_params": dict(PARAMS),
    })
    dialog = RefinementProgressDialog(worker)

    # Stop as soon as the first cycle has been drawn, then close the window the
    # way a user would once they have read the outcome
    def stop_when_running():
        if dialog._rwp_trace:
            dialog.request_stop()
            QTimer.singleShot(200, dialog.accept)
        else:
            QTimer.singleShot(100, stop_when_running)

    QTimer.singleShot(100, stop_when_running)
    dialog.exec_()
    worker.wait(30000)

    assert dialog.cancelled
    assert dialog.results is not None and dialog.results["cancelled"]
    assert "Stopped" in dialog.status.text()


def test_window_reports_a_failure_instead_of_vanishing(qt_app):
    class Broken:
        def perform_lebail_refinement(self, **_kwargs):
            raise ValueError("no phases to refine")

    worker = RefinementWorker(Broken(), {})
    dialog = RefinementProgressDialog(worker)
    QTimer.singleShot(600, dialog.accept)
    dialog.exec_()
    worker.wait(5000)

    assert dialog.results is None
    assert "no phases to refine" in dialog.error
    assert "no phases to refine" in dialog.status.text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
