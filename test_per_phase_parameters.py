#!/usr/bin/env python3
"""
Tests that each phase can be refined on its own terms, and that a value the
user pins stays pinned.

Two phases in one pattern rarely want the same treatment. Chlorite needs its
asymmetry free for stacking disorder while the quartz beside it is symmetric;
an internal standard weighed into the mount has a scale that is known and must
not be solved for. Until now every flag was one switch for the whole run, and
three separate places overwrote values before the refinement began: the trial
mode, the hand-off between staged refinement's two stages, and the least
squares that seeds the scale factors.
"""

import numpy as np
import pytest

from test_refinement_reporting import (
    BASE_PARAMS, INTEN_A, INTEN_B, PEAKS_A, PEAKS_B, _observed, _phase,
)
from utils.multi_phase_analyzer import MultiPhaseAnalyzer

PHASES = [_phase("Quartz", PEAKS_A, INTEN_A, 4.913),
          _phase("Albite", PEAKS_B, INTEN_B, 8.144)]


def _run(**extra):
    analyzer = MultiPhaseAnalyzer()
    analyzer.perform_lebail_refinement(
        _observed(shift=0.05), PHASES, max_iterations=4,
        refinement_params={**BASE_PARAMS, **extra},
    )
    return analyzer.lebail_engine


def _params(engine, name):
    for phase in engine.phases:
        info = phase["data"].get("phase", {})
        if info.get("mineral") == name:
            return phase["parameters"]
    raise AssertionError(f"{name} not in the refinement")


# --- flags that differ between phases --------------------------------------

def test_one_phase_can_refine_a_term_its_neighbour_holds_fixed():
    engine = _run(phase_overrides={
        "Quartz": {"refine_asymmetry": True},
        "Albite": {"refine_asymmetry": False, "asymmetry": 0.0},
    })
    assert _params(engine, "Quartz")["refine_asymmetry"] is True
    assert _params(engine, "Albite")["refine_asymmetry"] is False
    assert _params(engine, "Albite")["asymmetry"] == pytest.approx(0.0)


def test_a_starting_value_reaches_only_the_phase_it_was_given_to():
    engine = _run(phase_overrides={"Albite": {"microstrain": 2500.0}})
    assert _params(engine, "Albite")["microstrain"] != pytest.approx(
        _params(engine, "Quartz")["microstrain"]
    )


# --- pinning ---------------------------------------------------------------

def test_a_pinned_value_survives_the_refinement():
    engine = _run(phase_overrides={
        "Albite": {"microstrain": 1234.0, "_locked": ["microstrain"]},
    })
    assert _params(engine, "Albite")["microstrain"] == pytest.approx(1234.0)


def test_a_pinned_parameter_is_kept_out_of_the_optimiser():
    engine = _run(phase_overrides={
        "Albite": {"microstrain": 1234.0, "_locked": ["microstrain"]},
    })
    _, _, names = engine._create_parameter_vector(_params(engine, "Albite"))
    assert "microstrain" not in names
    _, _, others = engine._create_parameter_vector(_params(engine, "Quartz"))
    assert "microstrain" in others


def test_pinning_beats_a_refine_flag_that_contradicts_it():
    """Asking to fix and to refine the same term is a contradiction; fixing wins."""
    engine = _run(phase_overrides={
        "Albite": {"microstrain": 900.0, "refine_strain": True,
                   "_locked": ["microstrain"]},
    })
    assert _params(engine, "Albite")["microstrain"] == pytest.approx(900.0)
    _, _, names = engine._create_parameter_vector(_params(engine, "Albite"))
    assert "microstrain" not in names


def test_a_pinned_scale_is_not_solved_away_before_refinement():
    """
    The seeding least squares used to overwrite every scale factor. An internal
    standard weighed into the mount has a known scale, and losing it there would
    discard the one quantity the run was set up to exploit.
    """
    engine = _run(phase_overrides={
        "Albite": {"scale_factor": 0.375, "_locked": ["scale_factor"]},
    })
    assert _params(engine, "Albite")["scale_factor"] == pytest.approx(0.375)


def test_the_other_scales_still_solve_around_a_pinned_one():
    engine = _run(phase_overrides={
        "Albite": {"scale_factor": 0.375, "_locked": ["scale_factor"]},
    })
    assert _params(engine, "Quartz")["scale_factor"] > 0.0


def test_pinning_the_cell_holds_the_lattice():
    engine = _run(phase_overrides={
        "Albite": {"lattice_scale": 1.004, "_locked": ["lattice_scale"]},
    })
    assert _params(engine, "Albite")["lattice_scale"] == pytest.approx(1.004)
    _, _, names = engine._create_parameter_vector(_params(engine, "Albite"))
    assert "lattice_scale" not in names
    _, _, others = engine._create_parameter_vector(_params(engine, "Quartz"))
    assert "lattice_scale" in others


def test_staged_refinement_cannot_hand_back_a_pinned_term():
    """
    Stage 2 switches the profile flags back on for every phase. A pinned term
    has to stay out of the vector across that hand-off, which is why the lock is
    applied where the vector is built rather than at each flag.
    """
    engine = _run(phase_overrides={
        "Albite": {"crystallite_size": 0.5, "_locked": ["crystallite_size"]},
    }, refine_size=True)
    assert _params(engine, "Albite")["crystallite_size"] == pytest.approx(0.5)


def test_nothing_is_pinned_by_default():
    engine = _run()
    for phase in engine.phases:
        assert phase["parameters"]["_locked"] == frozenset()


def test_harmonics_pin_as_one_group():
    engine = _run(intensity_model="fixed", phase_overrides={
        "Albite": {"harmonic_order": 4, "harmonic_coeffs": [0.2, -0.1],
                   "refine_harmonics": True, "_locked": ["harmonic_coeffs"]},
    })
    params = _params(engine, "Albite")
    assert params["harmonic_coeffs"] == pytest.approx([0.2, -0.1])
    _, _, names = engine._create_parameter_vector(params)
    assert not any(n.startswith("harmonic_") for n in names)


# --- the defaults still work ------------------------------------------------

def test_a_run_without_overrides_is_unchanged():
    engine = _run()
    for name in ("Quartz", "Albite"):
        params = _params(engine, name)
        assert params["refine_strain"] is True
        assert params["scale_factor"] > 0.0
