#!/usr/bin/env python3
"""
Tests that each cell parameter refines on its own.

An isotropic dilation can only make a phase larger or smaller as a whole, which
is the wrong shape of answer for most real samples: heating a tetragonal
structure lengthens c and barely touches a, and a solid solution can expand one
axis while contracting another. Refining a, b, c and the angles separately needs
a Miller index per reflection to say how each peak responds to each parameter,
so the pattern here is built from a known cell, distorted by a known amount, and
the refinement is asked to find its way back.

What symmetry fixes must stay fixed. Freeing a and b apart in a tetragonal phase
would be fitting a direction no powder pattern can see, so equal axes have to
move together and a right angle has to stay a right angle.
"""

import numpy as np
import pytest

from utils import unit_cell as uc
from utils.multi_phase_analyzer import MultiPhaseAnalyzer

WAVELENGTH = 1.5406

TETRAGONAL = {'a': 5.0, 'b': 5.0, 'c': 8.3,
              'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
MONOCLINIC = {'a': 5.68, 'b': 15.2, 'c': 6.52,
              'alpha': 90.0, 'beta': 118.4, 'gamma': 90.0}
CUBIC = {'a': 5.64, 'b': 5.64, 'c': 5.64,
         'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}


# --- a pattern with a cell behind it ---------------------------------------

def _two_theta(d):
    return 2.0 * np.degrees(np.arcsin(np.clip(WAVELENGTH / (2.0 * np.asarray(d)),
                                              -1.0, 1.0)))


def _reflections(cell, limit=4, low=16.0, high=68.0, separation=0.35):
    """Well separated reflections of a cell, in the window a run would fit."""
    hkl = np.array([[h, k, l]
                    for h in range(limit) for k in range(limit) for l in range(limit + 1)
                    if (h, k, l) != (0, 0, 0)])
    d = uc.d_spacings(hkl, cell)
    two_theta = _two_theta(d)
    inside = (two_theta > low) & (two_theta < high)
    hkl, d, two_theta = hkl[inside], d[inside], two_theta[inside]

    order = np.argsort(two_theta)
    keep = []
    for index in order:
        if not keep or two_theta[index] - two_theta[keep[-1]] > separation:
            keep.append(index)
    return hkl[keep], d[keep], two_theta[keep]


def _intensities(count):
    # Falling away with angle, as a real pattern does, but with enough variety
    # that no two reflections are interchangeable
    steps = np.arange(count, dtype=float)
    return 100.0 * np.exp(-steps / 7.0) * (1.0 + 0.3 * np.cos(steps))


def _phase(name, cell, drop_d_spacing=False, rir=1.0):
    """A reference phase whose stored d-spacings match its peak positions."""
    hkl, d, two_theta = _reflections(cell)
    peaks = {
        "two_theta": two_theta,
        "intensity": _intensities(len(two_theta)),
    }
    if not drop_d_spacing:
        # Published tables round; the indexing has to survive that
        peaks["d_spacing"] = np.round(d, 4)
    return {
        "phase": {
            "mineral": name, "formula": "X", "rir": rir,
            "cell_a": cell['a'], "cell_b": cell['b'], "cell_c": cell['c'],
            "cell_alpha": cell['alpha'], "cell_beta": cell['beta'],
            "cell_gamma": cell['gamma'],
        },
        "theoretical_peaks": peaks,
        "optimized_scaling": 1.0,
        "_hkl": hkl,
    }


def _observed(cell, reference, width=0.08):
    """The pattern this cell would give, on the reference phase's intensities."""
    hkl = reference["_hkl"]
    intensity = np.asarray(reference["theoretical_peaks"]["intensity"], dtype=float)
    centres = _two_theta(uc.d_spacings(hkl, cell))

    two_theta = np.linspace(15.0, 70.0, 4000)
    pattern = np.zeros_like(two_theta)
    for centre, height in zip(centres, intensity):
        pattern += height * np.exp(-0.5 * ((two_theta - centre) / width) ** 2)
    return {"two_theta": two_theta, "intensity": pattern, "wavelength": WAVELENGTH}


# The peaks above are Gaussian, so the sample broadening terms start at nothing:
# a Lorentzian tail the data does not have would blunt every position the cell
# is being read from.
PARAMS = {
    "initial_u": 0.0, "initial_v": 0.0, "initial_w": 0.0064,
    "microstrain": 0.0, "crystallite_size": 10.0,
    "intensity_model": "fixed", "refine_cell": True, "refine_profile": False,
    "refine_strain": False, "refine_zero_shift": False,
    "refine_displacement": False, "max_scale": 100.0,
}


def _refine(phases, observed, **extra):
    analyzer = MultiPhaseAnalyzer()
    results = analyzer.perform_lebail_refinement(
        observed, phases, max_iterations=6,
        refinement_params={**PARAMS, **extra},
    )
    return analyzer.lebail_engine, results


def _row(results, name):
    for row in results["refinement_results"]["phase_summary"]:
        if row["name"] == name:
            return row
    raise AssertionError(f"{name} not in the results")


def _names(engine, index=0):
    _, _, names = engine._create_parameter_vector(engine.phases[index]['parameters'])
    return names


# --- indexing the reference reflections ------------------------------------

def test_reference_reflections_are_indexed_from_their_d_spacings():
    phase = _phase("Tetra", TETRAGONAL)
    engine, _ = _refine([phase], _observed(TETRAGONAL, phase))

    hkl = engine.phases[0]['hkl']
    assert hkl is not None
    assert len(hkl) == len(phase["theoretical_peaks"]["two_theta"])
    # Recovered indices must give back the d-spacings they were matched to
    recovered = uc.d_spacings(hkl, TETRAGONAL)
    assert recovered == pytest.approx(
        phase["theoretical_peaks"]["d_spacing"], abs=5e-4
    )


def test_a_pattern_without_d_spacings_falls_back_to_one_dilation():
    """
    The reference patterns are not all indexable, and that has to stay safe.

    Without a d-spacing to match there is no way to know which reflection is
    which, so the cell may only breathe as a whole rather than be refined on a
    guess about its axes.
    """
    phase = _phase("Tetra", TETRAGONAL, drop_d_spacing=True)
    engine, results = _refine([phase], _observed(TETRAGONAL, phase))

    assert engine.phases[0]['hkl'] is None
    assert "lattice_scale" in _names(engine)
    assert not any(name.startswith("cell_") for name in _names(engine))
    assert _row(results, "Tetra")["cell_free"] == []


def test_d_spacings_that_belong_to_another_cell_are_refused():
    phase = _phase("Tetra", TETRAGONAL)
    peaks = phase["theoretical_peaks"]
    peaks["d_spacing"] = np.full(len(peaks["two_theta"]), 2.0)

    engine, _ = _refine([phase], _observed(TETRAGONAL, phase))
    assert engine.phases[0]['hkl'] is None
    assert "lattice_scale" in _names(engine)


# --- what symmetry allows to move ------------------------------------------

def test_symmetry_decides_which_parameters_are_free():
    for cell, expected in (
        (CUBIC, ["cell_a"]),
        (TETRAGONAL, ["cell_a", "cell_c"]),
        (MONOCLINIC, ["cell_a", "cell_b", "cell_c", "cell_beta"]),
    ):
        phase = _phase("P", cell)
        engine, _ = _refine([phase], _observed(cell, phase))
        free = [name for name in _names(engine) if name.startswith("cell_")]
        assert free == expected, f"{cell} gave {free}"


def test_equal_axes_stay_equal_and_right_angles_stay_right():
    phase = _phase("Tetra", TETRAGONAL)
    truth = dict(TETRAGONAL, a=5.03, b=5.03, c=8.25)
    _, results = _refine([phase], _observed(truth, phase))

    cell = _row(results, "Tetra")["unit_cell"]
    assert cell["a"] == pytest.approx(cell["b"], rel=1e-9)
    for angle in ("alpha", "beta", "gamma"):
        assert cell[angle] == pytest.approx(90.0)


# --- refining the axes apart ------------------------------------------------

def test_one_axis_can_grow_while_another_shrinks():
    """The reported case: a is larger than the reference while c is smaller."""
    phase = _phase("Tetra", TETRAGONAL)
    truth = dict(TETRAGONAL, a=5.0 * 1.006, b=5.0 * 1.006, c=8.3 * 0.994)
    _, results = _refine([phase], _observed(truth, phase))

    row = _row(results, "Tetra")
    assert row["unit_cell"]["a"] == pytest.approx(truth["a"], rel=5e-4)
    assert row["unit_cell"]["c"] == pytest.approx(truth["c"], rel=5e-4)

    delta = row["cell_delta"]
    assert delta["a"] > 0.3 and delta["c"] < -0.3
    # A dilation could not have found this: the two axes moved opposite ways
    assert delta["a"] * delta["c"] < 0


def test_the_axes_beat_a_single_dilation_on_the_same_pattern():
    """
    An anisotropic distortion is not something one number can absorb.

    Fitting it with a dilation leaves the peaks systematically misplaced, so the
    comparison is on Rwp: if refining the axes apart did not improve the fit, it
    is not doing anything.
    """
    truth = dict(TETRAGONAL, a=5.0 * 1.006, b=5.0 * 1.006, c=8.3 * 0.994)
    observed = _observed(truth, _phase("Tetra", TETRAGONAL))

    _, anisotropic = _refine([_phase("Tetra", TETRAGONAL)], observed)
    _, isotropic = _refine(
        [_phase("Tetra", TETRAGONAL, drop_d_spacing=True)], observed
    )

    better = anisotropic["refinement_results"]["final_r_factors"]["Rwp"]
    worse = isotropic["refinement_results"]["final_r_factors"]["Rwp"]
    assert better < worse, f"Rwp {better:.3f} vs {worse:.3f}"
    # And the axes actually went opposite ways, which no single dilation can do
    delta = _row(anisotropic, "Tetra")["cell_delta"]
    assert delta["a"] * delta["c"] < 0


def test_a_monoclinic_angle_refines():
    phase = _phase("Mono", MONOCLINIC)
    truth = dict(MONOCLINIC, beta=118.4 + 0.25)
    _, results = _refine([phase], _observed(truth, phase))

    row = _row(results, "Mono")
    assert row["unit_cell"]["beta"] == pytest.approx(truth["beta"], abs=0.05)
    assert row["unit_cell"]["alpha"] == pytest.approx(90.0)
    assert row["unit_cell"]["gamma"] == pytest.approx(90.0)
    # The edges must not have wandered off to soak up the angle change
    assert row["unit_cell"]["a"] == pytest.approx(MONOCLINIC["a"], rel=1e-3)
    assert row["unit_cell"]["c"] == pytest.approx(MONOCLINIC["c"], rel=1e-3)


def test_a_cubic_cell_still_refines_as_one_number():
    phase = _phase("Cubic", CUBIC)
    truth = dict(CUBIC, a=5.64 * 1.004, b=5.64 * 1.004, c=5.64 * 1.004)
    _, results = _refine([phase], _observed(truth, phase))

    cell = _row(results, "Cubic")["unit_cell"]
    assert cell["a"] == pytest.approx(truth["a"], rel=5e-4)
    assert cell["a"] == pytest.approx(cell["b"]) == pytest.approx(cell["c"])


# --- holding one parameter while the others refine -------------------------

def test_one_axis_can_be_held_while_the_rest_refine():
    phase = _phase("Tetra", TETRAGONAL)
    truth = dict(TETRAGONAL, a=5.0 * 1.006, b=5.0 * 1.006, c=8.3 * 0.994)
    engine, results = _refine(
        [phase], _observed(truth, phase),
        phase_overrides={"Tetra": {"cell_c": 8.3, "_locked": ["cell_c"]}},
    )

    assert "cell_c" not in _names(engine)
    assert "cell_a" in _names(engine)
    row = _row(results, "Tetra")
    assert row["unit_cell"]["c"] == pytest.approx(8.3)
    assert row["unit_cell"]["a"] > TETRAGONAL["a"]


def test_a_typed_cell_edge_is_where_the_refinement_starts():
    phase = _phase("Tetra", TETRAGONAL)
    engine, _ = _refine(
        [phase], _observed(TETRAGONAL, phase),
        phase_overrides={"Tetra": {"cell_a": 5.02, "_locked": ["cell_a", "cell_c"]}},
    )
    params = engine.phases[0]['parameters']
    assert params["unit_cell"]["a"] == pytest.approx(5.02)
    assert params["unit_cell"]["b"] == pytest.approx(5.02), "the tie holds"
    assert params["_base_unit_cell"]["a"] == pytest.approx(5.0), (
        "the starting cell stays the one the reference positions belong to"
    )


# --- carrying the cell to the next run -------------------------------------

def test_the_refined_cell_starts_the_next_run():
    phase = _phase("Tetra", TETRAGONAL)
    truth = dict(TETRAGONAL, a=5.0 * 1.006, b=5.0 * 1.006, c=8.3 * 0.994)
    observed = _observed(truth, phase)

    _, first = _refine([phase], observed)
    refined = _row(first, "Tetra")["unit_cell"]

    engine, second = _refine(
        [phase], observed, refine_cell=False,
        carry_over={"Tetra": {"unit_cell": refined}},
    )
    row = _row(second, "Tetra")
    assert row["unit_cell"]["a"] == pytest.approx(refined["a"], rel=1e-9)
    assert row["unit_cell"]["c"] == pytest.approx(refined["c"], rel=1e-9)
    assert row["base_unit_cell"]["a"] == pytest.approx(5.0)


def test_an_old_isotropic_dilation_is_taken_up_by_the_cell():
    """
    A run saved before the axes were separable carries one dilation factor.

    It still means something -- the cell was that much larger -- so it is folded
    into the cell rather than dropped, and not applied twice.
    """
    phase = _phase("Tetra", TETRAGONAL)
    engine, results = _refine(
        [phase], _observed(TETRAGONAL, phase), refine_cell=False,
        carry_over={"Tetra": {"lattice_scale": 1.004}},
    )
    row = _row(results, "Tetra")
    assert row["unit_cell"]["a"] == pytest.approx(5.0 * 1.004, rel=1e-9)
    assert row["unit_cell"]["c"] == pytest.approx(8.3 * 1.004, rel=1e-9)
    assert engine.phases[0]['parameters']["lattice_scale"] == pytest.approx(1.0)
