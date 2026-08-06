#!/usr/bin/env python3
"""
Tests for modelling the Kα doublet instead of half of it.

A lab source emits two Kα lines, so every reflection appears twice: a Kα1 line
and a Kα2 satellite at slightly higher 2θ with about half the intensity. Data
that still contain the satellites, fitted with a model that has only one line per
reflection, cannot be fitted properly at all — the missing intensity has to be
made up out of width and skew, so every peak comes out too wide and leaning
towards high 2θ, and the residual keeps the shape of a peak.

The satellite belongs to its parent, not beside it. It has no intensity of its
own to extract and no cell parameter to refine, so it is built into the parent's
profile and the reflection list, the extracted intensities and the report all
stay one entry per reflection. These tests hold that line: the doublet has to
improve the fit of data that has it, stay out of the way of data that does not,
and change nothing about the bookkeeping either way.
"""

import numpy as np
import pytest

from utils import kalpha_filter as kalpha
from utils import unit_cell as uc
from utils.multi_phase_analyzer import MultiPhaseAnalyzer

# Kα1 of copper: with satellites modelled the parent line belongs here, not at
# the doublet average the pattern header often carries
WAVELENGTH = 1.540562
WAVELENGTH_RATIO = kalpha.alpha2_ratio(WAVELENGTH)

CUBIC = {'a': 5.64, 'b': 5.64, 'c': 5.64,
         'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
TETRAGONAL = {'a': 5.0, 'b': 5.0, 'c': 8.3,
              'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}

WIDTH = 0.10  # Gaussian FWHM of the synthetic peaks, in degrees


def _two_theta(d):
    return 2.0 * np.degrees(np.arcsin(np.clip(WAVELENGTH / (2.0 * np.asarray(d)),
                                              -1.0, 1.0)))


def _reflections(cell, limit=4, low=18.0, high=75.0, separation=0.6):
    hkl = np.array([[h, k, l]
                    for h in range(limit) for k in range(limit) for l in range(limit + 1)
                    if (h, k, l) != (0, 0, 0)])
    two_theta = _two_theta(uc.d_spacings(hkl, cell))
    inside = (two_theta > low) & (two_theta < high)
    hkl, two_theta = hkl[inside], two_theta[inside]

    order = np.argsort(two_theta)
    keep = []
    for index in order:
        if not keep or two_theta[index] - two_theta[keep[-1]] > separation:
            keep.append(index)
    return hkl[keep], two_theta[keep]


def _phase(name, cell, rir=1.0):
    hkl, two_theta = _reflections(cell)
    steps = np.arange(len(two_theta), dtype=float)
    return {
        "phase": {
            "mineral": name, "formula": "X", "rir": rir,
            "cell_a": cell['a'], "cell_b": cell['b'], "cell_c": cell['c'],
            "cell_alpha": cell['alpha'], "cell_beta": cell['beta'],
            "cell_gamma": cell['gamma'],
        },
        "theoretical_peaks": {
            "two_theta": two_theta,
            "intensity": 100.0 * np.exp(-steps / 8.0) * (1.0 + 0.3 * np.cos(steps)),
            "d_spacing": np.round(uc.d_spacings(hkl, cell), 4),
        },
        "optimized_scaling": 1.0,
    }


def _observed(*phases_and_scales, alpha2=0.0):
    """
    A pattern from one or more phases, optionally with Kα2 satellites.

    Peaks are Gaussian of a fixed width, so a fit that comes out wider than
    WIDTH has taken the satellite's intensity into the width — which is exactly
    what a model without the satellite must do.
    """
    two_theta = np.linspace(15.0, 80.0, 6500)
    pattern = np.zeros_like(two_theta)
    sigma = WIDTH / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def add(centre, height):
        return height * np.exp(-0.5 * ((two_theta - centre) / sigma) ** 2)

    for phase, scale in phases_and_scales:
        peaks = phase["theoretical_peaks"]
        for centre, height in zip(peaks["two_theta"], peaks["intensity"]):
            pattern += add(centre, scale * height)
            if alpha2 > 0:
                offset = float(kalpha.alpha2_separation(centre, WAVELENGTH_RATIO))
                pattern += add(centre + offset, alpha2 * scale * height)

    return {"two_theta": two_theta, "intensity": pattern, "wavelength": WAVELENGTH}


PARAMS = {
    "initial_u": 0.0, "initial_v": 0.0, "initial_w": WIDTH ** 2,
    "microstrain": 0.0, "crystallite_size": 10.0,
    "intensity_model": "fixed", "refine_cell": False, "refine_profile": False,
    "refine_strain": False, "refine_zero_shift": False,
    "refine_displacement": False, "max_scale": 100.0,
}


def _refine(phases, observed, **extra):
    analyzer = MultiPhaseAnalyzer()
    results = analyzer.perform_lebail_refinement(
        observed, phases, max_iterations=4,
        refinement_params={**PARAMS, **extra},
    )
    return analyzer.lebail_engine, results


def _rwp(results):
    return results["r_factors"]["Rwp"]


# --- where the satellite goes ----------------------------------------------

def test_the_satellite_sits_where_bragg_puts_it():
    """
    Both lines see the same d-spacing, so the satellite's angle follows from the
    wavelength ratio alone, and the separation grows as tan theta.
    """
    angles = np.array([20.0, 36.0, 60.0, 90.0])
    separation = kalpha.alpha2_separation(angles, WAVELENGTH_RATIO)

    expected = 2.0 * np.degrees(
        np.arcsin(np.sin(np.radians(angles / 2.0)) * WAVELENGTH_RATIO)
    ) - angles
    assert separation == pytest.approx(expected)
    assert np.all(np.diff(separation) > 0), "separation has to grow with angle"
    # A tenth of a degree in the middle of a lab pattern, three tenths by 90,
    # which is past a peak width and is why the satellites resolve at high angle
    assert separation[1] == pytest.approx(0.092, abs=0.005)
    assert separation[3] == pytest.approx(0.287, abs=0.01)


def test_the_engine_puts_the_satellite_at_the_same_place():
    engine, _ = _refine([_phase("A", CUBIC)], _observed((_phase("A", CUBIC), 1.0)))
    angles = np.array([25.0, 45.0, 70.0])
    assert engine._alpha2_separation(angles) == pytest.approx(
        kalpha.alpha2_separation(angles, WAVELENGTH_RATIO)
    )


# --- fitting data that has satellites in it --------------------------------

def test_modelling_the_doublet_fits_doublet_data_better():
    phase = _phase("A", CUBIC)
    observed = _observed((phase, 1.0), alpha2=0.5)

    _, without = _refine([_phase("A", CUBIC)], observed, alpha2_ratio=0.0)
    _, with_it = _refine([_phase("A", CUBIC)], observed, alpha2_ratio=0.5)

    assert _rwp(with_it) < _rwp(without) / 2.0, (
        f"Rwp {_rwp(with_it):.2f}% with the doublet against "
        f"{_rwp(without):.2f}% without it"
    )
    assert _rwp(with_it) < 2.0


def test_the_ratio_refines_to_the_one_in_the_data():
    phase = _phase("A", CUBIC)
    observed = _observed((phase, 1.0), alpha2=0.5)

    engine, _ = _refine([_phase("A", CUBIC)], observed,
                        alpha2_ratio=0.2, refine_alpha2_ratio=True)
    assert engine.global_parameters['alpha2_ratio'] == pytest.approx(0.5, abs=0.05)


def test_stripped_data_refines_the_satellite_away():
    """
    Monochromated or already-stripped data must not be given a satellite it does
    not have, and the refinement has to be able to find that out.
    """
    phase = _phase("A", CUBIC)
    observed = _observed((phase, 1.0), alpha2=0.0)

    engine, results = _refine([_phase("A", CUBIC)], observed,
                              alpha2_ratio=0.4, refine_alpha2_ratio=True)
    assert engine.global_parameters['alpha2_ratio'] < 0.05
    assert _rwp(results) < 2.0


def test_a_pattern_without_satellites_is_unchanged_by_the_setting_being_off():
    """The default has to be exactly the old single-line behaviour."""
    phase = _phase("A", CUBIC)
    observed = _observed((phase, 1.0))

    engine, results = _refine([_phase("A", CUBIC)], observed)
    assert engine.global_parameters['alpha2_ratio'] == 0.0
    assert _rwp(results) < 2.0


# --- what the satellite must not disturb -----------------------------------

def test_the_satellite_is_not_a_reflection_in_its_own_right():
    """
    One entry per reflection, doublet or not. The satellite has no intensity to
    extract and no index of its own, so anything counted per reflection --
    extracted intensities, the design matrix, the report -- has to keep its
    length.
    """
    phase = _phase("A", CUBIC)
    observed = _observed((phase, 1.0), alpha2=0.5)
    expected = len(phase["theoretical_peaks"]["two_theta"])

    for ratio in (0.0, 0.5):
        engine, results = _refine([_phase("A", CUBIC)], observed, alpha2_ratio=ratio)
        positions, intensities, _ = engine._phase_peaks(engine.phases[0])
        assert len(positions) == expected
        assert len(intensities) == expected
        extracted = engine.phases[0].get('_extracted_intensities')
        if extracted is not None:
            assert len(extracted) == expected
        assert len(results["refinement_results"]["phase_summary"]) == 1


def test_the_doublet_does_not_move_the_weight_percents():
    """
    Both lines of the doublet scale with the same reflection intensity, so
    modelling them multiplies every phase's peak areas by the same factor and the
    proportions between phases are left alone. If that were not so, turning the
    setting on would silently change every quantification.
    """
    major, minor = _phase("Major", CUBIC), _phase("Minor", TETRAGONAL)
    observed = _observed((major, 1.0), (minor, 0.25), alpha2=0.5)

    def weights(**extra):
        _, results = _refine([_phase("Major", CUBIC), _phase("Minor", TETRAGONAL)],
                             observed, **extra)
        return {row["name"]: row["weight_percent"]
                for row in results["refinement_results"]["phase_summary"]}

    without = weights(alpha2_ratio=0.0)
    with_it = weights(alpha2_ratio=0.5)

    assert with_it["Major"] == pytest.approx(without["Major"], abs=2.0)
    assert with_it["Minor"] == pytest.approx(without["Minor"], abs=2.0)
    assert with_it["Major"] + with_it["Minor"] == pytest.approx(100.0, abs=0.1)


def test_the_report_says_whether_satellites_were_modelled():
    phase = _phase("A", CUBIC)
    observed = _observed((phase, 1.0), alpha2=0.5)
    _, results = _refine([_phase("A", CUBIC)], observed,
                         alpha2_ratio=0.5, refine_alpha2_ratio=True)

    globals_ = results["refinement_results"]["global_parameters"]
    assert globals_["alpha2_ratio"] > 0.0
    assert globals_["refine_alpha2_ratio"] is True
