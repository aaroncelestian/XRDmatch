#!/usr/bin/env python3
"""
Tests that the weight percent depends on how much of a phase is present and
not on how wide its peaks came out.

I/Ic is defined on integrated intensities, but a refined scale factor gives a
peak height, and the two are proportional only while the width is held still.
Crystallite size and microstrain refine per phase, so a phase whose peaks come
out narrower gets a taller strongest line at unchanged area. Quantifying from
the height then reads that narrowness as abundance. On a sand spiked with a
known 10 wt% of forsterite, two refinements of nearly equal quality reported
25.7 and 8.3 wt% because their forsterite widths differed by a factor of two.

The mixtures below are built so the true Chung relation holds exactly: each
phase's scale is chosen to put an integrated strongest-line intensity of
w x RIR into the pattern. Anything the summary reports other than w is the
method's own error.
"""

import numpy as np
import pytest

from utils.lebail_refinement import LeBailRefinement

WAVELENGTH = 1.5406

PEAKS = {
    "Forsterite": (np.array([22.87, 32.30, 35.72, 36.55, 39.70, 52.20, 62.60]),
                   np.array([40.0, 55.0, 60.0, 100.0, 35.0, 25.0, 20.0]), 0.787),
    "Quartz": (np.array([20.86, 26.64, 36.54, 39.47, 42.45, 50.14, 59.96]),
               np.array([16.0, 100.0, 7.0, 8.0, 4.0, 14.0, 9.0]), 4.329),
    "Albite": (np.array([22.05, 23.55, 27.42, 27.94, 30.30, 35.30, 51.10]),
               np.array([25.0, 40.0, 60.0, 100.0, 30.0, 20.0, 15.0]), 0.600),
}


def _engine():
    engine = LeBailRefinement()
    engine.quiet = True
    engine.intensity_model = "fixed"
    two_theta = np.linspace(15.0, 70.0, 5500)
    engine.set_experimental_data(two_theta, np.ones_like(two_theta),
                                 wavelength=WAVELENGTH)
    return engine


def _add(engine, name, size, strain, scale=1.0):
    positions, intensities, rir = PEAKS[name]
    engine.add_phase(
        {"phase": {"mineral": name, "rir": rir},
         "theoretical_peaks": {"two_theta": positions, "intensity": intensities,
                               "d_spacing": np.full(len(positions), 2.0)}},
        {"scale_factor": scale, "crystallite_size": size, "microstrain": strain},
    )


def _mixture(weights, widths):
    """
    Build a mixture holding exactly `weights`, with each phase given its own
    peak width. The scale factors are solved so that the integrated intensity
    of each strongest line is w x RIR, which is the Chung relation itself.
    """
    engine = _engine()
    for name in weights:
        size, strain = widths[name]
        _add(engine, name, size, strain)

    for index, name in enumerate(weights):
        area, _ = engine._strongest_line_area(index)
        _, intensities, rir = PEAKS[name]
        reference_max = float(np.max(intensities))
        engine.phases[index]["parameters"]["scale_factor"] = (
            weights[name] * rir / (reference_max * area)
        )
    return engine


def _reported(engine):
    return {row["name"]: row["weight_percent"] for row in engine.phase_summary()}


# --- the property that was broken ------------------------------------------

EQUAL = {"Forsterite": (0.05, 500.0), "Quartz": (0.05, 500.0), "Albite": (0.05, 500.0)}
UNEQUAL = {"Forsterite": (0.038, 0.0), "Quartz": (0.185, 986.0), "Albite": (0.075, 154.0)}
SPIKE = {"Forsterite": 10.0, "Quartz": 60.0, "Albite": 30.0}


def test_weights_are_recovered_when_the_widths_match():
    reported = _reported(_mixture(SPIKE, EQUAL))
    for name, truth in SPIKE.items():
        assert reported[name] == pytest.approx(truth, abs=0.1)


def test_weights_are_recovered_when_the_widths_differ():
    """
    The real case: each phase refined to its own width. This is what the two
    exports from the spiked sand disagreed over.
    """
    reported = _reported(_mixture(SPIKE, UNEQUAL))
    for name, truth in SPIKE.items():
        assert reported[name] == pytest.approx(truth, abs=0.1)


def test_widening_one_phase_does_not_change_what_it_weighs():
    """
    Take a converged mixture and broaden one phase at constant integrated
    intensity. Its abundance must not move. Quantifying from the peak height
    would report it as less abundant purely for having spread out.
    """
    narrow = _mixture(SPIKE, EQUAL)
    before = _reported(narrow)

    index = list(SPIKE).index("Forsterite")
    params = narrow.phases[index]["parameters"]
    area_before, _ = narrow._strongest_line_area(index)
    params["microstrain"] = 4000.0                     # much broader peaks
    area_after, _ = narrow._strongest_line_area(index)
    params["scale_factor"] *= area_before / area_after  # same integrated intensity

    assert area_after > 1.5 * area_before, "the phase should have broadened"
    after = _reported(narrow)
    assert after["Forsterite"] == pytest.approx(before["Forsterite"], abs=0.1)


def test_quantifying_from_peak_height_would_get_it_wrong():
    """
    Guards the fix by showing the discarded formula fails the same mixture,
    so a regression to peak heights cannot pass silently.
    """
    engine = _mixture(SPIKE, UNEQUAL)
    rows = engine.phase_summary()
    terms = {r["name"]: r["line_intensity"] / r["rir"] for r in rows}
    total = sum(terms.values())
    by_height = {n: 100.0 * t / total for n, t in terms.items()}
    assert abs(by_height["Forsterite"] - SPIKE["Forsterite"]) > 3.0


# --- the width diagnostic --------------------------------------------------

def test_sample_terms_carrying_the_width_are_reported():
    """
    A crystallite size that is really standing in for an uncalibrated
    instrument profile should be visible as such.
    """
    engine = _mixture(SPIKE, UNEQUAL)
    shares = {r["name"]: r["sample_width_share"] for r in engine.phase_summary()}
    assert all(0.0 <= s <= 1.0 for s in shares.values())
    assert shares["Quartz"] > 0.2  # 0.185 um and 986 microstrain is mostly sample


def test_a_phase_with_no_sample_broadening_reports_none():
    engine = _engine()
    _add(engine, "Quartz", 0.0, 0.0)
    _, share = engine._strongest_line_area(0)
    assert share == pytest.approx(0.0, abs=1e-6)


def test_summary_survives_a_phase_with_no_peaks_in_range():
    engine = _engine()
    engine.add_phase(
        {"phase": {"mineral": "Ghost", "rir": 1.0},
         "theoretical_peaks": {"two_theta": np.array([]), "intensity": np.array([]),
                               "d_spacing": np.array([])}},
        {"scale_factor": 1.0},
    )
    assert engine.phase_summary()[0]["line_area"] == 0.0
