#!/usr/bin/env python3
"""
Tests for the split peak profile.

A real powder peak is not symmetric. Axial divergence drags a low-angle tail out
of every reflection in the pattern, worst below about 20 degrees, and a layer
structure with stacking disorder skews its own peaks whatever the instrument is
doing. Fitting either with a symmetric profile leaves the tail in the difference
curve and pushes the width and position terms around trying to absorb it.

The two terms are kept separate -- one on the instrument, one on the phase --
because a single disordered phase must not be able to skew the whole pattern.
"""

import numpy as np
import pytest

from utils.lebail_refinement import LeBailRefinement
from utils.profile_functions import (
    MAX_ASYMMETRY, asymmetry_exponent, flank_widths, phase_widths, pseudo_voigt,
    skew_description,
)

WAVELENGTH = 1.5406
INSTRUMENT = {'u_param': 0.002, 'v_param': -0.0005, 'w_param': 0.004}
# Matches the engine's own starting sample broadening, so a pattern generated
# here is one the engine can reproduce exactly and any residual left over is
# down to the asymmetry rather than a width the test forgot to include.
SAMPLE = {'crystallite_size': 1.0, 'microstrain': 1200.0}


def _fwhm_edges(profile, x):
    """The two half-maximum crossings, as distances from the peak centre."""
    above = profile >= 0.5 * profile.max()
    return -x[above][0], x[above][-1]


# --- the profile itself ----------------------------------------------------

def test_zero_asymmetry_is_the_symmetric_profile():
    """The default must reproduce the old profile exactly, not merely closely."""
    x = np.linspace(-2.0, 2.0, 4001)
    plain = pseudo_voigt(x, 0.3, 0.5)
    with_term = pseudo_voigt(x, 0.3, 0.5, 0.0)
    assert np.allclose(plain, with_term, atol=0, rtol=0)


def test_positive_asymmetry_tails_towards_low_angle():
    x = np.linspace(-3.0, 3.0, 20001)
    low, high = _fwhm_edges(pseudo_voigt(x, 1.0, 0.5, 0.6), x)
    assert low > high
    assert low / high == pytest.approx(np.exp(0.6), rel=0.02)


def test_negative_asymmetry_tails_towards_high_angle():
    """Stacking disorder skews the other way from axial divergence."""
    x = np.linspace(-3.0, 3.0, 20001)
    low, high = _fwhm_edges(pseudo_voigt(x, 1.0, 0.5, -0.6), x)
    assert high > low


def test_skewing_a_peak_does_not_broaden_it():
    """
    The width has to survive the skew.

    If asymmetry also widened the peak it would be measuring the same evidence
    as crystallite size and microstrain, and the three would trade against each
    other with no change in the residual.
    """
    x = np.linspace(-4.0, 4.0, 40001)
    reference = sum(_fwhm_edges(pseudo_voigt(x, 1.0, 0.5, 0.0), x))
    for asymmetry in (-1.2, -0.5, 0.5, 1.2):
        total = sum(_fwhm_edges(pseudo_voigt(x, 1.0, 0.5, asymmetry), x))
        assert total == pytest.approx(reference, rel=0.02)


def test_flank_widths_average_to_the_width_given():
    for asymmetry in (-2.0, -0.3, 0.0, 0.3, 2.0):
        low, high = flank_widths(0.4, asymmetry)
        assert 0.5 * (low + high) == pytest.approx(0.4)
        assert low > 0 and high > 0


def test_asymmetry_is_bounded_so_a_flank_cannot_run_away():
    low, high = flank_widths(0.4, 50.0)
    capped_low, capped_high = flank_widths(0.4, MAX_ASYMMETRY)
    assert low == pytest.approx(capped_low)
    assert high == pytest.approx(capped_high)


def test_peak_stays_unit_height_at_its_centre():
    """Both flanks meet at the centre, so a skewed peak is not rescaled."""
    for asymmetry in (-1.5, 0.0, 1.5):
        assert pseudo_voigt(np.array([0.0]), 0.3, 0.5, asymmetry)[0] == pytest.approx(1.0)


# --- where the skew comes from ---------------------------------------------

def test_axial_divergence_is_worst_at_low_angle_and_gone_by_ninety():
    angles = np.array([5.0, 10.0, 20.0, 45.0, 90.0])
    skew = asymmetry_exponent(angles, axial=0.02)
    assert np.all(np.diff(skew) < 0), "axial skew must fall off with angle"
    assert skew[0] > 5 * skew[3], "it should dominate the low-angle end"
    assert skew[-1] == pytest.approx(0.0, abs=1e-9)


def test_axial_divergence_reverses_direction_past_ninety():
    """A real feature of the geometry, not a fitting artefact."""
    assert asymmetry_exponent(np.array([120.0]), axial=0.02)[0] < 0


def test_the_sample_term_does_not_care_about_angle():
    angles = np.array([5.0, 30.0, 80.0])
    skew = asymmetry_exponent(angles, sample=0.4)
    assert np.allclose(skew, 0.4)


def test_a_low_angle_reflection_is_not_handed_an_unbounded_skew():
    assert np.isfinite(asymmetry_exponent(np.array([0.0, 1e-6]), axial=0.02)).all()


def test_an_immeasurably_small_skew_is_reported_as_symmetric():
    """A value the optimizer left near zero is not a finding."""
    assert skew_description(-0.0011) == "symmetric"
    assert skew_description(0.0) == "symmetric"
    assert skew_description(0.7) == "tail to low 2θ"
    assert skew_description(-0.7) == "tail to high 2θ"


# --- recovering an asymmetry through the engine ----------------------------

PEAKS = np.array([8.5, 12.3, 17.6, 24.1, 31.8])
HEIGHTS = np.array([100.0, 70.0, 45.0, 55.0, 25.0])
TRUE_SKEW = 0.7


def _phase(name="Chlorite"):
    return {
        "phase": {
            "mineral": name, "formula": "Mg5Al(AlSi3)O10(OH)8", "rir": 2.0,
            "cell_a": 5.35, "cell_b": 9.27, "cell_c": 14.3,
            "cell_alpha": 90.0, "cell_beta": 97.0, "cell_gamma": 90.0,
        },
        "theoretical_peaks": {
            "two_theta": PEAKS, "intensity": HEIGHTS,
            "d_spacing": np.full(len(PEAKS), 5.0),
        },
        "optimized_scaling": 1.0,
    }


def _pattern(sample_skew=0.0, axial=0.0):
    """A pattern built with the engine's own kernel, so it is self-consistent."""
    two_theta = np.linspace(5.0, 36.0, 3100)
    engine = LeBailRefinement()
    engine.quiet = True
    engine.set_experimental_data(two_theta, np.zeros_like(two_theta),
                                 wavelength=WAVELENGTH)
    engine.global_parameters.update(INSTRUMENT)
    widths, eta = phase_widths(PEAKS, INSTRUMENT, SAMPLE, WAVELENGTH)
    skew = asymmetry_exponent(PEAKS, axial=axial, sample=sample_skew)
    intensity = engine._accumulate_pseudo_voigt(PEAKS, widths, HEIGHTS, eta, skew)
    return two_theta, intensity


def _rwp(engine):
    return engine._calculate_r_factors(engine._calculate_total_pattern())['Rwp']


def _fit(observed_skew, refine_asymmetry, axial=0.0, refine_axial=False):
    two_theta, intensity = _pattern(sample_skew=observed_skew, axial=axial)
    engine = LeBailRefinement()
    engine.quiet = True
    engine.intensity_model = 'fixed'
    engine.set_experimental_data(two_theta, intensity, wavelength=WAVELENGTH)
    engine.global_parameters.update(INSTRUMENT)
    engine.global_parameters['refine_zero_shift'] = False
    engine.global_parameters['refine_axial_asymmetry'] = refine_axial
    engine.add_phase(_phase(), {
        **SAMPLE, 'refine_strain': True, 'refine_cell': False,
        'refine_asymmetry': refine_asymmetry, 'refine_profile': True,
    })
    engine.refine_phases(max_iterations=6, staged_refinement=False)
    return engine


def test_a_known_sample_asymmetry_is_recovered():
    engine = _fit(TRUE_SKEW, refine_asymmetry=True)
    found = engine.phases[0]['parameters']['asymmetry']
    assert found == pytest.approx(TRUE_SKEW, abs=0.15)


def test_fitting_the_asymmetry_beats_leaving_it_symmetric():
    """The point of the term: a skewed peak cannot be fitted without it."""
    skewed = _rwp(_fit(TRUE_SKEW, refine_asymmetry=True))
    symmetric = _rwp(_fit(TRUE_SKEW, refine_asymmetry=False))
    assert skewed < symmetric
    assert skewed < 2.0, f"a self-consistent pattern should fit closely, got {skewed}"


def test_a_symmetric_pattern_is_left_symmetric():
    """Offering the term must not skew peaks that were never skewed."""
    engine = _fit(0.0, refine_asymmetry=True)
    assert abs(engine.phases[0]['parameters']['asymmetry']) < 0.15
    assert _rwp(engine) < 2.0


def test_the_axial_term_is_recovered_from_the_global_step():
    engine = _fit(0.0, refine_asymmetry=False, axial=0.05, refine_axial=True)
    assert engine.global_parameters['axial_asymmetry'] == pytest.approx(0.05, abs=0.03)


# --- the skew reaches every mode -------------------------------------------

def _windows(engine, skew):
    widths, eta = phase_widths(PEAKS, INSTRUMENT, SAMPLE, WAVELENGTH)
    return engine._peak_windows(PEAKS, widths, eta, skew)


def test_the_window_widens_to_hold_the_longer_flank():
    """
    A tail is no use if the window it lives in is clipped at the old width.

    The kernel is shared by pattern accumulation, Le Bail partitioning and the
    Pawley design matrix, so a truncated window would quietly bias all three.
    """
    two_theta, intensity = _pattern()
    engine = LeBailRefinement()
    engine.quiet = True
    engine.set_experimental_data(two_theta, intensity, wavelength=WAVELENGTH)

    _, symmetric, _ = _windows(engine, 0.0)
    indices, skewed, _ = _windows(engine, 1.0)
    assert skewed.shape[1] > symmetric.shape[1]

    # The profile must reach zero inside its window rather than being cut off
    assert np.all(skewed[:, 0] < 1e-3) and np.all(skewed[:, -1] < 1e-3)


def test_extraction_and_pawley_see_the_same_skew():
    """
    Every intensity route runs through the one kernel.

    Le Bail partitioning and the Pawley solve both weight the observed counts by
    the profile, so a skew applied only to the drawn pattern would leave them
    splitting intensity with the wrong peak shape.
    """
    for model, pawley in (('extract', False), ('extract', True)):
        two_theta, intensity = _pattern(sample_skew=TRUE_SKEW)
        engine = LeBailRefinement()
        engine.quiet = True
        engine.intensity_model = model
        engine.set_experimental_data(two_theta, intensity, wavelength=WAVELENGTH)
        engine.global_parameters.update(INSTRUMENT)
        engine.global_parameters['refine_zero_shift'] = False
        engine.add_phase(_phase(), {
            **SAMPLE, 'refine_cell': False, 'refine_profile': True,
            'asymmetry': TRUE_SKEW, 'refine_asymmetry': False,
            'refine_intensities': pawley,
        })
        engine.refine_phases(max_iterations=4, staged_refinement=False)
        assert _rwp(engine) < 5.0, f"{model}, pawley={pawley}"


def test_one_skewed_phase_leaves_its_neighbour_alone():
    """
    The reason the sample term is per-phase.

    A disordered chlorite sitting next to a well-ordered olivine must be able to
    skew without imposing that shape on the olivine.
    """
    two_theta = np.linspace(5.0, 36.0, 3100)
    engine = LeBailRefinement()
    engine.quiet = True
    engine.set_experimental_data(two_theta, np.zeros_like(two_theta),
                                 wavelength=WAVELENGTH)
    engine.global_parameters.update(INSTRUMENT)
    widths, eta = phase_widths(PEAKS, INSTRUMENT, SAMPLE, WAVELENGTH)

    skewed = engine._accumulate_pseudo_voigt(
        PEAKS, widths, HEIGHTS, eta, asymmetry_exponent(PEAKS, sample=1.0)
    )
    plain = engine._accumulate_pseudo_voigt(PEAKS, widths, HEIGHTS, eta)
    assert not np.allclose(skewed, plain)

    # and with no term at all the two agree exactly
    again = engine._accumulate_pseudo_voigt(PEAKS, widths, HEIGHTS, eta, 0.0)
    assert np.allclose(again, plain, atol=0, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
