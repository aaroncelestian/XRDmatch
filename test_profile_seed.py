"""
Reading an instrument profile back off a peak that was built from a known one.

Every test here makes a peak from stated widths and asks the seeding code what
it thinks those widths were. That is the whole contract: what comes back has to
be usable as a starting point for the refinement, which means it has to be in
the same units and the same parameterization the engine expects, and it has to
put the width in the right component. A seed that gets the total width right and
the Gaussian/Lorentzian split wrong is worse than no seed at all, because the
sample terms then start from a value that has already absorbed the instrument.
"""

import numpy as np
import pytest

from utils import kalpha_filter as kalpha
from utils.profile_functions import (
    caglioti_fwhm, lorentzian_fwhm_strain, pseudo_voigt, tch_mix, tch_split,
)
from utils.profile_seed import candidate_peaks, fit_peak

WAVELENGTH = 1.540562
STEP = 0.01


def pattern(peaks, *, alpha2=0.0, background=0.0, noise=0.0,
            span=(28.0, 44.0), seed=0):
    """A pattern from explicit (centre, height, fwhm, eta, skew) peaks."""
    x = np.arange(span[0], span[1], STEP)
    y = np.full_like(x, background)
    ratio = kalpha.alpha2_ratio(WAVELENGTH)
    for centre, height, fwhm, eta, skew in peaks:
        y = y + height * pseudo_voigt(x - centre, fwhm, eta, skew)
        if alpha2 > 0:
            offset = float(kalpha.alpha2_separation(centre, ratio))
            y = y + alpha2 * height * pseudo_voigt(x - centre - offset, fwhm, eta, skew)
    if noise > 0:
        y = y + np.random.default_rng(seed).normal(0.0, noise, len(x))
    return x, y


def test_the_mixing_inverts_back_to_the_widths_it_came_from():
    for gauss, lorentz in [(0.10, 0.0), (0.0, 0.10), (0.10, 0.05),
                           (0.05, 0.20), (0.25, 0.25)]:
        fwhm, eta = tch_mix(gauss, lorentz)
        back_gauss, back_lorentz = tch_split(float(fwhm), float(eta))
        assert back_gauss == pytest.approx(gauss, abs=2e-3)
        assert back_lorentz == pytest.approx(lorentz, abs=2e-3)


def test_a_gaussian_peak_gives_its_width_to_the_instrument():
    """The whole point: a Gaussian peak is instrument, not microstrain."""
    x, y = pattern([(36.0, 100.0, 0.25, 0.0, 0.0)])
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)

    assert fit['fwhm'] == pytest.approx(0.25, abs=5e-3)
    assert fit['gauss_fwhm'] == pytest.approx(0.25, abs=1e-2)
    assert fit['lorentz_fwhm'] < 0.02
    # W is the square of the Gaussian width, which is what the engine is seeded with
    assert fit['w_param'] == pytest.approx(0.0625, rel=0.1)
    assert fit['microstrain'] < 200


def test_a_lorentzian_peak_gives_its_width_to_the_sample():
    x, y = pattern([(36.0, 100.0, 0.25, 1.0, 0.0)])
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)

    assert fit['lorentz_fwhm'] == pytest.approx(0.25, abs=2e-2)
    assert fit['gauss_fwhm'] < 0.05
    assert fit['w_param'] < 0.0025
    # A quarter degree of tan-theta broadening at 36 degrees is this much strain
    expected = float(lorentzian_fwhm_strain(36.0, fit['microstrain']))
    assert expected == pytest.approx(0.25, abs=2e-2)


def test_the_split_of_a_mixed_peak_lands_on_both_terms():
    gauss, lorentz = 0.18, 0.12
    fwhm, eta = tch_mix(gauss, lorentz)
    x, y = pattern([(36.0, 100.0, float(fwhm), float(eta), 0.0)])
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)

    assert fit['gauss_fwhm'] == pytest.approx(gauss, abs=0.02)
    assert fit['lorentz_fwhm'] == pytest.approx(lorentz, abs=0.02)


def test_a_seeded_w_reproduces_the_width_that_was_measured():
    """The seed has to be in the engine's units, not merely the right size."""
    x, y = pattern([(36.0, 100.0, 0.22, 0.15, 0.0)])
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)

    seeded = caglioti_fwhm(36.0, fit['u_param'], fit['v_param'], fit['w_param'])
    assert float(seeded) == pytest.approx(fit['gauss_fwhm'], abs=1e-6)


def test_a_doublet_is_seen_as_a_doublet_and_not_as_extra_width():
    """
    Kα2 in the data and not in the model is the failure this is meant to catch:
    the satellite has to come back as a satellite, leaving the parent's own width
    where it started.
    """
    x, y = pattern([(36.0, 100.0, 0.20, 0.2, 0.0)], alpha2=0.5)
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=True)

    assert fit['alpha2_ratio'] == pytest.approx(0.5, abs=0.05)
    assert fit['fwhm'] == pytest.approx(0.20, abs=0.02)
    assert abs(fit['skew']) < 0.1

    # Denied the satellite, the same peak has to be fitted with width and skew,
    # which is what the refinement has been doing
    blind = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)
    assert blind['fwhm'] > fit['fwhm'] * 1.1
    assert blind['skew'] < -0.05  # leaning towards high 2 theta


def test_stripped_data_asks_for_no_satellite():
    x, y = pattern([(36.0, 100.0, 0.20, 0.2, 0.0)], alpha2=0.0)
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=True)
    assert fit['alpha2_ratio'] < 0.05


def test_a_skewed_peak_reports_which_way_it_leans():
    x, y = pattern([(36.0, 100.0, 0.20, 0.3, 0.35)])
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)

    assert fit['skew'] == pytest.approx(0.35, abs=0.05)
    # Reported as the instrument term that would produce this lean here
    assert fit['axial_asymmetry'] == pytest.approx(
        0.35 * np.tan(np.radians(36.0)), rel=0.2
    )


def test_a_sloping_background_does_not_widen_the_peak():
    x, y = pattern([(36.0, 100.0, 0.22, 0.2, 0.0)])
    y = y + 8.0 - 0.4 * (x - 36.0)
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)

    assert fit['fwhm'] == pytest.approx(0.22, abs=0.01)
    assert fit['height'] == pytest.approx(100.0, rel=0.05)


def test_noise_moves_the_answer_but_not_far():
    x, y = pattern([(36.0, 100.0, 0.25, 0.2, 0.0)], noise=1.0)
    fit = fit_peak(x, y, 36.0, WAVELENGTH, fit_alpha2=False)
    assert fit['fwhm'] == pytest.approx(0.25, abs=0.02)
    assert fit['misfit'] < 10.0


def test_the_offered_peaks_are_the_strong_isolated_ones():
    x, y = pattern([
        (32.0, 100.0, 0.22, 0.2, 0.0),   # tall and alone
        (35.9, 90.0, 0.22, 0.2, 0.0),    # tall but crowded
        (36.2, 70.0, 0.22, 0.2, 0.0),
        (40.0, 20.0, 0.22, 0.2, 0.0),    # alone but weak
    ], noise=0.5)
    found = candidate_peaks(x, y)

    assert found, "no candidate peaks found"
    assert found[0]['two_theta'] == pytest.approx(32.0, abs=0.05)
    centres = [peak['two_theta'] for peak in found]
    assert any(abs(centre - 40.0) < 0.1 for centre in centres)
    crowded = next(peak for peak in found if abs(peak['two_theta'] - 36.2) < 0.1)
    assert crowded['isolation'] == pytest.approx(0.3, abs=0.05)


def test_asking_for_a_peak_where_there_is_none_says_so():
    x, y = pattern([(36.0, 100.0, 0.22, 0.2, 0.0)])
    with pytest.raises(ValueError):
        fit_peak(x, y, 30.0, WAVELENGTH)
