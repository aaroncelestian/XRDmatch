#!/usr/bin/env python3
"""Tests for profile widths, TCH mixing, and CIF reflection generation."""

import numpy as np
import pytest
from scipy.special import voigt_profile

from utils.profile_functions import (
    FWHM_TO_SIGMA, caglioti_fwhm, lorentzian_fwhm_size, lorentzian_fwhm_strain,
    phase_widths, pseudo_voigt, size_from_fwhm, strain_from_fwhm, tch_mix,
)
from utils.reflections import laue_class, reflections_from_cif


def _numeric_fwhm(x, y):
    """FWHM measured off a sampled curve, interpolating each flank."""
    half = y.max() / 2.0
    above = np.where(y >= half)[0]
    lo, hi = above[0], above[-1]
    left = np.interp(half, [y[lo - 1], y[lo]], [x[lo - 1], x[lo]])
    right = np.interp(half, [y[hi + 1], y[hi]], [x[hi + 1], x[hi]])
    return right - left


def test_tch_limits_are_exact():
    """A vanishing component must leave the other width untouched."""
    gamma, eta = tch_mix(np.array([0.1]), np.array([0.0]))
    assert gamma[0] == pytest.approx(0.1, abs=1e-9)
    assert eta[0] == pytest.approx(0.0, abs=1e-9)

    gamma, eta = tch_mix(np.array([0.0]), np.array([0.1]))
    assert gamma[0] == pytest.approx(0.1, abs=1e-9)
    assert eta[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("fwhm_g", [0.02, 0.05, 0.10, 0.20])
@pytest.mark.parametrize("fwhm_l", [0.02, 0.05, 0.10, 0.30])
def test_tch_width_matches_true_voigt(fwhm_g, fwhm_l):
    """TCH is an approximation to a Voigt; it should hold to well under 1%."""
    x = np.linspace(-3, 3, 200001)
    true = voigt_profile(x, fwhm_g * FWHM_TO_SIGMA, fwhm_l / 2.0)
    gamma, _ = tch_mix(np.array([fwhm_g]), np.array([fwhm_l]))
    assert gamma[0] == pytest.approx(_numeric_fwhm(x, true), rel=0.01)


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.7, 1.0])
def test_pseudo_voigt_is_unit_height_with_requested_fwhm(eta):
    x = np.linspace(-3, 3, 200001)
    profile = pseudo_voigt(x, 0.1, eta)
    assert profile.max() == pytest.approx(1.0, abs=1e-9)
    assert _numeric_fwhm(x, profile) == pytest.approx(0.1, rel=1e-4)


def test_caglioti_is_monotonic_for_positive_u():
    """With U > 0 and V = 0 the resolution curve must widen with angle."""
    two_theta = np.linspace(10, 140, 50)
    fwhm = caglioti_fwhm(two_theta, u=0.004, v=0.0, w=0.003)
    assert np.all(np.diff(fwhm) > 0)


def test_caglioti_floors_negative_polynomial():
    """A non-physical parameter set must not produce a NaN width."""
    fwhm = caglioti_fwhm(np.array([20.0, 60.0]), u=0.0, v=-1.0, w=0.0)
    assert np.all(np.isfinite(fwhm))
    assert np.all(fwhm > 0)


def test_size_broadening_scales_inversely_with_size():
    two_theta = np.array([40.0])
    wide = lorentzian_fwhm_size(two_theta, 0.05, 1.5406)[0]
    narrow = lorentzian_fwhm_size(two_theta, 0.50, 1.5406)[0]
    assert wide == pytest.approx(10.0 * narrow, rel=1e-9)
    # A micron is below the resolution of a laboratory diffractometer
    assert lorentzian_fwhm_size(two_theta, 1.0, 1.5406)[0] < 0.01


def test_size_and_strain_round_trip():
    two_theta = 40.0
    fwhm = lorentzian_fwhm_size(np.array([two_theta]), 0.05, 1.5406)[0]
    assert size_from_fwhm(two_theta, fwhm, 1.5406) == pytest.approx(0.05, rel=1e-9)

    fwhm = lorentzian_fwhm_strain(np.array([two_theta]), 1500.0)[0]
    assert strain_from_fwhm(two_theta, fwhm) == pytest.approx(1500.0, rel=1e-9)


def test_size_and_strain_have_different_angular_dependence():
    """Separating the two terms relies on 1/cos(theta) differing from tan(theta)."""
    two_theta = np.array([20.0, 80.0])
    size = lorentzian_fwhm_size(two_theta, 0.1, 1.5406)
    strain = lorentzian_fwhm_strain(two_theta, 1000.0)
    assert (size[1] / size[0]) == pytest.approx(1.29, abs=0.05)
    assert (strain[1] / strain[0]) == pytest.approx(4.75, abs=0.10)


def test_instrument_only_widths_are_gaussian():
    """With no sample broadening the profile must stay pure Gaussian."""
    instrument = {'u_param': 0.002, 'v_param': -0.0008, 'w_param': 0.004}
    _, eta = phase_widths(np.array([10.0, 90.0]), instrument, {}, 1.5406)
    assert np.allclose(eta, 0.0)


def test_sample_broadening_adds_lorentzian_character():
    instrument = {'u_param': 0.002, 'v_param': -0.0008, 'w_param': 0.004}
    two_theta = np.array([10.0, 30.0, 60.0, 90.0])
    bare, _ = phase_widths(two_theta, instrument, {}, 1.5406)
    broad, eta = phase_widths(
        two_theta, instrument, {'crystallite_size': 0.08, 'microstrain': 1200}, 1.5406
    )
    assert np.all(broad > bare)
    assert np.all(eta > 0.5)
    # Microstrain grows faster than the Gaussian instrument term
    assert np.all(np.diff(eta) > 0)


def test_anisotropic_extra_widens_selected_reflections():
    """The per-reflection hook used by the Stephens model must feed through."""
    instrument = {'u_param': 0.002, 'v_param': 0.0, 'w_param': 0.004}
    two_theta = np.array([20.0, 40.0, 60.0])
    extra = np.array([0.0, 0.05, 0.0])
    base, _ = phase_widths(two_theta, instrument, {'microstrain': 500}, 1.5406)
    with_extra, _ = phase_widths(
        two_theta, instrument, {'microstrain': 500, 'strain_extra': extra}, 1.5406
    )
    assert with_extra[1] > base[1]
    assert with_extra[0] == pytest.approx(base[0])
    assert with_extra[2] == pytest.approx(base[2])


@pytest.mark.parametrize("number,system,laue", [
    (1, 'triclinic', '-1'),
    (14, 'monoclinic', '2/m'),
    (62, 'orthorhombic', 'mmm'),
    (154, 'trigonal', '-3m'),
    (194, 'hexagonal', '6/mmm'),
    (225, 'cubic', 'm-3m'),
])
def test_laue_class_lookup(number, system, laue):
    assert laue_class(number) == (system, laue)


def test_laue_class_handles_missing():
    assert laue_class(None) == (None, None)


QUARTZ_CIF = """data_quartz
_chemical_name_mineral 'Quartz'
_cell_length_a 4.9160
_cell_length_b 4.9160
_cell_length_c 5.4054
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'P 32 2 1'
_symmetry_Int_Tables_number 154
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si 0.46970 0.00000 0.00000
O 0.41350 0.26690 0.11910
"""


def test_reflections_from_cif_indexes_quartz():
    """Miller indices, three of them, with the 101 reflection strongest."""
    result = reflections_from_cif(QUARTZ_CIF, 1.5406, (5.0, 90.0))
    assert result is not None
    assert result['source'] == 'cif'
    assert result['laue_class'] == '-3m'

    hkl = result['hkl']
    assert hkl.ndim == 2 and hkl.shape[1] == 3, "hexagonal 4-index must collapse to 3"
    assert hkl.shape[0] == len(result['two_theta'])

    strongest = int(np.argmax(result['intensity']))
    assert tuple(hkl[strongest]) == (1, 0, 1)
    assert result['two_theta'][strongest] == pytest.approx(26.63, abs=0.05)
    assert result['d_spacing'][strongest] == pytest.approx(3.345, abs=0.005)


def test_reflections_respect_two_theta_range():
    narrow = reflections_from_cif(QUARTZ_CIF, 1.5406, (5.0, 40.0))
    assert narrow is not None
    assert narrow['two_theta'].max() <= 40.0


def test_reflections_from_bad_cif_returns_none():
    assert reflections_from_cif("not a cif at all", 1.5406) is None
    assert reflections_from_cif("", 1.5406) is None
