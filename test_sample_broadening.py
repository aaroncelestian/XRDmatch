#!/usr/bin/env python3
"""Recovery tests for the instrument/sample profile split."""

import numpy as np
import pytest

from utils.lebail_refinement import LeBailRefinement
from utils.profile_functions import phase_widths, pseudo_voigt


def _make_pattern(two_theta, peaks, intensities, instrument, sample, wavelength=1.5406):
    """Synthetic pattern from the same width model the engine uses."""
    pattern = np.zeros_like(two_theta)
    gamma, eta = phase_widths(
        np.asarray(peaks, dtype=float), instrument, sample, wavelength
    )
    for centre, height, width, mix in zip(peaks, intensities, gamma, eta):
        pattern += height * pseudo_voigt(two_theta - centre, width, mix)
    return pattern


def _phase(peaks, intensities, cell=5.0):
    return {
        'phase': {
            'mineral': 'Test',
            'cell_a': cell, 'cell_b': cell, 'cell_c': cell,
            'cell_alpha': 90.0, 'cell_beta': 90.0, 'cell_gamma': 90.0,
        },
        'theoretical_peaks': {
            'two_theta': np.asarray(peaks, dtype=float),
            'intensity': np.asarray(intensities, dtype=float),
            'd_spacing': np.ones(len(peaks)),
        },
    }


INSTRUMENT = {'u_param': 0.002, 'v_param': -0.0005, 'w_param': 0.004}
PEAKS = [15.0, 25.0, 35.0, 45.0, 55.0, 70.0]
INTENSITIES = [80.0, 100.0, 60.0, 40.0, 30.0, 20.0]


def test_legacy_profile_params_route_to_instrument():
    """Callers that still pass U,V,W per phase should seed the global profile."""
    engine = LeBailRefinement()
    engine.set_experimental_data(np.linspace(10, 80, 500), np.ones(500))
    engine.add_phase(_phase(PEAKS, INTENSITIES), {
        'u_param': 0.012, 'v_param': -0.003, 'w_param': 0.008, 'eta_param': 0.7,
    })
    assert engine.global_parameters['u_param'] == pytest.approx(0.012)
    assert engine.global_parameters['v_param'] == pytest.approx(-0.003)
    assert engine.global_parameters['w_param'] == pytest.approx(0.008)
    assert 'u_param' not in engine.phases[0]['parameters']
    assert 'eta_param' not in engine.phases[0]['parameters']


def _peak_residual(engine) -> float:
    """RMS residual restricted to points near calculated peaks."""
    obs = engine.experimental_data['intensity']
    calc = engine._calculate_total_pattern()
    mask = calc > 0.05 * calc.max()
    return float(np.sqrt(np.mean((obs[mask] - calc[mask]) ** 2)))


def test_microstrain_recovery_with_fixed_instrument():
    """
    Hold the instrument fixed at the true values and recover a known strain.

    This is the TOPAS-style path: the instrument is not a free parameter, so
    the sample term has a unique angular signature to fit against.
    """
    true_strain = 2500.0
    two_theta = np.linspace(10, 80, 3500)
    pattern = _make_pattern(
        two_theta, PEAKS, INTENSITIES, INSTRUMENT,
        {'crystallite_size': 0.0, 'microstrain': true_strain},
    )

    engine = LeBailRefinement()
    engine.set_experimental_data(two_theta, pattern, wavelength=1.5406)
    engine.set_global_parameters(
        refine_zero_shift=False, refine_displacement=False,
        refine_instrument_profile=False, **INSTRUMENT,
    )
    engine.intensity_model = 'fixed'
    engine.add_phase(_phase(PEAKS, INTENSITIES), {
        'scale_factor': 0.5,
        'crystallite_size': 0.0,
        'microstrain': 200.0,          # start far from truth
        'refine_cell': False,
        'refine_profile': True,
        'refine_size': False,
        'refine_strain': True,
        'refine_scale': True,
    })

    engine.refine_phases(
        max_iterations=8, staged_refinement=False, mode='polish', quiet=True
    )
    recovered = engine.phases[0]['parameters']['microstrain']
    assert recovered == pytest.approx(true_strain, rel=0.20)
    assert _peak_residual(engine) < 2.0
    for key in ('u_param', 'v_param', 'w_param'):
        assert engine.global_parameters[key] == pytest.approx(INSTRUMENT[key])


def test_two_phases_keep_independent_strain():
    """Each phase must be free to carry its own microstrain."""
    two_theta = np.linspace(10, 80, 3500)
    peaks_a = [18.0, 30.0, 42.0, 60.0]
    peaks_b = [22.0, 34.0, 48.0, 68.0]
    inten = [100.0, 70.0, 40.0, 25.0]
    strain_a, strain_b = 800.0, 4000.0

    pattern = (
        _make_pattern(two_theta, peaks_a, inten, INSTRUMENT,
                      {'microstrain': strain_a})
        + _make_pattern(two_theta, peaks_b, inten, INSTRUMENT,
                        {'microstrain': strain_b})
    )

    engine = LeBailRefinement()
    engine.set_experimental_data(two_theta, pattern, wavelength=1.5406)
    engine.set_global_parameters(
        refine_zero_shift=False, refine_displacement=False, **INSTRUMENT,
    )
    engine.intensity_model = 'fixed'
    for peaks in (peaks_a, peaks_b):
        engine.add_phase(_phase(peaks, inten), {
            'scale_factor': 0.5,
            'crystallite_size': 0.0,
            'microstrain': 500.0,
            'refine_cell': False,
            'refine_profile': True,
            'refine_size': False,
            'refine_strain': True,
            'refine_scale': True,
        })

    engine.refine_phases(
        max_iterations=10, staged_refinement=False, mode='polish', quiet=True
    )
    recovered = [p['parameters']['microstrain'] for p in engine.phases]
    assert recovered[0] == pytest.approx(strain_a, rel=0.30)
    assert recovered[1] == pytest.approx(strain_b, rel=0.30)
    assert recovered[1] > 2.0 * recovered[0]


def test_size_recovery_when_enabled():
    """Nanomaterial-sized crystallites should refine back from a bad start."""
    true_size = 0.08  # um
    two_theta = np.linspace(10, 80, 3500)
    pattern = _make_pattern(
        two_theta, PEAKS, INTENSITIES, INSTRUMENT,
        {'crystallite_size': true_size, 'microstrain': 0.0},
    )

    engine = LeBailRefinement()
    engine.set_experimental_data(two_theta, pattern, wavelength=1.5406)
    engine.set_global_parameters(
        refine_zero_shift=False, refine_displacement=False, **INSTRUMENT,
    )
    engine.intensity_model = 'fixed'
    engine.add_phase(_phase(PEAKS, INTENSITIES), {
        'scale_factor': 0.5,
        'crystallite_size': 0.5,       # start an order of magnitude off
        'microstrain': 0.0,
        'refine_cell': False,
        'refine_profile': True,
        'refine_size': True,
        'refine_strain': False,
        'refine_scale': True,
    })

    engine.refine_phases(
        max_iterations=8, staged_refinement=False, mode='polish', quiet=True
    )
    recovered = engine.phases[0]['parameters']['crystallite_size']
    assert recovered == pytest.approx(true_size, rel=0.25)
    assert _peak_residual(engine) < 2.0
