#!/usr/bin/env python3
"""
Tests for Pawley intensity extraction.

Pawley intensities are linear in the calculated pattern, so they are solved by
non-negative least squares rather than searched for by the general optimizer.
These tests pin down that they are solved exactly, that they stay out of the
nonlinear parameter vector, and that the cost stays roughly linear in the number
of reflections -- the nonlinear formulation was quadratic.
"""

import time

import numpy as np
import pytest

from utils.lebail_refinement import LeBailRefinement
from utils.profile_functions import phase_widths

INSTRUMENT = {'u_param': 0.002, 'v_param': -0.0005, 'w_param': 0.004}
WAVELENGTH = 1.5406


def _phase(peaks, intensities, name='Test'):
    return {
        'phase': {
            'mineral': name,
            'cell_a': 5.0, 'cell_b': 5.0, 'cell_c': 5.0,
            'cell_alpha': 90.0, 'cell_beta': 90.0, 'cell_gamma': 90.0,
        },
        'theoretical_peaks': {
            'two_theta': np.asarray(peaks, dtype=float),
            'intensity': np.asarray(intensities, dtype=float),
            'd_spacing': np.ones(len(peaks)),
        },
    }


def _engine(two_theta, pattern):
    engine = LeBailRefinement()
    engine.set_experimental_data(two_theta, pattern, wavelength=WAVELENGTH)
    engine.set_global_parameters(
        refine_zero_shift=False, refine_displacement=False,
        refine_instrument_profile=False, **INSTRUMENT,
    )
    engine.intensity_model = 'fixed'
    return engine


def _synthesize(two_theta, peaks, intensities, strain):
    """
    Build a pattern with the engine's own windowed kernel.

    Using the engine's accumulator rather than an analytic pseudo-Voigt keeps the
    model exact: the profiles are truncated at a finite window, and a reference
    pattern with untruncated Lorentzian tails would leave a residual that has
    nothing to do with the intensity solve.
    """
    widths, eta = phase_widths(
        np.asarray(peaks, dtype=float), INSTRUMENT,
        {'microstrain': strain}, WAVELENGTH,
    )
    scratch = LeBailRefinement()
    scratch.set_experimental_data(two_theta, np.ones_like(two_theta),
                                  wavelength=WAVELENGTH)
    return scratch._accumulate_pseudo_voigt(
        np.asarray(peaks, dtype=float), widths,
        np.asarray(intensities, dtype=float), eta,
    )


def _pawley_params(strain=1500.0, **overrides):
    params = {
        'refine_cell': False,
        'refine_profile': False,
        'refine_strain': False,
        'refine_size': False,
        'refine_intensities': True,
        'microstrain': strain,
        # _synthesize applies no size broadening, so neither may the model
        'crystallite_size': 0.0,
    }
    params.update(overrides)
    return params


def _relative_error(solved, truth):
    """Compare up to a common factor; the engine renormalizes the pattern."""
    truth = np.asarray(truth, dtype=float)
    factor = np.sum(solved * truth) / np.sum(truth ** 2)
    return np.abs(solved / factor - truth) / truth


def test_solve_recovers_known_intensities():
    """At the correct profile the intensities are an exact linear solve."""
    rng = np.random.default_rng(3)
    peaks = np.linspace(12, 78, 60)
    truth = rng.uniform(10.0, 100.0, len(peaks))
    two_theta = np.linspace(10, 80, 4000)
    pattern = _synthesize(two_theta, peaks, truth, strain=1500.0)

    engine = _engine(two_theta, pattern)
    engine.add_phase(_phase(peaks, truth), _pawley_params())

    widths, eta = phase_widths(peaks, INSTRUMENT, {'microstrain': 1500.0}, WAVELENGTH)
    engine._freeze_extracted = False
    solved = engine._solve_pawley_intensities(0, peaks, widths, eta)

    assert np.max(_relative_error(solved, truth)) < 1e-6


def test_intensities_stay_out_of_the_nonlinear_vector():
    """The optimizer must not carry one parameter per reflection."""
    peaks = np.linspace(12, 78, 40)
    truth = np.full(len(peaks), 50.0)
    two_theta = np.linspace(10, 80, 2000)

    engine = _engine(two_theta, _synthesize(two_theta, peaks, truth, 1500.0))
    engine.add_phase(_phase(peaks, truth),
                     _pawley_params(refine_profile=True, refine_strain=True))

    _, _, names = engine._create_parameter_vector(engine.phases[0]['parameters'])
    assert not any(name.startswith('intensity_mult') for name in names)
    assert len(names) < 10

    # Scale is fully absorbed by the solve, so refining it would be a flat direction
    assert 'scale_factor' not in names


def test_solved_intensities_are_non_negative():
    """Overlapped reflections must not be split into cancelling +/- pairs."""
    rng = np.random.default_rng(11)
    # Deliberately crowded: spacing well inside one FWHM
    peaks = np.sort(rng.uniform(20.0, 40.0, 80))
    truth = rng.uniform(5.0, 100.0, len(peaks))
    two_theta = np.linspace(10, 80, 4000)

    engine = _engine(two_theta, _synthesize(two_theta, peaks, truth, 1500.0))
    engine.add_phase(_phase(peaks, truth), _pawley_params())

    widths, eta = phase_widths(peaks, INSTRUMENT, {'microstrain': 1500.0}, WAVELENGTH)
    engine._freeze_extracted = False
    solved = engine._solve_pawley_intensities(0, peaks, widths, eta)

    assert np.all(solved >= 0.0)
    # The group sum survives even where the individual split does not
    assert np.sum(solved) > 0.0


def test_two_pawley_phases_do_not_recurse():
    """A second Pawley phase is evaluated from cache, not by re-entering the solve."""
    two_theta = np.linspace(10, 80, 3500)
    peaks_a = np.array([18.0, 30.0, 42.0, 60.0])
    peaks_b = np.array([22.0, 34.0, 48.0, 68.0])
    truth_a = np.array([100.0, 70.0, 40.0, 25.0])
    truth_b = np.array([60.0, 90.0, 30.0, 50.0])

    pattern = (_synthesize(two_theta, peaks_a, truth_a, 1500.0)
               + _synthesize(two_theta, peaks_b, truth_b, 1500.0))

    engine = _engine(two_theta, pattern)
    engine.add_phase(_phase(peaks_a, truth_a, 'A'), _pawley_params())
    engine.add_phase(_phase(peaks_b, truth_b, 'B'), _pawley_params())

    result = engine.refine_phases(max_iterations=3, staged_refinement=False,
                                  mode='polish', quiet=True)

    assert result['final_r_factors']['Rwp'] < 1.0
    for index, truth in ((0, truth_a), (1, truth_b)):
        solved = engine.phases[index]['_pawley_intensities']
        assert np.max(_relative_error(solved, truth)) < 0.02


def test_pawley_beats_le_bail_on_overlapped_reflections():
    """
    The point of Pawley: reference intensities that are simply wrong.

    Where reflections are resolved, Le Bail hands each one the counts sitting
    under it and is exact no matter what the reference said, so the two methods
    are indistinguishable. The difference appears once reflections overlap: Le
    Bail can only split the shared counts in the ratio the reference suggests,
    and a textured sample never corrects that ratio. Pawley solves for the split.
    """
    # Partially overlapping doublets: close enough to share intensity, far
    # enough apart that a wrong split distorts the pattern rather than just
    # relabelling it.
    centres = np.arange(20.0, 60.0, 5.0)
    peaks = np.sort(np.concatenate([centres, centres + 0.15]))
    truth = np.tile([90.0, 15.0], len(centres))
    reference = np.tile([15.0, 90.0], len(centres))  # ratio inverted
    two_theta = np.linspace(10, 80, 6000)
    pattern = _synthesize(two_theta, peaks, truth, strain=1500.0)

    errors = {}
    for label, pawley in (('lebail', False), ('pawley', True)):
        engine = _engine(two_theta, pattern)
        engine.intensity_model = 'fixed' if pawley else 'lebail'
        engine.add_phase(_phase(peaks, reference),
                         _pawley_params(refine_intensities=pawley))
        engine.refine_phases(max_iterations=4, staged_refinement=False,
                             mode='polish', quiet=True)
        key = '_pawley_intensities' if pawley else '_extracted_intensities'
        errors[label] = float(np.median(
            _relative_error(engine.phases[0][key], truth)
        ))

    assert errors['pawley'] < 1e-3
    assert errors['lebail'] > 10 * max(errors['pawley'], 1e-4)


def test_cost_grows_about_linearly_with_reflection_count():
    """
    Guard against a return to the nonlinear formulation.

    Refining N intensities through a finite-difference Jacobian costs N pattern
    rebuilds per Jacobian, so the runtime grew as N^2 and a 200-reflection phase
    took fifty times as long as the same fit without Pawley. Solving them keeps
    the growth close to linear.
    """
    rng = np.random.default_rng(5)

    def elapsed(n_peaks):
        peaks = np.linspace(12, 78, n_peaks)
        truth = rng.uniform(10.0, 100.0, n_peaks)
        two_theta = np.linspace(10, 80, 3500)
        engine = _engine(two_theta, _synthesize(two_theta, peaks, truth, 1500.0))
        engine.add_phase(_phase(peaks, truth),
                         _pawley_params(refine_profile=True, refine_strain=True,
                                        strain=800.0))
        start = time.perf_counter()
        engine.refine_phases(max_iterations=3, staged_refinement=False,
                             mode='polish', quiet=True)
        return time.perf_counter() - start

    small = elapsed(50)
    large = elapsed(400)

    # An eightfold increase in reflections should cost far less than the ~64x a
    # quadratic method would need. Loose enough to survive a noisy machine.
    assert large < small * 20
