#!/usr/bin/env python3
"""
Tests for the statistics that describe how good a Le Bail fit really is.

Rwp over a background-subtracted pattern is dominated by the gaps between the
peaks: the data sits at zero there, which is where the Poisson error model puts
its smallest error and so its largest weight, while contributing almost nothing
to the denominator. The number that comes out is mostly counting noise from the
empty parts of the scan. These tests pin down the statistics added to see past
that, and pin down where R_Bragg must stay silent because it would flatter the
model rather than test it.
"""

import numpy as np
import pytest

from test_refinement_reporting import (
    BASE_PARAMS, INTEN_A, INTEN_B, PEAKS_A, PEAKS_B, _observed, _phase,
)
from gui import refinement_table
from utils.lebail_refinement import LeBailRefinement
from utils.multi_phase_analyzer import MultiPhaseAnalyzer

PHASES = [_phase("Quartz", PEAKS_A, INTEN_A, 4.913),
          _phase("Albite", PEAKS_B, INTEN_B, 8.144)]


def _noisy(sigma=0.6, seed=0):
    """The observed pattern with a realistic amount of baseline noise."""
    data = _observed(shift=0.05)
    noise = np.random.default_rng(seed).normal(0.0, sigma, len(data["intensity"]))
    data["intensity"] = data["intensity"] + noise
    return data


def _run(**extra):
    analyzer = MultiPhaseAnalyzer()
    results = analyzer.perform_lebail_refinement(
        _noisy(), PHASES, max_iterations=4,
        refinement_params={**BASE_PARAMS, **extra},
    )
    return analyzer.lebail_engine, results["r_factors"]


# --- the peak-region Rwp ---------------------------------------------------

def test_peak_region_covers_a_small_part_of_the_pattern():
    """A powder pattern is mostly gap, which is the whole reason for this."""
    _, factors = _run()
    assert 0.0 < factors["peak_coverage"] < 40.0


def test_peak_region_rwp_is_the_better_number():
    """
    The empty stretches inflate Rwp, so restricting to the peaks lowers it.

    This is not a nicer number for its own sake: the points removed are ones
    where the model predicts nothing and the data is noise about zero, so no
    refinement could ever have improved them.
    """
    _, factors = _run()
    assert factors["Rwp_peak"] < factors["Rwp"]


def test_baseline_noise_moves_rwp_but_not_the_peak_rwp():
    """
    Add noise only where there are no peaks and only the whole-pattern Rwp
    should notice. This isolates the effect being corrected for.
    """
    engine = LeBailRefinement()
    engine.quiet = True
    data = _observed(shift=0.0)
    engine.set_experimental_data(data["two_theta"], data["intensity"],
                                 wavelength=data["wavelength"])
    for phase in PHASES:
        engine.add_phase(phase, {})

    calc = engine._calculate_total_pattern()
    mask = engine._peak_region_mask()
    clean = engine._calculate_r_factors(calc)

    # Noise confined to the gaps between reflections
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.5, len(calc))
    noise[mask] = 0.0
    engine.experimental_data["intensity"] = engine.experimental_data["intensity"] + noise
    dirty = engine._calculate_r_factors(calc)

    assert dirty["Rwp"] > clean["Rwp"]
    assert dirty["Rwp_peak"] == pytest.approx(clean["Rwp_peak"], rel=1e-6)


# --- R_Bragg, and where it must stay quiet ---------------------------------

def test_r_bragg_is_withheld_from_le_bail():
    """
    Le Bail sets the calculated intensities to the values it partitioned out of
    the observed pattern, so R_Bragg is zero by construction. Reporting it would
    show a perfect score for an arbitrarily wrong model.
    """
    _, factors = _run(intensity_model="extract")
    assert factors["R_Bragg"] is None


def test_r_bragg_is_withheld_from_pawley():
    """Pawley fits the intensities as free parameters against the same data."""
    _, factors = _run(intensity_model="extract", refine_intensities=True)
    assert factors["R_Bragg"] is None


def test_r_bragg_is_reported_for_reference_intensities():
    """Here the intensities come from the structure, so the comparison is real."""
    _, factors = _run(intensity_model="fixed")
    assert factors["R_Bragg"] is not None
    assert factors["R_Bragg"] > 0.0


def test_r_bragg_grows_when_the_reference_intensities_are_wrong():
    """
    The point of R_Bragg is to catch a structural model whose relative
    intensities do not match the data -- preferred orientation, or the wrong
    polymorph. Scrambling the reference intensities must make it worse, while
    leaving the peak positions and therefore the profile fit untouched.
    """
    _, honest = _run(intensity_model="fixed")

    scrambled = [_phase("Quartz", PEAKS_A, INTEN_A[::-1], 4.913),
                 _phase("Albite", PEAKS_B, INTEN_B[::-1], 8.144)]
    analyzer = MultiPhaseAnalyzer()
    wrong = analyzer.perform_lebail_refinement(
        _noisy(), scrambled, max_iterations=4,
        refinement_params={**BASE_PARAMS, "intensity_model": "fixed"},
    )["r_factors"]

    assert wrong["R_Bragg"] > honest["R_Bragg"]


# --- Durbin-Watson ---------------------------------------------------------

def test_durbin_watson_near_two_for_uncorrelated_residuals():
    residual = np.random.default_rng(3).normal(0.0, 1.0, 4000)
    assert LeBailRefinement._durbin_watson(residual) == pytest.approx(2.0, abs=0.1)


def test_durbin_watson_near_zero_for_a_smooth_systematic_error():
    """A slow drift is what an unmodelled peak shape leaves behind."""
    x = np.linspace(0, 4 * np.pi, 4000)
    assert LeBailRefinement._durbin_watson(np.sin(x)) < 0.05


def test_durbin_watson_flags_a_systematic_misfit_in_a_real_fit():
    """
    Fit a pattern whose peaks are far broader than the model allows. Nothing
    can absorb the difference, so the residual is left with structure, and that
    is what the statistic must show.
    """
    engine = LeBailRefinement()
    engine.quiet = True
    two_theta = np.linspace(15.0, 65.0, 2500)
    broad = np.zeros_like(two_theta)
    for centre, height in zip(PEAKS_A, INTEN_A):
        broad += height * np.exp(-0.5 * ((two_theta - centre) / 0.6) ** 2)
    engine.set_experimental_data(two_theta, broad, wavelength=1.5406)
    engine.add_phase(_phase("Quartz", PEAKS_A, INTEN_A, 4.913),
                     {"crystallite_size": 10.0, "microstrain": 0.0})

    factors = engine._calculate_r_factors(engine._calculate_total_pattern())
    assert factors["durbin_watson"] < 1.0
    assert refinement_table._residual_character(factors["durbin_watson"]) == (
        "strongly correlated — systematic misfit"
    )


# --- fitting only near the peaks -------------------------------------------

def test_peak_region_fit_is_off_by_default():
    engine, _ = _run()
    assert engine._fit_mask is None


def test_peak_region_fit_restricts_the_residual():
    engine, _ = _run(fit_peak_regions_only=True)
    assert engine._fit_mask is not None
    assert engine._fit_mask.any()
    assert not engine._fit_mask.all()


def test_peak_region_fit_is_reported():
    """The results must say which region was fitted, or runs cannot be compared."""
    analyzer = MultiPhaseAnalyzer()
    results = analyzer.perform_lebail_refinement(
        _noisy(), PHASES, max_iterations=4,
        refinement_params={**BASE_PARAMS, "fit_peak_regions_only": True},
    )
    assert results["refinement_results"]["fit_peak_regions_only"] is True
    rows = dict(refinement_table.global_rows(results))
    assert rows["Fitted region"] == "Near modelled peaks only"


def test_peak_region_mask_is_fixed_at_the_start_of_the_run():
    """
    The mask must not follow the model. If it were recomputed each cycle the
    optimizer could lower the residual by sliding peaks until the points it
    fits badly fell outside the fitted region.
    """
    engine, _ = _run(fit_peak_regions_only=True)
    before = engine._fit_mask.copy()
    for phase in engine.phases:
        phase["parameters"]["crystallite_size"] = 0.02  # much broader peaks
    engine._calculate_total_pattern()
    assert np.array_equal(engine._fit_mask, before)


# --- how it reaches the user ----------------------------------------------

def test_headline_carries_the_new_statistics():
    _, factors = _run(intensity_model="fixed")
    headline = refinement_table.summary_headline({"r_factors": factors})
    joined = " ".join(headline)
    assert "Rwp(peaks)=" in joined
    assert "R_Bragg=" in joined
    assert "DW=" in joined


def test_headline_omits_r_bragg_when_it_would_be_meaningless():
    _, factors = _run(intensity_model="extract")
    joined = " ".join(refinement_table.summary_headline({"r_factors": factors}))
    assert "Rwp(peaks)=" in joined
    assert "R_Bragg" not in joined


def test_detail_rows_explain_the_residuals_in_words():
    _, factors = _run(intensity_model="fixed")
    rows = dict(refinement_table.global_rows({"r_factors": factors}))
    assert rows["Rwp near peaks (%)"] != "—"
    assert rows["R_Bragg (%)"] != "—"
    assert "correlated" in rows["Residuals look"] or "random" in rows["Residuals look"]


def test_missing_statistics_do_not_break_the_table():
    """Older saved sessions have none of these keys."""
    rows = dict(refinement_table.global_rows({"r_factors": {"Rwp": 12.0}}))
    assert rows["Rwp (%)"] == "12.000"
    assert rows["R_Bragg (%)"] == "—"
    assert rows["Residuals look"] == "—"
