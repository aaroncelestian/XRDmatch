#!/usr/bin/env python3
"""
Tests for the cell geometry helpers that anisotropic refinement rests on.

Indexing has to recover the Miller indices the stored patterns never kept, and
the free-parameter list has to say what a powder pattern can actually determine
from a starting cell: equal axes move together, a right angle stays put.
"""

import numpy as np
import pytest

from utils import unit_cell as uc

QUARTZ = {'a': 4.9134, 'b': 4.9134, 'c': 5.4052,
          'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0}
ALBITE = {'a': 8.144, 'b': 12.787, 'c': 7.1583,
          'alpha': 94.26, 'beta': 116.6, 'gamma': 87.68}
GYPSUM = {'a': 5.679, 'b': 15.202, 'c': 6.522,
          'alpha': 90.0, 'beta': 118.43, 'gamma': 90.0}
HALITE = {'a': 5.6402, 'b': 5.6402, 'c': 5.6402,
          'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
BARITE = {'a': 8.884, 'b': 5.457, 'c': 7.157,
          'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}


@pytest.mark.parametrize("cell, expected", [
    (HALITE, [("a", ("a", "b", "c"))]),
    (QUARTZ, [("a", ("a", "b")), ("c", ("c",))]),
    (BARITE, [("a", ("a",)), ("b", ("b",)), ("c", ("c",))]),
    (GYPSUM, [("a", ("a",)), ("b", ("b",)), ("c", ("c",)), ("beta", ("beta",))]),
    (ALBITE, [("a", ("a",)), ("b", ("b",)), ("c", ("c",)),
              ("alpha", ("alpha",)), ("beta", ("beta",)), ("gamma", ("gamma",))]),
])
def test_symmetry_of_the_starting_cell_decides_what_is_free(cell, expected):
    groups = uc.cell_groups(cell)
    assert [(g.key, g.tied) for g in groups] == expected


def test_too_few_reflections_collapse_the_model_to_one_dilation():
    groups = uc.cell_groups(ALBITE, reflections=3)
    assert len(groups) == 1
    assert set(groups[0].tied) == {"a", "b", "c"}


def test_a_typed_edge_moves_every_axis_it_is_tied_to():
    groups = uc.cell_groups(QUARTZ)
    values = uc.free_cell_values(QUARTZ, groups)
    values["cell_a"] = 4.95
    rebuilt = uc.cell_from_free(QUARTZ, values, groups)
    assert rebuilt["a"] == pytest.approx(4.95)
    assert rebuilt["b"] == pytest.approx(4.95)
    assert rebuilt["c"] == pytest.approx(QUARTZ["c"])
    assert rebuilt["gamma"] == pytest.approx(120.0)


@pytest.mark.parametrize("cell", [QUARTZ, ALBITE, GYPSUM, HALITE, BARITE])
def test_indexing_recovers_the_reflections_of_the_starting_cell(cell):
    truth = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0, 2], [1, 1, 1],
                      [2, 0, 0], [2, 0, 1], [1, 1, 2], [0, 0, 3], [2, 1, 1],
                      [3, 1, 2], [0, 4, 1]])
    d = np.round(uc.d_spacings(truth, cell), 4)
    hkl, matched = uc.index_reflections(d, cell)
    assert matched.all()
    assert uc.d_spacings(hkl, cell) == pytest.approx(
        uc.d_spacings(truth, cell), abs=1e-6
    )


def test_an_isotropic_dilation_scales_every_d_spacing_the_same():
    hkl = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 2]])
    grown = {**QUARTZ, "a": QUARTZ["a"] * 1.01, "b": QUARTZ["b"] * 1.01,
             "c": QUARTZ["c"] * 1.01}
    assert uc.d_spacing_ratio(hkl, grown, QUARTZ) == pytest.approx(1.01)


def test_an_anisotropic_change_moves_the_axes_apart():
    hkl = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 2]])
    skewed = {**QUARTZ, "a": QUARTZ["a"] * 1.005, "b": QUARTZ["b"] * 1.005,
              "c": QUARTZ["c"] * 0.997}
    ratio = uc.d_spacing_ratio(hkl, skewed, QUARTZ)
    assert ratio[0] == pytest.approx(1.005)
    assert ratio[1] == pytest.approx(0.997)
    assert ratio[0] != pytest.approx(ratio[1])


def test_deltas_report_percent_for_edges_and_degrees_for_angles():
    skewed = {**GYPSUM, "a": GYPSUM["a"] * 1.01, "beta": GYPSUM["beta"] + 0.3}
    delta = uc.cell_deltas(skewed, GYPSUM)
    assert delta["a"] == pytest.approx(1.0)
    assert delta["beta"] == pytest.approx(0.3)
    assert delta["b"] == pytest.approx(0.0)
