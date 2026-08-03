#!/usr/bin/env python3
"""
Tests for overlaying several patterns to compare them.

Ctrl-clicking more than one pattern in the file browser draws them together on a
common normalized scale. The comparison is a view only: the pattern loaded for
processing and refinement must survive it untouched.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from gui.pattern_io import normalize_for_comparison  # noqa: E402
from gui.session import AnalysisSession  # noqa: E402
from gui.widgets.file_browser import FileBrowser  # noqa: E402
from gui.workspace import AnalysisWorkspace  # noqa: E402
from matplotlib_config import get_overlay_colors  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_pattern(folder, name, scale=1.0, offset=0.0, peak_at=30.0):
    """A one-peak pattern, written in the plain two-column form."""
    two_theta = np.linspace(10.0, 60.0, 400)
    intensity = offset + scale * np.exp(-0.5 * ((two_theta - peak_at) / 0.4) ** 2)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        for x, y in zip(two_theta, intensity):
            handle.write(f"{x:.4f} {y:.6f}\n")
    return path


# --- normalization ---------------------------------------------------------

def test_normalization_puts_patterns_on_a_common_scale():
    """Exposure and background differences should not survive normalization."""
    two_theta = np.linspace(10.0, 60.0, 400)
    shape = np.exp(-0.5 * ((two_theta - 30.0) / 0.4) ** 2)

    faint = normalize_for_comparison(3.0 * shape + 1.0)
    strong = normalize_for_comparison(50000.0 * shape + 900.0)

    assert np.allclose(faint, strong, atol=1e-6)
    assert faint.max() == pytest.approx(100.0)
    assert faint.min() == pytest.approx(0.0, abs=1.0)


def test_normalization_ignores_a_single_low_outlier():
    """One bad channel must not drag the whole curve up off the baseline."""
    intensity = np.full(500, 10.0)
    intensity[100:110] = 110.0
    clean = normalize_for_comparison(intensity)

    spiked = intensity.copy()
    spiked[0] = -5000.0  # a negative excursion from background subtraction
    recovered = normalize_for_comparison(spiked)

    assert recovered[200] == pytest.approx(clean[200], abs=0.5)
    assert recovered[105] == pytest.approx(clean[105], abs=0.5)


def test_normalization_survives_degenerate_input():
    assert np.all(normalize_for_comparison(np.zeros(10)) == 0.0)
    assert np.all(normalize_for_comparison(np.full(10, 7.0)) == 0.0)
    assert np.all(normalize_for_comparison(np.array([])) == 0.0)


def test_overlay_colors_are_distinct_then_recycle():
    colors = get_overlay_colors("light", 8)
    assert len(set(colors)) == 8
    assert get_overlay_colors("light", 9)[8] == colors[0]
    assert get_overlay_colors("dark", 3) != get_overlay_colors("light", 3)


# --- file browser selection ------------------------------------------------

def test_browser_reports_a_multi_selection(qt_app, tmp_path):
    folder = str(tmp_path)
    paths = [_write_pattern(folder, f"p{i}.xy") for i in range(3)]

    browser = FileBrowser()
    browser.set_folder(folder)
    qt_app.processEvents()

    announced = []
    browser.comparison_changed.connect(announced.append)

    model = browser.tree.selectionModel()
    for path in paths[:2]:
        index = browser.model.index(path)
        model.select(index, model.Select | model.Rows)
    qt_app.processEvents()

    assert announced, "selecting a second pattern should announce a comparison"
    assert sorted(os.path.basename(p) for p in announced[-1]) == ["p0.xy", "p1.xy"]

    # Back down to one file ends the comparison
    announced.clear()
    model.clearSelection()
    model.select(browser.model.index(paths[0]), model.Select | model.Rows)
    qt_app.processEvents()
    assert announced[-1] == []


def test_single_selection_still_loads_normally(qt_app, tmp_path):
    """A plain click must keep opening the file for analysis."""
    folder = str(tmp_path)
    path = _write_pattern(folder, "only.xy")

    browser = FileBrowser()
    browser.set_folder(folder)
    qt_app.processEvents()

    opened = []
    browser.file_activated.connect(opened.append)

    index = browser.model.index(path)
    browser.tree.selectionModel().select(
        index, browser.tree.selectionModel().Select | browser.tree.selectionModel().Rows
    )
    browser._on_clicked(index)
    assert opened == [path]


def test_ctrl_click_does_not_replace_the_loaded_pattern(qt_app, tmp_path):
    """
    The second click of a comparison must not open that file for analysis.

    Otherwise building a comparison would quietly discard whatever processing
    or refinement was already under way on the loaded pattern.
    """
    folder = str(tmp_path)
    first = _write_pattern(folder, "a.xy")
    second = _write_pattern(folder, "b.xy")

    browser = FileBrowser()
    browser.set_folder(folder)
    qt_app.processEvents()

    opened = []
    browser.file_activated.connect(opened.append)

    model = browser.tree.selectionModel()
    for path in (first, second):
        model.select(browser.model.index(path), model.Select | model.Rows)
    browser._on_clicked(browser.model.index(second))

    assert opened == []


# --- workspace behaviour ---------------------------------------------------

def test_comparison_overlays_without_touching_the_session(qt_app, tmp_path):
    folder = str(tmp_path)
    loaded = _write_pattern(folder, "loaded.xy", scale=100.0)
    others = [
        _write_pattern(folder, "faint.xy", scale=2.0, offset=1.0, peak_at=28.0),
        _write_pattern(folder, "bright.xy", scale=90000.0, offset=800.0, peak_at=32.0),
    ]

    session = AnalysisSession()
    workspace = AnalysisWorkspace(session)
    workspace.open_pattern_file(loaded)
    before = session.raw_pattern
    assert before is not None

    workspace.set_comparison_patterns(others)

    assert len(workspace._comparison) == 2
    assert session.raw_pattern is before, "comparison must not disturb the session"

    curves = [line for line in workspace.ax.get_lines() if line.get_label() in
              ("faint.xy", "bright.xy")]
    assert len(curves) == 2
    for line in curves:
        assert max(line.get_ydata()) == pytest.approx(100.0)
    assert len({line.get_color() for line in curves}) == 2

    # Dropping back to a single selection restores the ordinary view
    workspace.set_comparison_patterns([])
    assert workspace._comparison == []
    assert not any(line.get_label() == "faint.xy" for line in workspace.ax.get_lines())


def test_comparison_is_capped_and_reports_unreadable_files(qt_app, tmp_path):
    folder = str(tmp_path)
    paths = [_write_pattern(folder, f"m{i:02d}.xy") for i in range(30)]

    broken = os.path.join(folder, "broken.xy")
    with open(broken, "w", encoding="utf-8") as handle:
        handle.write("not a diffraction pattern\n")

    session = AnalysisSession()
    workspace = AnalysisWorkspace(session)

    messages = []
    workspace.set_status_callback(messages.append)

    workspace.set_comparison_patterns(paths)
    assert len(workspace._comparison) == workspace.MAX_COMPARISON
    assert "first 24 of 30" in messages[-1]

    workspace.set_comparison_patterns([paths[0], broken])
    assert len(workspace._comparison) == 1
    assert "broken.xy" in messages[-1]


def test_mixed_wavelengths_are_called_out(qt_app, tmp_path):
    """Comparing 2θ across wavelengths lines up the wrong reflections."""
    folder = str(tmp_path)
    paths = [_write_pattern(folder, f"w{i}.xy") for i in range(2)]

    session = AnalysisSession()
    workspace = AnalysisWorkspace(session)
    workspace.set_comparison_patterns(paths)
    assert "mixed wavelengths" not in workspace.ax.get_title()

    workspace._comparison[1]["wavelength"] = 0.7107
    workspace.refresh_plot()
    assert "mixed wavelengths" in workspace.ax.get_title()
