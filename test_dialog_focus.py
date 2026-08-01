#!/usr/bin/env python3
"""
Tests that a dialog hands the focus back where it came from.

Quant Analysis is a non-modal tool window owned by the main window. When
anything modal belonging to it closes -- a file chooser, a message box, the
parameter window -- the platform hands activation back up the owner chain to
the main window rather than to the tool the user was working in. Every export
then drops them behind the main window.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QMainWindow, QWidget,
)

from gui.focus import hold_focus, restores_focus  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def _drain(app):
    """Let the deferred restore run; it is posted, not called inline."""
    for _ in range(5):
        app.processEvents()


def test_focus_is_restored_to_the_owning_window(qt_app):
    main = QMainWindow()
    tool = QDialog(main)
    inner = QWidget(tool)
    main.show()
    tool.show()
    _drain(qt_app)

    raised = []
    tool.raise_ = lambda: raised.append("raise")
    tool.activateWindow = lambda: raised.append("activate")

    # A widget deep inside the tool still restores the tool, not itself
    hold_focus(inner)
    assert raised == [], "the restore must be deferred, not run inline"
    _drain(qt_app)
    assert raised == ["raise", "activate"]

    tool.close()
    main.close()


def test_a_closed_window_is_not_resurrected(qt_app):
    main = QMainWindow()
    tool = QDialog(main)
    main.show()
    tool.show()
    _drain(qt_app)

    hold_focus(tool)
    tool.hide()
    _drain(qt_app)
    assert not tool.isVisible(), "a hidden window must not be raised again"

    main.close()


def test_a_hidden_widget_asks_for_nothing(qt_app):
    """No window on screen means there is no focus to restore."""
    orphan = QWidget()
    hold_focus(orphan)  # must not raise
    hold_focus(None)
    _drain(qt_app)


# --- the decorator ---------------------------------------------------------

class _Panel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ran = []

    @restores_focus
    def export_ok(self):
        self.ran.append("ok")
        return "written"

    @restores_focus
    def export_cancelled(self):
        self.ran.append("cancelled")
        return None  # the user dismissed the file chooser

    @restores_focus
    def export_broken(self):
        raise ValueError("disk full")


def _panel_in_tool():
    main = QMainWindow()
    tool = QDialog(main)
    panel = _Panel(tool)
    main.show()
    tool.show()
    calls = []
    tool.raise_ = lambda: calls.append("raise")
    tool.activateWindow = lambda: calls.append("activate")
    return main, tool, panel, calls


def test_a_completed_export_restores_focus(qt_app):
    main, tool, panel, calls = _panel_in_tool()
    assert panel.export_ok() == "written"
    _drain(qt_app)
    assert calls == ["raise", "activate"]
    tool.close(); main.close()


def test_a_cancelled_export_restores_focus_too(qt_app):
    """Backing out of the file chooser leaves the focus just as scrambled."""
    main, tool, panel, calls = _panel_in_tool()
    assert panel.export_cancelled() is None
    _drain(qt_app)
    assert calls == ["raise", "activate"]
    tool.close(); main.close()


def test_a_failed_export_restores_focus_and_still_raises(qt_app):
    main, tool, panel, calls = _panel_in_tool()
    with pytest.raises(ValueError):
        panel.export_broken()
    _drain(qt_app)
    assert calls == ["raise", "activate"]
    tool.close(); main.close()


def test_the_decorator_keeps_the_method_usable(qt_app):
    assert _Panel.export_ok.__name__ == "export_ok"


# --- the real windows ------------------------------------------------------

def test_the_quant_exports_are_all_covered(qt_app):
    """
    Every button that opens a file chooser has to be wrapped.

    Missing one is invisible until someone clicks that particular button, so
    the list is checked rather than trusted.
    """
    from gui.stages.refine_stage import RefineStage

    for name in ("export_plot", "export_csv_data", "export_pattern_csv"):
        method = getattr(RefineStage, name)
        assert getattr(method, "__wrapped__", None) is not None, (
            f"{name} opens a file chooser but does not restore focus"
        )


def test_closing_the_parameter_window_returns_to_quant(qt_app):
    from gui.session import AnalysisSession
    from gui.dialogs.quant_dialog import QuantDialog
    from test_refinement_reporting import _observed

    main = QMainWindow()
    session = AnalysisSession()
    session.set_raw_pattern(_observed())
    quant = QuantDialog(session, parent=main)
    main.show()
    quant.show()
    _drain(qt_app)

    quant.show_details()
    _drain(qt_app)

    calls = []
    quant.raise_ = lambda: calls.append("raise")
    quant.activateWindow = lambda: calls.append("activate")

    quant._details_dialog.reject()
    _drain(qt_app)
    assert calls == ["raise", "activate"]

    quant.close()
    main.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
