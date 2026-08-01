"""Keeping the user in the window they were working in."""

from __future__ import annotations

import functools

from PyQt5.QtCore import QTimer


def hold_focus(widget) -> None:
    """
    Give activation back to the window `widget` belongs to.

    A tool window such as Quant Analysis is a non-modal dialog owned by the main
    window. When a file chooser, a message box or a secondary window belonging
    to it is dismissed, the platform hands activation back up the owner chain --
    to the main window, not to the tool the user was actually working in. macOS
    is the strictest about this. Without a nudge, every export drops the user
    behind the main window and they have to click their way back.

    The restore is deferred by one turn of the event loop: at the moment a modal
    dialog closes the platform has not finished reassigning activation, and a
    call made now is simply overwritten a moment later.
    """
    window = widget.window() if widget is not None else None
    if window is None or not window.isVisible():
        return
    QTimer.singleShot(0, lambda: _activate(window))


def _activate(window) -> None:
    try:
        if window.isVisible():
            window.raise_()
            window.activateWindow()
    except RuntimeError:
        pass  # the window was closed while the restore was pending


def restores_focus(method):
    """
    Wrap an action that opens a dialog so focus returns when it closes.

    Applied to the whole method rather than to each dialog call inside it, so
    that the early return when a user cancels the file chooser is covered as
    well as the path that goes on to write the file.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            hold_focus(self)

    return wrapper
