"""Keeping the user in the window they were working in."""

from __future__ import annotations

import functools
import inspect

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

    Extra positional arguments are dropped to match what the method accepts.
    Qt reads the arity of a slot before calling it and passes only as many
    arguments as it can take, which is how `clicked.connect(self.export_csv)`
    works for a method with no parameters even though the signal carries a
    checked flag. A wrapper declared with *args advertises that it takes
    anything, so Qt starts passing that flag through and the call fails.
    """
    accepted = _positional_limit(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args[:accepted], **kwargs)
        finally:
            hold_focus(self)

    return wrapper


def _positional_limit(method):
    """How many arguments past `self` the method can take, or None for any."""
    try:
        parameters = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return None
    positional = [p for p in parameters if p.kind in (
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )]
    return max(len(positional) - 1, 0)  # less `self`
