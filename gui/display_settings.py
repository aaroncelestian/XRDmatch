"""
Plot display preferences for XRD Phase Matcher.

Holds the Settings -> Display values that affect drawing (grid, legend, error
bars, line width, marker size, export DPI), persists them in QSettings, and
notifies listeners so open plots restyle immediately. Mirrors gui/theme.py.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from PyQt5.QtCore import QSettings

from .theme import APP, ORG

DEFAULTS: Dict[str, object] = {
    "line_width": 1.2,
    "marker_size": 4,
    "show_grid": True,
    "show_legend": True,
    "show_error_bars": True,
    "plot_dpi": 300,
}

# Kept in step with the spin box ranges in gui/settings_tab.py
BOUNDS: Dict[str, tuple] = {
    "line_width": (0.5, 5.0),
    "marker_size": (2, 20),
    "plot_dpi": (72, 600),
}

_settings: Dict[str, object] = dict(DEFAULTS)
_listeners: List[Callable[[dict], None]] = []


def _coerce(key: str, value) -> object:
    """QSettings hands back strings, so every value is re-typed and clamped."""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isinstance(default, int):
        number = int(round(number))
    lo, hi = BOUNDS.get(key, (None, None))
    if lo is not None:
        number = max(lo, min(hi, number))
    return number


def current() -> dict:
    """All display preferences as a plain dict."""
    return dict(_settings)


def value(key: str):
    return _settings.get(key, DEFAULTS.get(key))


def load_saved() -> dict:
    """Read persisted preferences into the store. Call once at startup."""
    stored = QSettings(ORG, APP)
    for key, default in DEFAULTS.items():
        _settings[key] = _coerce(key, stored.value(f"display/{key}", default))
    return current()


def update(values: dict, persist: bool = True, notify: bool = True) -> dict:
    """Merge in new values; returns the full preference set."""
    changed = {}
    for key, raw in values.items():
        if key not in DEFAULTS:
            continue
        coerced = _coerce(key, raw)
        if coerced != _settings.get(key):
            _settings[key] = coerced
            changed[key] = coerced

    if persist and changed:
        stored = QSettings(ORG, APP)
        for key, val in changed.items():
            stored.setValue(f"display/{key}", val)

    if notify and changed:
        _notify_listeners()
    return current()


def reset() -> dict:
    return update(dict(DEFAULTS))


def add_listener(callback: Callable[[dict], None]) -> None:
    """Register a callback invoked after preferences change (receives dict)."""
    if callback not in _listeners:
        _listeners.append(callback)


def remove_listener(callback: Callable[[dict], None]) -> None:
    if callback in _listeners:
        _listeners.remove(callback)


def _notify_listeners() -> None:
    snapshot = current()
    for callback in list(_listeners):
        try:
            callback(snapshot)
        except Exception:
            pass


def line_width(factor: float = 1.0, minimum: float = 0.2) -> float:
    """Curve width scaled off the user's base width (0.7 for a faint trace)."""
    return max(minimum, float(value("line_width")) * factor)


def marker_size(factor: float = 1.0, minimum: float = 1.0) -> float:
    return max(minimum, float(value("marker_size")) * factor)


def show_grid() -> bool:
    return bool(value("show_grid"))


def show_legend() -> bool:
    return bool(value("show_legend"))


def show_error_bars() -> bool:
    return bool(value("show_error_bars"))


def export_dpi() -> int:
    return int(value("plot_dpi"))
