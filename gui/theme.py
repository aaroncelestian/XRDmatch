"""
Application theme manager for XRD Phase Matcher.

Provides Light (lab) and Dark (instrument) themes via QSS + Fusion palette,
persists the choice in QSettings, and notifies listeners so plots can restyle.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

# Theme modes
LIGHT = "light"
DARK = "dark"

ORG = "XRD Tools"
APP = "XRD Phase Matcher"

PALETTES = {
    LIGHT: {
        "bg": "#F5F7FA",
        "surface": "#FFFFFF",
        "surface_alt": "#EEF1F6",
        "border": "#C5CDD8",
        "text": "#1A2332",
        "muted": "#5A6A7A",
        "accent": "#1B6B7A",
        "accent_hover": "#155A66",
        "accent_pressed": "#0F4852",
        "danger": "#B33A3A",
        "success": "#2A7A4B",
        "warn": "#B36B00",
        "input_bg": "#FFFFFF",
        "selection": "#1B6B7A",
        "selection_text": "#FFFFFF",
        "tab_selected": "#FFFFFF",
        "toolbar": "#EEF1F6",
    },
    DARK: {
        "bg": "#1A1F26",
        "surface": "#222831",
        "surface_alt": "#2A313C",
        "border": "#3A4553",
        "text": "#E8EDF4",
        "muted": "#8A9BB0",
        "accent": "#4ECDC4",
        "accent_hover": "#3DB8B0",
        "accent_pressed": "#2FA39B",
        "danger": "#E06C6C",
        "success": "#5CB87A",
        "warn": "#E0A040",
        "input_bg": "#1A1F26",
        "selection": "#4ECDC4",
        "selection_text": "#1A1F26",
        "tab_selected": "#222831",
        "toolbar": "#222831",
    },
}

_theme_listeners: List[Callable[[str], None]] = []
_current_mode: str = LIGHT


def get_styles_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "styles")


def get_current_mode() -> str:
    return _current_mode


def get_palette(mode: Optional[str] = None) -> dict:
    m = (mode or _current_mode).lower()
    return PALETTES[DARK if m == DARK else LIGHT]


def load_saved_mode() -> str:
    settings = QSettings(ORG, APP)
    mode = str(settings.value("display/theme", LIGHT)).lower()
    return DARK if mode == DARK else LIGHT


def save_mode(mode: str) -> None:
    settings = QSettings(ORG, APP)
    settings.setValue("display/theme", DARK if mode == DARK else LIGHT)


def add_theme_listener(callback: Callable[[str], None]) -> None:
    """Register a callback invoked after theme changes (receives mode string)."""
    if callback not in _theme_listeners:
        _theme_listeners.append(callback)


def remove_theme_listener(callback: Callable[[str], None]) -> None:
    if callback in _theme_listeners:
        _theme_listeners.remove(callback)


def _notify_listeners(mode: str) -> None:
    for cb in list(_theme_listeners):
        try:
            cb(mode)
        except Exception:
            pass


def _build_palette(mode: str) -> QPalette:
    p = get_palette(mode)
    palette = QPalette()

    bg = QColor(p["bg"])
    surface = QColor(p["surface"])
    text = QColor(p["text"])
    muted = QColor(p["muted"])
    accent = QColor(p["accent"])
    sel_text = QColor(p["selection_text"])
    border = QColor(p["border"])
    input_bg = QColor(p["input_bg"])

    palette.setColor(QPalette.Window, bg)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, input_bg)
    palette.setColor(QPalette.AlternateBase, QColor(p["surface_alt"]))
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, surface)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Highlight, accent)
    palette.setColor(QPalette.HighlightedText, sel_text)
    palette.setColor(QPalette.ToolTipBase, surface)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.Link, accent)
    palette.setColor(QPalette.PlaceholderText, muted)
    palette.setColor(QPalette.Light, border)
    palette.setColor(QPalette.Midlight, border)
    palette.setColor(QPalette.Mid, border)
    palette.setColor(QPalette.Dark, muted)
    palette.setColor(QPalette.Shadow, QColor("#000000"))

    disabled = QPalette.Disabled
    palette.setColor(disabled, QPalette.WindowText, muted)
    palette.setColor(disabled, QPalette.Text, muted)
    palette.setColor(disabled, QPalette.ButtonText, muted)
    palette.setColor(disabled, QPalette.Highlight, border)
    palette.setColor(disabled, QPalette.HighlightedText, muted)

    return palette


def _load_qss(mode: str) -> str:
    filename = "dark.qss" if mode == DARK else "light.qss"
    path = os.path.join(get_styles_dir(), filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def apply_theme(app: Optional[QApplication] = None, mode: Optional[str] = None,
                persist: bool = True) -> str:
    """
    Apply Light or Dark theme to the application.

    Returns the mode that was applied.
    """
    global _current_mode

    app = app or QApplication.instance()
    if app is None:
        raise RuntimeError("No QApplication instance available")

    if mode is None:
        mode = load_saved_mode()
    mode = DARK if str(mode).lower() == DARK else LIGHT
    _current_mode = mode

    app.setPalette(_build_palette(mode))
    app.setStyleSheet(_load_qss(mode))

    if persist:
        save_mode(mode)

    _notify_listeners(mode)
    return mode


def toggle_theme(app: Optional[QApplication] = None) -> str:
    """Switch between light and dark; returns the new mode."""
    new_mode = DARK if _current_mode == LIGHT else LIGHT
    return apply_theme(app, new_mode, persist=True)
