"""Factory for themed matplotlib plot hosts (figure + canvas + toolbar)."""

from __future__ import annotations

from typing import Optional, Tuple

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from matplotlib_config import apply_plot_style, get_plot_palette
from gui import display_settings
from gui.theme import get_current_mode


def create_plot_host(
    parent: Optional[QWidget] = None,
    figsize: Tuple[float, float] = (8, 6),
    mode: Optional[str] = None,
    with_toolbar: bool = True,
) -> Tuple[QWidget, Figure, FigureCanvas, Optional[NavigationToolbar]]:
    """
    Create a widget containing navigation toolbar + canvas.

    Returns (host_widget, figure, canvas, toolbar). `toolbar` is None when the
    plot redraws itself from scratch, since pan and zoom would be undone by the
    next redraw.
    """
    mode = mode or get_current_mode()
    palette = get_plot_palette(mode)

    host = QWidget(parent)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    figure = Figure(figsize=figsize, facecolor=palette["figure_facecolor"])
    canvas = FigureCanvas(figure)
    toolbar = NavigationToolbar(canvas, host) if with_toolbar else None

    if toolbar is not None:
        layout.addWidget(toolbar)
    layout.addWidget(canvas)

    apply_plot_style(figure, mode, show_grid=display_settings.show_grid())
    return host, figure, canvas, toolbar
