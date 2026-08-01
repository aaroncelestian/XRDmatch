"""
Matplotlib styling for XRD Phase Matcher — follows Light / Dark app themes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Plot palettes aligned with gui/theme.py
PLOT_PALETTES = {
    "light": {
        "figure_facecolor": "#F5F7FA",
        "axes_facecolor": "#FFFFFF",
        "axes_edgecolor": "#C5CDD8",
        "text": "#1A2332",
        "tick": "#4A5568",
        "grid": "#D8DEE8",
        "spine": "#C5CDD8",
        "label": "#2D3748",
        "title": "#1A2332",
        "legend_facecolor": "#FFFFFF",
        "legend_edgecolor": "#C5CDD8",
        "exp_line": "#1B6B7A",
        "calc_line": "#C45C26",
        "diff_line": "#5A6A7A",
    },
    "dark": {
        "figure_facecolor": "#1A1F26",
        "axes_facecolor": "#222831",
        "axes_edgecolor": "#3A4553",
        "text": "#E8EDF4",
        "tick": "#A8B4C4",
        "grid": "#3A4553",
        "spine": "#3A4553",
        "label": "#C8D0DC",
        "title": "#E8EDF4",
        "legend_facecolor": "#222831",
        "legend_edgecolor": "#3A4553",
        "exp_line": "#4ECDC4",
        "calc_line": "#FF8C5A",
        "diff_line": "#8A9BB0",
    },
}


# Colors for drawing several patterns on one axis. Ordered so that neighbouring
# entries stay distinguishable, since overlaid patterns are usually told apart by
# color alone. The first entry matches each theme's exp_line, so a comparison of
# one pattern looks like the ordinary single-pattern view.
OVERLAY_CYCLES = {
    "light": [
        "#1B6B7A", "#C45C26", "#6A4C93", "#2E7D32",
        "#B3123F", "#1F5FA9", "#8C6D1F", "#A0439B",
    ],
    "dark": [
        "#4ECDC4", "#FF8C5A", "#B39DDB", "#7BC96F",
        "#FF6B8A", "#64B5F6", "#E0C36A", "#E39BD9",
    ],
}


def get_plot_palette(mode: str = "light") -> dict:
    """Return plot color palette for mode ('light' or 'dark')."""
    key = "dark" if str(mode).lower() in ("dark",) else "light"
    return PLOT_PALETTES[key]


def get_overlay_colors(mode: str = "light", count: int = 1) -> list:
    """
    Colors for `count` curves overlaid on one axis, recycled if there are many.

    Recycling is preferable to generating ever more hues: past eight or so curves
    no palette keeps them apart, and the legend is doing the work by then.
    """
    key = "dark" if str(mode).lower() in ("dark",) else "light"
    cycle = OVERLAY_CYCLES[key]
    return [cycle[i % len(cycle)] for i in range(max(0, int(count)))]


def apply_plot_style(figure, mode: str = "light", show_grid: bool = True) -> None:
    """
    Apply theme colors to a matplotlib Figure and all of its axes.

    Call after creating axes (or after clearing) and before draw().
    """
    palette = get_plot_palette(mode)
    figure.patch.set_facecolor(palette["figure_facecolor"])

    for ax in figure.get_axes():
        ax.set_facecolor(palette["axes_facecolor"])
        ax.tick_params(colors=palette["tick"], which="both")
        ax.xaxis.label.set_color(palette["label"])
        ax.yaxis.label.set_color(palette["label"])
        ax.title.set_color(palette["title"])

        for spine in ax.spines.values():
            spine.set_color(palette["spine"])

        if show_grid:
            ax.grid(True, color=palette["grid"], alpha=0.85, linewidth=0.6)
        else:
            ax.grid(False)

        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(palette["legend_facecolor"])
            legend.get_frame().set_edgecolor(palette["legend_edgecolor"])
            for text in legend.get_texts():
                text.set_color(palette["text"])


def draw_error_bars(ax, x, y, errors, color, scale: float = 1.0) -> None:
    """
    Overlay σ whiskers from an XYE third column onto an already-plotted curve.

    No-op when the file carried no usable errors. `scale` matches the errors to
    a normalized curve.
    """
    if errors is None:
        return
    err = np.asarray(errors, dtype=float) * scale
    if len(err) != len(y) or not np.any(err > 0):
        return
    ax.errorbar(
        x, y, yerr=err, fmt="none", ecolor=color, elinewidth=0.5,
        capsize=0, alpha=0.45, zorder=1,
    )


def style_new_figure(figsize=(8, 6), mode: str = "light", dpi: Optional[int] = None):
    """Create a Figure with theme-aware defaults."""
    import matplotlib.pyplot as plt

    palette = get_plot_palette(mode)
    kwargs = {"figsize": figsize, "facecolor": palette["figure_facecolor"]}
    if dpi is not None:
        kwargs["dpi"] = dpi
    return plt.figure(**kwargs)
