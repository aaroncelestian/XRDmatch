"""
Matplotlib styling for XRD Phase Matcher — follows Light / Dark app themes.
"""

from __future__ import annotations

from typing import Optional

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


def get_plot_palette(mode: str = "light") -> dict:
    """Return plot color palette for mode ('light' or 'dark')."""
    key = "dark" if str(mode).lower() in ("dark",) else "light"
    return PLOT_PALETTES[key]


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


def style_new_figure(figsize=(8, 6), mode: str = "light", dpi: Optional[int] = None):
    """Create a Figure with theme-aware defaults."""
    import matplotlib.pyplot as plt

    palette = get_plot_palette(mode)
    kwargs = {"figsize": figsize, "facecolor": palette["figure_facecolor"]}
    if dpi is not None:
        kwargs["dpi"] = dpi
    return plt.figure(**kwargs)
