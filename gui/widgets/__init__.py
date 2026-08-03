"""Shared lightweight UI widgets."""

from .section import SectionFrame, CollapsibleSection
from .plot_host import create_plot_host
from .file_browser import FileBrowser

__all__ = ["SectionFrame", "CollapsibleSection", "create_plot_host", "FileBrowser"]
