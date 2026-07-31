"""Analysis workspace — file browser, plot, and Background / Peaks / Phases tabs."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QMenu,
    QMessageBox, QPushButton, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from matplotlib.widgets import SpanSelector

from matplotlib_config import apply_plot_style, draw_error_bars, get_plot_palette
from gui import display_settings
from gui.theme import get_current_mode
from gui.widgets.file_browser import FileBrowser
from gui.widgets.plot_host import create_plot_host
from gui.stages import ProcessStage, IdentifyStage
from gui.pattern_io import load_pattern_file
from utils import emphasis
from utils.two_theta_shift import DISPLACEMENT, describe as describe_shift


LEFT_MIN_WIDTH = 220
LEFT_DEFAULT = 280
BOTTOM_DEFAULT = 280

PHASE_CONTROLS_MIN = 520
PHASE_LIST_MIN = 260
PHASE_LIST_DEFAULT = 380

# Checkbox column indices differ between the two result table layouts
CAND_SELECT_COL = 3
MATCH_SELECT_COL = 0


class AnalysisWorkspace(QWidget):
    """Main analysis layout: files | plot + bottom tool tabs."""

    TAB_BACKGROUND = 0
    TAB_PEAKS = 1
    TAB_PHASES = 2

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._status_callback = None
        self._candidate_results = []
        self._results_mode = None  # "candidates" | "matches"
        self._quant_dialog = None
        self._database_dialog = None
        self._details_dialog = None
        self._preview = None  # {"name", "two_theta", "intensity"}
        self._peak_highlight = None  # {"two_theta", "intensity", "d_spacing"}
        self._peak_edit = False  # clicks on the plot add/remove peaks
        self._emphasis_mode = False  # drags on the plot mark priority regions
        self._emphasis_selector = None

        self.process_stage = ProcessStage(session, self)
        self.identify_stage = IdentifyStage(session, self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        self.file_browser = FileBrowser()
        self.file_browser.setMinimumWidth(LEFT_MIN_WIDTH)
        self.file_browser.file_activated.connect(self.open_pattern_file)
        self.file_browser.wavelength_changed.connect(self._on_wavelength_changed)
        self.main_splitter.addWidget(self.file_browser)

        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setChildrenCollapsible(False)

        plot_host, self.figure, self.canvas, self.toolbar = create_plot_host(
            self, figsize=(10, 7)
        )
        self.ax = self.figure.add_subplot(111)
        # Back-compat aliases (search plot == main plot)
        self.search_figure = self.figure
        self.search_canvas = self.canvas
        self.search_ax = self.ax
        self._add_view_toggles()
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.right_splitter.addWidget(plot_host)

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self._build_background_tab(), "Background")
        self.bottom_tabs.addTab(self._build_peaks_tab(), "Peaks")
        self.bottom_tabs.addTab(self._build_phases_tab(), "Phases")
        self.right_splitter.addWidget(self.bottom_tabs)

        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        root.addWidget(self.main_splitter)

        session.pattern_changed.connect(self._on_session_changed)
        session.peaks_changed.connect(self._on_session_changed)
        session.candidates_changed.connect(self._on_session_changed)
        session.matches_changed.connect(self._on_session_changed)
        session.emphasis_changed.connect(self.refresh_plot)
        display_settings.add_listener(self.on_display_settings_changed)

        self.refresh_plot()
        QTimer.singleShot(0, self._apply_default_sizes)

    # --- plot view toggles ---

    def _add_view_toggles(self):
        """Show/hide curves straight from the plot toolbar."""
        self.toolbar.addSeparator()
        self.view_toggles = {}
        for key, label, tip, checked in (
            ("raw", "Raw", "Show the unprocessed pattern", True),
            ("processed", "Processed", "Show the background-subtracted pattern", True),
            ("background", "Background", "Show the fitted background curve", True),
            ("peaks", "Peaks", "Show detected peak markers", True),
            ("scaled", "Scaled",
             "Scale each phase's reference lines to the intensity it actually accounts "
             "for. Uncheck to draw every phase at full height, which is easier for "
             "checking peak positions alone.", True),
        ):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.setToolTip(tip)
            cb.stateChanged.connect(self.refresh_plot)
            self.toolbar.addWidget(cb)
            self.view_toggles[key] = cb

    def _visible(self, key: str) -> bool:
        cb = self.view_toggles.get(key)
        return cb.isChecked() if cb is not None else True

    # --- tab builders ---

    def _build_background_tab(self) -> QWidget:
        return self.process_stage.background_panel

    def _build_peaks_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.process_stage.peaks_panel)

        self.peaks_label = QLabel("Peaks")
        self.peaks_label.setObjectName("mutedLabel")
        self.peaks_label.setContentsMargins(8, 0, 8, 0)
        layout.addWidget(self.peaks_label)

        self.peaks_table = QTableWidget()
        self.peaks_table.setAlternatingRowColors(True)
        self.peaks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peaks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peaks_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.peaks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.peaks_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.peaks_table.customContextMenuRequested.connect(self._show_peaks_menu)
        self.peaks_table.currentCellChanged.connect(self._on_peak_row_changed)
        layout.addWidget(self.peaks_table, 1)
        return tab

    def _build_phases_tab(self) -> QWidget:
        """Controls grid on the left, a compact mineral list on the right."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls = self.identify_stage.control_panel
        controls.setMinimumWidth(PHASE_CONTROLS_MIN)
        self._add_phase_table_actions()
        splitter.addWidget(controls)

        list_wrap = QWidget()
        lw = QVBoxLayout(list_wrap)
        lw.setContentsMargins(4, 6, 6, 6)
        lw.setSpacing(3)
        self.phases_label = QLabel("Phases")
        self.phases_label.setObjectName("mutedLabel")
        self.phases_label.setWordWrap(True)
        lw.addWidget(self.phases_label)

        # Back-compat: results_table / results_label used by older helpers
        self.results_label = self.phases_label
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.verticalHeader().setDefaultSectionSize(20)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_results_menu)
        self.results_table.currentCellChanged.connect(self._on_results_row_changed)
        lw.addWidget(self.results_table, 1)
        list_wrap.setMinimumWidth(PHASE_LIST_MIN)
        splitter.addWidget(list_wrap)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([620, PHASE_LIST_DEFAULT])
        self.phases_splitter = splitter
        layout.addWidget(splitter)
        return tab

    def _add_phase_table_actions(self):
        """List-related buttons, placed inside the controls grid."""
        stage = self.identify_stage

        self.details_btn = QPushButton("Details…")
        self.details_btn.setToolTip("Show details for the highlighted phase (or right-click a row)")
        self.details_btn.clicked.connect(self.show_selected_phase_details)
        stage.add_action_widget(self.details_btn)

        self.clear_unselected_btn = QPushButton("Clear Unselected")
        self.clear_unselected_btn.setToolTip(
            "Remove unchecked phases from the list; keep only your selections"
        )
        self.clear_unselected_btn.clicked.connect(self.clear_unselected_phases)
        stage.add_action_widget(self.clear_unselected_btn)

        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.setToolTip("Remove every mineral from the list and start over")
        self.clear_all_btn.clicked.connect(self.clear_all_phases)
        stage.add_action_widget(self.clear_all_btn)

        self.quant_btn = QPushButton("Open Quant…")
        self.quant_btn.setToolTip("Open Le Bail / quantitative analysis window")
        self.quant_btn.clicked.connect(self.open_quant)
        stage.add_action_widget(self.quant_btn)
        stage.finish_action_row()

    def _apply_default_sizes(self):
        total_w = max(self.main_splitter.width(), 1200)
        left = min(max(LEFT_DEFAULT, LEFT_MIN_WIDTH), 360)
        self.main_splitter.setSizes([left, max(400, total_w - left)])
        total_h = max(self.right_splitter.height(), 700)
        bottom = min(BOTTOM_DEFAULT, total_h // 3)
        self.right_splitter.setSizes([max(300, total_h - bottom), bottom])

    # --- navigation ---

    def show_bottom_tab(self, key: str):
        mapping = {
            "background": self.TAB_BACKGROUND,
            "process": self.TAB_BACKGROUND,
            "load": self.TAB_BACKGROUND,
            "peaks": self.TAB_PEAKS,
            "identify": self.TAB_PHASES,
            "phases": self.TAB_PHASES,
        }
        idx = mapping.get(key, self.TAB_BACKGROUND)
        self.bottom_tabs.setCurrentIndex(idx)
        if idx == self.TAB_BACKGROUND:
            self.process_stage.on_enter()
        elif idx == self.TAB_PEAKS:
            self.process_stage.on_enter()
        elif idx == self.TAB_PHASES:
            self.identify_stage.on_enter()

    def set_stage(self, key: str):
        """Compatibility shim for older call sites."""
        if key == "refine":
            self.open_quant()
        else:
            self.show_bottom_tab(key)

    def show_search_tab(self, section: Optional[str] = None):
        self.show_bottom_tab(section or "phases")

    def show_quant_tab(self):
        self.open_quant()

    def show_database_tab(self):
        self.open_database()

    # --- dialogs ---

    def open_quant(self):
        from gui.dialogs.quant_dialog import QuantDialog

        if self._quant_dialog is None:
            parent = self.window()
            self._quant_dialog = QuantDialog(
                self.session, parent=parent, status_callback=self.set_status
            )
        self._quant_dialog.show()
        self._quant_dialog.raise_()
        self._quant_dialog.activateWindow()
        if hasattr(self._quant_dialog.refine_stage, "on_enter"):
            self._quant_dialog.refine_stage.on_enter()
        self._quant_dialog.refresh_plot()

    def open_database(self):
        from gui.dialogs.database_dialog import DatabaseManagerDialog

        if self._database_dialog is None:
            parent = self.window()
            self._database_dialog = DatabaseManagerDialog(parent)
            self._database_dialog.phases_selected.connect(self._on_db_phases)
        self._database_dialog.show()
        self._database_dialog.raise_()
        self._database_dialog.activateWindow()

    def _on_db_phases(self, phases: list):
        self.identify_stage.add_phases_from_database(phases)
        self.show_bottom_tab("phases")
        self.set_status(f"Added {len(phases)} phase(s) from database")

    # --- status / theme ---

    def set_status_callback(self, callback):
        self._status_callback = callback

    def set_status(self, message: str):
        if self._status_callback:
            self._status_callback(message)

    def _on_session_changed(self):
        self.refresh_plot()

    def on_theme_changed(self, mode: str):
        apply_plot_style(self.figure, mode, show_grid=display_settings.show_grid())
        self.refresh_plot()
        if self._quant_dialog is not None:
            self._quant_dialog.on_theme_changed(mode)

    def on_display_settings_changed(self, prefs: dict):
        """Settings -> Display changed; redraw with the new grid / widths."""
        self.refresh_plot()
        if self._quant_dialog is not None:
            self._quant_dialog.on_display_settings_changed(prefs)

    # --- file loading ---

    def _on_wavelength_changed(self, wl: float):
        self.session.set_wavelength(wl)
        self.refresh_plot()

    def open_folder(self, path: Optional[str] = None):
        if path:
            self.file_browser.set_folder(path)
        else:
            self.file_browser.choose_folder()

    def find_peaks(self):
        self.show_bottom_tab("peaks")
        self.process_stage.find_peaks()

    def open_pattern_file(self, path: str):
        try:
            wl = self.file_browser.current_wavelength()
            pattern = load_pattern_file(path, wl)
            if pattern.get("file_format") == "XML":
                self.file_browser.set_custom_wavelength(pattern["wavelength"])
            else:
                pattern["wavelength"] = wl

            self.session.set_raw_pattern(pattern)
            self.clear_peaks_table()
            name = os.path.basename(path)
            n = len(pattern["two_theta"])
            t0, t1 = float(pattern["two_theta"][0]), float(pattern["two_theta"][-1])
            meta = (
                f"{pattern['file_format']} · {n} points · "
                f"2θ {t0:.2f}–{t1:.2f}° · λ {pattern['wavelength']:.4f} Å"
            )
            self.file_browser.set_file_info(name, meta)
            self.file_browser.reveal_file(path)
            self.process_stage.on_enter()
            self.identify_stage.on_enter()
            self.show_bottom_tab("background")
            self.refresh_plot()
            self.set_status(f"Loaded {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load pattern:\n{e}")

    # --- results helpers ---

    def set_results_peaks(self, peaks: dict, select_row: Optional[int] = None):
        self.show_bottom_tab("peaks")
        n = len(peaks.get("two_theta", []))
        self.peaks_label.setText(
            f"Peaks ({n}) — click a row to mark it on the plot, right-click to delete"
        )
        self.peaks_table.blockSignals(True)
        self.peaks_table.clear()
        self.peaks_table.setColumnCount(4)
        self.peaks_table.setHorizontalHeaderLabels(
            ["2θ (°)", "Intensity", "d (Å)", "Source"]
        )
        tt = peaks.get("two_theta", [])
        inten = peaks.get("intensity", [])
        d = peaks.get("d_spacing", [])
        manual = np.asarray(peaks.get("manual", []), dtype=bool)
        self.peaks_table.setRowCount(len(tt))
        for i in range(len(tt)):
            self.peaks_table.setItem(i, 0, QTableWidgetItem(f"{tt[i]:.3f}"))
            self.peaks_table.setItem(i, 1, QTableWidgetItem(f"{inten[i]:.0f}"))
            dv = f"{d[i]:.4f}" if i < len(d) and np.isfinite(d[i]) else "—"
            self.peaks_table.setItem(i, 2, QTableWidgetItem(dv))
            src = "manual" if i < len(manual) and manual[i] else "detected"
            self.peaks_table.setItem(i, 3, QTableWidgetItem(src))
        self.peaks_table.blockSignals(False)

        self._peak_highlight = None
        if select_row is not None and n:
            row = int(np.clip(select_row, 0, n - 1))
            self.peaks_table.setCurrentCell(row, 0)
        else:
            self.refresh_plot()

    def clear_peaks_table(self):
        self.peaks_label.setText("Peaks")
        self.peaks_table.clearContents()
        self.peaks_table.setRowCount(0)
        self._peak_highlight = None

    # --- peak row interaction ---

    def _on_peak_row_changed(self, row: int, _col=0, _prow=-1, _pcol=0):
        """Mark the highlighted peak on the plot (clicks and arrow keys)."""
        peaks = self.session.peaks
        if peaks is None or row < 0:
            return
        tt = np.asarray(peaks.get("two_theta", []), dtype=float)
        if row >= len(tt):
            return
        inten = np.asarray(peaks.get("intensity", []), dtype=float)
        d = peaks.get("d_spacing")
        self._peak_highlight = {
            "two_theta": float(tt[row]),
            "intensity": float(inten[row]) if row < len(inten) else None,
            "d_spacing": float(d[row]) if d is not None and row < len(d) else None,
        }
        self.refresh_plot()

    def clear_peak_highlight(self):
        self._peak_highlight = None
        self.refresh_plot()

    def _show_peaks_menu(self, pos):
        row = self.peaks_table.rowAt(pos.y())
        if row < 0 or not self.session.has_peaks():
            return
        rows = sorted({idx.row() for idx in self.peaks_table.selectionModel().selectedRows()})
        if row not in rows:
            self.peaks_table.selectRow(row)
            rows = [row]

        menu = QMenu(self)
        label = "Delete peak" if len(rows) == 1 else f"Delete {len(rows)} peaks"
        menu.addAction(label, lambda: self.delete_peak_rows(rows))
        menu.addAction("Mark on plot", lambda: self._on_peak_row_changed(row))
        menu.addAction("Clear marker", self.clear_peak_highlight)
        menu.exec_(self.peaks_table.viewport().mapToGlobal(pos))

    def delete_peak_rows(self, rows: list):
        """Remove peaks from the session list and rebuild the table."""
        peaks = self.session.peaks
        if peaks is None or not rows:
            return
        n = len(np.asarray(peaks.get("two_theta", [])))
        drop = np.asarray(sorted({int(r) for r in rows if 0 <= int(r) < n}), dtype=int)
        if len(drop) == 0:
            return

        if len(drop) >= n:
            self.session.set_peaks(None)
            self.clear_peaks_table()
            self.process_stage.peak_status.setText("All peaks deleted.")
            self.set_status("Deleted all peaks")
            self.refresh_plot()
            return

        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        cleaned = dict(peaks)
        for key, value in peaks.items():
            arr = np.asarray(value)
            if arr.ndim == 1 and len(arr) == n:
                cleaned[key] = arr[keep]

        self.session.set_peaks(cleaned)
        self.set_results_peaks(cleaned, select_row=int(drop[0]))
        removed = len(drop)
        self.process_stage.peak_status.setText(
            f"Deleted {removed} peak{'s' if removed > 1 else ''}; "
            f"{len(cleaned['two_theta'])} remain."
        )
        self.set_status(f"Deleted {removed} peak{'s' if removed > 1 else ''}")

    # --- manual peak editing ---

    def set_peak_edit_mode(self, enabled: bool):
        """Turn plot clicks into peak edits."""
        self._peak_edit = bool(enabled)
        if self._peak_edit and self._emphasis_mode:
            # Both want the left button; emphasis has its own toggle to clear
            self.identify_stage.emphasis_btn.setChecked(False)
        self.canvas.setCursor(Qt.CrossCursor if self._peak_edit else Qt.ArrowCursor)
        if self._peak_edit:
            self.show_bottom_tab("peaks")
            self.process_stage.peak_status.setText(
                "Editing peaks: left-click the plot to add one, right-click a peak to "
                "remove it. Find Peaks rebuilds the list and discards manual peaks."
            )
            self.set_status("Peak editing on")
        else:
            self.set_status("Peak editing off")

    def _on_plot_click(self, event):
        """Add or remove a peak at the clicked 2θ while edit mode is on."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        # Pan and zoom own the mouse while a toolbar tool is armed
        if str(getattr(self.toolbar, "mode", "")):
            return
        if self._emphasis_mode and event.button == 3:
            self._remove_emphasis_at(float(event.xdata))
            return
        if not self._peak_edit:
            return
        if event.button == 1:
            self.add_peak_at(float(event.xdata))
        elif event.button == 3:
            self.remove_peak_near(float(event.xdata))

    # --- emphasised regions ---

    def set_emphasis_mode(self, enabled: bool):
        """Turn plot drags into priority regions for search/match."""
        self._emphasis_mode = bool(enabled)
        if self._emphasis_mode and self._peak_edit:
            # Both want the left button; peak editing has its own toggle to clear
            self.process_stage.edit_peaks_btn.setChecked(False)
        self.canvas.setCursor(
            Qt.SizeHorCursor if self._emphasis_mode else Qt.ArrowCursor
        )
        self._arm_emphasis_selector()
        if self._emphasis_mode:
            self.set_status(
                "Emphasis on: drag across the plot to prioritise a 2θ range, "
                "right-click a shaded band to drop it"
            )
        else:
            self.set_status("Emphasis drawing off")

    def _arm_emphasis_selector(self):
        """
        (Re)build the span selector.

        Every refresh_plot clears the axes, which throws away the selector's
        rectangle, so the selector has to be rebuilt alongside the plot rather
        than once at startup.
        """
        if self._emphasis_selector is not None:
            self._emphasis_selector.set_active(False)
            self._emphasis_selector = None
        if not self._emphasis_mode:
            return
        self._emphasis_selector = SpanSelector(
            self.ax,
            self._on_emphasis_span,
            "horizontal",
            useblit=False,
            button=1,
            minspan=emphasis.MIN_SPAN,
            props={"facecolor": "#7e57c2", "alpha": 0.25},
        )

    def _on_emphasis_span(self, lo: float, hi: float):
        if str(getattr(self.toolbar, "mode", "")):
            return
        weight = float(self.identify_stage.emphasis_weight.value())
        self.session.add_emphasis_region(lo, hi, weight)
        self.set_status(
            f"Emphasising {min(lo, hi):.2f}–{max(lo, hi):.2f}° at ×{weight:g} — "
            "run Search to use it"
        )

    def _remove_emphasis_at(self, two_theta: float):
        if self.session.remove_emphasis_at(two_theta):
            self.set_status(f"Dropped the emphasis region at {two_theta:.2f}°")

    def _draw_emphasis(self, ax):
        """Shade the prioritised ranges behind everything else."""
        regions = self.session.emphasis_regions
        if not regions:
            return
        weights = {r["weight"] for r in regions}
        label = f"Emphasis ×{weights.pop():g}" if len(weights) == 1 else "Emphasis"
        xlim = ax.get_xlim()  # a band drawn past the data must not pull the view
        for i, region in enumerate(regions):
            ax.axvspan(
                region["lo"], region["hi"],
                facecolor="#7e57c2", alpha=0.15, zorder=0,
                label=label if i == 0 else None,
            )
        ax.set_xlim(xlim)

    def _click_tolerance(self) -> float:
        """Hit window for a click: generous zoomed out, tight zoomed in."""
        xlim = self.ax.get_xlim()
        span = abs(float(xlim[1]) - float(xlim[0]))
        return max(float(self.process_stage.min_sep.value()), 0.01 * span)

    def add_peak_at(self, two_theta: float):
        """Add a peak the detector missed, snapped to the nearest apex."""
        pattern = self.session.active_pattern()
        if pattern is None:
            self.set_status("Load a pattern before adding peaks")
            return
        tt = np.asarray(pattern["two_theta"], dtype=float)
        inten = np.asarray(pattern["intensity"], dtype=float)
        if len(tt) == 0:
            return

        min_sep = float(self.process_stage.min_sep.value())
        step = ProcessStage._median_step(tt)
        half = max(2, int(round(min_sep / 2.0 / max(step, 1e-6))))
        idx = ProcessStage._refine_to_local_max(
            inten, int(np.argmin(np.abs(tt - two_theta))), half_window=half
        )
        peak_tt = float(tt[idx])

        peaks = self.session.peaks
        existing = np.asarray((peaks or {}).get("two_theta", []), dtype=float)
        if len(existing):
            nearest = int(np.argmin(np.abs(existing - peak_tt)))
            if abs(float(existing[nearest]) - peak_tt) < min_sep:
                self.peaks_table.setCurrentCell(nearest, 0)
                self.set_status(
                    f"A peak already sits at {float(existing[nearest]):.3f}°"
                )
                return

        with np.errstate(divide="ignore", invalid="ignore"):
            d = float(self.session.wavelength) / (
                2.0 * np.sin(np.radians(peak_tt / 2.0))
            )
        d = float(d) if np.isfinite(d) else float("nan")

        updated, row = self._insert_peak(
            peaks,
            peak_tt,
            {
                "indices": int(idx),
                "two_theta": peak_tt,
                "intensity": float(inten[idx]),
                "d_spacing": d,
                "manual": True,
            },
        )
        self.session.set_peaks(updated)
        self.set_results_peaks(updated, select_row=row)
        d_note = f" (d = {d:.4f} Å)" if np.isfinite(d) else ""
        self.process_stage.peak_status.setText(
            f"Added peak at {peak_tt:.3f}°{d_note}; "
            f"{len(updated['two_theta'])} peaks total."
        )
        self.set_status(f"Added peak at {peak_tt:.3f}°")
        self.refresh_plot()

    def remove_peak_near(self, two_theta: float):
        """Delete the peak closest to a click, if one is close enough."""
        tt = np.asarray((self.session.peaks or {}).get("two_theta", []), dtype=float)
        if len(tt) == 0:
            self.set_status("No peaks to remove")
            return
        row = int(np.argmin(np.abs(tt - two_theta)))
        tol = self._click_tolerance()
        if abs(float(tt[row]) - two_theta) > tol:
            self.set_status(f"No peak within {tol:.2f}° of {two_theta:.2f}°")
            return
        self.delete_peak_rows([row])

    def _insert_peak(self, peaks: Optional[dict], two_theta: float, values: dict):
        """
        Splice one peak into the session's peak arrays, keeping 2θ order.

        Every column is extended, not just the ones this widget knows about, so a
        peak list that has been through Kα2 stripping or residual bookkeeping
        stays internally consistent.
        """
        if peaks is None or len(np.asarray(peaks.get("two_theta", []))) == 0:
            fresh = {
                "indices": np.array([values["indices"]], dtype=int),
                "two_theta": np.array([two_theta], dtype=float),
                "intensity": np.array([values["intensity"]], dtype=float),
                "d_spacing": np.array([values["d_spacing"]], dtype=float),
                "manual": np.array([True]),
                "wavelength": float(self.session.wavelength),
            }
            return fresh, 0

        positions = np.asarray(peaks["two_theta"], dtype=float)
        n = len(positions)
        row = int(np.searchsorted(positions, two_theta))
        updated = dict(peaks)
        for key, value in peaks.items():
            arr = np.asarray(value)
            if arr.ndim != 1 or len(arr) != n:
                continue
            if key in values:
                fill = values[key]
            elif arr.dtype == bool:
                fill = False
            elif np.issubdtype(arr.dtype, np.integer):
                fill = 0
            else:
                fill = np.nan
            updated[key] = np.insert(arr, row, fill)
        if "manual" not in peaks:
            updated["manual"] = np.insert(np.zeros(n, dtype=bool), row, True)
        return updated, row

    @staticmethod
    def _lines_cell(result: dict) -> QTableWidgetItem:
        """Fingerprint line count, e.g. 9/10, flagged when the top line is absent."""
        fp = result.get("fingerprint")
        if not fp:
            item = QTableWidgetItem("—")
            item.setToolTip("No fingerprint scoring for this hit")
            return item
        text = f"{fp['n_found']}/{fp['n_expected']}"
        if not fp.get("top_found", True):
            text += " ⚠"
        item = QTableWidgetItem(text)
        tip = [
            f"Fingerprint score {fp['score']:.3f}",
            f"{fp['n_found']} of {fp['n_expected']} strong lines present",
            "Strongest line present" if fp.get("top_found") else "Strongest line MISSING",
        ]
        if fp.get("intensity_consistency") is not None:
            tip.append(
                f"Intensity consistency {fp['intensity_consistency']:.2f} "
                "(1.0 = every line has enough observed intensity)"
            )
        if fp.get("residual_score") is not None and fp["residual_score"] != fp["score"]:
            tip.append(f"Residual score {fp['residual_score']:.3f} (new peaks only)")
        if fp.get("shift"):
            tip.append(
                "Lines placed at "
                + describe_shift(fp["shift"], fp.get("shift_model", DISPLACEMENT))
            )
        missing = fp.get("missing_strong") or []
        if missing:
            tip.append("Missing: " + ", ".join(f"{m:.2f}°" for m in missing[:6]))
        item.setToolTip("\n".join(tip))
        return item

    @staticmethod
    def _mineral_item(name: str, result: dict) -> QTableWidgetItem:
        """Name cell carrying formula and cell data in its tooltip."""
        item = QTableWidgetItem(str(name))
        phase = result.get("phase", result)
        src = {**result, **phase} if isinstance(phase, dict) else result
        formula = src.get("chemical_formula") or src.get("formula") or "—"
        tip = [str(name), f"Formula: {formula}", f"Space group: {src.get('space_group') or '—'}"]
        cell = [src.get("cell_a"), src.get("cell_b"), src.get("cell_c")]
        if all(v for v in cell):
            tip.append("a, b, c: " + ", ".join(f"{float(v):.4f}" for v in cell))
        tip.append("Right-click for full details")
        item.setToolTip("\n".join(tip))
        return item

    def set_results_candidates(self, results: list):
        self._results_mode = "candidates"
        self._candidate_results = list(results)
        self._preview = None
        self.show_bottom_tab("phases")
        self.phases_label.setText(
            f"Candidates ({len(results)}) — click to preview, check to match, "
            "right-click for details"
        )
        self.results_table.blockSignals(True)
        self.results_table.clear()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Mineral", "Score", "Lines", "✓"])
        self.results_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.results_table.setItem(i, 0, self._mineral_item(r.get("mineral_name", ""), r))
            score = r.get(
                "fingerprint_score",
                r.get("ensemble_score",
                      r.get("combined_score", r.get("correlation", r.get("match_score", 0)))),
            )
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{float(score):.3f}"))
            self.results_table.setItem(i, 2, self._lines_cell(r))
            cb = QCheckBox()
            cb.setChecked(False)  # user must opt in
            cb.stateChanged.connect(self._on_candidate_checked)
            self.results_table.setCellWidget(i, CAND_SELECT_COL, cb)
        self.results_table.blockSignals(False)
        self._size_results_columns(stretch_cols=(0,))
        self.identify_stage.update_action_states()

    def check_candidate_rows(self, names: list):
        """Check rows whose mineral name matches (used for manually added phases)."""
        if self._results_mode != "candidates" or not names:
            return
        wanted = {str(n).lower() for n in names if n}
        for i in range(self.results_table.rowCount()):
            item = self.results_table.item(i, 0)
            if item and item.text().lower() in wanted:
                cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
                if cb:
                    cb.setChecked(True)

    def get_selected_candidates(self) -> list:
        """Return only explicitly checked candidates (no auto-fallback)."""
        results = getattr(self, "_candidate_results", None) or []
        if self._results_mode != "candidates" or not results:
            return []
        selected = []
        for i, r in enumerate(results):
            cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
            if cb is not None and cb.isChecked():
                selected.append(self.identify_stage._result_to_phase(r))
        return selected

    def set_results_matches(self, results: list, preselect: Optional[list] = None):
        """
        Show matched phases. Nothing is checked unless `preselect` provides
        keys/names (or match objects) that should stay selected.
        """
        self._results_mode = "matches"
        self._preview = None
        self.show_bottom_tab("phases")
        self.phases_label.setText(
            f"Matched ({len(results)}) — check to keep and plot, click to preview, "
            "right-click for details"
        )
        self.results_table.blockSignals(True)
        self.results_table.clear()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["✓", "Phase", "Score", "Lines"])
        self.results_table.setRowCount(len(results))

        from utils.residual_search import mineral_key

        preselect_keys = set()
        if preselect:
            for p in preselect:
                preselect_keys.add(mineral_key(p))

        for i, r in enumerate(results):
            phase = r.get("phase", r)
            name = phase.get("mineral", phase.get("mineral_name", f"Phase {i+1}"))
            cb = QCheckBox()
            cb.setChecked(bool(preselect_keys) and mineral_key(r) in preselect_keys)
            cb.stateChanged.connect(self._sync_selected_from_table)
            self.results_table.setCellWidget(i, MATCH_SELECT_COL, cb)
            item = self._mineral_item(name, r)
            cov = r.get("coverage", 0)
            if cov:
                item.setToolTip(f"{item.toolTip()}\nCoverage: {float(cov):.2f}")
            self.results_table.setItem(i, 1, item)
            score = r.get("combined_score", r.get("match_score", 0))
            score_item = QTableWidgetItem(f"{float(score):.3f}")
            score_item.setToolTip(
                f"Matched peaks: {len(r.get('matches', []))}"
                + (f"\nCoverage: {float(cov):.2f}" if cov else "")
            )
            self.results_table.setItem(i, 2, score_item)
            self.results_table.setItem(i, 3, self._lines_cell(r))
        self.results_table.blockSignals(False)
        self._size_results_columns(stretch_cols=(1,))
        self._sync_selected_from_table()

    def _size_results_columns(self, stretch_cols=(0,)):
        """Give the name columns the slack instead of the checkbox column."""
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(False)
        self.results_table.resizeColumnsToContents()
        for col in range(self.results_table.columnCount()):
            mode = QHeaderView.Stretch if col in stretch_cols else QHeaderView.ResizeToContents
            header.setSectionResizeMode(col, mode)

    def get_selected_matches(self) -> list:
        if self._results_mode != "matches":
            return list(self.session.selected_phases)
        selected = []
        matches = self.session.matched_phases
        for i in range(self.results_table.rowCount()):
            cb = self.results_table.cellWidget(i, MATCH_SELECT_COL)
            if cb and cb.isChecked() and i < len(matches):
                selected.append(matches[i])
        return selected

    # --- row preview / details / context menu ---

    def _result_at_row(self, row: int) -> Optional[dict]:
        if row < 0:
            return None
        if self._results_mode == "candidates":
            results = self._candidate_results
        elif self._results_mode == "matches":
            results = self.session.matched_phases
        else:
            return None
        return results[row] if row < len(results) else None

    def current_result(self) -> Optional[dict]:
        """The phase highlighted in the list, whatever the list is showing."""
        return self._result_at_row(self.results_table.currentRow())

    def _on_results_row_changed(self, row: int, _col=0, _prow=-1, _pcol=0):
        """Preview the highlighted phase; driven by clicks and arrow keys alike."""
        result = self._result_at_row(row)
        if result is None:
            return
        theo = self.identify_stage.reference_peaks_for(result)
        if not theo or len(theo.get("two_theta", [])) == 0:
            self._preview = None
            self.set_status("No reference peaks available for preview")
            self.refresh_plot()
            return
        phase = result.get("phase", result)
        name = (
            phase.get("mineral")
            or phase.get("mineral_name")
            or result.get("mineral_name")
            or "Preview"
        )
        self._preview = {
            "name": str(name),
            "two_theta": np.asarray(theo["two_theta"], dtype=float),
            "intensity": np.asarray(theo["intensity"], dtype=float),
            "result": result,
        }
        self.refresh_plot()

    def clear_preview(self):
        self._preview = None
        self.refresh_plot()

    def show_selected_phase_details(self):
        row = self.results_table.currentRow()
        if row < 0:
            QMessageBox.information(
                self, "No Phase Selected", "Click a phase in the table first."
            )
            return
        self.show_phase_details(row)

    def show_phase_details(self, row: int):
        result = self._result_at_row(row)
        if result is None:
            return
        from gui.dialogs.phase_details_dialog import PhaseDetailsDialog

        if self._details_dialog is None:
            self._details_dialog = PhaseDetailsDialog(self.window())
        theo = self.identify_stage.reference_peaks_for(result)
        self._details_dialog.show_phase(result, theo)

    def _show_results_menu(self, pos):
        row = self.results_table.rowAt(pos.y())
        if row < 0 or self._result_at_row(row) is None:
            return
        self.results_table.selectRow(row)

        select_col = CAND_SELECT_COL if self._results_mode == "candidates" else MATCH_SELECT_COL
        cb = self.results_table.cellWidget(row, select_col)

        menu = QMenu(self)
        menu.addAction("Details…", lambda: self.show_phase_details(row))
        menu.addAction("Preview peaks", lambda: self._on_results_row_changed(row))
        menu.addAction("Clear preview", self.clear_preview)
        if cb is not None:
            label = "Uncheck" if cb.isChecked() else "Check"
            menu.addSeparator()
            menu.addAction(label, lambda: cb.setChecked(not cb.isChecked()))
        menu.addSeparator()
        menu.addAction("Remove from list", lambda: self.remove_phase_row(row))
        menu.exec_(self.results_table.viewport().mapToGlobal(pos))

    def remove_phase_row(self, row: int):
        if self._results_mode == "candidates":
            checked = [
                i for i in range(self.results_table.rowCount())
                if (w := self.results_table.cellWidget(i, CAND_SELECT_COL)) and w.isChecked()
            ]
            results = [r for i, r in enumerate(self._candidate_results) if i != row]
            keep_names = [
                (self._candidate_results[i].get("mineral_name") or "")
                for i in checked if i != row
            ]
            self.session.set_candidates(
                [self.identify_stage._result_to_phase(r) for r in results]
            )
            self.identify_stage._search_results = results
            self.set_results_candidates(results)
            self.check_candidate_rows(keep_names)
            self.set_status("Removed candidate")
        elif self._results_mode == "matches":
            kept_selected = self.get_selected_matches()
            matches = [r for i, r in enumerate(self.session.matched_phases) if i != row]
            removed = self._result_at_row(row)
            kept_selected = [r for r in kept_selected if r is not removed]
            self.session.set_matched_phases(matches)
            self.session.set_selected_phases(kept_selected)
            self.set_results_matches(matches, preselect=kept_selected)
            self.set_status("Removed phase")

    def clear_all_phases(self):
        """Wipe candidates, matches, and selections."""
        if self.results_table.rowCount() == 0 and not self.session.matched_phases:
            self.set_status("Phase list is already empty")
            return
        if (
            QMessageBox.question(
                self, "Clear All Minerals",
                "Remove all minerals from the list, including matched and selected phases?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return

        self._candidate_results = []
        self._results_mode = None
        self._preview = None
        self.identify_stage.reset_results()
        self.session.set_candidates([])
        self.session.set_matched_phases([])
        self.session.set_selected_phases([])
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self.phases_label.setText("Phases — run a search or add a known mineral")
        self.identify_stage.on_enter()
        self.set_status("Cleared all minerals")
        self.refresh_plot()

    def clear_unselected_phases(self):
        """Drop unchecked rows; keep only user-selected phases."""
        if self._results_mode == "candidates":
            kept_results = []
            kept_phases = []
            for i, r in enumerate(self._candidate_results):
                cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
                if cb is not None and cb.isChecked():
                    kept_results.append(r)
                    kept_phases.append(self.identify_stage._result_to_phase(r))
            if not kept_phases:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "Nothing Selected",
                    "Select one or more candidates first, then Clear Unselected.",
                )
                return
            self.session.set_candidates(kept_phases)
            self.identify_stage._search_results = kept_results
            self.set_results_candidates(kept_results)
            # Re-check the kept ones so the user doesn't lose selection
            for i in range(self.results_table.rowCount()):
                cb = self.results_table.cellWidget(i, CAND_SELECT_COL)
                if cb:
                    cb.setChecked(True)
            self.set_status(f"Kept {len(kept_phases)} candidate(s)")
            return

        if self._results_mode == "matches":
            kept = self.get_selected_matches()
            if not kept:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "Nothing Selected",
                    "Select one or more matched phases first, then Clear Unselected.",
                )
                return
            self.session.set_matched_phases(kept)
            self.session.set_selected_phases(kept)
            self.set_results_matches(kept, preselect=kept)
            self.set_status(f"Kept {len(kept)} matched phase(s)")
            return

        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "No List", "Run a search or match first.")

    def _sync_selected_from_table(self):
        if self._results_mode != "matches":
            return
        selected = []
        matches = self.session.matched_phases
        for i in range(self.results_table.rowCount()):
            cb = self.results_table.cellWidget(i, MATCH_SELECT_COL)
            if cb and cb.isChecked() and i < len(matches):
                selected.append(matches[i])
        self.session.set_selected_phases(selected)
        self.identify_stage.update_action_states()
        self.refresh_plot()

    def _on_candidate_checked(self, *_):
        """Checked candidates count as accepted phases: plot them and enable actions."""
        if self._results_mode != "candidates":
            return
        accepted = self.identify_stage.accepted_phases()
        self.session.set_selected_phases(accepted)
        self.identify_stage.update_action_states()
        self.refresh_plot()

    # --- plotting ---

    def refresh_plot(self):
        mode = get_current_mode()
        ax = self.ax
        ax.clear()
        palette = get_plot_palette(mode)

        if self.session.has_matches() or self.session.selected_phases:
            self._plot_identify(ax, palette)
        elif self.session.processed_pattern is not None or self.session.has_peaks():
            self._plot_process(ax, palette)
        else:
            self._plot_load(ax, palette)

        self._draw_emphasis(ax)
        self._draw_preview(ax)
        self._draw_peak_highlight(ax)
        if display_settings.show_legend() and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)

        apply_plot_style(self.figure, mode, show_grid=display_settings.show_grid())
        self._arm_emphasis_selector()
        self.canvas.draw_idle()

    @staticmethod
    def _draw_error_bars(ax, two_theta, intensity, errors, color, scale=1.0):
        """Whiskers from an XYE file's third column, when the user wants them."""
        if not display_settings.show_error_bars():
            return
        draw_error_bars(ax, two_theta, intensity, errors, color, scale)

    def _draw_manual_peaks(self, ax, peaks, heights):
        """Ring hand-placed peaks so they read differently from detected ones."""
        manual = np.asarray(peaks.get("manual", []), dtype=bool)
        tt = np.asarray(peaks.get("two_theta", []), dtype=float)
        if len(manual) != len(tt) or not manual.any():
            return
        ax.plot(
            tt[manual], np.asarray(heights, dtype=float)[manual], "s",
            mfc="none", mec="#d81b60", mew=1.3,
            ms=display_settings.marker_size(2.0), ls="none",
            label="Manual peaks", zorder=6,
        )

    def _draw_peak_highlight(self, ax):
        """Mark the peak selected in the Peaks table."""
        if not self._peak_highlight:
            return
        tt = self._peak_highlight["two_theta"]
        xlim = ax.get_xlim()
        if not (xlim[0] <= tt <= xlim[1]):
            return
        d = self._peak_highlight.get("d_spacing")
        label = f"Selected peak {tt:.3f}°"
        if d is not None and np.isfinite(d):
            label += f"  (d = {d:.4f} Å)"
        ax.axvline(tt, color="#d81b60", lw=1.2, ls="--", alpha=0.9, label=label, zorder=6)
        ax.set_xlim(xlim)

    def _draw_preview(self, ax):
        """Overlay the highlighted candidate's reference lines."""
        if not self._preview:
            return
        # Re-resolve so the overlay tracks the 2θ shift as the user dials it
        theo = self.identify_stage.reference_peaks_for(self._preview.get("result"))
        if theo and len(theo.get("two_theta", [])) > 0:
            tt = np.asarray(theo["two_theta"], dtype=float)
            inten = np.asarray(theo["intensity"], dtype=float)
        else:
            tt = self._preview["two_theta"]
            inten = self._preview["intensity"]
        if len(tt) == 0:
            return
        top = ax.get_ylim()[1] or 1.0
        imax = float(np.max(inten)) if len(inten) and np.max(inten) > 0 else 1.0
        heights = inten / imax * top * 0.75

        label = f"Preview: {self._preview['name']}"
        shift = (theo or {}).get("two_theta_shift") or 0.0
        if shift:
            label += f" ({shift:+.3f}° shift)"

        # Reference lines often run past the measured range; keep the view fixed
        xlim = ax.get_xlim()
        inside = (tt >= xlim[0]) & (tt <= xlim[1])
        if not np.any(inside):
            return
        ax.vlines(
            tt[inside], 0, heights[inside], colors="#e0a300", lw=1.4, alpha=0.9,
            label=label, zorder=5,
        )
        ax.set_xlim(xlim)

    def _plot_load(self, ax, palette):
        pattern = self.session.raw_pattern
        ax.set_title("XRD Pattern")
        if not pattern:
            ax.text(
                0.5, 0.5, "Choose a folder and click a pattern file",
                ha="center", va="center", transform=ax.transAxes,
                color=palette.get("muted", palette["tick"]),
            )
            ax.set_xlabel("2θ (degrees)")
            ax.set_ylabel("Intensity")
            return
        ax.plot(
            pattern["two_theta"], pattern["intensity"],
            color=palette["exp_line"], lw=display_settings.line_width(),
            label="Experimental",
        )
        self._draw_error_bars(
            ax, pattern["two_theta"], pattern["intensity"],
            pattern.get("intensity_error"), palette["exp_line"],
        )
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity")
        fmt = pattern.get("file_format", "")
        ax.set_title(f"XRD Pattern ({fmt})" if fmt else "XRD Pattern")

    def _plot_process(self, ax, palette):
        raw = self.session.raw_pattern
        processed = self.session.processed_pattern
        if raw is not None and self._visible("raw"):
            ax.plot(
                raw["two_theta"], raw["intensity"],
                color=palette["diff_line"], lw=display_settings.line_width(0.7),
                alpha=0.5, label="Raw",
            )
        if processed is not None:
            if self._visible("processed"):
                ax.plot(
                    processed["two_theta"], processed["intensity"],
                    color=palette["exp_line"], lw=display_settings.line_width(),
                    label="Processed",
                )
                self._draw_error_bars(
                    ax, processed["two_theta"], processed["intensity"],
                    processed.get("intensity_error"), palette["exp_line"],
                )
            bg = self.session.background
            if bg is not None and raw is not None and self._visible("background"):
                ax.plot(
                    raw["two_theta"], bg, color=palette["calc_line"],
                    lw=display_settings.line_width(0.85), ls="--", label="Background",
                )
        peaks = self.session.peaks
        if peaks is not None and processed is not None and self._visible("peaks"):
            ax.plot(
                peaks["two_theta"], peaks["intensity"], "o",
                color=palette["calc_line"], ms=display_settings.marker_size(),
                label="Peaks",
            )
            self._draw_manual_peaks(ax, peaks, peaks["intensity"])
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity")
        ax.set_title("Processing Preview")

    def _reference_heights(self, result, tt, ti, two_theta, norm, max_i):
        """
        Reference line heights on the same 0-100 scale as the plotted pattern.

        Normalizing each phase to its own strongest line draws a trace phase as
        tall as a major one, which makes a correct minor phase look like a bad
        fit. Heights come from the joint Le Bail contribution when a refinement
        has produced one, and otherwise from fitting the phase's own lines to the
        observed curve.
        """
        tmax = float(np.max(ti)) if len(ti) else 0.0
        if tmax <= 0:
            return None
        rel = ti / tmax * 100.0
        if not self._visible("scaled"):
            return rel * 0.8
        if len(two_theta) < 2:
            return rel * 0.8

        contribution = result.get("contribution")
        if contribution is not None:
            contribution = np.asarray(contribution, dtype=float)
            if len(contribution) == len(two_theta) and np.max(contribution) > 0 and max_i > 0:
                return np.interp(tt, two_theta, contribution / max_i * 100.0)

        # Lines below a few percent sit in the noise, so they cannot pin a scale
        inside = (tt >= two_theta[0]) & (tt <= two_theta[-1])
        use = inside & (rel >= 10.0)
        if np.count_nonzero(use) < 2:
            use = inside & (rel > 0.0)
        if not np.any(use):
            return None

        # Peak apex within tolerance, so a slightly offset line is not read as absent
        tol = float(self.identify_stage.tolerance.value())
        lo = np.searchsorted(two_theta, tt[use] - tol, side="left")
        hi = np.searchsorted(two_theta, tt[use] + tol, side="right")
        observed = np.array([
            float(np.max(norm[a:b])) if b > a else 0.0 for a, b in zip(lo, hi)
        ])

        ratios = observed / rel[use]
        # A low quantile rather than the median: lines overlapping another phase
        # read high, and anchoring on those inflates the whole phase
        scale = (
            float(np.percentile(ratios, 25)) if len(ratios) >= 4
            else float(np.min(ratios))
        )
        if not np.isfinite(scale) or scale <= 0:
            # Nothing to anchor on; keep the lines faintly visible rather than
            # dropping the phase off the plot with no explanation
            scale = 0.03
        return rel * scale

    def _plot_identify(self, ax, palette):
        pattern = self.session.active_pattern()
        if pattern is None:
            ax.set_title("Identify")
            return
        inten = np.asarray(pattern["intensity"], dtype=float)
        two_theta = np.asarray(pattern["two_theta"], dtype=float)
        data_range = (float(np.min(two_theta)), float(np.max(two_theta))) if len(two_theta) else None
        max_i = np.max(inten) if len(inten) else 1.0
        norm = (inten / max_i * 100.0) if max_i > 0 else inten
        if self._visible("processed"):
            ax.plot(
                pattern["two_theta"], norm, color=palette["exp_line"],
                lw=display_settings.line_width(), label="Experimental",
            )
            # Errors ride the same 0-100 normalization as the pattern
            self._draw_error_bars(
                ax, pattern["two_theta"], norm, pattern.get("intensity_error"),
                palette["exp_line"], scale=(100.0 / max_i) if max_i > 0 else 1.0,
            )

        raw = self.session.raw_pattern
        if raw is not None and raw is not pattern and self._visible("raw"):
            rint = np.asarray(raw["intensity"], dtype=float)
            rmax = np.max(rint) if len(rint) else 1.0
            ax.plot(
                raw["two_theta"], (rint / rmax * 100.0) if rmax > 0 else rint,
                color=palette["diff_line"], lw=display_settings.line_width(0.7),
                alpha=0.45, label="Raw",
            )

        bg = self.session.background
        if bg is not None and raw is not None and self._visible("background"):
            bgn = np.asarray(bg, dtype=float)
            rint = np.asarray(raw["intensity"], dtype=float)
            rmax = np.max(rint) if len(rint) else 1.0
            ax.plot(
                raw["two_theta"], (bgn / rmax * 100.0) if rmax > 0 else bgn,
                color=palette["calc_line"], lw=display_settings.line_width(0.85),
                ls="--", label="Background",
            )

        colors = ["#c45c26", "#7a5cff", "#2a7a4b", "#b33a3a", "#5a6a7a"]
        selected = self.session.selected_phases
        tallest = 0.0
        for i, result in enumerate(selected[:5]):
            theo = self.identify_stage.reference_peaks_for(result)
            if not theo or len(theo.get("two_theta", [])) == 0:
                continue
            tt = np.asarray(theo["two_theta"])
            ti = np.asarray(theo["intensity"], dtype=float)
            tnorm = self._reference_heights(result, tt, ti, two_theta, norm, float(max_i))
            if tnorm is None:
                continue
            tallest = max(tallest, float(np.max(tnorm)) if len(tnorm) else 0.0)
            phase = result.get("phase", {})
            name = phase.get("mineral", f"Phase {i+1}")
            ax.vlines(
                tt, 0, tnorm, colors=colors[i % len(colors)],
                alpha=0.7, lw=display_settings.line_width(0.85), label=name,
            )

        peaks = self.session.peaks
        if peaks is not None and self._visible("peaks"):
            pi = np.asarray(peaks["intensity"], dtype=float)
            pmax = np.max(pi) if len(pi) else 1.0
            heights = (pi / pmax * 100.0) if pmax > 0 else pi
            ax.plot(
                peaks["two_theta"], heights, "o", color=palette["calc_line"],
                ms=display_settings.marker_size(0.75), label="Peaks",
            )
            self._draw_manual_peaks(ax, peaks, heights)

        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Normalized Intensity")
        ax.set_title("Phase Identification")
        # Leave room for a phase that over-predicts, but do not let one runaway
        # line squash the pattern into the baseline
        ax.set_ylim(0, max(110.0, min(160.0, tallest * 1.05)))
        # Reference lines run past the measurement; keep the view on the data
        if data_range is not None:
            ax.set_xlim(*data_range)
