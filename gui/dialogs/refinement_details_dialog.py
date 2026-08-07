"""Refinement control window — parameters, run, and export."""

from __future__ import annotations

import csv

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from gui import refinement_table
from gui.focus import restores_focus
from gui.stages.refine_stage import RefineStage
from gui.widgets.copyable_table import CopyableTable
from gui.widgets.parameter_matrix import ParameterMatrix


class RefinementDetailsDialog(QDialog):
    """
    Everything needed to set up and run a refinement.

    Global and per-phase controls, the editable parameter matrix, Run / Export,
    and the post-run tables live here so the Quant window can stay a plot.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Refinement Parameters")
        self.setWindowModality(Qt.NonModal)
        self.resize(1180, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # RefineStage uses this dialog as its workspace; plot/status forward to Quant
        self.refine_stage = RefineStage(session, self)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.refine_stage)
        left.setMinimumWidth(300)
        left.setMaximumWidth(440)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.headline = QLabel()
        self.headline.setObjectName("mutedLabel")
        self.headline.setWordWrap(True)
        right_layout.addWidget(self.headline)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_matrix_tab(), "Phases")
        self.tabs.addTab(self._build_table_tab("global_table"), "Statistics")
        self.tabs.addTab(self._build_table_tab("phase_table"), "All values")
        right_layout.addWidget(self.tabs, 1)

        buttons = QHBoxLayout()
        self.hint = QLabel()
        self.hint.setObjectName("mutedLabel")
        self.hint.setWordWrap(True)
        buttons.addWidget(self.hint, 1)
        copy_btn = QPushButton("Copy all")
        copy_btn.setToolTip("Copy both tables to the clipboard")
        copy_btn.clicked.connect(self.copy_all)
        buttons.addWidget(copy_btn)
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_csv)
        buttons.addWidget(export_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        right_layout.addLayout(buttons)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 820])
        root.addWidget(splitter)

        session.refinement_changed.connect(self.refresh)
        self.refresh()

    # --- workspace API for RefineStage (forwards to Quant parent) -----------

    def refresh_plot(self):
        parent = self.parent()
        if parent is not None and hasattr(parent, "refresh_plot"):
            parent.refresh_plot()

    def set_status(self, message: str):
        parent = self.parent()
        if parent is not None and hasattr(parent, "set_status"):
            parent.set_status(message)

    @property
    def quant_figure(self):
        parent = self.parent()
        return getattr(parent, "quant_figure", None) if parent is not None else None

    @property
    def figure(self):
        parent = self.parent()
        if parent is None:
            return None
        return getattr(parent, "quant_figure", None) or getattr(parent, "figure", None)

    # --- construction ------------------------------------------------------

    def _build_matrix_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(4)

        caption = QLabel(
            "Tick to refine a term for that phase; untick to hold it at the "
            "value shown, which you can edit. Changes apply to the next run. "
            "The phase checkboxes on the left set defaults for every phase; "
            "this grid overrides them one phase at a time."
        )
        caption.setObjectName("mutedLabel")
        caption.setWordWrap(True)
        layout.addWidget(caption)

        self.matrix = ParameterMatrix()
        self.matrix.changed.connect(self._store_overrides)
        layout.addWidget(self.matrix, 1)

        row = QHBoxLayout()
        row.addStretch()
        reset = QPushButton("Reset to defaults")
        reset.setToolTip("Discard every hand-set value and flag on this tab.")
        reset.clicked.connect(self._reset_overrides)
        row.addWidget(reset)
        layout.addLayout(row)
        return wrapper

    def _build_table_tab(self, attribute: str) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 6, 0, 0)
        table = CopyableTable()
        table.horizontalHeader().setStretchLastSection(True)
        setattr(self, attribute, table)
        layout.addWidget(table)
        return wrapper

    # --- state -------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self.refine_stage, "on_enter"):
            self.refine_stage.on_enter()
        self.refresh()

    def refresh(self):
        results = self.session.lebail_results
        overrides = getattr(self.session, "phase_overrides", {}) or {}

        if results and results.get("success"):
            parts = refinement_table.summary_headline(results)
            self.headline.setText("  ·  ".join(parts) if parts else "Refinement")
            self.global_table.set_content(
                ["Parameter", "Value"], refinement_table.global_rows(results)
            )
            names = refinement_table.phase_names(results)
            self.phase_table.set_content(
                ["Parameter"] + names, refinement_table.detail_rows(results)
            )
            values = refinement_table.phase_parameters(results, overrides)
            quantitative = (
                (results.get("refinement_results") or {}).get("intensity_model") == "fixed"
            )
        else:
            self.headline.setText(
                "No refinement yet — set starting values here, then run Le Bail."
            )
            self.global_table.set_content([], [])
            self.phase_table.set_content([], [])
            names, values = self._phases_before_first_run(overrides)
            quantitative = True

        self.matrix.set_phases(names, values, quantitative=quantitative)
        self._update_hint(overrides)

    # Matches the refine-stage checkboxes so the grid opens showing what a run
    # with no overrides would actually do, rather than a blank that looks like
    # every term is held.
    _DEFAULT_FLAGS = {
        "refine_scale": True,
        "refine_strain": True,
        "refine_size": False,
        "refine_asymmetry": False,
        "refine_cell": True,
        "refine_absorption": False,
        "refine_harmonics": False,
        "scale_factor": 1.0,
        "microstrain": 1000.0,
        "crystallite_size": 1.0,
        "asymmetry": 0.0,
        "lattice_scale": 1.0,
        "absorption": 0.0,
    }

    def _phases_before_first_run(self, overrides):
        """Seed the grid from the matched phases so the first run can be set up."""
        names = []
        for phase in self.session.selected_phases or self.session.matched_phases or []:
            info = phase.get("phase") or phase
            name = info.get("mineral") or info.get("mineral_name")
            if name and name not in names:
                names.append(name)
        values = {}
        for name in names:
            entry = dict(self._DEFAULT_FLAGS)
            entry.update(overrides.get(name) or {})
            values[name] = entry
        return names, values

    def _store_overrides(self):
        overrides = self.matrix.overrides()
        self.session.phase_overrides = overrides
        self._update_hint(overrides)

    def _reset_overrides(self):
        self.session.phase_overrides = {}
        self.refresh()

    def _update_hint(self, overrides):
        pinned = sum(len(entry.get("_locked") or ()) for entry in overrides.values())
        self.hint.setText(
            f"{pinned} parameter{'' if pinned == 1 else 's'} held at a set value"
            if pinned else ""
        )

    # --- copying and export ------------------------------------------------

    def _blocks(self):
        return (
            ("Global", self.global_table),
            ("Per phase", self.phase_table),
        )

    def copy_all(self):
        from PyQt5.QtWidgets import QApplication

        lines = []
        for title, table in self._blocks():
            rows = table.all_rows()
            if not rows:
                continue
            lines.append(title)
            lines.append("\t".join(table.headers()))
            lines.extend("\t".join(row) for row in rows)
            lines.append("")
        QApplication.clipboard().setText("\n".join(lines))

    @restores_focus
    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Refinement Parameters", "refinement_parameters.csv",
            "CSV (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for title, table in self._blocks():
                    rows = table.all_rows()
                    if not rows:
                        continue
                    writer.writerow([title])
                    writer.writerow(table.headers())
                    writer.writerows(rows)
                    writer.writerow([])
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return
        self.set_status(f"Exported {path}")
