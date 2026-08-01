"""Every refined parameter, for every phase, in one window."""

from __future__ import annotations

import csv

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from gui import refinement_table
from gui.widgets.copyable_table import CopyableTable


class RefinementDetailsDialog(QDialog):
    """
    The full parameter set behind the summary table.

    The summary shows what is read at a glance; this shows everything the
    refinement holds, including the quantities that are only meaningful once
    something has gone wrong with the fit.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Refinement Parameters")
        self.setWindowModality(Qt.NonModal)
        self.resize(820, 620)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.headline = QLabel()
        self.headline.setObjectName("mutedLabel")
        self.headline.setWordWrap(True)
        root.addWidget(self.headline)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._section(
            "Global — one value for the whole pattern", "global_table"
        ))
        splitter.addWidget(self._section(
            "Per phase", "phase_table"
        ))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 400])
        root.addWidget(splitter, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
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
        root.addLayout(buttons)

        session.refinement_changed.connect(self.refresh)
        self.refresh()

    def _section(self, title: str, attribute: str) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600;")
        layout.addWidget(label)
        table = CopyableTable()
        table.horizontalHeader().setStretchLastSection(True)
        setattr(self, attribute, table)
        layout.addWidget(table)
        return wrapper

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def refresh(self):
        results = self.session.lebail_results
        if not (results and results.get("success")):
            self.headline.setText("No refinement yet — run Le Bail to see its parameters.")
            self.global_table.set_content([], [])
            self.phase_table.set_content([], [])
            return

        parts = refinement_table.summary_headline(results)
        self.headline.setText("  ·  ".join(parts) if parts else "Refinement")

        self.global_table.set_content(
            ["Parameter", "Value"], refinement_table.global_rows(results)
        )
        names = refinement_table.phase_names(results)
        self.phase_table.set_content(
            ["Parameter"] + names, refinement_table.detail_rows(results)
        )

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
        if hasattr(self.parent(), "set_status"):
            self.parent().set_status(f"Exported {path}")
