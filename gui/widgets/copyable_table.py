"""A results table whose contents can be taken out of the window."""

from __future__ import annotations

import csv
import io
from typing import List, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QApplication, QMenu, QTableWidget,
    QTableWidgetItem,
)


class CopyableTable(QTableWidget):
    """
    Read-only table supporting ctrl-C, a copy context menu, and CSV rendering.

    Copying yields tab-separated text, which is what spreadsheets expect from
    the clipboard, while the CSV export of the same rows is comma-separated.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setShortcutContext(Qt.WidgetShortcut)
        copy_action.triggered.connect(self.copy_selection)
        self.addAction(copy_action)
        self._copy_action = copy_action

    # --- filling ---

    def set_content(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
                    tooltips: Optional[Sequence[Sequence[str]]] = None,
                    header_tooltips: Optional[Sequence[str]] = None):
        """Replace the whole table in one step."""
        self.clear()
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(list(headers))
        if header_tooltips:
            for column, tip in enumerate(header_tooltips):
                header_item = self.horizontalHeaderItem(column)
                if header_item is not None and tip:
                    header_item.setToolTip(tip)

        self.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                if tooltips and row < len(tooltips) and column < len(tooltips[row]):
                    tip = tooltips[row][column]
                    if tip:
                        item.setToolTip(tip)
                self.setItem(row, column, item)
        self.resizeColumnsToContents()

    # --- taking the contents out ---

    def headers(self) -> List[str]:
        return [
            self.horizontalHeaderItem(column).text()
            if self.horizontalHeaderItem(column) else ""
            for column in range(self.columnCount())
        ]

    def all_rows(self) -> List[List[str]]:
        return [
            [
                self.item(row, column).text() if self.item(row, column) else ""
                for column in range(self.columnCount())
            ]
            for row in range(self.rowCount())
        ]

    def _selected_block(self) -> List[List[str]]:
        """
        The selection as a rectangle, so a partial selection pastes in shape.

        Empty cells inside the bounding box are kept as blanks rather than
        closing up, which is what a spreadsheet does with a sparse selection.
        """
        ranges = self.selectedRanges()
        if not ranges:
            return []
        top = min(r.topRow() for r in ranges)
        bottom = max(r.bottomRow() for r in ranges)
        left = min(r.leftColumn() for r in ranges)
        right = max(r.rightColumn() for r in ranges)

        block = []
        for row in range(top, bottom + 1):
            line = []
            for column in range(left, right + 1):
                item = self.item(row, column)
                selected = item is not None and item.isSelected()
                line.append(item.text() if (item and selected) else "")
            block.append(line)
        return block

    def copy_selection(self):
        """Copy the selection, or the whole table when nothing is selected."""
        block = self._selected_block()
        include_headers = False
        if not block:
            block = self.all_rows()
            include_headers = True
        if not block:
            return

        lines = []
        if include_headers:
            lines.append("\t".join(self.headers()))
        lines.extend("\t".join(row) for row in block)
        QApplication.clipboard().setText("\n".join(lines))

    def copy_all(self):
        rows = self.all_rows()
        if not rows:
            return
        lines = ["\t".join(self.headers())] + ["\t".join(row) for row in rows]
        QApplication.clipboard().setText("\n".join(lines))

    def as_csv_text(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.headers())
        writer.writerows(self.all_rows())
        return buffer.getvalue()

    def _show_context_menu(self, position):
        if self.rowCount() == 0:
            return
        menu = QMenu(self)
        menu.addAction("Copy", self.copy_selection)
        menu.addAction("Copy whole table", self.copy_all)
        menu.exec_(self.viewport().mapToGlobal(position))
