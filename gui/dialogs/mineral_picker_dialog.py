"""Choose one database record for a mineral, ranked by how well it fits the data.

A common mineral has hundreds of records in the archive — 594 spinels, 343
periclases — differing in composition, and in the pressure and temperature they
were measured at. Those differences move the lines: the whole point of picking
one record rather than another is which cell reproduces the peaks in front of
you. So the list is ranked by the same fingerprint score the search uses, and
the highlighted record is drawn over the measured pattern while the dialog is
open, which is why this dialog is modeless.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from utils.conditions import Conditions

COLUMNS = [
    "Mineral", "Fit", "Lines", "Δ2θ", "Formula", "Cell a, b, c (Å)",
    "Space group", "Conditions", "Density", "AMCSD",
]
COL_NAME, COL_FIT, COL_LINES, COL_OFFSET = 0, 1, 2, 3


def _fmt(value, spec: str = "", dash: str = "—") -> str:
    if value is None or value == "":
        return dash
    if spec:
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _cell_text(hit: dict) -> str:
    values = [hit.get("cell_a"), hit.get("cell_b"), hit.get("cell_c")]
    if not all(v for v in values):
        return "—"
    try:
        return ", ".join(f"{float(v):.4f}" for v in values)
    except (TypeError, ValueError):
        return "—"


def _conditions(hit: dict) -> Conditions:
    return Conditions(
        pressure_gpa=hit.get("pressure_gpa"), temperature_k=hit.get("temperature_k")
    )


class _SortItem(QTableWidgetItem):
    """Cell that sorts on a number while displaying formatted text."""

    def __init__(self, text: str, key: float):
        super().__init__(text)
        self._key = key

    def __lt__(self, other):
        if isinstance(other, _SortItem):
            return self._key < other._key
        return super().__lt__(other)


class MineralPickerDialog(QDialog):
    """Modeless chooser: ranked records, live overlay, one gets added."""

    def __init__(self, hits: list, query: str, *,
                 score_fn: Optional[Callable[[dict], Optional[dict]]] = None,
                 preview_fn: Optional[Callable[[Optional[dict]], None]] = None,
                 tolerance: float = 0.2, ambient_only: bool = True,
                 truncated: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Choose a record — “{query}”")
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(940, 460)

        self._preview_fn = preview_fn
        self._tolerance = float(tolerance)
        self._chosen = None

        if score_fn is not None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                for hit in hits:
                    hit["fingerprint"] = score_fn(hit)
            finally:
                QApplication.restoreOverrideCursor()
            # Records of one mineral often tie on score — same lines, same
            # count — and then what separates them is how close those lines
            # land, so the mean offset is the tiebreak
            hits = sorted(hits, key=self._rank_key)
        self._hits = hits
        self._scored = score_fn is not None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._query = query
        self._truncated = truncated
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.title)

        hint = QLabel(
            "The highlighted record is drawn on the pattern in gold. Records of the "
            "same mineral differ in composition and in the pressure and temperature "
            "they were measured at, which moves their lines — pick the one whose "
            "lines land on your peaks."
        )
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        layout.addWidget(hint)

        filters = QHBoxLayout()
        filters.setSpacing(8)
        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText(
            "Filter these records — formula, space group, AMCSD id"
        )
        self.filter_box.textChanged.connect(self._apply_filter)
        filters.addWidget(self.filter_box, 1)

        self.ambient_box = QCheckBox("Ambient only")
        self.ambient_box.setChecked(bool(ambient_only))
        self.ambient_box.setToolTip(
            "Hide records measured at high pressure or high temperature.\n\n"
            "Their cells are compressed or expanded, so their lines sit at shifted "
            "2θ. Uncheck this only if your sample really was measured off-ambient."
        )
        self.ambient_box.toggled.connect(self._apply_filter)
        filters.addWidget(self.ambient_box)
        layout.addLayout(filters)

        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(self._on_row_changed)
        self.table.itemDoubleClicked.connect(lambda *_: self.accept())
        layout.addWidget(self.table, 1)

        self.detail = QLabel(" ")
        self.detail.setWordWrap(True)
        self.detail.setObjectName("mutedLabel")
        layout.addWidget(self.detail)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        add = buttons.addButton("Add Selected", QDialogButtonBox.AcceptRole)
        add.setObjectName("primaryButton")
        add.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._fill()
        self._place_beside_parent()

    # --- table ---

    def _update_title(self):
        shown = self.table.rowCount()
        total = len(self._hits)
        headline = (
            f"{shown} records for “{self._query}”" if shown == total
            else f"{shown} of {total} records for “{self._query}”"
        )
        if self._truncated:
            headline += " (first page — type more of the name to narrow it)"
        headline += (
            " — ranked by how well each one fits your peaks."
            if self._scored else
            " — no peaks detected yet, so these cannot be ranked by fit."
        )
        self.title.setText(headline)

    def _fill(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for index, hit in enumerate(self._hits):
            if not self._passes_filter(hit):
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._fill_row(row, index, hit)
        self.table.setSortingEnabled(True)
        self._update_title()
        if self.table.rowCount():
            self.table.setCurrentCell(0, COL_NAME)
            self._on_row_changed(0)
        else:
            self.detail.setText("No records left — loosen the filter.")
            self._preview(None)

    def _fill_row(self, row: int, index: int, hit: dict):
        name = QTableWidgetItem(str(hit.get("mineral_name") or "?"))
        # The row's position in the unfiltered list, not the record itself: Qt
        # copies a dict on its way through item data, and these carry whole CIFs
        name.setData(Qt.UserRole, index)
        self.table.setItem(row, COL_NAME, name)

        fp = hit.get("fingerprint") or {}
        score = fp.get("score")
        if score is None:
            self.table.setItem(row, COL_FIT, _SortItem("—", -1.0))
            self.table.setItem(row, COL_LINES, _SortItem("—", -1.0))
            self.table.setItem(row, COL_OFFSET, _SortItem("—", 9e9))
        else:
            fit = _SortItem(f"{score:.3f}", float(score))
            fit.setToolTip(self._fit_tooltip(fp))
            self.table.setItem(row, COL_FIT, fit)

            lines_text = f"{fp['n_found']}/{fp['n_expected']}"
            if not fp.get("top_found", True):
                lines_text += " ⚠"
            lines = _SortItem(lines_text, float(fp["n_found"]) / max(fp["n_expected"], 1))
            lines.setToolTip(
                "How many of this record's strong lines are present in your pattern"
                + ("" if fp.get("top_found", True) else "\nIts strongest line is MISSING")
            )
            self.table.setItem(row, COL_LINES, lines)

            offset = self._mean_offset(fp)
            item = _SortItem("—" if offset is None else f"{offset:.4f}°",
                             9e9 if offset is None else offset)
            item.setToolTip(
                "How far this record's lines land from your peaks, on average.\n"
                "Records of the same mineral often tie on score, and this is\n"
                "what separates them."
            )
            self.table.setItem(row, COL_OFFSET, item)

        for col, text in (
            (4, _fmt(hit.get("chemical_formula"))),
            (5, _cell_text(hit)),
            (6, _fmt(hit.get("space_group"))),
            (7, _conditions(hit).describe() or "ambient"),
            (8, _fmt(hit.get("density"), ".3f")),
            (9, _fmt(hit.get("amcsd_id"))),
        ):
            self.table.setItem(row, col, QTableWidgetItem(text))

    def _fit_tooltip(self, fp: dict) -> str:
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
        missing = fp.get("missing_strong") or []
        if missing:
            tip.append("Missing: " + ", ".join(f"{m:.2f}°" for m in missing[:6]))
        return "\n".join(tip)

    def _mean_offset(self, fp: dict) -> Optional[float]:
        """Mean |Δ2θ| of the matched lines, recovered from the position quality."""
        quality = fp.get("position_quality")
        if quality is None or not fp.get("n_found"):
            return None
        return (1.0 - float(quality)) * self._tolerance

    def _rank_key(self, hit: dict) -> tuple:
        fp = hit.get("fingerprint") or {}
        score = fp.get("score")
        if score is None:
            return (1.0, 0.0)
        offset = self._mean_offset(fp)
        return (-float(score), self._tolerance if offset is None else offset)

    # --- filtering ---

    def _passes_filter(self, hit: dict) -> bool:
        if self.ambient_box.isChecked() and not _conditions(hit).is_ambient:
            return False
        needle = self.filter_box.text().strip().lower()
        if not needle:
            return True
        haystack = " ".join(
            str(hit.get(key) or "") for key in (
                "mineral_name", "chemical_formula", "space_group",
                "authors", "year", "amcsd_id",
            )
        ).lower()
        return all(word in haystack for word in needle.split())

    def _apply_filter(self):
        self._fill()

    # --- selection / preview ---

    def _on_row_changed(self, row: int, _col=0, _prow=-1, _pcol=0):
        hit = self.current_hit()
        if hit is None:
            return
        self.detail.setText(self._describe(hit))
        self._preview(hit)

    def current_hit(self) -> Optional[dict]:
        item = self.table.item(self.table.currentRow(), COL_NAME)
        if item is None:
            return None
        index = item.data(Qt.UserRole)
        return self._hits[index] if index is not None else None

    def _describe(self, hit: dict) -> str:
        parts = []
        fp = hit.get("fingerprint") or {}
        if not self._scored:
            parts.append(
                "Detect peaks on the Process stage to rank these records by fit"
            )
        elif fp.get("score") is None:
            parts.append("No reference pattern stored for this record")
        else:
            parts.append(
                f"{fp['n_found']} of {fp['n_expected']} strong lines present"
                + ("" if fp.get("top_found") else ", strongest line missing")
            )
            offset = self._mean_offset(fp)
            if offset is not None:
                parts.append(f"lines land {offset:.4f}° from your peaks on average")
            if fp.get("intensity_consistency") is not None:
                parts.append(f"intensity consistency {fp['intensity_consistency']:.2f}")
        reference = " ".join(
            str(hit.get(key) or "") for key in ("authors", "journal", "year")
        ).strip()
        if reference:
            parts.append(reference)
        return " · ".join(parts) if parts else " "

    def _preview(self, hit: Optional[dict]):
        if self._preview_fn is not None:
            self._preview_fn(hit)

    # --- window placement ---

    def _place_beside_parent(self):
        """Keep clear of the main window if the screen allows, so the plot stays visible."""
        parent = self.parentWidget().window() if self.parentWidget() else None
        if parent is None:
            return
        screen = QApplication.desktop().availableGeometry(parent)
        window = parent.frameGeometry()
        geometry = self.frameGeometry()

        beside = QRect(geometry)
        beside.moveTopLeft(window.topRight())
        below = QRect(geometry)
        below.moveTopLeft(window.bottomLeft())
        for candidate in (beside, below):
            if screen.contains(candidate):
                self.move(candidate.topLeft())
                return

        # No room outside the window: the lower edge covers the phase tabs
        # rather than the plot, which is what the user needs to see
        geometry.moveBottomLeft(window.bottomLeft())
        geometry.moveLeft(max(geometry.left(), screen.left()))
        geometry.moveBottom(min(geometry.bottom(), screen.bottom()))
        self.move(geometry.topLeft())

    # --- result ---

    def accept(self):
        self._chosen = self.current_hit()
        if self._chosen is None:
            return
        super().accept()

    def chosen(self) -> Optional[dict]:
        return self._chosen
