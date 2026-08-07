"""
Every refinable parameter, for every phase, in one editable grid.

A powder refinement is a negotiation between phases: chlorite wants its
asymmetry free for stacking disorder while the quartz beside it stays
symmetric, and a standard weighed into the mount has a scale that is known and
must not be fitted. That is a per-phase question, so it is laid out per phase --
parameters down, phases across -- rather than as one switch per term applied to
everything at once.

A ticked box means the term is refined for that phase. Unticked means it is held
at the value in the cell, which is editable: typing a number and leaving the box
clear is how you fix a parameter somewhere the optimiser would not have gone,
whether to break a correlation or to hold a quantity you already know.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem


class ParameterRow:
    """One refinable quantity: where its value lives and what frees it."""

    def __init__(self, label: str, value_key: str, refine_key: str,
                 fmt: str = ".6g", tooltip: str = "", quantitative: bool = False):
        self.label = label
        self.value_key = value_key
        self.refine_key = refine_key
        self.fmt = fmt
        self.tooltip = tooltip
        # Terms that only mean something when intensities are tied to the
        # reference pattern; Le Bail extraction absorbs them.
        self.quantitative = quantitative


_CELL_HINT = (
    "Cell parameters refine separately, so a can grow while c contracts. What "
    "the symmetry of the starting cell fixes is held: equal axes move together "
    "and a right angle stays a right angle, since a pattern cannot tell those "
    "directions apart.\n\n"
    "Untick to hold this one at the value shown while the rest of the cell "
    "refines around it."
)


PHASE_ROWS: tuple = (
    ParameterRow("Scale factor", "scale_factor", "refine_scale", ".6g",
                 "How much of this phase the pattern holds. Untick and type a "
                 "value to hold a phase you have weighed in as a standard.",
                 quantitative=True),
    ParameterRow("Microstrain (×10⁻⁶)", "microstrain", "refine_strain", ".1f",
                 "Lorentzian broadening going as tan θ, from a spread of "
                 "lattice constants. Usually the dominant sample term."),
    ParameterRow("Crystallite size (µm)", "crystallite_size", "refine_size", ".5g",
                 "Lorentzian broadening going as 1/cos θ. Negligible above "
                 "about a micron, so leave it fixed unless the sample is "
                 "genuinely fine."),
    ParameterRow("Peak asymmetry", "asymmetry", "refine_asymmetry", "+.4f",
                 "Skew of this phase's peaks alone. Stacking disorder in the "
                 "layered minerals is the usual cause."),
    ParameterRow("a (Å)", "cell_a", "refine_cell", ".5f", _CELL_HINT),
    ParameterRow("b (Å)", "cell_b", "refine_cell", ".5f", _CELL_HINT),
    ParameterRow("c (Å)", "cell_c", "refine_cell", ".5f", _CELL_HINT),
    ParameterRow("α (°)", "cell_alpha", "refine_cell", ".4f", _CELL_HINT),
    ParameterRow("β (°)", "cell_beta", "refine_cell", ".4f", _CELL_HINT),
    ParameterRow("γ (°)", "cell_gamma", "refine_cell", ".4f", _CELL_HINT),
    ParameterRow("Absorption", "absorption", "refine_absorption", "+.5f",
                 "Angle-dependent intensity loss, exp(-a / sin θ). Absorbs "
                 "microabsorption contrast between phases.", quantitative=True),
)

_REFINE_COLUMN_HINT = (
    "Ticked: refined for this phase.\n"
    "Unticked: held at the value shown, which you can edit."
)


class ParameterMatrix(QTableWidget):
    """Parameters down the side, phases across the top."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phases: List[str] = []
        self._rows = list(PHASE_ROWS)
        self._quantitative = True
        self._loading = False

        self.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.itemChanged.connect(self._on_item_changed)

    # --- filling it in -----------------------------------------------------

    def set_phases(self, names: List[str], values: Dict[str, Dict[str, Any]],
                   quantitative: bool = True) -> None:
        """
        Lay out one column per phase.

        `values` maps a phase name to its current parameters, which is what the
        cells start at: the point of showing them is that the next run begins
        where the last one finished, so a number can be nudged rather than
        guessed from nothing.
        """
        self._loading = True
        try:
            self._phases = list(names)
            self._quantitative = quantitative
            self.clear()
            self.setColumnCount(1 + len(self._phases))
            self.setRowCount(len(self._rows))
            self.setHorizontalHeaderLabels(["Parameter"] + self._phases)
            self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            for column in range(1, self.columnCount()):
                self.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)

            for index, row in enumerate(self._rows):
                self._build_row(index, row, values)
        finally:
            self._loading = False

    def _build_row(self, index: int, row: ParameterRow,
                   values: Dict[str, Dict[str, Any]]) -> None:
        label = QTableWidgetItem(row.label)
        label.setFlags(Qt.ItemIsEnabled)
        if row.tooltip:
            label.setToolTip(row.tooltip)
        self.setItem(index, 0, label)

        available = self._quantitative or not row.quantitative
        for column, phase in enumerate(self._phases, start=1):
            params = values.get(phase) or {}
            item = QTableWidgetItem(self._format(params.get(row.value_key), row.fmt))
            item.setToolTip(
                _REFINE_COLUMN_HINT if available else
                "Not determinable under Le Bail extraction, which absorbs it."
            )
            if available:
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
                refined = bool(params.get(row.refine_key, False))
                item.setCheckState(Qt.Checked if refined else Qt.Unchecked)
            else:
                item.setFlags(Qt.ItemIsEnabled)
            self.setItem(index, column, item)

    @staticmethod
    def _format(value, fmt: str) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(f"{float(v):.4f}" for v in value)
        try:
            return format(float(value), fmt)
        except (TypeError, ValueError):
            return str(value)

    # --- reading it back ---------------------------------------------------

    def overrides(self) -> Dict[str, Dict[str, Any]]:
        """
        Per-phase parameters in the form the refinement takes.

        An unticked box is reported as a lock rather than merely as a refine
        flag left off, because the two differ in what happens next: a flag can
        be switched back on by the staged refinement as it moves between its
        stages, whereas a lock holds the value the user typed all the way
        through the run.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for column, phase in enumerate(self._phases, start=1):
            entry: Dict[str, Any] = {}
            locked: List[str] = []
            for index, row in enumerate(self._rows):
                item = self.item(index, column)
                if item is None or not (item.flags() & Qt.ItemIsUserCheckable):
                    continue
                value = self._parse(item.text())
                if value is not None:
                    entry[row.value_key] = value
                refined = item.checkState() == Qt.Checked
                # The six cell parameters share one flag, so the cell is refined
                # if any of them is ticked and the unticked ones are held. Were
                # the last row to decide the flag on its own, clearing γ would
                # quietly stop a and c from refining as well.
                entry[row.refine_key] = bool(entry.get(row.refine_key)) or refined
                if not refined and value is not None:
                    locked.append(row.value_key)
            if locked:
                entry["_locked"] = locked
            if entry:
                result[phase] = entry
        return result

    @staticmethod
    def _parse(text: str) -> Optional[float]:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    # --- editing -----------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() == 0:
            return
        self.changed.emit()

    def set_all(self, row_label: str, refined: bool) -> None:
        """Tick or clear one parameter across every phase at once."""
        for index, row in enumerate(self._rows):
            if row.label != row_label:
                continue
            for column in range(1, self.columnCount()):
                item = self.item(index, column)
                if item is not None and item.flags() & Qt.ItemIsUserCheckable:
                    item.setCheckState(Qt.Checked if refined else Qt.Unchecked)
