"""Phase details popup — shown from the Phases table context menu."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


def _fmt(value, spec: str = "", dash: str = "—") -> str:
    if value is None or value == "":
        return dash
    if spec:
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


class PhaseDetailsDialog(QDialog):
    """Metadata, scores, and reference peaks for one phase."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phase Details")
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(620, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title = QLabel("—")
        self.title.setStyleSheet("font-weight: 600; font-size: 15px;")
        layout.addWidget(self.title)

        self.subtitle = QLabel("")
        self.subtitle.setObjectName("mutedLabel")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        columns = QHBoxLayout()
        columns.setSpacing(16)

        cell_wrap = QWidget()
        self.cell_form = QFormLayout(cell_wrap)
        self.cell_form.setLabelAlignment(Qt.AlignLeft)
        columns.addWidget(cell_wrap, 1)

        score_wrap = QWidget()
        self.score_form = QFormLayout(score_wrap)
        self.score_form.setLabelAlignment(Qt.AlignLeft)
        columns.addWidget(score_wrap, 1)
        layout.addLayout(columns)

        self.peaks_label = QLabel("Reference peaks")
        self.peaks_label.setObjectName("mutedLabel")
        layout.addWidget(self.peaks_label)

        self.peaks_table = QTableWidget()
        self.peaks_table.setAlternatingRowColors(True)
        self.peaks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.peaks_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.peaks_table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.hide)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.clicked.connect(self.hide)
        layout.addWidget(buttons)

    def show_phase(self, result: dict, theo: Optional[dict] = None):
        """Populate from a search hit or match result, plus optional peak data."""
        phase = result.get("phase", result) if isinstance(result, dict) else {}
        name = (
            phase.get("mineral")
            or phase.get("mineral_name")
            or result.get("mineral_name")
            or "Unknown phase"
        )
        formula = (
            phase.get("formula")
            or phase.get("chemical_formula")
            or result.get("chemical_formula")
        )
        space_group = phase.get("space_group") or result.get("space_group")
        self.title.setText(str(name))
        self.subtitle.setText(
            f"{_fmt(formula)}   ·   space group {_fmt(space_group)}   ·   "
            f"database id {_fmt(phase.get('id') or result.get('mineral_id'))}"
        )

        self._clear_form(self.cell_form)
        src = {**result, **phase} if isinstance(phase, dict) else result
        for label, key, spec in (
            ("a (Å)", "cell_a", ".4f"),
            ("b (Å)", "cell_b", ".4f"),
            ("c (Å)", "cell_c", ".4f"),
            ("α (°)", "cell_alpha", ".2f"),
            ("β (°)", "cell_beta", ".2f"),
            ("γ (°)", "cell_gamma", ".2f"),
            ("RIR", "rir", ".3f"),
        ):
            self.cell_form.addRow(label, QLabel(_fmt(src.get(key), spec)))

        self._clear_form(self.score_form)
        fp = result.get("fingerprint") or {}
        rows = [
            ("Match score", _fmt(result.get("match_score"), ".3f")),
            ("Combined score", _fmt(result.get("combined_score"), ".3f")),
            ("Correlation", _fmt(result.get("correlation"), ".3f")),
            ("Coverage", _fmt(result.get("coverage"), ".3f")),
            ("Matched peaks", str(len(result.get("matches") or [])) if result.get("matches") else "—"),
        ]
        if fp:
            rows.extend([
                ("Fingerprint", _fmt(fp.get("score"), ".3f")),
                ("Lines found", f"{fp.get('n_found', 0)}/{fp.get('n_expected', 0)}"),
                ("Strongest line", "present" if fp.get("top_found") else "missing"),
                ("Position quality", _fmt(fp.get("position_quality"), ".2f")),
                ("Intensity consistency", _fmt(fp.get("intensity_consistency"), ".2f")),
            ])
            if fp.get("residual_score") is not None and fp.get("residual_score") != fp.get("score"):
                rows.append(("Residual score", _fmt(fp.get("residual_score"), ".3f")))
            missing = fp.get("missing_strong") or []
            if missing:
                preview = ", ".join(f"{m:.2f}°" for m in missing[:6])
                rows.append(("Missing lines", preview))
        for label, value in rows:
            self.score_form.addRow(label, QLabel(value))

        self._fill_peaks(theo or result.get("theoretical_peaks"))
        self.show()
        self.raise_()
        self.activateWindow()

    def _fill_peaks(self, theo: Optional[dict]):
        self.peaks_table.clear()
        self.peaks_table.setColumnCount(3)
        self.peaks_table.setHorizontalHeaderLabels(["2θ (°)", "Rel. intensity", "d (Å)"])
        if not theo:
            self.peaks_table.setRowCount(0)
            self.peaks_label.setText("Reference peaks — none available")
            return

        tt = np.asarray(theo.get("two_theta", []), dtype=float)
        inten = np.asarray(theo.get("intensity", []), dtype=float)
        d = theo.get("d_spacing")
        d = np.asarray(d, dtype=float) if d is not None else None
        if len(tt) == 0:
            self.peaks_table.setRowCount(0)
            self.peaks_label.setText("Reference peaks — none available")
            return

        imax = float(np.max(inten)) if len(inten) and np.max(inten) > 0 else 1.0
        self.peaks_label.setText(f"Reference peaks ({len(tt)})")
        self.peaks_table.setRowCount(len(tt))
        for i in range(len(tt)):
            self.peaks_table.setItem(i, 0, QTableWidgetItem(f"{tt[i]:.3f}"))
            rel = inten[i] / imax * 100.0 if i < len(inten) else 0.0
            self.peaks_table.setItem(i, 1, QTableWidgetItem(f"{rel:.1f}"))
            dv = f"{d[i]:.4f}" if d is not None and i < len(d) else "—"
            self.peaks_table.setItem(i, 2, QTableWidgetItem(dv))

    @staticmethod
    def _clear_form(form: QFormLayout):
        while form.rowCount():
            form.removeRow(0)
