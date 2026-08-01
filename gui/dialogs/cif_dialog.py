"""Dialog for viewing raw CIF text from the local CIF archive."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from gui.focus import restores_focus
from utils.cif_repository import get_cif_repository


class CifViewerDialog(QDialog):
    """Scrollable CIF text view with copy and save actions."""

    def __init__(self, amcsd_id, mineral_name: str = "", parent=None):
        super().__init__(parent)
        self.amcsd_id = amcsd_id
        self.mineral_name = mineral_name or "Unknown"
        self.repo = get_cif_repository()

        self.setWindowTitle(f"CIF — {self.mineral_name} (AMCSD {amcsd_id})")
        self.resize(760, 640)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.source_label = QLabel()
        self.source_label.setObjectName("mutedLabel")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        self.text_view = QPlainTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text_view.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self.text_view, 1)

        controls = QHBoxLayout()
        self.original_check = QCheckBox("Show original deposition")
        self.original_check.setToolTip(
            "AMCSD keeps an unedited copy of some depositions alongside the "
            "standardized file"
        )
        self.original_check.setEnabled(self.repo.has_original(amcsd_id))
        self.original_check.toggled.connect(self._load)
        controls.addWidget(self.original_check)
        controls.addStretch()

        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy)
        controls.addWidget(copy_btn)

        save_btn = QPushButton("Save As…")
        save_btn.clicked.connect(self._save)
        controls.addWidget(save_btn)
        layout.addLayout(controls)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

        self._load()

    def _current_text(self) -> Optional[str]:
        return self.repo.get_cif_text(
            self.amcsd_id, original=self.original_check.isChecked()
        )

    def _load(self):
        text = self._current_text()
        if text is None:
            self.text_view.setPlainText(
                f"No CIF found for AMCSD ID {self.amcsd_id}.\n\n"
                f"Expected an entry in {self.repo.zip_path.name} named "
                f"<Mineral>__{self.amcsd_id}.cif"
            )
            self.source_label.setText("Source: not available")
            return

        self.text_view.setPlainText(text)
        name = self.repo.source_name(
            self.amcsd_id, original=self.original_check.isChecked()
        )
        lines = text.count("\n") + 1
        self.source_label.setText(
            f"Source: {name}  ·  {lines} lines  ·  {len(text):,} chars"
        )

    def _copy(self):
        text = self.text_view.toPlainText()
        if text:
            QApplication.clipboard().setText(text)

    @restores_focus
    def _save(self):
        text = self._current_text()
        if not text:
            QMessageBox.warning(self, "Nothing to Save", "No CIF content loaded.")
            return

        suggested = self.repo.source_name(
            self.amcsd_id, original=self.original_check.isChecked()
        ) or f"{self.amcsd_id}.cif"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CIF", suggested, "CIF files (*.cif);;All files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
