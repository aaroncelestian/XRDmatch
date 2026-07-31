"""Search diagnosis popup — shown from the Phases panel 'Why not?' button."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout,
)


class SearchReportDialog(QDialog):
    """Read-only monospace report of what a search did to one phase."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Why Not?")
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(760, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title = QLabel("—")
        self.title.setStyleSheet("font-weight: 600; font-size: 15px;")
        layout.addWidget(self.title)

        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        layout.addWidget(self.verdict)

        self.body = QPlainTextEdit()
        self.body.setReadOnly(True)
        self.body.setLineWrapMode(QPlainTextEdit.NoWrap)
        # Columns of 2θ and intensity only line up in a fixed-width font
        self.body.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self.body, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.hide)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.clicked.connect(self.hide)
        layout.addWidget(buttons)

    def show_report(self, title: str, verdict: str, body: str):
        self.title.setText(title)
        self.verdict.setText(verdict)
        self.body.setPlainText(body)
        self.body.moveCursor(self.body.textCursor().Start)
        self.show()
        self.raise_()
        self.activateWindow()
