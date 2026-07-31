"""Search diagnosis popup — shown from the Phases panel 'Why not?' button."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase, QGuiApplication
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout,
)


SECTION_GUIDE = (
    "Below: 1 what happened · 2 what would change it · "
    "3 how well it fits · 4 line by line · 5 raw search counts"
)


class SearchReportDialog(QDialog):
    """Read-only monospace report of what a search did to one phase."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Why is this phase not in my results?")
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(820, 660)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title = QLabel("—")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600; font-size: 15px;")
        layout.addWidget(self.title)

        # The verdict answers the question on its own; everything below it is
        # supporting evidence for the user who does not believe the answer
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.verdict)

        guide = QLabel(SECTION_GUIDE)
        guide.setObjectName("mutedLabel")
        guide.setWordWrap(True)
        layout.addWidget(guide)

        self.body = QPlainTextEdit()
        self.body.setReadOnly(True)
        self.body.setLineWrapMode(QPlainTextEdit.NoWrap)
        # Columns of 2θ and intensity only line up in a fixed-width font
        self.body.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self.body, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        copy_btn = buttons.addButton("Copy report", QDialogButtonBox.ActionRole)
        copy_btn.setToolTip("Copy the whole report to the clipboard")
        copy_btn.clicked.connect(self._copy_report)
        buttons.rejected.connect(self.hide)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.clicked.connect(self.hide)
        layout.addWidget(buttons)

    def _copy_report(self):
        clipboard = QGuiApplication.clipboard()
        if clipboard:
            clipboard.setText(
                f"{self.title.text()}\n\n{self.verdict.text()}\n\n"
                f"{self.body.toPlainText()}"
            )

    def show_report(self, title: str, verdict: str, body: str):
        self.title.setText(title)
        self.verdict.setText(verdict)
        self.body.setPlainText(body)
        self.body.moveCursor(self.body.textCursor().Start)
        self.show()
        self.raise_()
        self.activateWindow()
