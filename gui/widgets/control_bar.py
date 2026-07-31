"""Compact horizontal control bars and a reusable options popup.

The bottom tool tabs are short, so controls are laid out in wide rows with the
primary action always visible and rarely-touched parameters moved into a popup.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QSizePolicy, QVBoxLayout, QWidget,
)


FIELD_WIDTH = 96


def compact(widget: QWidget, width: int = FIELD_WIDTH) -> QWidget:
    """Stop spin boxes and combos from stretching across a wide row."""
    widget.setMaximumWidth(width)
    widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return widget


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class ControlRow(QWidget):
    """A single horizontal row of labelled controls."""

    def __init__(self, parent=None, margins=(6, 4, 6, 4), spacing=6):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(spacing)

    def layout(self) -> QHBoxLayout:
        return self._layout

    def add_field(self, label: str, widget: QWidget, width: int = FIELD_WIDTH):
        if label:
            text = QLabel(label)
            text.setObjectName("mutedLabel")
            self._layout.addWidget(text)
        self._layout.addWidget(compact(widget, width))
        return widget

    def add_widget(self, widget: QWidget, stretch: int = 0):
        self._layout.addWidget(widget, stretch)
        return widget

    def add_separator(self):
        self._layout.addWidget(separator())

    def add_stretch(self, stretch: int = 1):
        self._layout.addStretch(stretch)


class OptionsDialog(QDialog):
    """Non-modal popup that hosts advanced parameter widgets.

    Widgets are reparented, not copied, so their values stay live wherever the
    owning stage reads them.
    """

    def __init__(self, title: str, parent=None, description: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if description:
            note = QLabel(description)
            note.setObjectName("mutedLabel")
            note.setWordWrap(True)
            layout.addWidget(note)

        self.form = QFormLayout()
        self.form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.form.setLabelAlignment(Qt.AlignLeft)
        layout.addLayout(self.form)
        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.hide)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.clicked.connect(self.hide)
        layout.addWidget(buttons)

    def add_row(self, label: str, widget: QWidget):
        if label:
            self.form.addRow(label, widget)
        else:
            self.form.addRow(widget)
        return widget

    def add_heading(self, text: str):
        heading = QLabel(text)
        heading.setStyleSheet("font-weight: 600;")
        self.form.addRow(heading)
        return heading

    def show_centered(self):
        parent = self.parentWidget()
        if parent is not None and not self.isVisible():
            center = parent.window().geometry().center()
            geo = self.frameGeometry()
            geo.moveCenter(center)
            self.move(geo.topLeft())
        self.show()
        self.raise_()
        self.activateWindow()
