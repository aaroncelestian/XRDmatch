"""Titled section frame and collapsible accordion sections."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget, QFormLayout,
)


class SectionFrame(QGroupBox):
    """Thin titled section used in place of ad-hoc nested GroupBoxes."""

    def __init__(self, title: str = "", parent=None, layout_type: str = "vbox"):
        super().__init__(title, parent)
        if layout_type == "form":
            self.body = QFormLayout(self)
            self.body.setContentsMargins(8, 12, 8, 8)
            self.body.setSpacing(8)
            self.body.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        elif layout_type == "hbox":
            self.body = QHBoxLayout(self)
            self.body.setContentsMargins(8, 12, 8, 8)
            self.body.setSpacing(8)
        else:
            self.body = QVBoxLayout(self)
            self.body.setContentsMargins(8, 12, 8, 8)
            self.body.setSpacing(8)


class CollapsibleSection(QWidget):
    """Expandable header + content body for the Search/Match control column."""

    def __init__(self, title: str, content: QWidget, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title
        self._content = content

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.toggle_btn = QPushButton(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setObjectName("collapsibleHeader")
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_btn.setMinimumHeight(32)
        self.toggle_btn.toggled.connect(self._on_toggled)
        root.addWidget(self.toggle_btn)

        self.body_frame = QFrame()
        self.body_frame.setObjectName("collapsibleBody")
        body_layout = QVBoxLayout(self.body_frame)
        body_layout.setContentsMargins(0, 0, 0, 4)
        body_layout.setSpacing(0)
        body_layout.addWidget(content)
        root.addWidget(self.body_frame)

        self._on_toggled(expanded)

    def _on_toggled(self, expanded: bool):
        self.body_frame.setVisible(expanded)
        arrow = "▼" if expanded else "▶"
        self.toggle_btn.setText(f"{arrow} {self._title}")

    def set_expanded(self, expanded: bool):
        self.toggle_btn.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self.toggle_btn.isChecked()
