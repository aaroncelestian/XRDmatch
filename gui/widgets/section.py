"""Titled section frame with consistent margins."""

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QFormLayout, QHBoxLayout


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
