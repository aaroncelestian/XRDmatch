"""Left-rail stage navigation for the guided workspace."""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QButtonGroup, QPushButton, QVBoxLayout, QWidget


STAGES = [
    ("load", "1  Load"),
    ("process", "2  Process"),
    ("identify", "3  Identify"),
    ("refine", "4  Refine"),
]


class StageRail(QWidget):
    """Vertical stage buttons with enabled / complete state."""

    stage_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buttons = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(6)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for i, (key, label) in enumerate(STAGES):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setProperty("stageKey", key)
            if i == 0:
                btn.setChecked(True)
            else:
                btn.setEnabled(False)
            self._group.addButton(btn, i)
            self._buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()
        self._group.buttonClicked.connect(self._on_clicked)

    def _on_clicked(self, button):
        key = button.property("stageKey")
        if key:
            self.stage_selected.emit(key)

    def set_current(self, key: str):
        btn = self._buttons.get(key)
        if btn and btn.isEnabled():
            btn.setChecked(True)

    def update_availability(self, session):
        """Enable stages based on session progress."""
        self._buttons["load"].setEnabled(True)
        self._buttons["process"].setEnabled(session.has_pattern())
        self._buttons["identify"].setEnabled(session.has_pattern())
        # Refine needs selected/matched phases ideally; allow when matches exist
        self._buttons["refine"].setEnabled(session.has_matches() or len(session.selected_phases) > 0)

        # Visual complete markers via objectName for QSS if desired
        for key, done in (
            ("load", session.has_pattern()),
            ("process", session.has_peaks()),
            ("identify", session.has_matches()),
            ("refine", session.lebail_results is not None),
        ):
            btn = self._buttons[key]
            btn.setObjectName("stageComplete" if done else "stagePending")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
