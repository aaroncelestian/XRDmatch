"""Settings dialog — hosts SettingsTab outside the stage rail."""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox

from gui.settings_tab import SettingsTab


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(640, 520)

        layout = QVBoxLayout(self)
        self.settings_tab = SettingsTab()
        layout.addWidget(self.settings_tab)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

    @property
    def theme_changed(self):
        return self.settings_tab.theme_changed

    def sync_theme_combo(self, mode: str):
        self.settings_tab.sync_theme_combo(mode)
