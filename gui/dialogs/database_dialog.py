"""Database Manager dialog — hosts LocalDatabaseTab outside the main workflow."""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox

from gui.local_database_tab import LocalDatabaseTab


class DatabaseManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Database Manager")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        self.db_tab = LocalDatabaseTab()
        layout.addWidget(self.db_tab)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.Close)
        if close_btn:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

    @property
    def phases_selected(self):
        return self.db_tab.phases_selected
