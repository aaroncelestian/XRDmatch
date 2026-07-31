"""
Main window for XRD Phase Matching — file browser + tool tabs workspace.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QStatusBar, QToolBar, QAction, QFileDialog,
    QMessageBox, QApplication,
)
from PyQt5.QtGui import QKeySequence

from .session import AnalysisSession
from .workspace import AnalysisWorkspace
from .dialogs.settings_dialog import SettingsDialog
from .theme import (
    apply_theme, get_current_mode, add_theme_listener,
)


class XRDMainWindow(QMainWindow):
    """Main application window hosting the analysis workspace."""

    def __init__(self):
        super().__init__()
        self.session = AnalysisSession(self)
        self.settings_dialog = None
        self.init_ui()
        self.setup_menus()
        self.setup_toolbar()
        self.setup_statusbar()
        add_theme_listener(self.on_theme_changed)

    def init_ui(self):
        self.setWindowTitle("XRD Phase Matcher")
        self.setGeometry(80, 60, 1440, 920)

        self.workspace = AnalysisWorkspace(self.session)
        self.setCentralWidget(self.workspace)
        self.workspace.set_status_callback(self._show_status)

    def _show_status(self, message: str):
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(message)

    def setup_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_folder_action = QAction("Open &Folder…", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        open_action = QAction("&Open Pattern...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_pattern)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Results...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = menubar.addMenu("&Tools")
        peak_action = QAction("&Find Peaks", self)
        peak_action.triggered.connect(self.workspace.find_peaks)
        tools_menu.addAction(peak_action)

        identify_action = QAction("&Phases (Search / Match)", self)
        identify_action.triggered.connect(lambda: self.workspace.show_bottom_tab("phases"))
        tools_menu.addAction(identify_action)

        quant_action = QAction("&Quant Analysis…", self)
        quant_action.triggered.connect(self.workspace.open_quant)
        tools_menu.addAction(quant_action)

        tools_menu.addSeparator()
        db_action = QAction("&Database…", self)
        db_action.triggered.connect(self.workspace.open_database)
        tools_menu.addAction(db_action)

        settings_action = QAction("&Settings…", self)
        settings_action.triggered.connect(self.open_settings)
        tools_menu.addAction(settings_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        settings_action = QAction("Settings", self)
        settings_action.setToolTip("Application settings (theme and more)")
        settings_action.triggered.connect(self.open_settings)
        toolbar.addAction(settings_action)

        toolbar.addSeparator()
        db_action = QAction("Database", self)
        db_action.setToolTip("Browse and manage the local diffraction database")
        db_action.triggered.connect(self.workspace.open_database)
        toolbar.addAction(db_action)

    def apply_theme_mode(self, mode: str):
        apply_theme(QApplication.instance(), mode)

    def on_theme_changed(self, mode: str):
        if hasattr(self, "workspace"):
            self.workspace.on_theme_changed(mode)

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — open a folder and select a pattern")

    def open_folder(self):
        self.workspace.open_folder()

    def open_pattern(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Diffraction Pattern",
            self.workspace.file_browser.folder() or "",
            "Data files (*.xy *.xye *.chi *.xml *.txt *.dat *.csv);;All files (*.*)",
        )
        if file_path:
            self.workspace.open_pattern_file(file_path)
            self.status_bar.showMessage(f"Loaded pattern: {file_path}")

    def save_results(self):
        matches = self.session.matched_phases
        if not matches:
            QMessageBox.information(self, "No Results", "No matching results to save yet.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        if not file_path:
            return
        try:
            with open(file_path, "w") as f:
                f.write("Phase\tScore\tCoverage\tMatches\n")
                for r in matches:
                    phase = r.get("phase", r)
                    name = phase.get("mineral", "Unknown")
                    score = r.get("combined_score", r.get("match_score", 0))
                    cov = r.get("coverage", 0)
                    n = len(r.get("matches", []))
                    f.write(f"{name}\t{score:.4f}\t{cov}\t{n}\n")
            self.status_bar.showMessage(f"Results saved: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def open_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self)
            self.settings_dialog.theme_changed.connect(self.apply_theme_mode)
        self.settings_dialog.sync_theme_combo(get_current_mode())
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def show_about(self):
        QMessageBox.about(
            self,
            "About XRD Phase Matcher",
            """<h3>XRD Phase Matcher</h3>
            <p>XRD phase identification and quantitative analysis</p>
            <ul>
            <li>Browse folders and load diffraction patterns</li>
            <li>Background, peak finding, and phase matching</li>
            <li>Le Bail refinement in a dedicated Quant window</li>
            <li>Local CIF database management</li>
            </ul>
            <p>Built with PyQt5 and scientific Python libraries</p>""",
        )
