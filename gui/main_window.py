"""
Main window for XRD Phase Matching — tabbed analysis workspace.
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
    apply_theme, toggle_theme, get_current_mode, add_theme_listener, DARK,
)


class XRDMainWindow(QMainWindow):
    """Main application window hosting the tabbed workspace."""

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

        view_menu = menubar.addMenu("&View")
        self.compress_menu_action = QAction("Compress Control Panel", self)
        self.compress_menu_action.setShortcut("Ctrl+[")
        self.compress_menu_action.triggered.connect(self.toggle_compress_panel)
        view_menu.addAction(self.compress_menu_action)

        tools_menu = menubar.addMenu("&Tools")
        peak_action = QAction("&Find Peaks", self)
        peak_action.triggered.connect(self.workspace.find_peaks)
        tools_menu.addAction(peak_action)

        tools_menu.addSeparator()
        identify_action = QAction("&Identify (Pattern Search)", self)
        identify_action.triggered.connect(lambda: self.workspace.show_search_tab("identify"))
        tools_menu.addAction(identify_action)

        quant_action = QAction("&Quant Analysis", self)
        quant_action.triggered.connect(self.workspace.show_quant_tab)
        tools_menu.addAction(quant_action)

        db_action = QAction("&Database", self)
        db_action.triggered.connect(self.workspace.show_database_tab)
        tools_menu.addAction(db_action)

        tools_menu.addSeparator()
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

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_pattern)
        toolbar.addAction(open_action)

        toolbar.addSeparator()
        find_peaks_action = QAction("Find Peaks", self)
        find_peaks_action.triggered.connect(self.workspace.find_peaks)
        toolbar.addAction(find_peaks_action)

        toolbar.addSeparator()
        self.compress_action = QAction("Compress Panel", self)
        self.compress_action.setToolTip("Toggle narrow control panel for a wider plot")
        self.compress_action.triggered.connect(self.toggle_compress_panel)
        toolbar.addAction(self.compress_action)

        toolbar.addSeparator()
        self.theme_action = QAction(self._theme_action_label(), self)
        self.theme_action.setToolTip("Toggle Light / Dark theme")
        self.theme_action.triggered.connect(self.toggle_app_theme)
        toolbar.addAction(self.theme_action)

        toolbar.addSeparator()
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        toolbar.addAction(settings_action)

    def toggle_compress_panel(self):
        compressed = self.workspace.toggle_controls_compressed()
        label = "Expand Panel" if compressed else "Compress Panel"
        self.compress_action.setText(label)
        self.compress_menu_action.setText(
            "Expand Control Panel" if compressed else "Compress Control Panel"
        )
        self.status_bar.showMessage(
            "Control panel compressed" if compressed else "Control panel restored",
            3000,
        )

    def _theme_action_label(self) -> str:
        return "Light Mode" if get_current_mode() == DARK else "Dark Mode"

    def toggle_app_theme(self):
        toggle_theme(QApplication.instance())
        if self.settings_dialog is not None:
            self.settings_dialog.sync_theme_combo(get_current_mode())

    def apply_theme_mode(self, mode: str):
        apply_theme(QApplication.instance(), mode)

    def on_theme_changed(self, mode: str):
        if hasattr(self, "theme_action"):
            self.theme_action.setText(self._theme_action_label())
        if hasattr(self, "workspace"):
            self.workspace.on_theme_changed(mode)

    def setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — load a pattern to begin")

    def open_pattern(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Diffraction Pattern",
            "",
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
            <p><b>Tabs:</b> Search / Match · Quant Analysis · Database</p>
            <ul>
            <li>Pattern loading and preprocessing</li>
            <li>Ultra-fast pattern search and phase matching</li>
            <li>Le Bail refinement and export</li>
            <li>Local CIF database management</li>
            </ul>
            <p>Built with PyQt5 and scientific Python libraries</p>""",
        )
