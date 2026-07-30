"""Load stage — open/drop pattern and set wavelength."""

from __future__ import annotations

import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from gui.pattern_io import SUPPORTED_EXTENSIONS, load_pattern_file


class LoadStage(QWidget):
    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel("Load Pattern")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(title)

        hint = QLabel("Open a diffraction file or drag & drop onto this panel.")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.open_btn = QPushButton("Open Pattern…")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.clicked.connect(self.open_dialog)
        layout.addWidget(self.open_btn)

        form = QFormLayout()
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.addItems([
            "Cu Kα1 (1.5406)",
            "Cu Kα (1.5418)",
            "Co Kα1 (1.7890)",
            "Fe Kα1 (1.9373)",
            "Cr Kα1 (2.2897)",
            "Mo Kα1 (0.7107)",
            "17 BM (0.24105)",
            "Custom",
        ])
        self.wavelength_combo.currentTextChanged.connect(self._wavelength_changed)
        form.addRow("Wavelength:", self.wavelength_combo)

        self.custom_wavelength = QDoubleSpinBox()
        self.custom_wavelength.setRange(0.1, 10.0)
        self.custom_wavelength.setDecimals(4)
        self.custom_wavelength.setValue(1.5406)
        self.custom_wavelength.setVisible(False)
        self.custom_wavelength.valueChanged.connect(self._custom_wl_changed)
        form.addRow("", self.custom_wavelength)
        layout.addLayout(form)

        self.file_label = QLabel("No file loaded")
        self.file_label.setObjectName("mutedLabel")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("mutedLabel")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        next_row = QHBoxLayout()
        next_row.addStretch()
        self.next_btn = QPushButton("Continue to Process →")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(lambda: self.workspace.set_stage("process"))
        next_row.addWidget(self.next_btn)
        layout.addLayout(next_row)

        layout.addStretch()

    def current_wavelength(self) -> float:
        text = self.wavelength_combo.currentText()
        if "Custom" in text:
            return self.custom_wavelength.value()
        return float(text.split("(")[1].split(")")[0])

    def _wavelength_changed(self, text: str):
        self.custom_wavelength.setVisible("Custom" in text)
        wl = self.current_wavelength()
        self.session.set_wavelength(wl)
        self.workspace.refresh_plot()

    def _custom_wl_changed(self, value: float):
        self.session.set_wavelength(value)
        self.workspace.refresh_plot()

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Diffraction Pattern",
            "",
            "Data files (*.xy *.xye *.chi *.xml *.txt *.dat *.csv);;All files (*.*)",
        )
        if path:
            self.load_file(path)

    def load_file(self, file_path: str):
        try:
            pattern = load_pattern_file(file_path, self.current_wavelength())
            # Sync UI wavelength if XML provided one
            if pattern.get("file_format") == "XML":
                self.wavelength_combo.setCurrentText("Custom")
                self.custom_wavelength.setValue(pattern["wavelength"])
                self.custom_wavelength.setVisible(True)
            else:
                pattern["wavelength"] = self.current_wavelength()

            self.session.set_raw_pattern(pattern)
            name = os.path.basename(file_path)
            self.file_label.setObjectName("")
            self.file_label.setText(f"Loaded: {name}")
            self.file_label.style().unpolish(self.file_label)
            self.file_label.style().polish(self.file_label)

            n = len(pattern["two_theta"])
            t0, t1 = float(pattern["two_theta"][0]), float(pattern["two_theta"][-1])
            self.meta_label.setText(
                f"{pattern['file_format']} · {n} points · "
                f"2θ {t0:.2f}–{t1:.2f}° · λ {pattern['wavelength']:.4f} Å"
            )
            self.next_btn.setEnabled(True)
            self.workspace.refresh_plot()
            self.workspace.set_status(f"Loaded {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load pattern:\n{e}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile().lower()
                    if any(path.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if any(path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                        self.load_file(path)
                        event.acceptProposedAction()
                        return
        event.ignore()

    def on_enter(self):
        pass
