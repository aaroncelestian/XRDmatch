"""Left-panel folder / file browser for pattern loading."""

from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import QDir, QModelIndex, QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFileSystemModel, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeView, QVBoxLayout, QWidget,
)

from gui.pattern_io import SUPPORTED_EXTENSIONS


class FileBrowser(QWidget):
    """Browse a folder and emit a path when a diffraction file is activated."""

    file_activated = pyqtSignal(str)
    wavelength_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._folder = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Folder…")
        self.path_edit.setReadOnly(True)
        header.addWidget(self.path_edit, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setToolTip("Choose a folder of diffraction patterns")
        browse_btn.clicked.connect(self.choose_folder)
        header.addWidget(browse_btn)
        layout.addLayout(header)

        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        name_filters = [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]
        self.model.setNameFilters(name_filters)
        self.model.setNameFilterDisables(False)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setAnimated(False)
        self.tree.setSortingEnabled(True)
        self.tree.setHeaderHidden(False)
        self.tree.setSelectionMode(QTreeView.SingleSelection)
        self.tree.doubleClicked.connect(self._on_activated)
        self.tree.clicked.connect(self._on_clicked)
        # Show name column only
        for col in range(1, 4):
            self.tree.hideColumn(col)
        layout.addWidget(self.tree, 1)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
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
        self.wavelength_combo.currentTextChanged.connect(self._on_wavelength_ui)
        form.addRow("Wavelength:", self.wavelength_combo)

        self.custom_wavelength = QDoubleSpinBox()
        self.custom_wavelength.setRange(0.1, 10.0)
        self.custom_wavelength.setDecimals(4)
        self.custom_wavelength.setValue(1.5406)
        self.custom_wavelength.setVisible(False)
        self.custom_wavelength.valueChanged.connect(self._emit_wavelength)
        form.addRow("", self.custom_wavelength)
        layout.addLayout(form)

        self.file_label = QLabel("Select a folder, then click a pattern file.")
        self.file_label.setObjectName("mutedLabel")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        self.meta_label = QLabel("")
        self.meta_label.setObjectName("mutedLabel")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        last = QSettings().value("file_browser/last_folder", "", type=str)
        if last and os.path.isdir(last):
            self.set_folder(last)

    def current_wavelength(self) -> float:
        text = self.wavelength_combo.currentText()
        if "Custom" in text:
            return self.custom_wavelength.value()
        return float(text.split("(")[1].split(")")[0])

    def _on_wavelength_ui(self, text: str):
        self.custom_wavelength.setVisible("Custom" in text)
        self._emit_wavelength()

    def _emit_wavelength(self, *_args):
        self.wavelength_changed.emit(self.current_wavelength())

    def set_custom_wavelength(self, value: float):
        self.wavelength_combo.setCurrentText("Custom")
        self.custom_wavelength.setValue(value)
        self.custom_wavelength.setVisible(True)

    def choose_folder(self, start: Optional[str] = None):
        start_dir = start if isinstance(start, str) and start else (self._folder or os.path.expanduser("~"))
        path = QFileDialog.getExistingDirectory(self, "Open Pattern Folder", start_dir)
        if path:
            self.set_folder(path)

    def set_folder(self, path: str):
        if not path or not os.path.isdir(path):
            return
        self._folder = path
        self.path_edit.setText(path)
        root = self.model.setRootPath(path)
        self.tree.setRootIndex(root)
        self.file_label.setText("Click a pattern file to load.")
        QSettings().setValue("file_browser/last_folder", path)

    def folder(self) -> str:
        return self._folder

    def reveal_file(self, file_path: str):
        """Select a file in the tree if it lives under the current folder."""
        if not file_path or not os.path.isfile(file_path):
            return
        parent = os.path.dirname(file_path)
        if parent and parent != self._folder:
            self.set_folder(parent)
        index = self.model.index(file_path)
        if index.isValid():
            self.tree.setCurrentIndex(index)
            self.tree.scrollTo(index)

    def set_file_info(self, name: str, meta: str = ""):
        self.file_label.setText(f"Loaded: {name}" if name else "No file loaded")
        self.meta_label.setText(meta or "")

    def _path_from_index(self, index: QModelIndex) -> Optional[str]:
        if not index.isValid():
            return None
        path = self.model.filePath(index)
        if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            return path
        return None

    def _on_clicked(self, index: QModelIndex):
        path = self._path_from_index(index)
        if path:
            self.file_activated.emit(path)

    def _on_activated(self, index: QModelIndex):
        path = self._path_from_index(index)
        if path:
            self.file_activated.emit(path)
        elif index.isValid() and self.model.isDir(index):
            # Expand / navigate into subdirectory
            self.tree.setExpanded(index, not self.tree.isExpanded(index))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    lower = path.lower()
                    if os.path.isdir(path) or any(lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.set_folder(path)
                event.acceptProposedAction()
                return
            if any(path.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                self.set_folder(os.path.dirname(path))
                self.reveal_file(path)
                self.file_activated.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()
