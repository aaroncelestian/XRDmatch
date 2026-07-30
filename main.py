#!/usr/bin/env python3
"""
XRD Phase Matching Program
A GUI application for X-ray diffraction phase matching using AMCSD database
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import XRDMainWindow
from gui.theme import apply_theme


def main():
    """Main entry point for the XRD Phase Matching application"""
    app = QApplication(sys.argv)
    app.setApplicationName("XRD Phase Matcher")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("XRD Tools")

    app.setStyle('Fusion')
    apply_theme(app)  # Light / Dark from QSettings

    window = XRDMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
