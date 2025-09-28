#!/usr/bin/env python3
"""
XRD Phase Matching Program
A GUI application for X-ray diffraction phase matching using AMCSD database
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from gui.main_window import XRDMainWindow

def main():
    """Main entry point for the XRD Phase Matching application"""
    app = QApplication(sys.argv)
    app.setApplicationName("XRD Phase Matcher")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("XRD Tools")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = XRDMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
