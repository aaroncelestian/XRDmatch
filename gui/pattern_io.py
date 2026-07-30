"""Pattern file I/O helpers extracted for the Load stage."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


SUPPORTED_EXTENSIONS = (".xy", ".xye", ".chi", ".xml", ".txt", ".dat", ".csv")


def parse_text_file(file_path: str):
    """Parse XY / XYE / CHI / CSV / DAT text diffraction files."""
    two_theta = []
    intensity = []
    intensity_error = []
    has_errors = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            # Skip header-ish lines
            lower = line.lower()
            if any(k in lower for k in ("2theta", "two_theta", "intensity", "counts")):
                if not any(ch.isdigit() for ch in line.split()[0:1] or [""]):
                    continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                two_theta.append(x)
                intensity.append(y)
                if len(parts) >= 3:
                    intensity_error.append(float(parts[2]))
                    has_errors = True
                else:
                    intensity_error.append(0.0)
            except ValueError:
                continue

    if not two_theta:
        raise ValueError("No numeric diffraction data found in file")

    tt = np.asarray(two_theta, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    err = np.asarray(intensity_error, dtype=float) if has_errors else None
    return tt, inten, err


def parse_xml_file(file_path: str):
    """Parse simple XML intensity format used by this project."""
    import xml.etree.ElementTree as ET

    two_theta = []
    intensity = []
    intensity_error = []
    wavelength = None

    tree = ET.parse(file_path)
    root = tree.getroot()

    w_elem = root.find("w")
    if w_elem is not None:
        try:
            wavelength = float(w_elem.text)
        except (ValueError, TypeError):
            pass

    for intensity_elem in root.findall("intensity"):
        try:
            x_val = float(intensity_elem.get("X"))
            y_val = float(intensity_elem.get("Y"))
            t_val = float(intensity_elem.get("T", 1.0))
            two_theta.append(x_val)
            intensity.append(y_val)
            if t_val > 0:
                intensity_error.append(np.sqrt(y_val) if y_val > 0 else 0.0)
        except (ValueError, TypeError):
            continue

    if not two_theta:
        raise ValueError("No valid intensity data found in XML file")

    err = np.asarray(intensity_error, dtype=float) if intensity_error else None
    return (
        np.asarray(two_theta, dtype=float),
        np.asarray(intensity, dtype=float),
        err,
        wavelength,
    )


def load_pattern_file(file_path: str, wavelength: float = 1.5406) -> dict:
    """Load a pattern file into the session dict format."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    wl = wavelength
    if file_path.lower().endswith(".xml"):
        two_theta, intensity, intensity_error, xml_wl = parse_xml_file(file_path)
        file_format = "XML"
        if xml_wl:
            wl = xml_wl
    else:
        two_theta, intensity, intensity_error = parse_text_file(file_path)
        file_format = "XYE" if intensity_error is not None else "XY"

    return {
        "two_theta": two_theta,
        "intensity": intensity,
        "intensity_error": intensity_error,
        "file_path": file_path,
        "file_format": file_format,
        "wavelength": wl,
    }
