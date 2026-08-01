"""Pattern file I/O helpers extracted for the Load stage."""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np


SUPPORTED_EXTENSIONS = (".xy", ".xye", ".chi", ".xml", ".txt", ".dat", ".csv")

COMMENT_PREFIXES = ("#", "!", ";", "'", "//")

# TOPAS / jEdit style XYE files wrap headers in C comments, sometimes multi-line
_C_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_comments(text: str) -> str:
    """Remove C-style comment spans, including unterminated trailing blocks."""
    text = _C_COMMENT.sub(" ", text)
    start = text.find("/*")
    return text[:start] if start != -1 else text


def parse_text_file(file_path: str):
    """Parse XY / XYE / CHI / CSV / DAT text diffraction files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = strip_comments(f.read())

    two_theta = []
    intensity = []
    intensity_error = []
    has_errors = False

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(COMMENT_PREFIXES):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue  # header or label row
        two_theta.append(x)
        intensity.append(y)

        err = 0.0
        if len(parts) >= 3:
            try:
                err = float(parts[2])
                has_errors = True
            except ValueError:
                err = 0.0
        intensity_error.append(err)

    if not two_theta:
        raise ValueError("No numeric diffraction data found in file")

    tt = np.asarray(two_theta, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    err = np.asarray(intensity_error, dtype=float) if has_errors else None

    valid = np.isfinite(tt) & np.isfinite(inten)
    if err is not None:
        valid &= np.isfinite(err)
    if not np.all(valid):
        tt, inten = tt[valid], inten[valid]
        if err is not None:
            err = err[valid]
    if len(tt) == 0:
        raise ValueError("No finite diffraction data found in file")

    # Zero-filled error columns are unusable as Le Bail weights
    if err is not None and not np.any(err > 0):
        err = None
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


def normalize_for_comparison(intensity) -> np.ndarray:
    """
    Rescale a pattern to 0-100 so patterns of different exposure can be overlaid.

    Both ends are set from the data: the top from the strongest peak, the bottom
    from a low percentile rather than the outright minimum, so that one dead
    channel or a negative excursion left by background subtraction cannot drag
    the whole curve down. Removing the floor as well as the ceiling matters
    because patterns collected on different instruments sit on very different
    backgrounds, and scaling by the peak alone would leave them stacked at
    different heights.
    """
    values = np.asarray(intensity, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)

    floor = float(np.percentile(finite, 1.0))
    span = float(finite.max()) - floor
    if span <= 0:
        return np.zeros_like(values)
    return (values - floor) / span * 100.0


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
