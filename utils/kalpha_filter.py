"""
Kα2 satellite rejection.

Lab sources emit a Kα doublet, so every reflection appears twice: a Kα1 line
and a weaker Kα2 line at slightly higher 2θ (about half the intensity). At low
angles the pair is unresolved, but above roughly 40° the satellite is often
detected as a separate peak, which pollutes peak lists and phase matching.

A detected peak is treated as a satellite when a stronger peak sits at exactly
the Kα1 position implied by the doublet geometry and the intensity ratio is
consistent with Kα2/Kα1.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# anode: (Kα1, Kα2) in Angstroms
KALPHA_DOUBLETS = {
    "Cu": (1.540562, 1.544398),
    "Cr": (2.289700, 2.293606),
    "Fe": (1.936042, 1.939980),
    "Co": (1.788965, 1.792850),
    "Mo": (0.709300, 0.713590),
    "Ag": (0.559408, 0.563813),
}

# Typical Kα2/Kα1 wavelength ratio, used when the anode cannot be identified
GENERIC_RATIO = 1.00249

DEFAULT_MAX_INTENSITY_RATIO = 0.75  # nominal is 0.5; overlap inflates it


def identify_anode(wavelength: float, tol: float = 0.004) -> Optional[str]:
    """Match a wavelength against Kα1 or the Kα1/Kα2 weighted average."""
    for anode, (a1, a2) in KALPHA_DOUBLETS.items():
        weighted = (2.0 * a1 + a2) / 3.0
        if abs(wavelength - a1) <= tol or abs(wavelength - weighted) <= tol:
            return anode
    return None


def alpha2_ratio(wavelength: float) -> float:
    anode = identify_anode(wavelength)
    if anode is None:
        return GENERIC_RATIO
    a1, a2 = KALPHA_DOUBLETS[anode]
    return a2 / a1


def alpha2_position(two_theta_alpha1: float, ratio: float) -> float:
    """2θ of the Kα2 satellite of a line observed at `two_theta_alpha1`."""
    sin_theta = np.sin(np.radians(two_theta_alpha1 / 2.0)) * ratio
    if sin_theta >= 1.0:
        return float("nan")
    return float(2.0 * np.degrees(np.arcsin(sin_theta)))


def alpha1_position(two_theta_alpha2: float, ratio: float) -> float:
    """Inverse of `alpha2_position`: where the parent Kα1 line would sit."""
    sin_theta = np.sin(np.radians(two_theta_alpha2 / 2.0)) / ratio
    if sin_theta >= 1.0:
        return float("nan")
    return float(2.0 * np.degrees(np.arcsin(sin_theta)))


def identify_alpha2_peaks(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    wavelength: float,
    *,
    max_intensity_ratio: float = DEFAULT_MAX_INTENSITY_RATIO,
    position_tolerance: Optional[float] = None,
    min_separation: float = 0.0,
) -> Tuple[List[int], List[Dict]]:
    """
    Return indices of peaks that look like Kα2 satellites.

    `position_tolerance` defaults to a fraction of the doublet separation, which
    keeps the test tight at low angles where the splitting is small.
    """
    tt = np.asarray(two_theta, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if len(tt) < 2 or len(tt) != len(inten):
        return [], []

    ratio = alpha2_ratio(wavelength)
    flagged: List[int] = []
    details: List[Dict] = []

    for j in range(len(tt)):
        parent_tt = alpha1_position(tt[j], ratio)
        if not np.isfinite(parent_tt):
            continue
        separation = tt[j] - parent_tt
        if separation < min_separation:
            continue
        tol = position_tolerance if position_tolerance is not None else max(0.3 * separation, 0.015)

        diffs = np.abs(tt - parent_tt)
        candidates = np.where((diffs <= tol) & (np.arange(len(tt)) != j))[0]
        # already-flagged peaks cannot be somebody else's parent
        candidates = [int(k) for k in candidates if k not in flagged and tt[k] < tt[j]]
        if not candidates:
            continue

        parent = max(candidates, key=lambda k: inten[k])
        if inten[parent] <= 0 or inten[j] > max_intensity_ratio * inten[parent]:
            continue

        flagged.append(j)
        details.append({
            "index": int(j),
            "two_theta": float(tt[j]),
            "parent_index": parent,
            "parent_two_theta": float(tt[parent]),
            "separation": float(tt[j] - tt[parent]),
            "intensity_ratio": float(inten[j] / inten[parent]),
        })

    return flagged, details


def strip_alpha2_peaks(peaks: Dict, wavelength: float, **kwargs) -> Tuple[Dict, List[Dict]]:
    """Drop Kα2 satellites from a peak dictionary, keeping arrays aligned."""
    tt = np.asarray(peaks.get("two_theta", []), dtype=float)
    if len(tt) == 0:
        return peaks, []

    flagged, details = identify_alpha2_peaks(
        tt, peaks.get("intensity", []), wavelength, **kwargs
    )
    if not flagged:
        return peaks, []

    keep = np.ones(len(tt), dtype=bool)
    keep[np.asarray(flagged, dtype=int)] = False
    cleaned = dict(peaks)
    for key, value in peaks.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == len(tt):
            cleaned[key] = arr[keep]
    return cleaned, details
