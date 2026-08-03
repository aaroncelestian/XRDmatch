"""
Emphasised 2theta regions — user-declared priorities for search/match.

A region says "the lines in here matter more to me than the rest of the
pattern". It becomes a per-peak weight, which is the channel the fingerprint
scorer already uses for residual search, so an emphasised region shifts both
the coverage screen and the final ranking towards phases that explain the
highlighted peaks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_WEIGHT = 5.0
MIN_SPAN = 0.05  # a click-sized drag is not a region


def make_region(lo: float, hi: float, weight: float = DEFAULT_WEIGHT) -> Dict[str, float]:
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    return {"lo": lo, "hi": hi, "weight": max(float(weight), 1.0)}


def normalize(regions: Optional[Sequence[dict]]) -> List[Dict[str, float]]:
    """Drop degenerate spans and merge overlaps, keeping the stronger weight."""
    clean = []
    for r in regions or []:
        try:
            region = make_region(r["lo"], r["hi"], r.get("weight", DEFAULT_WEIGHT))
        except (KeyError, TypeError, ValueError):
            continue
        if region["hi"] - region["lo"] >= MIN_SPAN:
            clean.append(region)
    if not clean:
        return []

    clean.sort(key=lambda r: r["lo"])
    merged = [clean[0]]
    for region in clean[1:]:
        last = merged[-1]
        if region["lo"] <= last["hi"]:
            last["hi"] = max(last["hi"], region["hi"])
            last["weight"] = max(last["weight"], region["weight"])
        else:
            merged.append(region)
    return merged


def region_at(regions: Optional[Sequence[dict]], two_theta: float) -> Optional[dict]:
    """The region containing a 2theta, for hit-testing a click."""
    for region in regions or []:
        if float(region["lo"]) <= two_theta <= float(region["hi"]):
            return region
    return None


def inside(two_theta: Sequence[float], regions: Optional[Sequence[dict]]) -> np.ndarray:
    """Boolean mask of the values that fall in any region."""
    tt = np.asarray(two_theta, dtype=float)
    mask = np.zeros(len(tt), dtype=bool)
    for region in regions or []:
        mask |= (tt >= float(region["lo"])) & (tt <= float(region["hi"]))
    return mask


def peak_weights(
    two_theta: Sequence[float],
    regions: Optional[Sequence[dict]],
    base: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Per-peak weights, or None when nothing is emphasised.

    Overlapping regions are already merged by `normalize`, but take the max
    anyway so a raw list still behaves sensibly.
    """
    regions = normalize(regions)
    tt = np.asarray(two_theta, dtype=float)
    if not regions or len(tt) == 0:
        return None
    weights = np.full(len(tt), float(base), dtype=float)
    for region in regions:
        in_region = (tt >= region["lo"]) & (tt <= region["hi"])
        weights[in_region] = np.maximum(weights[in_region], region["weight"])
    return weights


def describe(regions: Optional[Sequence[dict]]) -> str:
    """One-line summary for a status bar or report."""
    regions = normalize(regions)
    if not regions:
        return "none"
    spans = ", ".join(
        f"{r['lo']:.2f}-{r['hi']:.2f}° (x{r['weight']:g})" for r in regions
    )
    return spans
