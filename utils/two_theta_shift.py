"""
Zero-shift and sample-displacement corrections for reference line positions.

A powder mount sitting off the focusing circle moves every line by
Δ2θ = -(2s/R)·cos θ — largest at low angle, shrinking towards back-reflection.
A miscalibrated zero point moves them all by the same amount instead. Either
way the database pattern no longer lands on the observed peaks, and a search
that compares positions inside a fixed tolerance simply fails to find the
phase: the lines are all there, just not where they are being looked for.

These helpers move reference positions onto the measured scale and fit the
shift when it is not known in advance, which is the usual case — the
displacement is only obvious once the phase has been identified.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


ZERO_SHIFT = "zero"
DISPLACEMENT = "displacement"

# Label / key pairs for a combo box, best default first
SHIFT_MODELS = [
    ("Sample displacement (∝cos θ)", DISPLACEMENT),
    ("Zero shift (constant)", ZERO_SHIFT),
]

# A rigid shift fitted to one or two lines is not a correction, it is an
# excuse: almost any pattern can be slid onto a couple of peaks.
MIN_LINES_TO_FIT = 3

# Cap on the scan so a wide range with many peaks cannot stall the search
MAX_FIT_STEPS = 81
MAX_FIT_CELLS = 2_000_000


def shift_basis(two_theta: Sequence[float], model: str = DISPLACEMENT) -> np.ndarray:
    """Angular dependence of the shift, so one parameter covers the pattern."""
    tt = np.asarray(two_theta, dtype=float)
    if model == ZERO_SHIFT:
        return np.ones_like(tt)
    return np.cos(np.radians(tt / 2.0))


def apply_shift(
    two_theta: Sequence[float], amount: float, model: str = DISPLACEMENT
) -> np.ndarray:
    """
    Move reference positions to where the sample puts them.

    `amount` is the shift in degrees: constant for a zero shift, and the value
    extrapolated to 2θ = 0 for a displacement, where the actual shift is
    `amount·cos θ`.
    """
    tt = np.asarray(two_theta, dtype=float)
    if not amount or len(tt) == 0:
        return tt
    return tt + float(amount) * shift_basis(tt, model)


def remove_shift(
    two_theta: Sequence[float], amount: float, model: str = DISPLACEMENT
) -> np.ndarray:
    """
    Move measured positions back onto the reference scale.

    Exact for a zero shift. For a displacement the basis is evaluated at the
    observed angle instead of the true one, which for the sub-degree shifts
    this corrects stays far inside any usable match tolerance.
    """
    tt = np.asarray(two_theta, dtype=float)
    if not amount or len(tt) == 0:
        return tt
    return tt - float(amount) * shift_basis(tt, model)


def shift_pattern(
    pattern: Optional[Dict], amount: float, model: str = DISPLACEMENT
) -> Optional[Dict]:
    """
    Copy of a reference pattern with its lines moved to observed 2θ.

    The original positions are kept alongside, so re-shifting a pattern that
    has already been shifted replaces the correction rather than compounding
    it. `d_spacing` is deliberately left untouched: the spacing belongs to the
    material, the shift belongs to the mount.
    """
    if not pattern:
        return pattern
    base = pattern.get("two_theta_unshifted")
    if base is None:
        base = pattern.get("two_theta")
    base = np.asarray(base if base is not None else [], dtype=float)
    if len(base) == 0:
        return pattern

    out = dict(pattern)
    out["two_theta_unshifted"] = base
    out["two_theta"] = apply_shift(base, amount, model)
    out["two_theta_shift"] = float(amount)
    out["two_theta_shift_model"] = model
    return out


def unshift_pattern(pattern: Optional[Dict]) -> Optional[Dict]:
    """Reference pattern with any applied shift undone."""
    if not pattern or pattern.get("two_theta_unshifted") is None:
        return pattern
    out = dict(pattern)
    out["two_theta"] = np.asarray(pattern["two_theta_unshifted"], dtype=float)
    out.pop("two_theta_shift", None)
    out.pop("two_theta_shift_model", None)
    return out


def describe(amount: float, model: str = DISPLACEMENT) -> str:
    """Short human-readable form of a shift, for status lines and tooltips."""
    if not amount:
        return "no 2θ shift"
    if model == ZERO_SHIFT:
        return f"zero shift {amount:+.3f}°"
    return f"displacement {amount:+.3f}°·cos θ"


def fit_shift(
    exp_two_theta: Sequence[float],
    line_two_theta: Sequence[float],
    line_weights: Optional[Sequence[float]] = None,
    *,
    tolerance: float = 0.2,
    center: float = 0.0,
    span: float = 0.0,
    model: str = DISPLACEMENT,
) -> Tuple[float, int]:
    """
    Best rigid shift bringing a phase's lines onto the observed peaks.

    Scans `center ± span` on a grid no coarser than half the match tolerance,
    scoring each trial by intensity-weighted, distance-tapered agreement, then
    takes one weighted least-squares step from the winning grid point so the
    answer is not quantized to the scan.

    Returns `(shift, lines_matched)`. The shift stays at `center` unless at
    least `MIN_LINES_TO_FIT` lines land on peaks — a single free parameter is
    only evidence when several lines have to agree on it.
    """
    exp = np.asarray(exp_two_theta, dtype=float)
    lines = np.asarray(line_two_theta, dtype=float)
    center = float(center)
    if len(exp) == 0 or len(lines) == 0:
        return center, 0

    weights = (
        np.ones(len(lines), dtype=float) if line_weights is None
        else np.asarray(line_weights, dtype=float)
    )
    if len(weights) != len(lines) or not np.any(weights > 0):
        weights = np.ones(len(lines), dtype=float)

    tolerance = max(float(tolerance), 1e-6)
    span = abs(float(span))
    if span <= 0:
        grid = np.array([center])
    else:
        steps = int(np.clip(2.0 * span / (tolerance / 2.0) + 1, 5, MAX_FIT_STEPS))
        budget = MAX_FIT_CELLS // max(1, len(lines) * len(exp))
        steps = int(max(5, min(steps, max(budget, 5))))
        grid = np.linspace(center - span, center + span, steps)

    basis = shift_basis(lines, model)
    trial = lines[None, :] + grid[:, None] * basis[None, :]      # (steps, lines)
    distance = np.abs(trial[:, :, None] - exp[None, None, :])    # (steps, lines, peaks)
    nearest = np.argmin(distance, axis=2)
    closest = np.take_along_axis(distance, nearest[:, :, None], axis=2)[:, :, 0]

    found = closest <= tolerance
    # Taper with distance so the scan settles on the best-centred trial rather
    # than the first one that happens to bring the same lines inside tolerance
    agreement = np.where(found, 1.0 - closest / tolerance, 0.0) @ weights
    best = int(np.argmax(agreement))

    matched = found[best]
    n_found = int(np.count_nonzero(matched))
    if span <= 0 or n_found < MIN_LINES_TO_FIT:
        return center, n_found

    residual = exp[nearest[best][matched]] - trial[best][matched]
    b = basis[matched]
    w = weights[matched]
    denominator = float(np.sum(w * b * b))
    step = float(np.sum(w * b * residual) / denominator) if denominator > 0 else 0.0
    fitted = float(np.clip(grid[best] + step, center - span, center + span))

    refined = np.abs(apply_shift(lines, fitted, model)[:, None] - exp[None, :])
    n_refined = int(np.count_nonzero(np.min(refined, axis=1) <= tolerance))
    if n_refined < n_found:  # the least-squares step overshot; keep the scan result
        return float(grid[best]), n_found
    return fitted, n_refined
