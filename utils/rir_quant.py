"""
RIR (reference intensity ratio) quantification — Chung's matrix-flushing method.

    w_i = (I_i / RIR_i) / sum_j (I_j / RIR_j)

RIR is the AMCSD I/I_corundum value stored per mineral. I_i is the fitted
intensity of the phase's strongest line, obtained by fitting all selected
reference patterns to the observed pattern at once with non-negative least
squares. Fitting jointly rather than phase by phase matters in a mixture:
overlapping lines get shared between phases instead of counted twice for each.

This is deliberately not a refinement. Positions, relative intensities, and
peak width are held fixed and only one scale per phase is fitted, which makes
it fast enough to run interactively while identifying phases. Use the Le Bail
refinement in the Quant window when the cell, profile, and correction terms
need to move.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import nnls

DEFAULT_FWHM = 0.12
DEFAULT_ETA = 0.3  # Lorentzian fraction


def phase_name(entry: dict) -> str:
    """Display name for a match result, search hit, or bare phase dict."""
    if not isinstance(entry, dict):
        return str(entry)
    phase = entry.get("phase", entry)
    for src in (phase, entry):
        if not isinstance(src, dict):
            continue
        name = src.get("mineral") or src.get("mineral_name")
        if name:
            return str(name)
    return "Unknown"


def phase_rir(*sources) -> Optional[float]:
    """First usable RIR among the given dicts, unwrapping nested phase dicts."""
    for src in sources:
        if not isinstance(src, dict):
            continue
        candidates = [src]
        if isinstance(src.get("phase"), dict):
            candidates.append(src["phase"])
        for candidate in candidates:
            value = candidate.get("rir")
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return value
    return None


def pseudo_voigt(two_theta: np.ndarray, centers: Sequence[float],
                 intensities: Sequence[float], fwhm: float = DEFAULT_FWHM,
                 eta: float = DEFAULT_ETA, window: float = 8.0) -> np.ndarray:
    """
    Reference lines broadened onto the measured grid.

    Accumulated per line over a window a few FWHM wide, so cost scales with the
    number of lines rather than lines times data points.
    """
    out = np.zeros(len(two_theta), dtype=float)
    centers = np.asarray(centers, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    if len(centers) == 0 or len(two_theta) == 0 or fwhm <= 0:
        return out

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    eta = float(np.clip(eta, 0.0, 1.0))
    half = window * fwhm

    lo = np.searchsorted(two_theta, centers - half, side="left")
    hi = np.searchsorted(two_theta, centers + half, side="right")
    for center, intensity, a, b in zip(centers, intensities, lo, hi):
        if intensity <= 0 or b <= a:
            continue
        offset = two_theta[a:b] - center
        gaussian = np.exp(-0.5 * (offset / sigma) ** 2)
        lorentzian = 1.0 / (1.0 + (offset / gamma) ** 2)
        out[a:b] += intensity * ((1.0 - eta) * gaussian + eta * lorentzian)
    return out


def _weighted_rwp(observed: np.ndarray, calculated: np.ndarray) -> float:
    weights = 1.0 / np.maximum(observed, 1.0)
    denominator = float(np.sum(weights * observed ** 2))
    if denominator <= 0:
        return float("nan")
    numerator = float(np.sum(weights * (observed - calculated) ** 2))
    return 100.0 * float(np.sqrt(numerator / denominator))


def fit_phase_intensities(
    pattern: dict,
    phases: Sequence[dict],
    *,
    fwhm: float = DEFAULT_FWHM,
    eta: float = DEFAULT_ETA,
    max_shift: float = 0.3,
    theoretical_for: Optional[Callable[[dict], Optional[dict]]] = None,
) -> Optional[Dict]:
    """
    Fit one scale per phase to the observed pattern with non-negative least squares.

    A single 2-theta shift is searched alongside the scales. Real patterns carry
    zero-point and specimen-displacement errors of a tenth of a degree or more,
    which is enough to move a line off its own peak; fitting scales against
    misaligned lines pushes a phase that is present towards zero and lets one
    that is absent pick up a neighbour's intensity. Pass max_shift=0 to fit at
    the reference positions exactly.

    Returns None when no phase has usable reference lines inside the measured
    range. A phase fitted to zero is one the pattern gives no room for.
    """
    two_theta = np.asarray(pattern["two_theta"], dtype=float)
    observed = np.maximum(np.asarray(pattern["intensity"], dtype=float), 0.0)
    if len(two_theta) == 0:
        return None

    usable: List[Dict] = []
    skipped: List[str] = []

    for entry in phases:
        theo = entry.get("theoretical_peaks") if isinstance(entry, dict) else None
        if (not theo or len(theo.get("two_theta", [])) == 0) and theoretical_for is not None:
            theo = theoretical_for(entry)
        name = phase_name(entry)
        if not theo:
            skipped.append(name)
            continue

        positions = np.asarray(theo.get("two_theta", []), dtype=float)
        intensities = np.asarray(theo.get("intensity", []), dtype=float)
        if len(positions) == 0 or len(intensities) != len(positions):
            skipped.append(name)
            continue
        reference_max = float(np.max(intensities)) if len(intensities) else 0.0
        if reference_max <= 0:
            skipped.append(name)
            continue

        inside = (positions >= two_theta[0] - fwhm) & (positions <= two_theta[-1] + fwhm)
        if not np.any(inside):  # every line falls outside the measured range
            skipped.append(name)
            continue

        usable.append({
            "entry": entry,
            "name": name,
            "positions": positions,
            "intensities": intensities,
            "reference_max": reference_max,
            "rir": phase_rir(entry, theo),
        })

    if not usable:
        return None

    def column_for(index: int, shift: float) -> np.ndarray:
        info = usable[index]
        return pseudo_voigt(
            two_theta, info["positions"] + shift, info["intensities"], fwhm, eta
        )

    def solve(columns: List[np.ndarray]):
        design = np.column_stack(columns)
        scales, residual = nnls(design, observed)
        return design, scales, float(residual)

    shifts = [0.0] * len(usable)
    columns = [column_for(i, 0.0) for i in range(len(usable))]
    design, scales, residual = solve(columns)

    if max_shift > 0:
        # One shift for all phases first: that is the instrument error, and
        # fitting it jointly uses every line in the pattern to pin it down
        step = max(fwhm / 3.0, 0.01)
        best_shift = 0.0
        for candidate in np.arange(-max_shift, max_shift + 1e-9, step):
            trial_columns = [column_for(i, float(candidate)) for i in range(len(usable))]
            trial = solve(trial_columns)
            if trial[2] < residual:
                design, scales, residual = trial
                columns, best_shift = trial_columns, float(candidate)
        shifts = [best_shift] * len(usable)

        # Then a small shift per phase, one phase at a time. A phase whose cell
        # differs from the reference entry sits slightly off even after the
        # instrument shift is removed, and misaligned lines cost it intensity
        # that a neighbouring phase then picks up.
        fine = step / 2.0
        for _ in range(2):
            improved = False
            for index in range(len(usable)):
                for offset in (-2 * fine, -fine, fine, 2 * fine):
                    candidate = shifts[index] + offset
                    if abs(candidate - best_shift) > max_shift:
                        continue
                    trial_columns = list(columns)
                    trial_columns[index] = column_for(index, candidate)
                    trial = solve(trial_columns)
                    if trial[2] < residual:
                        design, scales, residual = trial
                        columns = trial_columns
                        shifts[index] = candidate
                        improved = True
            if not improved:
                break
        shift = best_shift

    calculated = design @ scales

    results = []
    for info, column, scale, phase_shift in zip(usable, columns, scales, shifts):
        scale = float(scale)
        results.append({
            "entry": info["entry"],
            "name": info["name"],
            "scale": scale,
            "shift": float(phase_shift),
            "rir": info["rir"],
            # Strongest-line intensity: the quantity RIR is defined against.
            # Every phase shares one peak shape here, so the area factor that
            # turns this into an integrated intensity cancels in Chung's ratio.
            "line_intensity": scale * info["reference_max"],
            "pattern_intensity": float(np.sum(column) * scale),
            "profile": column * scale,
        })

    total_observed = float(np.sum(observed))
    return {
        "phases": results,
        "two_theta": two_theta,
        "observed": observed,
        "calculated": calculated,
        "rwp": _weighted_rwp(observed, calculated),
        "explained_fraction": (
            float(np.sum(np.minimum(calculated, observed)) / total_observed)
            if total_observed > 0 else 0.0
        ),
        "skipped": skipped,
        "fwhm": float(fwhm),
        "shift": float(shift),
    }


def quantify(
    pattern: dict,
    phases: Sequence[dict],
    *,
    fwhm: float = DEFAULT_FWHM,
    eta: float = DEFAULT_ETA,
    max_shift: float = 0.3,
    theoretical_for: Optional[Callable[[dict], Optional[dict]]] = None,
) -> Optional[Dict]:
    """
    Fit phase intensities and convert them to RIR weight percents.

    Phases without a RIR value get `weight_percent` None and are left out of the
    normalization, so the reported percentages are of the quantified phases only.
    """
    fit = fit_phase_intensities(
        pattern, phases, fwhm=fwhm, eta=eta, max_shift=max_shift,
        theoretical_for=theoretical_for,
    )
    if fit is None:
        return None

    terms = []
    for phase in fit["phases"]:
        rir = phase.get("rir")
        intensity = phase.get("line_intensity", 0.0)
        if rir and intensity > 0:
            term = intensity / rir
            phase["rir_term"] = term
            terms.append(term)
        else:
            phase["rir_term"] = None

    total = float(sum(terms))
    for phase in fit["phases"]:
        term = phase.get("rir_term")
        phase["weight_percent"] = (
            100.0 * term / total if (term is not None and total > 0) else None
        )

    fit["phases"].sort(
        key=lambda p: (p.get("weight_percent") is None, -(p.get("weight_percent") or 0.0))
    )
    fit["n_quantified"] = len(terms)
    fit["missing_rir"] = [p["name"] for p in fit["phases"] if p.get("rir") is None]
    fit["absent"] = [p["name"] for p in fit["phases"] if p.get("scale", 0.0) <= 0]
    return fit


def summary_lines(result: Dict) -> List[str]:
    """Human-readable report, used for the status line and the details panel."""
    if not result:
        return ["No phases could be quantified."]
    lines = [
        f"RIR quantification — fit Rwp {result['rwp']:.1f}%, "
        f"{result['explained_fraction'] * 100:.0f}% of observed intensity explained, "
        f"2θ shift {result.get('shift', 0.0):+.3f}°"
    ]
    for phase in result["phases"]:
        wt = phase.get("weight_percent")
        rir = phase.get("rir")
        if wt is None:
            reason = "no RIR in database" if rir is None else "fitted to zero"
            lines.append(f"  {phase['name']}: not quantified ({reason})")
        else:
            lines.append(
                f"  {phase['name']}: {wt:.1f} wt%  "
                f"(I={phase['line_intensity']:.3g}, RIR={rir:.3f})"
            )
    if result.get("skipped"):
        lines.append(f"  No reference pattern: {', '.join(result['skipped'])}")
    return lines
