"""
Fingerprint scoring for multi-phase mixtures.

Whole-pattern correlation and coverage-based scores punish a candidate for the
experimental peaks it does not explain. In a mixture that is exactly wrong: a
minor phase only ever explains a handful of peaks. These helpers score a
candidate on *its own* strong lines instead — "are the peaks this mineral must
show actually present?" — so unknown extra peaks are simply ignored.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


DEFAULT_TOLERANCE = 0.20
DEFAULT_MIN_REL_INTENSITY = 5.0
DEFAULT_N_FINGERPRINT = 10

# A phase cannot be present if its most intense line is absent
MISSING_TOP_PENALTY = 0.45


def select_fingerprint_peaks(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    *,
    n_peaks: int = DEFAULT_N_FINGERPRINT,
    min_rel_intensity: float = DEFAULT_MIN_REL_INTENSITY,
    two_theta_range: Optional[tuple] = None,
) -> Dict[str, np.ndarray]:
    """Pick the strong lines that define a phase, normalized to I_max = 100."""
    tt = np.asarray(two_theta, dtype=float)
    inten = np.asarray(intensity, dtype=float)
    if len(tt) == 0 or len(tt) != len(inten):
        return {"two_theta": np.array([]), "intensity": np.array([])}

    keep = np.isfinite(tt) & np.isfinite(inten) & (inten > 0)
    if two_theta_range is not None:
        lo, hi = two_theta_range
        keep &= (tt >= lo) & (tt <= hi)
    tt, inten = tt[keep], inten[keep]
    if len(tt) == 0:
        return {"two_theta": np.array([]), "intensity": np.array([])}

    norm = inten / np.max(inten) * 100.0
    strong = norm >= min_rel_intensity
    tt, norm = tt[strong], norm[strong]
    if len(tt) == 0:
        return {"two_theta": np.array([]), "intensity": np.array([])}

    order = np.argsort(norm)[::-1][:n_peaks]
    order = order[np.argsort(tt[order])]  # keep 2θ order for readability
    return {"two_theta": tt[order], "intensity": norm[order]}


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Rank correlation without pulling in scipy.stats; None when undefined."""
    if len(a) < 4:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def fingerprint_score(
    exp_two_theta: Sequence[float],
    exp_intensity: Sequence[float],
    theo_two_theta: Sequence[float],
    theo_intensity: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    n_peaks: int = DEFAULT_N_FINGERPRINT,
    min_rel_intensity: float = DEFAULT_MIN_REL_INTENSITY,
    exp_range: Optional[tuple] = None,
) -> Dict:
    """
    Score a candidate by how many of its own strong lines are present.

    Unexplained experimental peaks carry no penalty, so minor phases in a
    mixture are scored on equal footing with the dominant phase. `exp_range`
    should be the measured 2θ range; reference lines outside it are ignored,
    while lines inside it with no matching peak count as missing.
    """
    exp_tt = np.asarray(exp_two_theta, dtype=float)
    exp_int = np.asarray(exp_intensity, dtype=float)
    empty = {
        "score": 0.0,
        "presence": 0.0,
        "n_expected": 0,
        "n_found": 0,
        "top_found": False,
        "position_quality": 0.0,
        "intensity_agreement": 0.0,
        "missing_strong": [],
        "matched_two_theta": [],
    }
    if len(exp_tt) == 0:
        return empty

    window = exp_range or (float(np.min(exp_tt)) - tolerance, float(np.max(exp_tt)) + tolerance)
    fp = select_fingerprint_peaks(
        theo_two_theta,
        theo_intensity,
        n_peaks=n_peaks,
        min_rel_intensity=min_rel_intensity,
        two_theta_range=window,
    )
    fp_tt, fp_int = fp["two_theta"], fp["intensity"]
    if len(fp_tt) == 0:
        return empty

    exp_max = float(np.max(exp_int)) if len(exp_int) and np.max(exp_int) > 0 else 1.0
    strongest_idx = int(np.argmax(fp_int))

    found_mask = np.zeros(len(fp_tt), dtype=bool)
    offsets = np.zeros(len(fp_tt), dtype=float)
    exp_matched = np.full(len(fp_tt), np.nan)
    theo_found, exp_found = [], []

    for i, tt in enumerate(fp_tt):
        diffs = np.abs(exp_tt - tt)
        j = int(np.argmin(diffs))
        if diffs[j] <= tolerance:
            found_mask[i] = True
            offsets[i] = float(diffs[j])
            exp_matched[i] = float(exp_tt[j])
            theo_found.append(float(fp_int[i]))
            exp_found.append(float(exp_int[j] / exp_max * 100.0))

    weights = fp_int.astype(float)
    presence = float(np.sum(weights[found_mask]) / np.sum(weights))

    pos_quality = 0.0
    if np.any(found_mask):
        pos_quality = float(np.mean(1.0 - offsets[found_mask] / max(tolerance, 1e-6)))

    agreement = _spearman(np.asarray(theo_found), np.asarray(exp_found))

    score = presence * (0.85 + 0.15 * pos_quality)
    if agreement is not None:
        # Only nudge the score when there are enough lines to rank meaningfully
        score *= 0.9 + 0.1 * max(0.0, agreement)
    top_found = bool(found_mask[strongest_idx])
    if not top_found:
        score *= MISSING_TOP_PENALTY

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "presence": presence,
        "n_expected": int(len(fp_tt)),
        "n_found": int(np.sum(found_mask)),
        "top_found": top_found,
        "position_quality": pos_quality,
        "intensity_agreement": float(agreement) if agreement is not None else None,
        "missing_strong": [float(t) for t in fp_tt[~found_mask]],
        "matched_two_theta": [float(t) for t in exp_matched[found_mask]],
    }


def rank_by_fingerprint(
    results: Sequence[Dict],
    exp_peaks: Dict,
    theo_lookup: Callable[[Dict], Optional[Dict]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    n_peaks: int = DEFAULT_N_FINGERPRINT,
    min_rel_intensity: float = DEFAULT_MIN_REL_INTENSITY,
    min_score: float = 0.35,
    min_found: int = 2,
    require_top_peak: bool = False,
    max_results: Optional[int] = None,
    exp_range: Optional[tuple] = None,
) -> List[Dict]:
    """
    Rescore search hits by fingerprint presence and drop weak ones.

    `theo_lookup` maps a result dict to its theoretical peaks (or None); the
    caller owns any database access and caching.
    """
    exp_tt = exp_peaks.get("two_theta", [])
    exp_int = exp_peaks.get("intensity", [])
    scored: List[Dict] = []

    for result in results:
        theo = theo_lookup(result)
        if not theo:
            continue
        info = fingerprint_score(
            exp_tt,
            exp_int,
            theo.get("two_theta", []),
            theo.get("intensity", []),
            tolerance=tolerance,
            n_peaks=n_peaks,
            min_rel_intensity=min_rel_intensity,
            exp_range=exp_range,
        )
        if info["n_found"] < min_found or info["score"] < min_score:
            continue
        if require_top_peak and not info["top_found"]:
            continue
        enriched = dict(result)
        enriched["fingerprint"] = info
        enriched["fingerprint_score"] = info["score"]
        scored.append(enriched)

    scored.sort(key=lambda r: r["fingerprint_score"], reverse=True)
    if max_results is not None:
        scored = scored[:max_results]
    return scored


def unexplained_peaks(
    exp_peaks: Dict,
    phase_theo_patterns: Sequence[Dict],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict:
    """Summarize which experimental peaks no accepted phase accounts for."""
    tt = np.asarray(exp_peaks.get("two_theta", []), dtype=float)
    inten = np.asarray(exp_peaks.get("intensity", []), dtype=float)
    if len(tt) == 0:
        return {"two_theta": np.array([]), "intensity": np.array([]), "fraction": 0.0}

    claimed = np.zeros(len(tt), dtype=bool)
    for theo in phase_theo_patterns:
        theo_tt = np.asarray(theo.get("two_theta", []), dtype=float)
        if len(theo_tt) == 0:
            continue
        for i, t in enumerate(tt):
            if claimed[i]:
                continue
            if np.min(np.abs(theo_tt - t)) <= tolerance:
                claimed[i] = True

    total = float(np.sum(inten)) or 1.0
    return {
        "two_theta": tt[~claimed],
        "intensity": inten[~claimed],
        "fraction": float(np.sum(inten[~claimed]) / total),
        "n_unexplained": int(np.sum(~claimed)),
    }
