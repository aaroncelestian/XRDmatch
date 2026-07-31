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

from utils.two_theta_shift import DISPLACEMENT, apply_shift, fit_shift


DEFAULT_TOLERANCE = 0.20
DEFAULT_MIN_REL_INTENSITY = 5.0
DEFAULT_N_FINGERPRINT = 10

# A phase cannot be present if its most intense line is absent
MISSING_TOP_PENALTY = 0.45

# Matching a handful of lines is weak evidence: in a 30-peak pattern a two- or
# three-line reference can be satisfied by coincidence, so such candidates are
# demoted rather than allowed to outrank a well-matched 10-line phase.
INFORMATIVE_LINES = 5


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


def coincidence_fraction(
    exp_two_theta: Sequence[float],
    tolerance: float,
    exp_range: Optional[tuple] = None,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """
    Fraction of the measured range that sits within tolerance of some peak.

    This is the chance that an arbitrary reference line lands on a peak by luck.
    A dense peak list or a wide tolerance can cover most of the pattern, at
    which point "all its lines are present" says nothing, so scores have to be
    judged against this baseline.
    """
    tt = np.asarray(exp_two_theta, dtype=float)
    if len(tt) == 0:
        return 0.0
    lo, hi = exp_range or (float(np.min(tt)) - tolerance, float(np.max(tt)) + tolerance)
    span = float(hi - lo)
    if span <= 0:
        return 0.0

    w = np.ones(len(tt)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(tt) or not np.any(w > 0):
        w = np.ones(len(tt))
    w = w / float(np.max(w))

    # Highest weight wins where windows overlap, matching build_peak_mask
    step = max(tolerance / 4.0, 1e-3)
    grid = np.arange(lo, hi + step, step)
    mask = build_peak_mask(grid, tt, w, tolerance)
    return float(np.clip(np.mean(mask), 0.0, 0.95))


def _enrichment(presence: float, chance: float) -> float:
    """Presence rescaled so a chance-level match scores zero."""
    chance = float(np.clip(chance, 0.0, 0.95))
    if chance <= 0:
        return float(np.clip(presence, 0.0, 1.0))
    return float(np.clip((presence - chance) / (1.0 - chance), 0.0, 1.0))


def _intensity_consistency(theo: np.ndarray, exp: np.ndarray) -> float:
    """
    How well the observed intensities can support this phase's line pattern.

    In a mixture other phases only ever *add* intensity, so once a scale factor
    is fitted every reference line needs at least its share of the observed
    peak. A candidate whose strongest lines land on tiny peaks cannot be
    present, however well its positions happen to line up.
    """
    if len(theo) == 0 or len(theo) != len(exp):
        return 0.0
    theo = np.asarray(theo, dtype=float)
    exp = np.asarray(exp, dtype=float)
    valid = theo > 0
    if not np.any(valid):
        return 0.0
    theo, exp = theo[valid], exp[valid]

    scale = float(np.median(exp / theo))
    if scale <= 0:
        return 0.0
    required = scale * theo
    deficit = np.clip((required - exp) / np.maximum(required, 1e-9), 0.0, 1.0)
    return float(1.0 - np.average(deficit, weights=theo))


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
    exp_weights: Optional[Sequence[float]] = None,
    chance: Optional[float] = None,
    chance_weighted: Optional[float] = None,
    shift: float = 0.0,
    shift_span: float = 0.0,
    shift_model: str = DISPLACEMENT,
) -> Dict:
    """
    Score a candidate by how many of its own strong lines are present.

    Unexplained experimental peaks carry no penalty, so minor phases in a
    mixture are scored on equal footing with the dominant phase. `exp_range`
    should be the measured 2θ range; reference lines outside it are ignored,
    while lines inside it with no matching peak count as missing.

    `exp_weights` (residual search) down-weights peaks already claimed by
    accepted phases, giving a `residual_score` that rewards candidates which
    explain what is still unaccounted for.

    `shift` moves the reference lines to where a displaced sample puts them.
    With `shift_span` above zero the shift is fitted per candidate inside
    `shift ± shift_span` and reported back as `shift`, which is what finds a
    phase whose displacement is not yet known.
    """
    exp_tt = np.asarray(exp_two_theta, dtype=float)
    exp_int = np.asarray(exp_intensity, dtype=float)
    empty = {
        "score": 0.0,
        "presence": 0.0,
        "enrichment": 0.0,
        "chance_match": 0.0,
        "residual_score": 0.0,
        "specificity": 0.0,
        "intensity_consistency": 0.0,
        "n_expected": 0,
        "n_found": 0,
        "top_found": False,
        "position_quality": 0.0,
        "intensity_agreement": 0.0,
        "missing_strong": [],
        "matched_two_theta": [],
        "shift": float(shift),
        "shift_model": shift_model,
    }
    if len(exp_tt) == 0:
        return empty

    weights = None
    if exp_weights is not None:
        weights = np.asarray(exp_weights, dtype=float)
        if len(weights) != len(exp_tt) or not np.any(weights > 0):
            weights = None
        else:
            weights = weights / float(np.max(weights))

    window = exp_range or (float(np.min(exp_tt)) - tolerance, float(np.max(exp_tt)) + tolerance)
    # A shift moves lines across the ends of the measured range, so select from
    # a padded window and drop whatever ends up outside once the shift is known
    pad = abs(float(shift)) + abs(float(shift_span))
    fp = select_fingerprint_peaks(
        theo_two_theta,
        theo_intensity,
        n_peaks=n_peaks,
        min_rel_intensity=min_rel_intensity,
        two_theta_range=(window[0] - pad, window[1] + pad),
    )
    fp_tt, fp_int = fp["two_theta"], fp["intensity"]
    if len(fp_tt) == 0:
        return empty

    fitted_shift = float(shift)
    if shift_span > 0:
        fitted_shift, _ = fit_shift(
            exp_tt, fp_tt, fp_int,
            tolerance=tolerance, center=shift, span=shift_span, model=shift_model,
        )
    obs_tt = apply_shift(fp_tt, fitted_shift, shift_model)
    if pad > 0:
        inside = (obs_tt >= window[0]) & (obs_tt <= window[1])
        if not np.any(inside):
            return {**empty, "shift": fitted_shift}
        fp_tt, fp_int, obs_tt = fp_tt[inside], fp_int[inside], obs_tt[inside]

    exp_max = float(np.max(exp_int)) if len(exp_int) and np.max(exp_int) > 0 else 1.0
    strongest_idx = int(np.argmax(fp_int))

    found_mask = np.zeros(len(fp_tt), dtype=bool)
    offsets = np.zeros(len(fp_tt), dtype=float)
    exp_matched = np.full(len(fp_tt), np.nan)
    match_weight = np.zeros(len(fp_tt), dtype=float)
    theo_found, exp_found = [], []

    for i, tt in enumerate(obs_tt):
        diffs = np.abs(exp_tt - tt)
        j = int(np.argmin(diffs))
        if diffs[j] <= tolerance:
            found_mask[i] = True
            offsets[i] = float(diffs[j])
            exp_matched[i] = float(exp_tt[j])
            match_weight[i] = float(weights[j]) if weights is not None else 1.0
            theo_found.append(float(fp_int[i]))
            exp_found.append(float(exp_int[j] / exp_max * 100.0))

    line_weights = fp_int.astype(float)
    total_weight = float(np.sum(line_weights))
    presence = float(np.sum(line_weights[found_mask]) / total_weight)
    residual_presence = float(
        np.sum(line_weights[found_mask] * match_weight[found_mask]) / total_weight
    )

    if chance is None:
        chance = coincidence_fraction(exp_tt, tolerance, exp_range)
    if weights is None:
        chance_weighted = chance
    elif chance_weighted is None:
        chance_weighted = coincidence_fraction(exp_tt, tolerance, exp_range, weights)
    enrichment = _enrichment(presence, chance)
    residual_enrichment = _enrichment(residual_presence, chance_weighted)

    pos_quality = 0.0
    if np.any(found_mask):
        pos_quality = float(np.mean(1.0 - offsets[found_mask] / max(tolerance, 1e-6)))

    agreement = _spearman(np.asarray(theo_found), np.asarray(exp_found))
    consistency = _intensity_consistency(np.asarray(theo_found), np.asarray(exp_found))

    top_found = bool(found_mask[strongest_idx])
    penalty = 1.0 if top_found else MISSING_TOP_PENALTY

    n_expected = int(len(fp_tt))
    n_found = int(np.sum(found_mask))
    specificity = min(1.0, n_expected / INFORMATIVE_LINES) * (
        0.6 + 0.4 * min(1.0, n_found / INFORMATIVE_LINES)
    )
    quality = (
        (0.75 + 0.25 * pos_quality)
        * (0.35 + 0.65 * consistency)
        * specificity
    )

    score = enrichment * quality * penalty
    residual_score = residual_enrichment * quality * penalty

    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "presence": presence,
        "enrichment": enrichment,
        "chance_match": float(chance),
        "residual_score": float(np.clip(residual_score, 0.0, 1.0)),
        "specificity": float(specificity),
        "intensity_consistency": float(consistency),
        "n_expected": n_expected,
        "n_found": n_found,
        "top_found": top_found,
        "position_quality": pos_quality,
        "intensity_agreement": float(agreement) if agreement is not None else None,
        "missing_strong": [float(t) for t in obs_tt[~found_mask]],
        "matched_two_theta": [float(t) for t in exp_matched[found_mask]],
        "shift": float(fitted_shift),
        "shift_model": shift_model,
    }


def build_peak_mask(
    grid: np.ndarray,
    peak_two_theta: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """
    Indicator over a 2θ grid marking where experimental peaks were observed.

    Values are peak weights (1.0 by default), so a weighted mask can favour
    peaks that no accepted phase explains yet.
    """
    grid = np.asarray(grid, dtype=float)
    mask = np.zeros(len(grid), dtype=np.float32)
    tt = np.asarray(peak_two_theta, dtype=float)
    if len(grid) == 0 or len(tt) == 0:
        return mask

    w = np.ones(len(tt), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(tt):
        w = np.ones(len(tt), dtype=float)

    step = float(np.mean(np.diff(grid))) if len(grid) > 1 else 0.02
    half = max(1, int(np.ceil(tolerance / max(step, 1e-9))))
    centers = np.searchsorted(grid, tt)
    for center, weight in zip(centers, w):
        lo = max(0, int(center) - half)
        hi = min(len(grid), int(center) + half + 1)
        if lo < hi:
            np.maximum(mask[lo:hi], np.float32(weight), out=mask[lo:hi])
    return mask


def line_coverage(
    pattern_matrix: np.ndarray,
    grid: np.ndarray,
    peak_two_theta: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    row_sums: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fraction of each indexed phase's own intensity that lands on observed peaks.

    One matrix-vector product screens the whole database on peak positions
    rather than whole-pattern similarity, which is what lets a minor phase in a
    mixture survive to detailed scoring.
    """
    mask = build_peak_mask(grid, peak_two_theta, weights, tolerance)
    if not np.any(mask):
        return np.zeros(pattern_matrix.shape[0], dtype=float)

    explained = pattern_matrix @ mask
    totals = pattern_matrix.sum(axis=1) if row_sums is None else row_sums
    totals = np.where(totals > 0, totals, np.inf)
    return np.asarray(explained / totals, dtype=float)


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
    dedupe_by_name: bool = True,
    exp_weights: Optional[Sequence[float]] = None,
    shift: float = 0.0,
    shift_span: float = 0.0,
    shift_model: str = DISPLACEMENT,
) -> List[Dict]:
    """
    Rescore search hits by fingerprint presence and drop weak ones.

    `theo_lookup` maps a result dict to its theoretical peaks (or None), in
    their unshifted database positions; the caller owns any database access and
    caching. Databases hold many records per mineral, so by default only the
    best-scoring record per mineral name is kept — otherwise a dozen quartz
    entries bury the minor phases.
    """
    exp_tt = exp_peaks.get("two_theta", [])
    exp_int = exp_peaks.get("intensity", [])
    scored: List[Dict] = []

    # Same for every candidate, and the dominant cost otherwise
    chance = coincidence_fraction(exp_tt, tolerance, exp_range)
    chance_weighted = (
        coincidence_fraction(exp_tt, tolerance, exp_range, exp_weights)
        if exp_weights is not None else chance
    )

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
            exp_weights=exp_weights,
            chance=chance,
            chance_weighted=chance_weighted,
            shift=shift,
            shift_span=shift_span,
            shift_model=shift_model,
        )
        # In residual mode rank on newly explained lines, not overall presence
        rank_score = info["residual_score"] if exp_weights is not None else info["score"]
        if info["n_found"] < min_found or rank_score < min_score:
            continue
        if require_top_peak and not info["top_found"]:
            continue
        enriched = dict(result)
        enriched["fingerprint"] = info
        enriched["fingerprint_score"] = rank_score
        scored.append(enriched)

    scored.sort(key=lambda r: r["fingerprint_score"], reverse=True)

    if dedupe_by_name:
        best: Dict[str, Dict] = {}
        for result in scored:
            key = str(result.get("mineral_name", "")).strip().lower()
            if not key:
                key = f"__row{len(best)}"
            if key in best:
                best[key]["duplicate_records"] = best[key].get("duplicate_records", 0) + 1
                continue
            result["duplicate_records"] = 0
            best[key] = result
        scored = list(best.values())

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
