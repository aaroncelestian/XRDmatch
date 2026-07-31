"""Residual pattern / peak helpers for iterative multiphase search."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _phase_theo(result: dict) -> Optional[dict]:
    theo = result.get("theoretical_peaks")
    if theo and len(theo.get("two_theta", [])) > 0:
        return theo
    return None


def explained_peak_fractions(
    peak_two_theta: np.ndarray,
    selected_results: Sequence[dict],
    tolerance: float,
) -> np.ndarray:
    """
    For each experimental peak, return how strongly it is explained by selected phases.

    0 = unmatched, 1 = strongly matched by at least one selected phase.
    Soft: multiple weak matches accumulate but clamp at 1.
    """
    n = len(peak_two_theta)
    explained = np.zeros(n, dtype=float)
    if n == 0 or not selected_results:
        return explained

    for result in selected_results:
        theo = _phase_theo(result)
        if theo is None:
            # Fall back to matched peak list from peak-matching
            for m in result.get("matches") or []:
                exp_tt = m.get("exp_2theta")
                if exp_tt is None:
                    continue
                diffs = np.abs(peak_two_theta - float(exp_tt))
                idx = int(np.argmin(diffs))
                if diffs[idx] <= tolerance:
                    quality = float(m.get("intensity_similarity", 0.7))
                    pos = max(0.0, 1.0 - float(m.get("difference", 0.0)) / max(tolerance, 1e-6))
                    explained[idx] = min(1.0, explained[idx] + 0.5 * quality + 0.5 * pos)
            continue

        theo_tt = np.asarray(theo["two_theta"], dtype=float)
        theo_int = np.asarray(theo["intensity"], dtype=float)
        if len(theo_tt) == 0:
            continue
        tmax = float(np.max(theo_int)) if len(theo_int) else 1.0
        for i, tt in enumerate(peak_two_theta):
            diffs = np.abs(theo_tt - tt)
            j = int(np.argmin(diffs))
            if diffs[j] > tolerance:
                continue
            pos = max(0.0, 1.0 - float(diffs[j]) / max(tolerance, 1e-6))
            inten_w = float(theo_int[j] / tmax) if tmax > 0 else 0.5
            explained[i] = min(1.0, explained[i] + 0.55 * pos + 0.45 * inten_w)
    return explained


def residual_peak_weights(
    explained: np.ndarray,
    *,
    unmatched_boost: float = 1.5,
    overlap_keep: float = 0.35,
) -> np.ndarray:
    """
    Convert explained fractions into residual search weights.

    - Unmatched peaks (explained≈0): weight = unmatched_boost
    - Fully explained peaks: weight = overlap_keep (not zero — overlaps still count)
    - Partial: linear blend between boost and overlap_keep
    """
    explained = np.clip(np.asarray(explained, dtype=float), 0.0, 1.0)
    unmatched_boost = max(float(unmatched_boost), 0.1)
    overlap_keep = float(np.clip(overlap_keep, 0.0, 1.0))
    return unmatched_boost * (1.0 - explained) + overlap_keep * explained


def build_residual_peaks(
    peaks: dict,
    selected_results: Sequence[dict],
    tolerance: float,
    *,
    unmatched_boost: float = 1.5,
    overlap_keep: float = 0.35,
) -> Tuple[dict, dict]:
    """
    Return (residual_peaks, info) where intensities are reweighted for residual search.
    """
    tt = np.asarray(peaks.get("two_theta", []), dtype=float)
    inten = np.asarray(peaks.get("intensity", []), dtype=float)
    d = peaks.get("d_spacing")
    explained = explained_peak_fractions(tt, selected_results, tolerance)
    weights = residual_peak_weights(
        explained, unmatched_boost=unmatched_boost, overlap_keep=overlap_keep
    )
    residual = {
        "two_theta": tt.copy(),
        "intensity": inten * weights,
        "d_spacing": np.asarray(d, dtype=float).copy() if d is not None else None,
        "wavelength": peaks.get("wavelength"),
        "indices": peaks.get("indices"),
        "residual_weights": weights,
        "explained_fraction": explained,
    }
    info = {
        "n_peaks": int(len(tt)),
        "n_unmatched": int(np.sum(explained < 0.15)),
        "n_overlap": int(np.sum((explained >= 0.15) & (explained < 0.85))),
        "n_explained": int(np.sum(explained >= 0.85)),
        "overlap_keep": overlap_keep,
        "unmatched_boost": unmatched_boost,
    }
    return residual, info


def build_residual_pattern(
    pattern: dict,
    selected_results: Sequence[dict],
    *,
    overlap_keep: float = 0.35,
    fwhm: float = 0.12,
) -> Tuple[dict, dict]:
    """
    Soft-subtract selected phase contributions from the continuous pattern.

    residual = max(0, obs - (1 - overlap_keep) * contribution)
    so overlapping regions keep overlap_keep of the explained intensity.
    """
    tt = np.asarray(pattern["two_theta"], dtype=float)
    obs = np.asarray(pattern["intensity"], dtype=float).astype(float)
    contribution = np.zeros_like(obs)
    overlap_keep = float(np.clip(overlap_keep, 0.0, 1.0))

    for result in selected_results:
        theo = _phase_theo(result)
        if theo is None:
            continue
        theo_tt = np.asarray(theo["two_theta"], dtype=float)
        theo_int = np.asarray(theo["intensity"], dtype=float)
        if len(theo_tt) == 0:
            continue
        # Scale theo peaks into obs intensity units via matched peaks if available
        scale = _estimate_scale(obs, tt, theo_tt, theo_int, result)
        contribution += _pseudo_voigt(tt, theo_tt, theo_int * scale, fwhm=fwhm)

    subtract = (1.0 - overlap_keep) * contribution
    residual_inten = np.maximum(obs - subtract, 0.0)
    residual = {
        "two_theta": tt,
        "intensity": residual_inten,
        "intensity_error": pattern.get("intensity_error"),
        "wavelength": pattern.get("wavelength"),
        "file_path": pattern.get("file_path"),
        "file_format": pattern.get("file_format"),
        "processed": True,
        "residual": True,
    }
    obs_sum = float(np.sum(obs)) or 1.0
    info = {
        "fraction_remaining": float(np.sum(residual_inten) / obs_sum),
        "overlap_keep": overlap_keep,
        "n_phases": len(selected_results),
    }
    return residual, info


def _estimate_scale(obs, obs_tt, theo_tt, theo_int, result: dict) -> float:
    matches = result.get("matches") or []
    if matches:
        ratios = []
        for m in matches:
            ti = m.get("norm_theo_int") or m.get("theo_int")
            ei = m.get("norm_exp_int") or m.get("exp_int")
            if ti and ei and float(ti) > 0:
                ratios.append(float(ei) / float(ti))
        if ratios:
            return float(np.median(ratios))
    # Fallback: match strongest theo peak to nearest obs
    if len(theo_int) == 0 or len(obs) == 0:
        return 1.0
    j = int(np.argmax(theo_int))
    i = int(np.argmin(np.abs(obs_tt - theo_tt[j])))
    if theo_int[j] <= 0:
        return 1.0
    return float(obs[i] / theo_int[j])


def _pseudo_voigt(x, centers, intensities, fwhm: float = 0.12) -> np.ndarray:
    pattern = np.zeros_like(x, dtype=float)
    sigma_g = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma_l = fwhm / 2
    for center, intensity in zip(centers, intensities):
        if intensity <= 0:
            continue
        gaussian = np.exp(-0.5 * ((x - center) / sigma_g) ** 2)
        lorentzian = 1.0 / (1.0 + ((x - center) / gamma_l) ** 2)
        pattern += float(intensity) * (0.7 * gaussian + 0.3 * lorentzian)
    return pattern


def mineral_key(phase_or_result: dict) -> str:
    if not isinstance(phase_or_result, dict):
        return str(phase_or_result).lower()
    phase = phase_or_result.get("phase", phase_or_result)
    name = (
        phase.get("mineral")
        or phase.get("mineral_name")
        or phase_or_result.get("mineral_name")
        or ""
    )
    return str(name).strip().lower()


def mineral_ids(phase_or_result: dict) -> set:
    """Collect all identifying IDs for a phase / match / search hit."""
    if not isinstance(phase_or_result, dict):
        return set()
    phase = phase_or_result.get("phase", phase_or_result)
    ids = set()
    for src in (phase_or_result, phase):
        if not isinstance(src, dict):
            continue
        for key in ("id", "amcsd_id", "mineral_id"):
            val = src.get(key)
            if val is None or val == "":
                continue
            ids.add(str(val))
    return ids


def exclusion_sets(kept_phases: Sequence[dict]) -> Tuple[set, set]:
    """
    Return (excluded_ids, excluded_names) for already-found phases.

    Residual / multiphase search should drop any hit whose mineral ID is in
    excluded_ids OR whose mineral name is in excluded_names (e.g. no more quartz
    once the user has accepted a quartz).
    """
    ids: set = set()
    names: set = set()
    for p in kept_phases or []:
        ids |= mineral_ids(p)
        name = mineral_key(p)
        if name:
            names.add(name)
    return ids, names


def is_excluded_hit(hit: dict, excluded_ids: set, excluded_names: set) -> bool:
    """True if a search/match result duplicates an already-found mineral."""
    if mineral_ids(hit) & excluded_ids:
        return True
    name = mineral_key(hit)
    if name and name in excluded_names:
        return True
    # Search engine rows often use mineral_name / mineral_id at top level
    if isinstance(hit, dict):
        mid = hit.get("mineral_id")
        if mid is not None and str(mid) in excluded_ids:
            return True
        mname = str(hit.get("mineral_name") or "").strip().lower()
        if mname and mname in excluded_names:
            return True
    return False


def filter_new_hits(hits: Sequence[dict], kept_phases: Sequence[dict]) -> list:
    """Drop hits that match already-found mineral IDs or names."""
    excluded_ids, excluded_names = exclusion_sets(kept_phases)
    if not excluded_ids and not excluded_names:
        return list(hits)
    return [h for h in hits if not is_excluded_hit(h, excluded_ids, excluded_names)]
