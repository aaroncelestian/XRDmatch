"""
Formatting of Le Bail results for display, copying and export.

The on-screen table, the clipboard, the CSV file and the details window all show
the same numbers, so they are all formatted here rather than each rebuilding the
strings from the results dict. The summary table carries what is read at a
glance; everything the refinement holds is in the detail rows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.profile_functions import skew_description

# (label, tooltip) for the per-phase summary table
SUMMARY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Phase", "Mineral name and formula"),
    ("wt%", "Chung RIR weight percent where every phase has an I/Ic, "
            "otherwise each phase's share of the fitted pattern"),
    ("Scale", "Refined scale factor"),
    ("Strain", "Microstrain, Δd/d × 10⁻⁶"),
    ("a (Å)", "Refined a axis"),
    ("b (Å)", "Refined b axis"),
    ("c (Å)", "Refined c axis"),
    ("α (°)", "Refined cell angle. An angle fixed at 90° by symmetry is held there"),
    ("β (°)", "Refined cell angle. An angle fixed at 90° by symmetry is held there"),
    ("γ (°)", "Refined cell angle. An angle fixed at 90° by symmetry is held there"),
    ("V (Å³)", "Refined cell volume"),
    ("ΔV %", "Cell volume change from the starting cell. The per-axis changes "
             "are in the parameter window, since they need not agree"),
    ("Contrib.%", "Share of the calculated pattern intensity"),
)

RIR_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Phase", ""),
    ("wt%", "Chung RIR weight percent"),
    ("Scale", "Refined scale factor"),
    ("Fitted I", "Integrated intensity of the strongest line, which is what I/Ic is defined on"),
    ("RIR", "I/I_corundum from AMCSD"),
    ("Pattern share %", "Share of the fitted pattern intensity, before the RIR conversion"),
)

_MISSING = "—"

# Durbin-Watson runs 0 to 4 and sits at 2 when successive residuals are
# independent. Departures either way mean the residuals still have structure the
# model has not taken up, so the bands below are symmetric about 2.
_DURBIN_BANDS = (
    (1.0, "strongly correlated — systematic misfit"),
    (1.6, "correlated — some systematic misfit"),
    (2.4, "close to random — near the noise floor"),
    (3.0, "correlated — some systematic misfit"),
)
_DURBIN_EXTREME = "strongly anti-correlated — check the error estimates"


# Above this the peak width is mostly coming from the refined sample terms
# rather than from the instrument, which for an ordinary ground powder means
# the crystallite size is standing in for a resolution curve nobody calibrated.
_WIDTH_SHARE_LIMIT = 0.5


def _percent(fraction):
    try:
        return 100.0 * float(fraction)
    except (TypeError, ValueError):
        return None


def _size_warning(phase: Dict) -> str:
    share = phase.get("sample_width_share")
    try:
        share = float(share)
    except (TypeError, ValueError):
        return _MISSING
    if share < _WIDTH_SHARE_LIMIT:
        return "none"
    return (f"{share * 100:.0f}% of the width is sample broadening — calibrate "
            "U, V, W against a standard before reading the size as a particle size")


def _residual_character(value) -> str:
    """Say in words what the Durbin-Watson number means."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return _MISSING
    if not np.isfinite(value):
        return _MISSING
    for limit, description in _DURBIN_BANDS:
        if value < limit:
            return description
    return _DURBIN_EXTREME


_WEIGHT_NOTES = {
    "rir": "Chung RIR weight percent",
    "contribution": (
        "Share of the fitted pattern intensity, standing in for the RIR weight "
        "percent because the database has no I/Ic for every phase. It assumes "
        "the phases scatter equally, so treat it as an estimate."
    ),
}


def weight_basis(results: Optional[Dict]) -> Optional[str]:
    """'rir', 'contribution' or None — what the wt% column was derived from."""
    inner = (results or {}).get("refinement_results") or {}
    for phase in inner.get("phase_summary") or []:
        return phase.get("weight_percent_basis")
    return None


def weight_basis_note(results: Optional[Dict]) -> str:
    """A sentence explaining where the wt% column came from, for a tooltip."""
    return _WEIGHT_NOTES.get(weight_basis(results), "")


def phases_missing_rir(results: Optional[Dict]) -> List[str]:
    inner = (results or {}).get("refinement_results") or {}
    return [
        str(phase.get("name") or f"Phase {index + 1}")
        for index, phase in enumerate(inner.get("phase_summary") or [])
        if not phase.get("rir")
    ]


def _number(value, spec: str, missing: str = _MISSING) -> str:
    if value is None:
        return missing
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return missing


def _cell_change(phase: Dict, key: str) -> Optional[float]:
    """
    How far one cell parameter moved: percent for the edges and the volume,
    degrees for the angles.

    Taken from what the refinement reported where it said, and worked out from
    the two cells where it did not, so a result saved by an older version still
    fills the column.
    """
    delta = phase.get("cell_delta") or {}
    if key in delta:
        return delta[key]
    now = (phase.get("unit_cell") or {}).get(key)
    start = (phase.get("base_unit_cell") or {}).get(key)
    if now is None or start is None:
        return None
    try:
        now, start = float(now), float(start)
    except (TypeError, ValueError):
        return None
    if key in ("alpha", "beta", "gamma"):
        return now - start
    return (now / start - 1.0) * 100.0 if start > 0 else None


def summary_rows(results: Optional[Dict]) -> List[List[str]]:
    """One formatted row per phase, matching SUMMARY_COLUMNS."""
    inner = (results or {}).get("refinement_results") or {}
    rows = []
    for index, phase in enumerate(inner.get("phase_summary") or []):
        cell = phase.get("unit_cell") or {}
        rows.append([
            str(phase.get("name") or f"Phase {index + 1}"),
            _number(phase.get("weight_percent"), ".1f"),
            _number(phase.get("scale"), ".4g"),
            _number(phase.get("microstrain"), ".0f"),
            _number(cell.get("a"), ".4f"),
            _number(cell.get("b"), ".4f"),
            _number(cell.get("c"), ".4f"),
            _number(cell.get("alpha"), ".3f"),
            _number(cell.get("beta"), ".3f"),
            _number(cell.get("gamma"), ".3f"),
            _number(cell.get("volume"), ".2f"),
            _number(_cell_change(phase, "volume"), "+.3f"),
            _number(phase.get("contribution_percent"), ".1f"),
        ])
    return rows


def summary_tooltips(results: Optional[Dict]) -> List[List[str]]:
    """Per-cell tooltips for the summary table, same shape as summary_rows."""
    inner = (results or {}).get("refinement_results") or {}
    tooltips = []
    for phase in inner.get("phase_summary") or []:
        base = phase.get("base_unit_cell") or {}
        wt_note = _WEIGHT_NOTES.get(
            phase.get("weight_percent_basis"),
            "Not available for a Le Bail extraction",
        )
        free = ", ".join(phase.get("cell_free") or ())
        cell_note = (
            f"Free: {free}. Anything not listed is held by the symmetry of the "
            "starting cell" if free else
            "Reflections could not be indexed, so the cell was dilated as a whole"
        )
        tooltips.append([
            str(phase.get("formula") or ""),
            wt_note, "", "",
            _start_note(base.get("a")), _start_note(base.get("b")),
            _start_note(base.get("c")),
            _start_note(base.get("alpha"), "°"), _start_note(base.get("beta"), "°"),
            _start_note(base.get("gamma"), "°"),
            _start_note(base.get("volume"), "Å³"),
            cell_note, "",
        ])
    return tooltips


def _skew_direction(value) -> str:
    return _MISSING if value is None else skew_description(value)


def _start_note(value, unit: str = "Å") -> str:
    return "" if value is None else f"start {float(value):.4f} {unit}"


def rir_rows(result: Optional[Dict]) -> List[List[str]]:
    """One formatted row per phase of an RIR quantification."""
    phases = (result or {}).get("phases") or []
    total = sum(p.get("pattern_intensity", 0.0) for p in phases) or 1.0
    rows = []
    for index, phase in enumerate(phases):
        rows.append([
            str(phase.get("name") or f"Phase {index + 1}"),
            _number(phase.get("weight_percent"), ".1f"),
            _number(phase.get("scale"), ".4g"),
            _number(phase.get("line_area"), ".4g"),
            _number(phase.get("rir") or None, ".3f"),
            _number(phase.get("pattern_intensity", 0.0) / total * 100.0, ".1f"),
        ])
    return rows


def summary_headline(results: Optional[Dict]) -> List[str]:
    """Short statistics shown above the table."""
    if not results:
        return []
    inner = results.get("refinement_results") or {}
    factors = inner.get("final_r_factors") or results.get("r_factors") or {}
    globals_ = inner.get("global_parameters") or {}

    parts = []
    for key, label in (("Rwp", "Rwp"), ("Rwp_peak", "Rwp(peaks)"), ("Rp", "Rp"),
                       ("R_Bragg", "R_Bragg"), ("GoF", "GoF")):
        value = factors.get(key)
        if value is not None and np.isfinite(float(value)):
            parts.append(f"{label}={float(value):.2f}" + ("%" if key != "GoF" else ""))
    durbin = factors.get("durbin_watson")
    if durbin is not None and np.isfinite(float(durbin)):
        parts.append(f"DW={float(durbin):.2f}")
    parts.append(f"zero={float(globals_.get('zero_shift', 0.0)):+.4f}°")
    parts.append(f"disp={float(globals_.get('displacement', 0.0)):+.4f}°")
    if inner.get("iterations"):
        parts.append(f"{inner['iterations']} cycles")
    if inner.get("intensity_model") == "extract":
        parts.append("Le Bail extraction — wt% unavailable")
    elif weight_basis(results) == "contribution":
        missing = phases_missing_rir(results)
        named = ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
        parts.append(f"wt% from pattern contribution — no RIR for {named}")
    return parts


def phase_parameters(results: Optional[Dict],
                     overrides: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    Current per-phase values and refine flags, for the editable grid.

    The last run's values are the starting point, so the grid opens where the
    refinement finished rather than at a default nobody chose. Anything the user
    has already set by hand wins over that, otherwise their edits would be
    silently undone every time a run completed.
    """
    inner = (results or {}).get("refinement_results") or {}
    out: Dict[str, Dict] = {}
    for row in inner.get("phase_summary") or []:
        name = row.get("name")
        if not name:
            continue
        cell = row.get("unit_cell") or {}
        entry = {
            "scale_factor": row.get("scale"),
            "microstrain": row.get("microstrain"),
            "crystallite_size": row.get("crystallite_size"),
            "asymmetry": row.get("asymmetry"),
            "absorption": row.get("absorption"),
            **{f"cell_{key}": cell.get(key)
               for key in ("a", "b", "c", "alpha", "beta", "gamma")},
            # Defaults match the refine-stage checkboxes; an older result that
            # never recorded its flags still opens looking like a fresh run
            "refine_scale": True,
            "refine_strain": True,
            "refine_size": False,
            "refine_asymmetry": False,
            "refine_cell": True,
            "refine_absorption": False,
            "refine_harmonics": False,
        }
        entry.update(row.get("refine_flags") or {})
        out[name] = entry
    for name, override in (overrides or {}).items():
        out.setdefault(name, {}).update(
            {k: v for k, v in override.items() if k != "_locked"}
        )
    return out


# --- everything the refinement holds, for the details window and the CSV ----

def _alpha2_description(ratio) -> str:
    """
    What the doublet setting was, in terms of what it does to the peaks.

    Worth spelling out rather than printing a bare number: a reader comparing
    two runs needs to see at a glance that one modelled two lines per reflection
    and the other one, since almost every peak-shape quantity beside it means
    something different depending on which.
    """
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        return "not modelled"
    if ratio <= 0.0:
        return "not modelled (one line per reflection)"
    return f"modelled at {ratio:.3f} of each parent line"


def global_rows(results: Optional[Dict]) -> List[Tuple[str, str]]:
    """(name, value) for the parameters shared by every phase."""
    inner = (results or {}).get("refinement_results") or {}
    factors = inner.get("final_r_factors") or (results or {}).get("r_factors") or {}
    globals_ = inner.get("global_parameters") or {}

    model = inner.get("intensity_model")
    rows = [
        ("Rwp (%)", _number(factors.get("Rwp"), ".3f")),
        ("Rwp near peaks (%)", _number(factors.get("Rwp_peak"), ".3f")),
        ("Points near peaks (%)", _number(factors.get("peak_coverage"), ".1f")),
        ("Rp (%)", _number(factors.get("Rp"), ".3f")),
        ("Rexp (%)", _number(factors.get("Rexp"), ".3f")),
        ("R_Bragg (%)", _number(factors.get("R_Bragg"), ".3f")),
        ("Durbin-Watson", _number(factors.get("durbin_watson"), ".3f")),
        ("Residuals look", _residual_character(factors.get("durbin_watson"))),
        ("Goodness of fit", _number(factors.get("GoF"), ".3f")),
        ("Fitted region", "Near modelled peaks only"
         if inner.get("fit_peak_regions_only") else "Whole pattern"),
        ("Cycles", str(inner.get("iterations") or _MISSING)),
        ("Intensity model", {
            "fixed": "Reference intensities (quantitative)",
            "extract": "Le Bail extraction (profile only)",
        }.get(model, str(model or _MISSING))),
        ("Zero shift (°)", _number(globals_.get("zero_shift"), "+.5f")),
        ("Sample displacement (°)", _number(globals_.get("displacement"), "+.5f")),
        ("Instrument U", _number(globals_.get("u_param"), ".6f")),
        ("Instrument V", _number(globals_.get("v_param"), ".6f")),
        ("Instrument W", _number(globals_.get("w_param"), ".6f")),
        ("Axial asymmetry", _number(globals_.get("axial_asymmetry"), "+.5f")),
        ("Kα2 satellites", _alpha2_description(globals_.get("alpha2_ratio"))),
    ]
    bg_coeffs = globals_.get("background_coeffs") or []
    if globals_.get("refine_background") or bg_coeffs:
        order = globals_.get("background_order")
        if order is None and bg_coeffs:
            order = len(bg_coeffs) - 1
        rows.append(("Background order", str(order if order is not None else _MISSING)))
        if bg_coeffs:
            rows.append((
                "Background coeffs",
                ", ".join(f"{float(c):+.4g}" for c in bg_coeffs[:6])
                + ("…" if len(bg_coeffs) > 6 else ""),
            ))
    for label, key in (
        ("Zero shift refined", "refine_zero_shift"),
        ("Displacement refined", "refine_displacement"),
        ("Instrument profile refined", "refine_instrument_profile"),
        ("Axial asymmetry refined", "refine_axial_asymmetry"),
        ("Background refined", "refine_background"),
        ("Kα2 ratio refined", "refine_alpha2_ratio"),
    ):
        rows.append((label, "yes" if globals_.get(key) else "no"))
    return rows


# Every per-phase quantity, in the order the details window lists them
_DETAIL_FIELDS = (
    ("Formula", lambda p: str(p.get("formula") or _MISSING)),
    ("Weight percent", lambda p: _number(p.get("weight_percent"), ".2f")),
    ("Weight percent from", lambda p: {
        "rir": "Chung RIR",
        "contribution": "Pattern contribution (RIR incomplete)",
    }.get(p.get("weight_percent_basis"), _MISSING)),
    ("Contribution (%)", lambda p: _number(p.get("contribution_percent"), ".2f")),
    ("Scale factor", lambda p: _number(p.get("scale"), ".6g")),
    ("Strongest-line height", lambda p: _number(p.get("line_intensity"), ".6g")),
    ("Strongest-line area", lambda p: _number(p.get("line_area"), ".6g")),
    ("Integrated intensity", lambda p: _number(p.get("integrated_intensity"), ".6g")),
    ("RIR (I/Ic)", lambda p: _number(p.get("rir"), ".3f")),
    ("Width from sample terms (%)",
     lambda p: _number(_percent(p.get("sample_width_share")), ".0f")),
    ("Crystallite size warning", lambda p: _size_warning(p)),
    ("Microstrain (×10⁻⁶)", lambda p: _number(p.get("microstrain"), ".1f")),
    ("Crystallite size (µm)", lambda p: _number(p.get("crystallite_size"), ".4g")),
    ("Phase asymmetry", lambda p: _number(p.get("asymmetry"), "+.4f")),
    ("Peak skew", lambda p: _skew_direction(p.get("asymmetry"))),
    ("Cell parameters free", lambda p: ", ".join(p.get("cell_free") or ())
     or "none refined separately — dilated as a whole"),
    ("a (Å)", lambda p: _number((p.get("unit_cell") or {}).get("a"), ".5f")),
    ("b (Å)", lambda p: _number((p.get("unit_cell") or {}).get("b"), ".5f")),
    ("c (Å)", lambda p: _number((p.get("unit_cell") or {}).get("c"), ".5f")),
    ("α (°)", lambda p: _number((p.get("unit_cell") or {}).get("alpha"), ".4f")),
    ("β (°)", lambda p: _number((p.get("unit_cell") or {}).get("beta"), ".4f")),
    ("γ (°)", lambda p: _number((p.get("unit_cell") or {}).get("gamma"), ".4f")),
    ("Volume (Å³)", lambda p: _number((p.get("unit_cell") or {}).get("volume"), ".3f")),
    ("Δa (%)", lambda p: _number(_cell_change(p, "a"), "+.4f")),
    ("Δb (%)", lambda p: _number(_cell_change(p, "b"), "+.4f")),
    ("Δc (%)", lambda p: _number(_cell_change(p, "c"), "+.4f")),
    ("Δα (°)", lambda p: _number(_cell_change(p, "alpha"), "+.4f")),
    ("Δβ (°)", lambda p: _number(_cell_change(p, "beta"), "+.4f")),
    ("Δγ (°)", lambda p: _number(_cell_change(p, "gamma"), "+.4f")),
    ("ΔV (%)", lambda p: _number(_cell_change(p, "volume"), "+.4f")),
    ("Starting a (Å)", lambda p: _number((p.get("base_unit_cell") or {}).get("a"), ".5f")),
    ("Starting b (Å)", lambda p: _number((p.get("base_unit_cell") or {}).get("b"), ".5f")),
    ("Starting c (Å)", lambda p: _number((p.get("base_unit_cell") or {}).get("c"), ".5f")),
    ("Starting α (°)",
     lambda p: _number((p.get("base_unit_cell") or {}).get("alpha"), ".4f")),
    ("Starting β (°)",
     lambda p: _number((p.get("base_unit_cell") or {}).get("beta"), ".4f")),
    ("Starting γ (°)",
     lambda p: _number((p.get("base_unit_cell") or {}).get("gamma"), ".4f")),
    ("Starting volume (Å³)",
     lambda p: _number((p.get("base_unit_cell") or {}).get("volume"), ".3f")),
    ("Absorption", lambda p: _number(p.get("absorption"), "+.5f")),
    ("Harmonic coefficients", lambda p: ", ".join(
        f"{c:+.4f}" for c in (p.get("harmonic_coeffs") or [])) or _MISSING),
)


def phase_names(results: Optional[Dict]) -> List[str]:
    inner = (results or {}).get("refinement_results") or {}
    return [
        str(phase.get("name") or f"Phase {index + 1}")
        for index, phase in enumerate(inner.get("phase_summary") or [])
    ]


def detail_rows(results: Optional[Dict]) -> List[List[str]]:
    """
    Every per-phase parameter, one row per parameter and one column per phase.

    Parameters run down the page rather than across it because there are far
    more of them than there are phases, and because comparing one quantity
    between phases is the usual reason for opening the window.
    """
    inner = (results or {}).get("refinement_results") or {}
    phases = inner.get("phase_summary") or []
    if not phases:
        return []
    return [
        [label] + [render(phase) for phase in phases]
        for label, render in _DETAIL_FIELDS
    ]
