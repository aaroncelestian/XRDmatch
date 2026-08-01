"""
Formatting of Le Bail results for display, copying and export.

The on-screen table, the clipboard, the CSV file and the details window all show
the same numbers, so they are all formatted here rather than each rebuilding the
strings from the results dict. The summary table carries what is read at a
glance; everything the refinement holds is in the detail rows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    ("α (°)", "Cell angle; unchanged by an isotropic dilation"),
    ("β (°)", "Cell angle; unchanged by an isotropic dilation"),
    ("γ (°)", "Cell angle; unchanged by an isotropic dilation"),
    ("V (Å³)", "Refined cell volume"),
    ("Δlattice %", "Isotropic lattice dilation from the starting cell"),
    ("Contrib.%", "Share of the calculated pattern intensity"),
)

RIR_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Phase", ""),
    ("wt%", "Chung RIR weight percent"),
    ("Scale", "Refined scale factor"),
    ("Fitted I", "Strongest-line intensity from the fit"),
    ("RIR", "I/I_corundum from AMCSD"),
    ("Pattern share %", "Share of the fitted pattern intensity, before the RIR conversion"),
)

_MISSING = "—"

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


def summary_rows(results: Optional[Dict]) -> List[List[str]]:
    """One formatted row per phase, matching SUMMARY_COLUMNS."""
    inner = (results or {}).get("refinement_results") or {}
    rows = []
    for index, phase in enumerate(inner.get("phase_summary") or []):
        cell = phase.get("unit_cell") or {}
        lattice = phase.get("lattice_scale")
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
            _number(None if lattice is None else (float(lattice) - 1.0) * 100.0, "+.3f"),
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
        tooltips.append([
            str(phase.get("formula") or ""),
            wt_note, "", "",
            _start_note(base.get("a")), _start_note(base.get("b")),
            _start_note(base.get("c")),
            "", "", "",
            _start_note(base.get("volume"), "Å³"),
            "Isotropic lattice dilation", "",
        ])
    return tooltips


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
            _number(phase.get("line_intensity"), ".4g"),
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
    for key, label in (("Rwp", "Rwp"), ("Rp", "Rp"), ("GoF", "GoF")):
        value = factors.get(key)
        if value is not None:
            parts.append(f"{label}={float(value):.2f}" + ("%" if key != "GoF" else ""))
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


# --- everything the refinement holds, for the details window and the CSV ----

def global_rows(results: Optional[Dict]) -> List[Tuple[str, str]]:
    """(name, value) for the parameters shared by every phase."""
    inner = (results or {}).get("refinement_results") or {}
    factors = inner.get("final_r_factors") or (results or {}).get("r_factors") or {}
    globals_ = inner.get("global_parameters") or {}

    model = inner.get("intensity_model")
    rows = [
        ("Rwp (%)", _number(factors.get("Rwp"), ".3f")),
        ("Rp (%)", _number(factors.get("Rp"), ".3f")),
        ("Rexp (%)", _number(factors.get("Rexp"), ".3f")),
        ("Goodness of fit", _number(factors.get("GoF"), ".3f")),
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
    ]
    for label, key in (
        ("Zero shift refined", "refine_zero_shift"),
        ("Displacement refined", "refine_displacement"),
        ("Instrument profile refined", "refine_instrument_profile"),
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
    ("Strongest-line intensity", lambda p: _number(p.get("line_intensity"), ".6g")),
    ("Integrated intensity", lambda p: _number(p.get("integrated_intensity"), ".6g")),
    ("RIR (I/Ic)", lambda p: _number(p.get("rir"), ".3f")),
    ("Microstrain (×10⁻⁶)", lambda p: _number(p.get("microstrain"), ".1f")),
    ("Crystallite size (µm)", lambda p: _number(p.get("crystallite_size"), ".4g")),
    ("Lattice scale", lambda p: _number(p.get("lattice_scale"), ".6f")),
    ("Δlattice (%)", lambda p: _number(
        None if p.get("lattice_scale") is None
        else (float(p["lattice_scale"]) - 1.0) * 100.0, "+.4f")),
    ("a (Å)", lambda p: _number((p.get("unit_cell") or {}).get("a"), ".5f")),
    ("b (Å)", lambda p: _number((p.get("unit_cell") or {}).get("b"), ".5f")),
    ("c (Å)", lambda p: _number((p.get("unit_cell") or {}).get("c"), ".5f")),
    ("α (°)", lambda p: _number((p.get("unit_cell") or {}).get("alpha"), ".4f")),
    ("β (°)", lambda p: _number((p.get("unit_cell") or {}).get("beta"), ".4f")),
    ("γ (°)", lambda p: _number((p.get("unit_cell") or {}).get("gamma"), ".4f")),
    ("Volume (Å³)", lambda p: _number((p.get("unit_cell") or {}).get("volume"), ".3f")),
    ("Starting a (Å)", lambda p: _number((p.get("base_unit_cell") or {}).get("a"), ".5f")),
    ("Starting b (Å)", lambda p: _number((p.get("base_unit_cell") or {}).get("b"), ".5f")),
    ("Starting c (Å)", lambda p: _number((p.get("base_unit_cell") or {}).get("c"), ".5f")),
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
