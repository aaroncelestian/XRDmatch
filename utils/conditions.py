"""Recover the pressure/temperature conditions an AMCSD structure was measured at.

None of the AMCSD CIFs carry the standard ``_diffrn_ambient_pressure`` or
``_diffrn_ambient_temperature`` tags, so the only record of non-ambient
conditions is a convention AMCSD appends to ``_publ_section_title``, e.g.
``"... compressibility of coesite P = 21.8 kbar"`` or ``"... at T = 1273 K"``.
Roughly a quarter of the archive is such high-pressure or high-temperature work,
which matters for phase matching: a compressed or heated cell has shifted
lattice parameters, so its lines sit at shifted 2theta and both create false
matches and crowd correct phases out of the rankings.

Two details this module exists to get right:

* Units are inconsistent (GPa, kbar, the truncated ``kb``, kPa, atm, bar, MPa,
  mbar). Reading ``P = 31 kb`` as GPa would be a 10x error. The ``mbar`` entries
  are millibar water-vapour pressures from zeolite dehydration studies, not
  megabar.
* Only explicit numeric annotations are trusted. Several hundred titles mention
  "high pressure" in prose while describing samples that merely *formed* at
  depth -- mantle xenoliths, "chondrodite of high-pressure origin" -- and were
  measured at ambient. Filtering on prose would discard good data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# An ambient measurement, generously bounded: room-temperature work spans a
# fair range of reported values, and pressures below this are vacuum, vapour
# pressure, or 1 atm expressed in assorted units.
AMBIENT_MAX_PRESSURE_GPA = 0.1
AMBIENT_MIN_TEMPERATURE_K = 250.0
AMBIENT_MAX_TEMPERATURE_K = 350.0

_ABSOLUTE_ZERO_OFFSET = 273.15

_PRESSURE_TO_GPA = {
    "gpa": 1.0,
    "kbar": 0.1,
    "kb": 0.1,
    "mpa": 1e-3,
    "kpa": 1e-6,
    "mbar": 1e-7,
    "bar": 1e-4,
    "pa": 1e-9,
    "atm": 1.01325e-4,
}

# Longest unit first so "kbar" wins over "bar" and "kpa"/"mpa" over "pa"
_PRESSURE_UNITS = "|".join(
    sorted((re.escape(u) for u in _PRESSURE_TO_GPA), key=len, reverse=True)
)
_PRESSURE_RE = re.compile(
    rf"\bP\s*=\s*(-?\d+(?:\.\d+)?)\s*({_PRESSURE_UNITS})?", re.IGNORECASE
)

_CELSIUS_UNITS = ("c", "deg", "degc", "degs", "degree", "degrees", "degreec")
_TEMPERATURE_RE = re.compile(
    r"\bT\s*=\s*(-?\d+(?:\.\d+)?)\s*"
    r"(K\b|deg(?:ree)?s?\s*C\b|deg(?:ree)?s?\b|C\b)?",
    re.IGNORECASE,
)

# Below this a bare number is more plausibly Celsius than Kelvin: AMCSD reports
# room temperature as both "T = 25 C" and "T = 298", never "T = 25 K".
_BARE_KELVIN_FLOOR = 200.0


@dataclass(frozen=True)
class Conditions:
    """Measurement conditions recovered from a publication title."""

    pressure_gpa: Optional[float] = None
    temperature_k: Optional[float] = None

    @property
    def is_ambient(self) -> bool:
        return is_ambient(self.pressure_gpa, self.temperature_k)

    @property
    def is_high_pressure(self) -> bool:
        return (
            self.pressure_gpa is not None
            and self.pressure_gpa > AMBIENT_MAX_PRESSURE_GPA
        )

    @property
    def is_non_ambient_temperature(self) -> bool:
        if self.temperature_k is None:
            return False
        return not (
            AMBIENT_MIN_TEMPERATURE_K
            <= self.temperature_k
            <= AMBIENT_MAX_TEMPERATURE_K
        )

    def describe(self) -> str:
        """Short human-readable summary, empty when nothing was annotated."""
        parts = []
        if self.pressure_gpa is not None:
            # Vacuum, vapour pressure and 1 atm all read as "ambient" to a user
            if self.pressure_gpa > AMBIENT_MAX_PRESSURE_GPA:
                parts.append(f"{self.pressure_gpa:.2f} GPa")
            else:
                parts.append("ambient P")
        if self.temperature_k is not None:
            parts.append(f"{self.temperature_k:.0f} K")
        return ", ".join(parts)


def is_ambient(
    pressure_gpa: Optional[float],
    temperature_k: Optional[float],
    max_pressure_gpa: float = AMBIENT_MAX_PRESSURE_GPA,
    min_temperature_k: float = AMBIENT_MIN_TEMPERATURE_K,
    max_temperature_k: float = AMBIENT_MAX_TEMPERATURE_K,
) -> bool:
    """True when nothing indicates the structure was measured off-ambient.

    Unannotated entries count as ambient: AMCSD only records conditions when
    they are noteworthy, so absence of an annotation is the common case.
    """
    if pressure_gpa is not None and pressure_gpa > max_pressure_gpa:
        return False
    if temperature_k is not None and not (
        min_temperature_k <= temperature_k <= max_temperature_k
    ):
        return False
    return True


def ambient_sql_filter(alias: str = "") -> Tuple[str, List[float]]:
    """SQL predicate and parameters restricting a query to ambient entries.

    Mirrors :func:`is_ambient`, including its treatment of NULL (unannotated)
    conditions as ambient, so browsing and matching agree on what is excluded.
    """
    prefix = f"{alias}." if alias else ""
    clause = (
        f"({prefix}pressure_gpa IS NULL OR {prefix}pressure_gpa <= ?) "
        f"AND ({prefix}temperature_k IS NULL "
        f"OR {prefix}temperature_k BETWEEN ? AND ?)"
    )
    params = [
        AMBIENT_MAX_PRESSURE_GPA,
        AMBIENT_MIN_TEMPERATURE_K,
        AMBIENT_MAX_TEMPERATURE_K,
    ]
    return clause, params


def _parse_pressure(text: str) -> Optional[float]:
    match = _PRESSURE_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = (match.group(2) or "").lower()
    if unit:
        return value * _PRESSURE_TO_GPA[unit]
    # No unit given. Zero is unambiguous; otherwise assume GPa, the most common
    # unit by an order of magnitude. Any misread unit is still far above the
    # ambient cutoff, so the ambient/non-ambient verdict is unaffected.
    return 0.0 if value == 0 else value


def _parse_temperature(text: str) -> Optional[float]:
    match = _TEMPERATURE_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = re.sub(r"[\s.]", "", (match.group(2) or "")).lower()
    if unit == "k":
        return value
    if unit in _CELSIUS_UNITS or unit.startswith("deg"):
        return value + _ABSOLUTE_ZERO_OFFSET
    return value if value > _BARE_KELVIN_FLOOR else value + _ABSOLUTE_ZERO_OFFSET


def parse_conditions(title: Optional[str]) -> Conditions:
    """Pull ``P =`` / ``T =`` annotations out of a publication title.

    The first annotation of each kind wins, which is what AMCSD's own ordering
    implies: a title like "study at temperatures up to 1273 K ... T = 293 K"
    describes the 293 K member of that series.
    """
    if not title:
        return Conditions()
    return Conditions(
        pressure_gpa=_parse_pressure(title),
        temperature_k=_parse_temperature(title),
    )
