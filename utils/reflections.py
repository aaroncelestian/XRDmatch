"""
Reflection lists generated from a phase's CIF.

The stored reference patterns carry only 2-theta, d-spacing and intensity. Le
Bail refinement of anything direction-dependent -- an anisotropic cell, or the
Stephens microstrain model -- needs a Miller index per reflection, so those
reflections are regenerated here from the deposited structure.

Falling back to the stored pattern is always allowed; a phase without a CIF
simply cannot use the direction-dependent models.
"""

from __future__ import annotations

import io
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.cif_repository import get_cif_repository, normalize_amcsd_id

# Space group number -> (crystal system, Laue class). The Laue class is what
# actually constrains the cell and the strain tensor; the crystal system alone
# does not distinguish 4/m from 4/mmm.
_LAUE_RANGES: Sequence[Tuple[int, int, str, str]] = (
    (1, 2, 'triclinic', '-1'),
    (3, 15, 'monoclinic', '2/m'),
    (16, 74, 'orthorhombic', 'mmm'),
    (75, 88, 'tetragonal', '4/m'),
    (89, 142, 'tetragonal', '4/mmm'),
    (143, 148, 'trigonal', '-3'),
    (149, 167, 'trigonal', '-3m'),
    (168, 176, 'hexagonal', '6/m'),
    (177, 194, 'hexagonal', '6/mmm'),
    (195, 206, 'cubic', 'm-3'),
    (207, 230, 'cubic', 'm-3m'),
)

# Regenerating a reflection list costs a CIF parse plus a structure-factor
# calculation, roughly half a second, and the same phase is asked for on every
# refinement. Keyed by structure, wavelength and angular range.
_CACHE: Dict[Tuple, Optional[Dict]] = {}
_CACHE_LIMIT = 64


def laue_class(space_group_number: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    """(crystal system, Laue class) for a space group number."""
    if not space_group_number:
        return None, None
    for low, high, system, laue in _LAUE_RANGES:
        if low <= int(space_group_number) <= high:
            return system, laue
    return None, None


def _resolve_amcsd_id(phase: Dict, database=None) -> Optional[str]:
    """Find the AMCSD id for a phase, looking it up by row id if need be."""
    if not isinstance(phase, dict):
        return None
    inner = phase.get('phase') if isinstance(phase.get('phase'), dict) else phase

    for source in (inner, phase):
        value = source.get('amcsd_id')
        if value:
            return normalize_amcsd_id(value)

    if database is None:
        return None
    for key in ('mineral_id', 'id'):
        for source in (inner, phase):
            row_id = source.get(key)
            if row_id in (None, ''):
                continue
            try:
                record = database.get_mineral_by_id(int(row_id))
            except (TypeError, ValueError, AttributeError):
                continue
            if record and record.get('amcsd_id'):
                return normalize_amcsd_id(record['amcsd_id'])
    return None


def _to_three_index(hkl: Sequence[int]) -> Tuple[int, int, int]:
    """
    Collapse Bravais-Miller (hkil) to (hkl).

    pymatgen reports four indices for trigonal and hexagonal lattices, where
    i = -(h + k) carries no independent information.
    """
    values = [int(round(v)) for v in hkl]
    if len(values) == 4:
        return values[0], values[1], values[3]
    return values[0], values[1], values[2]


def _trim_cache():
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))


def reflections_from_cif(cif_text: str, wavelength: float,
                         two_theta_range: Tuple[float, float] = (5.0, 90.0),
                         quiet: bool = True) -> Optional[Dict]:
    """
    Indexed reflections and structure-factor intensities from CIF text.

    Returns two_theta, intensity, d_spacing, hkl, multiplicity, together with
    the cell and symmetry needed by the direction-dependent refinement models.
    """
    if not cif_text:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from pymatgen.io.cif import CifParser
            from pymatgen.analysis.diffraction.xrd import XRDCalculator

            parser = CifParser(io.StringIO(cif_text))
            # The conventional cell is what the published Miller indices refer to
            structures = parser.parse_structures(primitive=False)
            if not structures:
                return None
            structure = structures[0]

            pattern = XRDCalculator(wavelength=float(wavelength)).get_pattern(
                structure, two_theta_range=tuple(two_theta_range)
            )
            try:
                sg_symbol, sg_number = structure.get_space_group_info()
            except Exception:
                sg_symbol, sg_number = None, None
    except Exception as e:
        if not quiet:
            print(f"Reflection generation failed: {e}")
        return None

    two_theta: List[float] = []
    intensity: List[float] = []
    d_spacing: List[float] = []
    indices: List[Tuple[int, int, int]] = []
    multiplicity: List[int] = []

    for angle, height, families, d in zip(
        pattern.x, pattern.y, pattern.hkls, pattern.d_hkls
    ):
        if not families:
            continue
        # pymatgen groups reflections within 1e-5 degrees, which merges the
        # symmetry-equivalent set into one entry (wanted) but also merges
        # accidental coincidences such as cubic 333/511 (not wanted, since the
        # two families broaden differently under an anisotropic strain model).
        # Splitting by multiplicity is exact for the former and approximate for
        # the latter; Le Bail re-partitions these intensities regardless.
        weights = np.array(
            [float(f.get('multiplicity', 1) or 1) for f in families], dtype=float
        )
        total = weights.sum()
        shares = weights / total if total > 0 else np.full(len(families), 1.0 / len(families))
        for family, share in zip(families, shares):
            two_theta.append(float(angle))
            intensity.append(float(height) * float(share))
            d_spacing.append(float(d))
            indices.append(_to_three_index(family['hkl']))
            multiplicity.append(int(family.get('multiplicity', 1) or 1))

    if not two_theta:
        return None

    lattice = structure.lattice
    system, laue = laue_class(sg_number)

    return {
        'two_theta': np.asarray(two_theta, dtype=float),
        'intensity': np.asarray(intensity, dtype=float),
        'd_spacing': np.asarray(d_spacing, dtype=float),
        'hkl': np.asarray(indices, dtype=int),
        'multiplicity': np.asarray(multiplicity, dtype=int),
        'unit_cell': {
            'a': float(lattice.a), 'b': float(lattice.b), 'c': float(lattice.c),
            'alpha': float(lattice.alpha), 'beta': float(lattice.beta),
            'gamma': float(lattice.gamma), 'volume': float(lattice.volume),
        },
        'space_group': sg_symbol,
        'space_group_number': int(sg_number) if sg_number else None,
        'crystal_system': system,
        'laue_class': laue,
        'wavelength': float(wavelength),
        'source': 'cif',
    }


def reflections_for_phase(phase: Dict, wavelength: float,
                          two_theta_range: Tuple[float, float] = (5.0, 90.0),
                          database=None, quiet: bool = True) -> Optional[Dict]:
    """
    Indexed reflections for a matched phase, or None when no CIF is available.

    Callers fall back to the phase's stored reference pattern, which supports
    every isotropic model but none of the direction-dependent ones.
    """
    amcsd_id = _resolve_amcsd_id(phase, database)
    if not amcsd_id:
        return None

    key = (amcsd_id, round(float(wavelength), 6),
           round(float(two_theta_range[0]), 3), round(float(two_theta_range[1]), 3))
    if key in _CACHE:
        return _CACHE[key]

    repository = get_cif_repository()
    cif_text = repository.get_cif_text(amcsd_id) if repository.available else None
    result = reflections_from_cif(cif_text, wavelength, two_theta_range, quiet=quiet)
    if result is not None:
        result['amcsd_id'] = amcsd_id

    _CACHE[key] = result
    _trim_cache()
    return result


def clear_cache():
    """Drop cached reflection lists; used when the CIF source changes."""
    _CACHE.clear()
