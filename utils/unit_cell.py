"""
Unit cell geometry: d-spacings, reflection indexing, and what may refine freely.

Refining a, b, c and the angles separately needs a Miller index for every
reflection, because the index is what says how that peak responds to each cell
parameter: 200 answers to a alone, 002 to c alone, and hk0 to both. The stored
reference patterns carry d-spacings but no indices, so they are recovered here by
asking which reflection of the starting cell has that d-spacing. The starting
cell is the one the pattern was calculated from, so where a match exists it is
exact to the rounding in the table; a pattern that will not index is left to an
isotropic dilation instead.

Which parameters may move is read from the starting cell rather than from a space
group symbol. A symmetry equality is a statement about the metric -- hexagonal
means a = b and gamma = 120 -- and the metric is what a diffraction pattern can
speak to. Refining a and b apart in a hexagonal phase would be fitting a
direction the data cannot distinguish, so equal edges move together and a right
angle stays a right angle.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

EDGE_KEYS: Tuple[str, ...] = ('a', 'b', 'c')
ANGLE_KEYS: Tuple[str, ...] = ('alpha', 'beta', 'gamma')
CELL_KEYS: Tuple[str, ...] = EDGE_KEYS + ANGLE_KEYS

# How closely two cell parameters must agree to be treated as one. A real cell
# states a symmetry equality exactly, so these only have to absorb the rounding
# of a four-decimal table, not a physical difference.
_EDGE_TOL = 1e-5      # relative
_ANGLE_TOL = 1e-2     # degrees

# A d-spacing has to land this close to a calculated one, relatively, to be
# called that reflection. The reference d-spacings were computed from this very
# cell, so the window is set by how many digits the table carries.
INDEX_TOLERANCE = 3e-3

# Enumerating candidate reflections is cheap but not free, and a nonsensical
# cell or d-spacing can ask for an unbounded number of them.
_MAX_CANDIDATES = 2_000_000


class CellGroup(NamedTuple):
    """One free cell parameter, and the parameters that follow it."""

    key: str
    tied: Tuple[str, ...]
    kind: str  # 'edge' or 'angle'

    @property
    def name(self) -> str:
        return f'cell_{self.key}'


def is_cell_parameter(name: str) -> bool:
    return name.startswith('cell_') and name[5:] in CELL_KEYS


# --- geometry ---------------------------------------------------------------

def cell_volume(cell: Dict) -> float:
    """Triclinic cell volume, valid for every crystal system."""
    try:
        a, b, c = float(cell['a']), float(cell['b']), float(cell['c'])
        alpha, beta, gamma = (np.radians(float(cell.get(key, 90.0)))
                             for key in ANGLE_KEYS)
    except (KeyError, TypeError, ValueError):
        return 0.0
    term = (
        1.0
        - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2
        + 2.0 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )
    return float(a * b * c * np.sqrt(max(term, 0.0)))


def metric_tensor(cell: Dict) -> Optional[np.ndarray]:
    """The real-space metric G, whose inverse turns hkl into 1/d²."""
    try:
        a, b, c = float(cell['a']), float(cell['b']), float(cell['c'])
        alpha, beta, gamma = (np.radians(float(cell.get(key, 90.0)))
                             for key in ANGLE_KEYS)
    except (KeyError, TypeError, ValueError):
        return None
    if not (a > 0 and b > 0 and c > 0):
        return None
    return np.array([
        [a * a, a * b * np.cos(gamma), a * c * np.cos(beta)],
        [a * b * np.cos(gamma), b * b, b * c * np.cos(alpha)],
        [a * c * np.cos(beta), b * c * np.cos(alpha), c * c],
    ], dtype=float)


def inv_d_squared(hkl: np.ndarray, cell: Dict) -> Optional[np.ndarray]:
    """1/d² for each reflection, from the reciprocal metric tensor."""
    metric = metric_tensor(cell)
    if metric is None:
        return None
    try:
        reciprocal = np.linalg.inv(metric)
    except np.linalg.LinAlgError:
        return None
    indices = np.atleast_2d(np.asarray(hkl, dtype=float))
    values = np.einsum('ij,jk,ik->i', indices, reciprocal, indices)
    return values


def d_spacings(hkl: np.ndarray, cell: Dict) -> Optional[np.ndarray]:
    """d for each reflection of a cell, in the same order as hkl."""
    values = inv_d_squared(hkl, cell)
    if values is None:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(values > 0, 1.0 / np.sqrt(np.abs(values)), np.inf)


def d_spacing_ratio(hkl: np.ndarray, cell: Dict, base: Dict) -> Optional[np.ndarray]:
    """
    How far each reflection's d-spacing moves between two cells.

    This is what turns a cell into peak positions without recalculating the
    pattern: sin(theta) scales by the reciprocal of the ratio, so the reference
    positions can be moved where the refined cell puts them while keeping
    whatever else those positions already carried.
    """
    q_new = inv_d_squared(hkl, cell)
    q_base = inv_d_squared(hkl, base)
    if q_new is None or q_base is None:
        return None
    ratio = np.ones(len(q_new), dtype=float)
    good = (q_new > 0) & (q_base > 0)
    ratio[good] = np.sqrt(q_base[good] / q_new[good])
    return ratio


# --- indexing ---------------------------------------------------------------

def _candidate_indices(cell: Dict, d_min: float, d_max: float
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Every reflection of a cell in a d-spacing window, with its d."""
    if not (d_min > 0) or d_max < d_min:
        return None
    # h is the projection of the reciprocal vector on a, so |h| <= a/d exactly,
    # and likewise for k and l. The bound is therefore not an estimate.
    limits = []
    for key in EDGE_KEYS:
        try:
            edge = float(cell[key])
        except (KeyError, TypeError, ValueError):
            return None
        limits.append(int(np.floor(edge / d_min)) + 1)
    if any(limit < 1 for limit in limits):
        return None
    if np.prod([2 * limit + 1 for limit in limits], dtype=float) > _MAX_CANDIDATES:
        return None

    grids = np.meshgrid(*[np.arange(-limit, limit + 1) for limit in limits],
                        indexing='ij')
    hkl = np.stack([grid.ravel() for grid in grids], axis=1)

    # hkl and -h-k-l have the same d-spacing, so keep one of each pair. This
    # halves the search and removes a whole class of ties.
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    half = (h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (l > 0))
    hkl = hkl[half]

    d = d_spacings(hkl, cell)
    if d is None:
        return None
    keep = (d >= d_min * 0.98) & (d <= d_max * 1.02)
    return hkl[keep], d[keep]


def index_reflections(observed_d: Sequence[float], cell: Dict,
                      tolerance: float = INDEX_TOLERANCE
                      ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    A Miller index for each observed d-spacing, from the cell that produced it.

    Returns the indices and a mask of which ones were identified. An unmatched
    reflection is left at 000 and flagged, so the caller can decide whether
    enough of the pattern indexed to be worth refining anisotropically.
    """
    observed = np.asarray(observed_d, dtype=float)
    matched = np.zeros(len(observed), dtype=bool)
    hkl = np.zeros((len(observed), 3), dtype=int)
    usable = np.isfinite(observed) & (observed > 0)
    if not usable.any():
        return None, matched

    candidates = _candidate_indices(
        cell, float(np.min(observed[usable])), float(np.max(observed[usable]))
    )
    if candidates is None:
        return None, matched
    cand_hkl, cand_d = candidates
    if not len(cand_d):
        return None, matched

    # Sorting by d and then by index sum means the first candidate reached at a
    # given d-spacing is the simplest one, which is the right choice when two
    # unrelated reflections coincide.
    order = np.lexsort((np.abs(cand_hkl).sum(axis=1), cand_d))
    cand_hkl, cand_d = cand_hkl[order], cand_d[order]

    sources = np.flatnonzero(usable)
    positions = np.searchsorted(cand_d, observed[sources])
    for source, index in zip(sources, positions):
        target = observed[source]
        low = max(0, index - 4)
        gaps = np.abs(cand_d[low:index + 5] - target)
        if not len(gaps):
            continue
        best = int(np.argmin(gaps))
        if gaps[best] <= tolerance * target:
            hkl[source] = cand_hkl[low + best]
            matched[source] = True

    return hkl, matched


# --- what may refine --------------------------------------------------------

def _edge_groups(cell: Dict) -> List[CellGroup]:
    """Equal edges move together; everything else moves on its own."""
    groups: List[CellGroup] = []
    for key in EDGE_KEYS:
        value = float(cell[key])
        for index, group in enumerate(groups):
            reference = float(cell[group.key])
            if reference > 0 and abs(value - reference) <= _EDGE_TOL * reference:
                groups[index] = group._replace(tied=group.tied + (key,))
                break
        else:
            groups.append(CellGroup(key, (key,), 'edge'))
    return groups


def _angle_groups(cell: Dict) -> List[CellGroup]:
    """
    Angles fixed by symmetry stay fixed; equal ones move together.

    A right angle in a real cell is always a consequence of symmetry, so it is
    held rather than refined. So is a gamma of 120 alongside a = b, which is the
    hexagonal setting; 120 anywhere else is just a number and stays free.
    """
    hexagonal = (
        abs(float(cell['a']) - float(cell['b'])) <= _EDGE_TOL * float(cell['a'])
        and abs(float(cell['gamma']) - 120.0) <= _ANGLE_TOL
    )
    groups: List[CellGroup] = []
    for key in ANGLE_KEYS:
        value = float(cell[key])
        if abs(value - 90.0) <= _ANGLE_TOL:
            continue
        if key == 'gamma' and hexagonal:
            continue
        for index, group in enumerate(groups):
            if abs(value - float(cell[group.key])) <= _ANGLE_TOL:
                groups[index] = group._replace(tied=group.tied + (key,))
                break
        else:
            groups.append(CellGroup(key, (key,), 'angle'))
    return groups


def cell_groups(cell: Dict, reflections: Optional[int] = None,
                per_parameter: int = 2) -> Tuple[CellGroup, ...]:
    """
    The cell parameters a pattern of this cell can be refined on.

    With too few indexed reflections to go round, the angles are given up first
    and then the edges are dilated as one: an under-determined cell refinement
    does not fail loudly, it wanders, and a phase whose axial ratios have gone
    somewhere the data never asked for is worse than one that only breathed.
    """
    try:
        groups = _edge_groups(cell) + _angle_groups(cell)
    except (KeyError, TypeError, ValueError):
        return ()

    if reflections is None:
        return tuple(groups)

    budget = max(1, int(reflections) // max(1, per_parameter))
    while len(groups) > budget:
        angles = [index for index, group in enumerate(groups)
                  if group.kind == 'angle']
        if angles:
            groups.pop(angles[-1])
            continue
        edges = [group for group in groups if group.kind == 'edge']
        if len(edges) <= 1:
            break
        tied = tuple(key for group in edges for key in group.tied)
        groups = [CellGroup(edges[0].key, tied, 'edge')]
    return tuple(groups)


def free_cell_values(cell: Dict, groups: Sequence[CellGroup]) -> Dict[str, float]:
    """The current value of each free parameter, keyed by parameter name."""
    return {group.name: float(cell[group.key]) for group in groups
            if cell.get(group.key) is not None}


def cell_bounds(group: CellGroup, value: float,
                edge_span: float = 0.05, angle_span: float = 2.0
                ) -> Tuple[float, float]:
    """
    How far a cell parameter may travel.

    A few percent covers thermal expansion, hydrostatic strain and solid
    solution; beyond that the refinement has stopped describing this phase and
    has started indexing someone else's peaks.
    """
    if group.kind == 'edge':
        return value * (1.0 - edge_span), value * (1.0 + edge_span)
    return max(1.0, value - angle_span), min(179.0, value + angle_span)


def cell_from_free(base: Dict, values: Dict[str, float],
                   groups: Sequence[CellGroup]) -> Dict:
    """
    Rebuild a cell from the free parameters, honouring the ties.

    Edges tied together keep their ratio to the starting cell rather than their
    difference, so one free parameter dilates a whole group. That makes a single
    edge group exactly an isotropic dilation, which is what a pattern with too
    few reflections falls back to.
    """
    cell = dict(base)
    for group in groups:
        value = values.get(group.name)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if group.kind == 'edge':
            reference = float(base.get(group.key) or 0.0)
            if reference <= 0:
                continue
            factor = value / reference
            for key in group.tied:
                cell[key] = float(base[key]) * factor
        else:
            for key in group.tied:
                cell[key] = value
    cell['volume'] = cell_volume(cell)
    return cell


def cell_deltas(cell: Optional[Dict], base: Optional[Dict]) -> Dict[str, float]:
    """
    Per-parameter change from the starting cell: percent for edges and volume,
    degrees for angles.
    """
    if not cell or not base:
        return {}
    deltas: Dict[str, float] = {}
    for key in EDGE_KEYS + ('volume',):
        try:
            start, now = float(base[key]), float(cell[key])
        except (KeyError, TypeError, ValueError):
            continue
        if start > 0:
            deltas[key] = (now / start - 1.0) * 100.0
    for key in ANGLE_KEYS:
        try:
            deltas[key] = float(cell[key]) - float(base[key])
        except (KeyError, TypeError, ValueError):
            continue
    return deltas
