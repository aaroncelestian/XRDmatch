"""
Resolve CIF files by AMCSD ID from the local CIF archive.

The database stores AMCSD IDs but not CIF text. The archive at data/cif.zip holds
files named ``<MineralName>__<amcsd_id>.cif``, with some entries also providing an
``<MineralName>__original__<amcsd_id>.cif`` variant (the unedited deposition).
This module indexes the archive once and serves CIF text on demand, so the
several hundred MB of uncompressed CIF text never has to enter the database.
"""

from __future__ import annotations

import os
import re
import threading
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from utils.conditions import parse_conditions

# Trailing digits before .cif are the AMCSD id, e.g. Abernathyite__0000130.cif
_ID_PATTERN = re.compile(r"(\d+)\.cif$", re.IGNORECASE)

AUTHOR_LOOP_TAG = "_publ_author_name"

# Single-value tags surfaced in the UI
_SCALAR_TAGS = (
    "_chemical_name_mineral",
    "_chemical_formula_sum",
    "_chemical_compound_source",
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
    "_cell_volume",
    "_exptl_crystal_density_diffrn",
    "_symmetry_space_group_name_H-M",
    "_space_group_name_H-M_alt",
    "_database_code_amcsd",
    "_journal_name_full",
    "_journal_year",
    "_journal_volume",
    "_journal_page_first",
    "_publ_section_title",
)


def normalize_amcsd_id(amcsd_id) -> str:
    """Return a 7-digit zero-padded AMCSD id string ('130' -> '0000130')."""
    if amcsd_id is None:
        return ""
    text = str(amcsd_id).strip()
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return digits.zfill(7)


def _strip_cif_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    # CIF marks unknown/inapplicable values with ? and .
    if value in ("?", "."):
        return ""
    return value


def parse_cif_metadata(cif_text: str) -> Dict[str, object]:
    """Extract the handful of fields the UI displays from raw CIF text.

    Uses a light line scanner rather than a full CIF parser: these AMCSD files
    are simple, and this avoids a pymatgen import on every table selection.
    """
    scalars: Dict[str, str] = {}
    authors: List[str] = []
    lines = cif_text.splitlines()

    # None outside a loop, "tags" while reading its header, "body" in its rows
    loop_state: Optional[str] = None
    loop_tags: List[str] = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        # Semicolon-delimited text block: skip its body
        if line.startswith(";"):
            while i < len(lines) and not lines[i].strip().startswith(";"):
                i += 1
            i += 1
            continue

        if line.lower().startswith("loop_"):
            loop_tags = []
            loop_state = "tags"
            continue

        if loop_state == "tags":
            if line.startswith("_"):
                loop_tags.append(line.split()[0])
                continue
            loop_state = "body"

        if loop_state == "body":
            if line.startswith("_") or line.startswith("data_"):
                # A tag after loop rows means the loop has ended
                loop_state = None
            else:
                # Author loops in AMCSD files list one quoted name per row
                if loop_tags == [AUTHOR_LOOP_TAG]:
                    name = _strip_cif_value(line)
                    if name:
                        authors.append(name)
                continue

        if line.startswith("_"):
            loop_state = None
            parts = line.split(None, 1)
            tag = parts[0]
            if len(parts) == 1:
                # Value lives on a following line (possibly a ; block)
                if i < len(lines) and lines[i].strip().startswith(";"):
                    body: List[str] = []
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith(";"):
                        body.append(lines[i].strip())
                        i += 1
                    i += 1
                    scalars[tag] = " ".join(body).strip()
                elif i < len(lines):
                    scalars[tag] = _strip_cif_value(lines[i])
                    i += 1
            else:
                scalars[tag] = _strip_cif_value(parts[1])

    def num(tag: str) -> Optional[float]:
        value = scalars.get(tag, "")
        if not value:
            return None
        # CIF may append an estimated standard deviation, e.g. 7.176(2)
        cleaned = re.sub(r"\(.*?\)", "", value).strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    space_group = (
        scalars.get("_symmetry_space_group_name_H-M")
        or scalars.get("_space_group_name_H-M_alt")
        or ""
    )

    title = scalars.get("_publ_section_title", "")
    conditions = parse_conditions(title)

    return {
        "mineral_name": scalars.get("_chemical_name_mineral", ""),
        "chemical_formula": scalars.get("_chemical_formula_sum", ""),
        "space_group": space_group,
        "locality": scalars.get("_chemical_compound_source", ""),
        "cell_a": num("_cell_length_a"),
        "cell_b": num("_cell_length_b"),
        "cell_c": num("_cell_length_c"),
        "cell_alpha": num("_cell_angle_alpha"),
        "cell_beta": num("_cell_angle_beta"),
        "cell_gamma": num("_cell_angle_gamma"),
        "cell_volume": num("_cell_volume"),
        "density": num("_exptl_crystal_density_diffrn"),
        "amcsd_id": scalars.get("_database_code_amcsd", ""),
        "authors": authors,
        "journal": scalars.get("_journal_name_full", ""),
        "journal_volume": scalars.get("_journal_volume", ""),
        "journal_page": scalars.get("_journal_page_first", ""),
        "year": scalars.get("_journal_year", ""),
        "title": title,
        "pressure_gpa": conditions.pressure_gpa,
        "temperature_k": conditions.temperature_k,
        "conditions": conditions.describe(),
        "is_ambient": conditions.is_ambient,
    }


def infer_crystal_system(a, b, c, alpha, beta, gamma, tol=1e-3) -> str:
    """Classify the lattice from its cell metric.

    Metric-based only, so a rhombohedral cell in hexagonal setting reports as
    hexagonal; callers should present this as inferred rather than authoritative.
    """
    if None in (a, b, c, alpha, beta, gamma):
        return ""

    def eq(x, y):
        return abs(x - y) <= tol

    ab, bc, ac = eq(a, b), eq(b, c), eq(a, c)
    all_90 = eq(alpha, 90) and eq(beta, 90) and eq(gamma, 90)

    if all_90:
        if ab and bc:
            return "Cubic"
        if ab or bc or ac:
            return "Tetragonal"
        return "Orthorhombic"

    if ab and eq(alpha, 90) and eq(beta, 90) and eq(gamma, 120):
        return "Hexagonal"

    if ab and bc and eq(alpha, beta) and eq(beta, gamma):
        return "Rhombohedral"

    if eq(alpha, 90) and eq(gamma, 90) and not eq(beta, 90):
        return "Monoclinic"
    if eq(alpha, 90) and eq(beta, 90) and not eq(gamma, 90):
        return "Monoclinic"

    return "Triclinic"


class CifRepository:
    """Index of the local CIF archive, keyed by AMCSD ID."""

    def __init__(self, zip_path: Optional[str] = None, cif_dir: Optional[str] = None):
        base = Path(__file__).resolve().parent.parent
        self.zip_path = Path(zip_path) if zip_path else base / "data" / "cif.zip"
        self.cif_dir = Path(cif_dir) if cif_dir else base / "data" / "cif"

        self._lock = threading.Lock()
        self._zip: Optional[zipfile.ZipFile] = None
        self._index: Optional[Dict[str, Dict[str, str]]] = None
        self._metadata_cache: Dict[str, Dict[str, object]] = {}

    # --- indexing ---

    def _build_index(self) -> Dict[str, Dict[str, str]]:
        index: Dict[str, Dict[str, str]] = {}

        def register(key: str, name: str, is_original: bool):
            entry = index.setdefault(key, {})
            entry["original" if is_original else "primary"] = name

        if self.zip_path.exists():
            try:
                self._zip = zipfile.ZipFile(self.zip_path)
                for name in self._zip.namelist():
                    if not name.lower().endswith(".cif"):
                        continue
                    match = _ID_PATTERN.search(name)
                    if not match:
                        continue
                    register(
                        normalize_amcsd_id(match.group(1)),
                        name,
                        "__original__" in name,
                    )
            except (zipfile.BadZipFile, OSError) as exc:
                print(f"CifRepository: could not read {self.zip_path}: {exc}")
                self._zip = None

        # Loose files on disk take precedence, letting users drop in corrections
        if self.cif_dir.is_dir():
            for path in self.cif_dir.glob("*.cif"):
                match = _ID_PATTERN.search(path.name)
                if match:
                    register(
                        normalize_amcsd_id(match.group(1)),
                        str(path),
                        "__original__" in path.name,
                    )

        return index

    def _ensure_index(self) -> Dict[str, Dict[str, str]]:
        with self._lock:
            if self._index is None:
                self._index = self._build_index()
            return self._index

    @property
    def available(self) -> bool:
        return bool(self._ensure_index())

    def count(self) -> int:
        return len(self._ensure_index())

    def has(self, amcsd_id) -> bool:
        return normalize_amcsd_id(amcsd_id) in self._ensure_index()

    def has_original(self, amcsd_id) -> bool:
        entry = self._ensure_index().get(normalize_amcsd_id(amcsd_id), {})
        return "original" in entry

    def source_name(self, amcsd_id, original: bool = False) -> Optional[str]:
        entry = self._ensure_index().get(normalize_amcsd_id(amcsd_id), {})
        name = entry.get("original" if original else "primary")
        return os.path.basename(name) if name else None

    # --- reading ---

    def get_cif_text(self, amcsd_id, original: bool = False) -> Optional[str]:
        """Return CIF text for an AMCSD id, or None when unavailable."""
        entry = self._ensure_index().get(normalize_amcsd_id(amcsd_id), {})
        name = entry.get("original" if original else "primary")
        if name is None and original:
            name = entry.get("primary")
        if not name:
            return None

        try:
            if os.path.isabs(name) and os.path.exists(name):
                with open(name, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()
            with self._lock:
                if self._zip is None:
                    return None
                return self._zip.read(name).decode("utf-8", errors="replace")
        except (KeyError, OSError, zipfile.BadZipFile) as exc:
            print(f"CifRepository: failed reading {name}: {exc}")
            return None

    def get_metadata(self, amcsd_id) -> Optional[Dict[str, object]]:
        """Parsed CIF fields for an AMCSD id, cached per id."""
        key = normalize_amcsd_id(amcsd_id)
        if not key:
            return None
        if key in self._metadata_cache:
            return self._metadata_cache[key]

        text = self.get_cif_text(key)
        if text is None:
            return None

        meta = parse_cif_metadata(text)
        meta["crystal_system"] = infer_crystal_system(
            meta.get("cell_a"), meta.get("cell_b"), meta.get("cell_c"),
            meta.get("cell_alpha"), meta.get("cell_beta"), meta.get("cell_gamma"),
        )
        self._metadata_cache[key] = meta
        return meta

    def close(self):
        with self._lock:
            if self._zip is not None:
                self._zip.close()
                self._zip = None


_shared_repository: Optional[CifRepository] = None


def get_cif_repository() -> CifRepository:
    """Shared repository instance (the archive index is built once per process)."""
    global _shared_repository
    if _shared_repository is None:
        _shared_repository = CifRepository()
    return _shared_repository
