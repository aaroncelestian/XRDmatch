"""
A user-curated list of minerals that outlives the pattern being analysed.

Search results are rebuilt from scratch on every run and the session drops
everything the moment another file is loaded, so the phases a user has decided
matter have nowhere to live. This store keeps them on disk: the shortlist
itself, which minerals in it are currently checked, and any named combinations
worth coming back to — so several mixtures can be quantified against the same
pattern, and the same suite of minerals can follow the user from one pattern to
the next.

Only identity is stored. Cell parameters, RIR values, and reference lines are
read back from the local database when the list is loaded, because those belong
to the database and change when it is rebuilt. The AMCSD id is what ties a
record to a mineral: the local row id is an autoincrement artefact of import
order and will not survive a rebuild.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from utils.cif_repository import normalize_amcsd_id


STORE_VERSION = 1

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_STORE_PATH = os.path.join(_DATA_DIR, "phase_shortlist.json")


def entry_key(entry: Dict) -> str:
    """
    Stable identity for one shortlisted mineral.

    The AMCSD id when there is one, since it survives a database rebuild;
    otherwise the mineral name, which is all a hand-added phase may have.
    """
    if not isinstance(entry, dict):
        return str(entry).strip().lower()
    amcsd = normalize_amcsd_id(entry.get("amcsd_id"))
    if amcsd:
        return f"amcsd:{amcsd}"
    name = entry.get("mineral_name") or entry.get("mineral") or ""
    return f"name:{str(name).strip().lower()}"


def make_entry(amcsd_id=None, mineral_name: str = "", mineral_id=None,
               checked: bool = True) -> Dict:
    """One shortlist record, in the shape that gets written to disk."""
    return {
        "amcsd_id": normalize_amcsd_id(amcsd_id),
        "mineral_name": str(mineral_name or "").strip(),
        # A hint only: re-resolved against the database on load, since the
        # local row id moves when the database is rebuilt
        "mineral_id": mineral_id,
        "checked": bool(checked),
    }


class PhaseShortlist:
    """The shortlist and its named sets, persisted as one small JSON file."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_STORE_PATH
        self._entries: List[Dict] = []
        self._sets: Dict[str, List[Dict]] = {}
        self._listeners: List[Callable[[], None]] = []
        self.load()

    # --- persistence ---

    def load(self) -> None:
        self._entries = []
        self._sets = {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return

        for record in data.get("phases", []):
            if not isinstance(record, dict):
                continue
            entry = make_entry(
                amcsd_id=record.get("amcsd_id"),
                mineral_name=record.get("mineral_name", ""),
                mineral_id=record.get("mineral_id"),
                checked=record.get("checked", True),
            )
            if entry_key(entry) != "name:" and not self.contains(entry_key(entry)):
                self._entries.append(entry)

        for name, members in (data.get("sets") or {}).items():
            if not isinstance(members, list):
                continue
            self._sets[str(name)] = [
                make_entry(
                    amcsd_id=m.get("amcsd_id"),
                    mineral_name=m.get("mineral_name", ""),
                    mineral_id=m.get("mineral_id"),
                )
                for m in members if isinstance(m, dict)
            ]

    def save(self) -> None:
        payload = {
            "version": STORE_VERSION,
            "phases": self._entries,
            "sets": self._sets,
        }
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # Write beside the target first so a crash cannot truncate a list
            # the user has been building up over many sessions
            temp = f"{self.path}.tmp"
            with open(temp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(temp, self.path)
        except OSError as exc:
            print(f"Could not save the phase shortlist: {exc}")

    # --- listeners ---

    def add_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _changed(self) -> None:
        self.save()
        for callback in list(self._listeners):
            try:
                callback()
            except Exception as exc:
                print(f"Phase shortlist listener failed: {exc}")

    # --- the list ---

    def entries(self) -> List[Dict]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def contains(self, key: str) -> bool:
        return any(entry_key(e) == key for e in self._entries)

    def find(self, key: str) -> Optional[Dict]:
        for entry in self._entries:
            if entry_key(entry) == key:
                return entry
        return None

    def add(self, amcsd_id=None, mineral_name: str = "", mineral_id=None,
            checked: bool = True) -> bool:
        """Add a mineral. False when it was already there."""
        entry = make_entry(amcsd_id, mineral_name, mineral_id, checked)
        if entry_key(entry) == "name:":
            return False
        existing = self.find(entry_key(entry))
        if existing is not None:
            # Keep the record fresh, but never silently flip the user's check
            if mineral_id is not None:
                existing["mineral_id"] = mineral_id
            if entry["mineral_name"]:
                existing["mineral_name"] = entry["mineral_name"]
            self._changed()
            return False
        self._entries.append(entry)
        self._changed()
        return True

    def remove(self, key: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if entry_key(e) != key]
        if len(self._entries) != before:
            self._changed()
            return True
        return False

    def clear(self) -> None:
        if self._entries:
            self._entries = []
            self._changed()

    def set_checked(self, key: str, checked: bool) -> None:
        entry = self.find(key)
        if entry is not None and entry["checked"] != bool(checked):
            entry["checked"] = bool(checked)
            self._changed()

    def set_all_checked(self, checked: bool) -> None:
        changed = False
        for entry in self._entries:
            if entry["checked"] != bool(checked):
                entry["checked"] = bool(checked)
                changed = True
        if changed:
            self._changed()

    def checked(self) -> List[Dict]:
        return [e for e in self._entries if e.get("checked")]

    # --- named sets ---

    def set_names(self) -> List[str]:
        return sorted(self._sets.keys())

    def save_set(self, name: str) -> int:
        """
        Store the current check marks under a name.

        The members are kept as full records rather than keys so that applying
        the set later can put minerals back that have since been removed from
        the shortlist.
        """
        name = str(name).strip()
        if not name:
            return 0
        self._sets[name] = [dict(e, checked=True) for e in self.checked()]
        self._changed()
        return len(self._sets[name])

    def apply_set(self, name: str) -> int:
        """
        Check exactly the minerals in a set, adding any that are missing.

        Returns how many are checked afterwards.
        """
        members = self._sets.get(str(name))
        if members is None:
            return 0
        wanted = {entry_key(m) for m in members}
        for member in members:
            if not self.contains(entry_key(member)):
                self._entries.append(make_entry(
                    member.get("amcsd_id"), member.get("mineral_name"),
                    member.get("mineral_id"), checked=False,
                ))
        for entry in self._entries:
            entry["checked"] = entry_key(entry) in wanted
        self._changed()
        return len(wanted)

    def delete_set(self, name: str) -> bool:
        if str(name) in self._sets:
            del self._sets[str(name)]
            self._changed()
            return True
        return False


_shortlist: Optional[PhaseShortlist] = None


def get_shortlist() -> PhaseShortlist:
    """The one shortlist shared by the whole application."""
    global _shortlist
    if _shortlist is None:
        _shortlist = PhaseShortlist()
    return _shortlist
