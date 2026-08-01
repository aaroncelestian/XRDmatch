"""
The Shortlist tab: minerals the user keeps, independent of any one pattern.

A search list is disposable — it is rebuilt on every run and thrown away when
another file is loaded. The shortlist is the opposite: the handful of minerals
the user has decided are worth carrying, with check marks that say which of
them make up the mixture being quantified right now. Checking a different
combination and running RIR again is the whole point, so nothing here re-runs a
search or touches the candidate list.

Only identity is persisted (see `utils.phase_shortlist`). Everything the
analysis needs — cell, RIR, reference lines — is rebuilt from the local
database against the current wavelength each time a pattern is loaded, so a
shortlisted mineral arrives at a new pattern with no stale fit attached to it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from utils.phase_shortlist import entry_key, get_shortlist


SELECT_COL = 0
NAME_COL = 1
SPACE_GROUP_COL = 2
RIR_COL = 3


class ShortlistPanel(QWidget):
    """Table of kept minerals with check marks, named sets, and quant actions."""

    selection_changed = pyqtSignal()

    def __init__(self, session, workspace, parent=None):
        super().__init__(parent)
        self.session = session
        self.workspace = workspace
        self.store = get_shortlist()

        # Analysis entries keyed by shortlist key, rebuilt whenever the pattern
        # changes so no phase carries a shift fitted against different data
        self._entry_cache: Dict[str, Optional[Dict]] = {}
        # AMCSD id and database name per local row id. Identity is asked for
        # once per visible row every time a result list is built, and the
        # answer only moves when the database does.
        self._identity_cache: Dict[str, tuple] = {}
        self._loading = False
        # A check mark must not rebuild the table: the checkbox that emitted the
        # signal would be destroyed while it is still delivering it
        self._suppress_reload = False

        self._build_ui()
        self.store.add_listener(self.reload)
        self.session.pattern_changed.connect(self._on_pattern_changed)
        self.reload()

    # --- UI ---

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.header = QLabel()
        self.header.setObjectName("mutedLabel")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["✓", "Mineral", "Space group", "RIR"])
        self.table.itemDoubleClicked.connect(self._toggle_current_row)
        layout.addWidget(self.table, 1)

        layout.addLayout(self._build_list_actions())
        layout.addLayout(self._build_set_actions())

        self.status = QLabel("")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _build_list_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self.check_all_btn = QPushButton("Check All")
        self.check_all_btn.clicked.connect(lambda: self._set_all(True))
        row.addWidget(self.check_all_btn)

        self.uncheck_all_btn = QPushButton("Uncheck All")
        self.uncheck_all_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(self.uncheck_all_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setToolTip("Drop the highlighted minerals from the shortlist")
        self.remove_btn.clicked.connect(self.remove_selected)
        row.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.setToolTip("Empty the shortlist; saved sets are kept")
        self.clear_btn.clicked.connect(self.clear_list)
        row.addWidget(self.clear_btn)

        row.addStretch()

        self.rir_btn = QPushButton("RIR Quant")
        self.rir_btn.setObjectName("primaryButton")
        self.rir_btn.setToolTip(
            "Reference intensity ratio weight percents for the checked minerals, "
            "plus anything still checked in the Phases list"
        )
        self.rir_btn.clicked.connect(self._run_rir)
        row.addWidget(self.rir_btn)

        self.quant_btn = QPushButton("Open Quant…")
        self.quant_btn.setToolTip("Open the Le Bail / quantitative analysis window")
        self.quant_btn.clicked.connect(self.workspace.open_quant)
        row.addWidget(self.quant_btn)
        return row

    def _build_set_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        label = QLabel("Saved set:")
        label.setObjectName("mutedLabel")
        row.addWidget(label)

        self.set_combo = QComboBox()
        self.set_combo.setMinimumWidth(180)
        self.set_combo.setToolTip(
            "A named combination of minerals — load one to check exactly those "
            "and nothing else"
        )
        row.addWidget(self.set_combo)

        self.load_set_btn = QPushButton("Load")
        self.load_set_btn.setToolTip(
            "Check exactly the minerals in this set, adding back any that have "
            "been removed from the shortlist"
        )
        self.load_set_btn.clicked.connect(self.load_set)
        row.addWidget(self.load_set_btn)

        self.save_set_btn = QPushButton("Save As…")
        self.save_set_btn.setToolTip("Store the current check marks under a name")
        self.save_set_btn.clicked.connect(self.save_set)
        row.addWidget(self.save_set_btn)

        self.delete_set_btn = QPushButton("Delete")
        self.delete_set_btn.setToolTip("Forget this saved set")
        self.delete_set_btn.clicked.connect(self.delete_set)
        row.addWidget(self.delete_set_btn)

        row.addStretch()
        return row

    # --- table ---

    def reload(self):
        """Rebuild the table from the store."""
        if self._suppress_reload:
            return
        entries = self.store.entries()
        self._loading = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(entries))
            for i, entry in enumerate(entries):
                key = entry_key(entry)
                resolved = self._resolve(key, entry)

                cb = QCheckBox()
                cb.setChecked(bool(entry.get("checked")))
                cb.stateChanged.connect(
                    lambda state, k=key: self._on_check(k, state)
                )
                self.table.setCellWidget(i, SELECT_COL, cb)

                name = entry.get("mineral_name") or "Unknown"
                item = QTableWidgetItem(name if resolved else f"{name} ⚠")
                item.setData(Qt.UserRole, key)
                item.setToolTip(self._entry_tooltip(entry, resolved))
                self.table.setItem(i, NAME_COL, item)

                phase = (resolved or {}).get("phase", {})
                self.table.setItem(
                    i, SPACE_GROUP_COL, QTableWidgetItem(str(phase.get("space_group") or "—"))
                )
                rir = phase.get("rir")
                rir_item = QTableWidgetItem(
                    f"{float(rir):.2f}" if isinstance(rir, (int, float)) and rir else "—"
                )
                if not rir:
                    rir_item.setToolTip(
                        "No reference intensity ratio in the database, so this "
                        "mineral cannot get a weight percent"
                    )
                self.table.setItem(i, RIR_COL, rir_item)
            self.table.resizeColumnsToContents()
            self._refresh_set_combo()
        finally:
            self._loading = False
        self._update_header()

    def _entry_tooltip(self, entry: Dict, resolved: Optional[Dict]) -> str:
        lines = [entry.get("mineral_name") or "Unknown"]
        if entry.get("amcsd_id"):
            lines.append(f"AMCSD {entry['amcsd_id']}")
        if resolved is None:
            lines.append("Not found in the local database — it may have been rebuilt")
            return "\n".join(lines)
        phase = resolved.get("phase", {})
        cell = [phase.get("cell_a"), phase.get("cell_b"), phase.get("cell_c")]
        if all(v for v in cell):
            lines.append("a, b, c: " + ", ".join(f"{float(v):.4f}" for v in cell))
        theo = resolved.get("theoretical_peaks") or {}
        n_lines = len(theo.get("two_theta", []))
        if n_lines:
            lines.append(f"{n_lines} reference lines")
        return "\n".join(lines)

    def _update_header(self):
        total = len(self.store)
        checked = len(self.store.checked())
        if total == 0:
            self.header.setText(
                "Shortlist is empty — every mineral you check in the Phases list "
                "lands here automatically. The shortlist is saved between sessions "
                "and follows you from one pattern to the next."
            )
        else:
            self.header.setText(
                f"{checked} of {total} checked. Checked minerals are plotted and are "
                "what RIR Quant and the Quant window analyse."
            )
        self.rir_btn.setEnabled(checked > 0 and self.session.has_pattern())

    def _refresh_set_combo(self):
        current = self.set_combo.currentText()
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        self.set_combo.addItems(self.store.set_names())
        if current:
            index = self.set_combo.findText(current)
            if index >= 0:
                self.set_combo.setCurrentIndex(index)
        self.set_combo.blockSignals(False)
        has_sets = self.set_combo.count() > 0
        self.load_set_btn.setEnabled(has_sets)
        self.delete_set_btn.setEnabled(has_sets)

    # --- resolving against the database ---

    def _on_pattern_changed(self):
        """
        A new pattern means new reference positions and no fitted shift.

        Dropping the cache forces every shortlisted mineral to be rebuilt from
        the database at the current wavelength. Re-applying the checks is the
        workspace's job, once it has cleared the stale candidate list.
        """
        self._entry_cache.clear()
        self._identity_cache.clear()
        self.reload()

    def _resolve(self, key: str, entry: Dict) -> Optional[Dict]:
        """
        The analysis entry for one shortlisted mineral, or None if it is gone.

        Shaped like a match result — a `phase` dict plus reference lines — which
        is what matching, RIR, and Le Bail all consume.
        """
        if key in self._entry_cache:
            return self._entry_cache[key]

        stage = self.workspace.identify_stage
        row = None
        try:
            if entry.get("amcsd_id"):
                row = stage.local_db.get_mineral_by_amcsd_id(entry["amcsd_id"])
            if row is None and entry.get("mineral_id") is not None:
                row = stage.local_db.get_mineral_by_id(int(entry["mineral_id"]))
            if row is None and entry.get("mineral_name"):
                hits = stage.local_db.search_by_mineral_name(entry["mineral_name"], limit=1)
                row = hits[0] if hits else None
        except Exception as exc:
            print(f"Could not resolve shortlisted mineral {entry.get('mineral_name')}: {exc}")
            row = None

        resolved = None
        if row is not None:
            phase = stage._db_row_to_phase(row)
            resolved = {
                "phase": phase,
                "mineral_id": phase.get("id"),
                "mineral_name": phase.get("mineral"),
                "match_score": 1.0,
                "shortlisted": True,
            }
            theo = stage.reference_peaks_for(phase)
            if theo:
                resolved["theoretical_peaks"] = theo
        self._entry_cache[key] = resolved
        return resolved

    def checked_entries(self) -> List[Dict]:
        """Analysis entries for every checked mineral that still resolves."""
        entries = []
        for entry in self.store.checked():
            resolved = self._resolve(entry_key(entry), entry)
            if resolved is not None:
                entries.append(resolved)
        return entries

    # --- actions ---

    def identity_for(self, result: Dict) -> Optional[Dict]:
        """
        The stable identity of a search hit, match, or phase dict.

        The real AMCSD id has to be looked up: search hits do not carry one, and
        the phase dicts built from them fill the field with the local row id.
        """
        if not isinstance(result, dict):
            return None
        phase = result.get("phase", result)
        name = (
            phase.get("mineral")
            or phase.get("mineral_name")
            or result.get("mineral_name")
            or ""
        )
        mineral_id = (
            result.get("mineral_id")
            or phase.get("id")
            or result.get("id")
        )

        amcsd_id = None
        if mineral_id is not None:
            cache_key = str(mineral_id)
            if cache_key not in self._identity_cache:
                found = (None, "")
                try:
                    row = self.workspace.identify_stage.local_db.get_mineral_by_id(
                        int(mineral_id)
                    )
                    if row:
                        found = (row.get("amcsd_id"), row.get("mineral_name") or "")
                except Exception:
                    pass
                self._identity_cache[cache_key] = found
            amcsd_id, db_name = self._identity_cache[cache_key]
            name = name or db_name

        if not amcsd_id and not name:
            return None
        return {"amcsd_id": amcsd_id, "mineral_name": name, "mineral_id": mineral_id}

    def add_result(self, result: Dict) -> bool:
        """Put a search hit, match, or phase dict on the shortlist, checked."""
        identity = self.identity_for(result)
        if identity is None:
            return False
        added = self.store.add(**identity, checked=True)
        self.store.set_checked(entry_key(identity), True)
        return added

    def add_results(self, results: List[Dict]) -> int:
        """Shortlist a whole list at once; returns how many were new."""
        added = 0
        self._suppress_reload = True
        try:
            for result in results:
                if self.add_result(result):
                    added += 1
        finally:
            self._suppress_reload = False
        self.reload()
        return added

    def checked_state(self, result: Dict) -> Optional[bool]:
        """Whether this mineral is checked here, or None if it is not on the list."""
        identity = self.identity_for(result)
        if identity is None:
            return None
        entry = self.store.find(entry_key(identity))
        return bool(entry.get("checked")) if entry is not None else None

    def set_result_checked(self, result: Dict, checked: bool) -> None:
        """
        Carry a check mark from the phase list onto the shortlist.

        Checking a mineral in the results list is the same statement as
        checking it here, so a new one is added rather than needing to be
        remembered separately. Unchecking only clears the mark: the mineral
        stays on the list, which is the whole point of having one.
        """
        identity = self.identity_for(result)
        if identity is None:
            return
        key = entry_key(identity)
        self._suppress_reload = True
        try:
            if checked:
                self.store.add(**identity, checked=True)
                self.store.set_checked(key, True)
            elif self.store.contains(key):
                self.store.set_checked(key, False)
        finally:
            self._suppress_reload = False
        self.reload()

    def _on_check(self, key: str, state):
        if self._loading:
            return
        self._suppress_reload = True
        try:
            self.store.set_checked(key, state == Qt.Checked)
        finally:
            self._suppress_reload = False
        self._publish()

    def _toggle_current_row(self, item):
        cb = self.table.cellWidget(item.row(), SELECT_COL)
        if cb is not None:
            cb.setChecked(not cb.isChecked())

    def _set_all(self, checked: bool):
        self.store.set_all_checked(checked)
        self._publish()

    def _selected_keys(self) -> List[str]:
        keys = []
        for index in self.table.selectionModel().selectedRows(NAME_COL):
            item = self.table.item(index.row(), NAME_COL)
            if item is not None:
                keys.append(item.data(Qt.UserRole))
        return keys

    def remove_selected(self):
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(
                self, "Nothing Highlighted",
                "Click the minerals you want to drop, then Remove.\n\n"
                "(Unchecking keeps a mineral on the list but leaves it out of "
                "the analysis.)",
            )
            return
        self._suppress_reload = True
        try:
            for key in keys:
                self.store.remove(key)
                self._entry_cache.pop(key, None)
        finally:
            self._suppress_reload = False
        self.reload()
        self._publish()
        self.status.setText(f"Removed {len(keys)} mineral(s) from the shortlist.")

    def clear_list(self):
        if len(self.store) == 0:
            self.status.setText("The shortlist is already empty.")
            return
        if QMessageBox.question(
            self, "Clear Shortlist",
            f"Remove all {len(self.store)} minerals from the shortlist?\n\n"
            "Saved sets are kept, so you can load one to get a combination back.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.store.clear()
        self._entry_cache.clear()
        self._publish()
        self.status.setText("Shortlist cleared.")

    def save_set(self):
        checked = self.store.checked()
        if not checked:
            QMessageBox.information(
                self, "Nothing Checked",
                "Check the minerals that make up this combination, then Save As.",
            )
            return
        suggestion = self.set_combo.currentText()
        name, ok = QInputDialog.getText(
            self, "Save Set", "Name for this combination:", text=suggestion
        )
        if not ok or not name.strip():
            return
        count = self.store.save_set(name.strip())
        index = self.set_combo.findText(name.strip())
        if index >= 0:
            self.set_combo.setCurrentIndex(index)
        self.status.setText(f"Saved “{name.strip()}” with {count} mineral(s).")

    def load_set(self):
        name = self.set_combo.currentText()
        if not name:
            return
        count = self.store.apply_set(name)
        self._publish()
        self.status.setText(f"Loaded “{name}” — {count} mineral(s) checked.")

    def delete_set(self):
        name = self.set_combo.currentText()
        if not name:
            return
        if QMessageBox.question(
            self, "Delete Set", f"Forget the saved set “{name}”?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.store.delete_set(name)
        self.status.setText(f"Deleted “{name}”.")

    def _run_rir(self):
        self.workspace.identify_stage.run_rir_quant()
        self.status.setText(self.workspace.identify_stage.status.text())

    def _publish(self):
        """Push the checked minerals into the session and redraw."""
        self._update_header()
        # The same mineral may be sitting in the phase list with its own
        # check box; leaving the two disagreeing is worse than either state
        self.workspace.apply_shortlist_checks()
        self.workspace.sync_selected_phases()
        self.selection_changed.emit()
