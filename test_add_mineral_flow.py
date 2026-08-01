"""Headless walk through Add mineral: picker opens, previews, adds the record.

Uses the real workspace so the plot overlay, the candidates table, and the
session all see what the picker chose.
"""

import sys
import types

if "pymatgen" not in sys.modules:
    for name in ("pymatgen", "pymatgen.io", "pymatgen.io.cif", "pymatgen.core"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["pymatgen.io.cif"].CifParser = object
    sys.modules["pymatgen.core"].Structure = object

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from gui.session import AnalysisSession
from gui.workspace import AnalysisWorkspace
from gui.dialogs.mineral_picker_dialog import COL_NAME, COL_FIT, COL_OFFSET
from utils.local_database import get_local_database


def main():
    app = QApplication([])
    db = get_local_database()

    # A pattern built from one real spinel record, so a right answer exists
    truth = next(
        h for h in db.search_by_mineral_name("spinel", limit=300)
        if abs((h.get("cell_a") or 0) - 8.0843) < 0.001
    )
    reference = db.get_diffraction_pattern(int(truth["id"]), 1.5406)
    tt = np.asarray(reference["two_theta"], dtype=float)
    inten = np.asarray(reference["intensity"], dtype=float)
    keep = (tt >= 10) & (tt <= 80)
    tt, inten = tt[keep], inten[keep]

    grid = np.arange(10.0, 80.0, 0.02)
    profile = np.zeros_like(grid)
    for centre, height in zip(tt, inten):
        profile += height * np.exp(-0.5 * ((grid - centre) / 0.06) ** 2)

    session = AnalysisSession()
    session.set_raw_pattern({"two_theta": grid, "intensity": profile})
    session.set_processed_pattern({"two_theta": grid, "intensity": profile})
    session.set_peaks({"two_theta": tt, "intensity": inten})

    workspace = AnalysisWorkspace(session)
    stage = workspace.identify_stage

    stage.mineral_search.setText("spinel")
    stage.add_mineral_by_name()

    picker = stage._picker
    assert picker is not None, "picker did not open"
    print(f"picker rows: {picker.table.rowCount()}")
    print("top 3:")
    for r in range(min(3, picker.table.rowCount())):
        print(f"    {picker.table.item(r, COL_NAME).text()}  "
              f"fit={picker.table.item(r, COL_FIT).text()}  "
              f"d2th={picker.table.item(r, COL_OFFSET).text()}")

    preview = workspace.current_preview()
    assert preview is not None, "highlighting a record did not preview it"
    print(f"preview: {preview['name']} ({len(preview['two_theta'])} lines)")

    best = picker.current_hit()
    assert best["id"] == truth["id"], f"expected id {truth['id']}, got {best['id']}"
    print(f"best record is the source record (id {best['id']}), fit "
          f"{best['fingerprint']['score']:.3f}")

    picker.accept()
    app.processEvents()

    assert stage._picker is None, "picker reference not cleared"
    candidates = workspace._candidate_results
    assert len(candidates) == 1, f"expected 1 candidate, got {len(candidates)}"
    added = candidates[0]
    assert added["mineral_id"] == truth["id"]
    assert added.get("fingerprint"), "the picker's score was not carried over"
    print(f"added: {added['mineral_name']} id={added['mineral_id']} "
          f"fingerprint={added['fingerprint_score']:.3f}")
    assert session.search_candidates and \
        session.search_candidates[0]["id"] == truth["id"]

    checkbox = workspace.results_table.cellWidget(0, 3)
    assert checkbox is not None and checkbox.isChecked(), "added row was not checked"

    # A second record of the same mineral is a separate candidate, not a dupe
    other = next(
        h for h in db.search_by_mineral_name("spinel", limit=300)
        if h["id"] != truth["id"] and h.get("cell_a")
    )
    stage._add_mineral_record(other)
    print(f"after adding a second record: {len(workspace._candidate_results)} candidates")
    assert len(workspace._candidate_results) == 2, "second record was swallowed as a duplicate"

    # Cancelling puts back whatever overlay was on the plot
    before = workspace.current_preview()
    stage.mineral_search.setText("spinel")
    stage.add_mineral_by_name()
    stage._picker.reject()
    app.processEvents()
    assert workspace.current_preview() is before, "cancel did not restore the overlay"
    print("cancel restored the previous overlay — OK")

    workspace.deleteLater()
    app.quit()


if __name__ == "__main__":
    main()
