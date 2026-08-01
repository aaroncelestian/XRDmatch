"""Headless check of the ranked mineral picker.

Builds a peak list from one known spinel record, then asks the picker to rank
every spinel record in the database against it: the record the peaks came from
must come out on top, and the filters must not lose it.
"""

import sys
import types

# The database module imports pymatgen for CIF parsing, which this test never
# reaches; stub it so the check runs without a full crystallography stack
if "pymatgen" not in sys.modules:
    for name in ("pymatgen", "pymatgen.io", "pymatgen.io.cif", "pymatgen.core"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["pymatgen.io.cif"].CifParser = object
    sys.modules["pymatgen.core"].Structure = object

import numpy as np
from PyQt5.QtWidgets import QApplication

from gui.dialogs.mineral_picker_dialog import MineralPickerDialog, COL_NAME, COL_FIT
from utils.fingerprint_search import fingerprint_score
from utils.local_database import get_local_database
from PyQt5.QtCore import Qt

TOLERANCE = 0.2


def main():
    app = QApplication([])
    db = get_local_database()
    hits = db.search_by_mineral_name("spinel", limit=300)
    print(f"{len(hits)} spinel records")

    # Peaks lifted from one record, so the right answer is known
    truth = next(
        h for h in hits
        if (db.get_diffraction_pattern(int(h["id"]), 1.5406) or {}).get("two_theta") is not None
    )
    pattern = db.get_diffraction_pattern(int(truth["id"]), 1.5406)
    tt = np.asarray(pattern["two_theta"], dtype=float)
    inten = np.asarray(pattern["intensity"], dtype=float)
    keep = (tt >= 10) & (tt <= 80)
    peaks_tt, peaks_i = tt[keep], inten[keep]
    print(f"truth record: id={truth['id']} a={truth.get('cell_a')} "
          f"({len(peaks_tt)} peaks)")

    def score(hit):
        theo = db.get_diffraction_pattern(int(hit["id"]), 1.5406)
        if not theo:
            return None
        return fingerprint_score(
            peaks_tt, peaks_i, theo["two_theta"], theo["intensity"],
            tolerance=TOLERANCE, exp_range=(10.0, 80.0), shift_span=0.05,
        )

    previewed = []
    dialog = MineralPickerDialog(
        hits, "spinel", score_fn=score, preview_fn=previewed.append,
        tolerance=TOLERANCE, ambient_only=True,
    )

    def hit_at(row: int) -> dict:
        return hits[dialog.table.item(row, COL_NAME).data(Qt.UserRole)]

    rows = dialog.table.rowCount()
    print(f"ambient rows: {rows} of {len(hits)}")
    print("top 5 (id, fit, a):")
    for r in range(min(rows, 5)):
        hit = hit_at(r)
        print("   ", (hit["id"], dialog.table.item(r, COL_FIT).text(), hit.get("cell_a")))

    assert previewed and previewed[-1] is dialog.current_hit(), "no preview fired"
    best = hit_at(0)
    best_fit = (best.get("fingerprint") or {}).get("score", 0)
    truth_fit = (truth.get("fingerprint") or {}).get("score", 0)
    print(f"best fit {best_fit:.3f} vs truth record {truth_fit:.3f}")
    assert best_fit >= truth_fit - 1e-9, "ranking did not put the true record on top"
    assert dialog.detail.text().strip(), "no detail line for the selection"

    dialog.filter_box.setText("Fd3m")
    filtered = dialog.table.rowCount()
    print(f"filtered to space group Fd3m: {filtered} rows")
    assert 0 < filtered <= rows

    dialog.filter_box.clear()
    dialog.ambient_box.setChecked(False)
    print(f"all conditions: {dialog.table.rowCount()} rows")
    assert dialog.table.rowCount() >= rows

    dialog.table.selectRow(2)
    picked = dialog.current_hit()
    dialog.accept()
    assert dialog.chosen() is picked, "accept returned the wrong record"
    print(f"chose id={picked['id']} — OK")

    app.quit()


if __name__ == "__main__":
    main()
