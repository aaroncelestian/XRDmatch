"""Headless check that the shortlist decides what RIR Quant reports.

The shortlist is where the user says which minerals make up the mixture, so
unchecking or removing one there has to change the next RIR run. It used to
change only the shortlist: the phase list row the mineral had been checked in
was left ticked, and RIR quantified it anyway.

Runs against the real database and the real workspace, with the shortlist
pointed at a temporary file so the user's own list is left alone.
"""

import os
import sys
import tempfile
import types

if "pymatgen" not in sys.modules:
    for name in ("pymatgen", "pymatgen.io", "pymatgen.io.cif", "pymatgen.core"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["pymatgen.io.cif"].CifParser = object
    sys.modules["pymatgen.core"].Structure = object

import numpy as np
from PyQt5.QtWidgets import QApplication

from utils import phase_shortlist
from utils.local_database import get_local_database

WAVELENGTH = 1.5406
LO, HI, STEP, FWHM = 10.0, 80.0, 0.02, 0.06


def _first_record(db, name):
    """A record of this mineral that has reference lines in range."""
    for hit in db.search_by_mineral_name(name, limit=50):
        reference = db.get_diffraction_pattern(int(hit["id"]), WAVELENGTH)
        if reference is None:
            continue
        tt = np.asarray(reference["two_theta"], dtype=float)
        if np.any((tt >= LO) & (tt <= HI)):
            return hit, reference
    raise AssertionError(f"no usable {name} record in the local database")


def _two_phase_pattern(references, weights):
    grid = np.arange(LO, HI, STEP)
    profile = np.zeros_like(grid)
    lines_tt, lines_i = [], []
    for reference, weight in zip(references, weights):
        tt = np.asarray(reference["two_theta"], dtype=float)
        inten = np.asarray(reference["intensity"], dtype=float)
        keep = (tt >= LO) & (tt <= HI)
        for centre, height in zip(tt[keep], inten[keep]):
            profile += weight * height * np.exp(-0.5 * ((grid - centre) / FWHM) ** 2)
        lines_tt.extend(tt[keep])
        lines_i.extend(weight * inten[keep])
    order = np.argsort(lines_tt)
    peaks = {
        "two_theta": np.asarray(lines_tt)[order],
        "intensity": np.asarray(lines_i)[order],
    }
    return grid, profile, peaks


def _shortlist_row(panel, name):
    from gui.widgets.shortlist_panel import NAME_COL
    for row in range(panel.table.rowCount()):
        item = panel.table.item(row, NAME_COL)
        if item is not None and item.text().startswith(name):
            return row
    raise AssertionError(f"{name} is not on the shortlist")


def _accepted_names(stage):
    names = set()
    for entry in stage.accepted_phases():
        phase = entry.get("phase", entry)
        names.add(phase.get("mineral") or entry.get("mineral_name") or "?")
    return names


def _rir_names(session):
    assert session.rir_results is not None, "RIR produced no result"
    return {p["name"] for p in session.rir_results["phases"]}


def main():
    app = QApplication([])

    store_path = os.path.join(tempfile.mkdtemp(prefix="xrd_shortlist_"), "shortlist.json")
    phase_shortlist._shortlist = phase_shortlist.PhaseShortlist(store_path)

    from gui.session import AnalysisSession
    from gui.workspace import AnalysisWorkspace

    db = get_local_database()
    quartz, quartz_ref = _first_record(db, "quartz")
    calcite, calcite_ref = _first_record(db, "calcite")
    print(f"records: {quartz['mineral_name']} id={quartz['id']}, "
          f"{calcite['mineral_name']} id={calcite['id']}")

    grid, profile, peaks = _two_phase_pattern([quartz_ref, calcite_ref], [0.7, 0.3])

    session = AnalysisSession()
    session.set_raw_pattern({"two_theta": grid, "intensity": profile})
    session.set_processed_pattern({"two_theta": grid, "intensity": profile})
    session.set_peaks(peaks)

    workspace = AnalysisWorkspace(session)
    stage = workspace.identify_stage
    panel = workspace.shortlist_panel

    stage.add_phases_from_database([
        stage._db_row_to_phase(quartz), stage._db_row_to_phase(calcite),
    ])
    app.processEvents()

    q_name = quartz["mineral_name"]
    c_name = calcite["mineral_name"]

    accepted = _accepted_names(stage)
    print(f"after adding both: accepted={sorted(accepted)}")
    assert accepted == {q_name, c_name}, accepted
    assert len(panel.store.checked()) == 2, "checked candidates were not shortlisted"

    stage.run_rir_quant()
    print(f"RIR with both: {sorted(_rir_names(session))}")
    assert _rir_names(session) == {q_name, c_name}

    # Unchecking on the shortlist takes the mineral out of the analysis, even
    # though the candidates table is still showing the row it came from
    from gui.widgets.shortlist_panel import SELECT_COL
    row = _shortlist_row(panel, c_name)
    panel.table.cellWidget(row, SELECT_COL).setChecked(False)
    app.processEvents()

    accepted = _accepted_names(stage)
    print(f"after unchecking {c_name}: accepted={sorted(accepted)}")
    assert accepted == {q_name}, accepted

    stage.run_rir_quant()
    print(f"RIR after unchecking: {sorted(_rir_names(session))}")
    assert _rir_names(session) == {q_name}, "RIR kept a mineral the shortlist dropped"

    panel.table.cellWidget(row, SELECT_COL).setChecked(True)
    app.processEvents()
    assert _accepted_names(stage) == {q_name, c_name}, "re-checking did not bring it back"

    # A phase carried over from a residual round is not in the candidates table
    # any more, so the shortlist is the only place left to take it out of a run
    stage._kept_phases = list(stage.accepted_phases())
    panel.table.cellWidget(row, SELECT_COL).setChecked(False)
    app.processEvents()

    accepted = _accepted_names(stage)
    print(f"after unchecking kept {c_name}: accepted={sorted(accepted)}")
    assert accepted == {q_name}, accepted

    stage.run_rir_quant()
    print(f"RIR with a kept phase unchecked: {sorted(_rir_names(session))}")
    assert _rir_names(session) == {q_name}, "RIR kept a phase from an earlier round"

    stage._kept_phases = []
    panel.table.cellWidget(row, SELECT_COL).setChecked(True)
    app.processEvents()

    # Removing behaves the same way: gone from the list is gone from the answer
    panel.table.selectRow(_shortlist_row(panel, q_name))
    panel.remove_selected()
    app.processEvents()

    accepted = _accepted_names(stage)
    print(f"after removing {q_name}: accepted={sorted(accepted)}")
    assert accepted == {c_name}, accepted

    stage.run_rir_quant()
    print(f"RIR after removing: {sorted(_rir_names(session))}")
    assert _rir_names(session) == {c_name}, "RIR kept a mineral removed from the shortlist"

    print("the shortlist decides what RIR quantifies — OK")
    workspace.deleteLater()
    app.quit()


if __name__ == "__main__":
    main()
