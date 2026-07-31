"""Temporary smoke test for the rebuilt Peaks/Phases layout."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from gui.main_window import XRDMainWindow
from utils.fingerprint_search import fingerprint_score, select_fingerprint_peaks


def make_pattern(path):
    tt = np.arange(5.0, 75.0, 0.02)
    y = 40 + 8 * np.exp(-(tt - 8) ** 2 / 200.0)
    # quartz-ish lines
    for pos, amp in [(20.86, 200), (26.64, 1000), (36.54, 90), (39.47, 90),
                     (40.30, 45), (42.45, 65), (45.79, 40), (50.14, 150),
                     (54.87, 40), (59.96, 110), (68.14, 70)]:
        y += amp * np.exp(-(tt - pos) ** 2 / (2 * 0.05 ** 2))
    # calcite-ish lines
    for pos, amp in [(23.02, 120), (29.40, 600), (35.96, 80), (39.40, 180),
                     (43.15, 140), (47.49, 170), (48.51, 180)]:
        y += amp * np.exp(-(tt - pos) ** 2 / (2 * 0.05 ** 2))
    rng = np.random.default_rng(7)
    y += rng.normal(0, 2.0, len(tt))
    with open(path, "w") as fh:
        fh.write("/* synthetic two-phase test */\n")
        fh.write("# 2theta intensity error\n")
        for a, b in zip(tt, y):
            fh.write(f"{a:.4f} {max(b, 0):.3f} {np.sqrt(max(b, 1)):.3f}\n")


def check_unit_math():
    fp = select_fingerprint_peaks([10, 20, 30, 40], [5, 100, 50, 2], n_peaks=3)
    assert list(np.round(fp["two_theta"], 3)) == [10.0, 20.0, 30.0], fp
    perfect = fingerprint_score([20, 30], [100, 50], [20, 30], [100, 50], tolerance=0.2)
    assert perfect["score"] > 0.95, perfect
    assert perfect["n_found"] == 2
    # extra unexplained experimental peaks must not lower the score
    with_extra = fingerprint_score([20, 25, 30, 33], [100, 90, 50, 70],
                                   [20, 30], [100, 50], tolerance=0.2)
    assert abs(with_extra["score"] - perfect["score"]) < 1e-9, with_extra
    # missing strongest line is heavily penalized
    missing_top = fingerprint_score([30], [50], [20, 30], [100, 50],
                                    tolerance=0.2, exp_range=(5.0, 70.0))
    assert missing_top["score"] < 0.25, missing_top
    assert not missing_top["top_found"]
    # lines outside the measured range are not held against a candidate
    out_of_range = fingerprint_score([30], [50], [3, 30], [100, 50],
                                     tolerance=0.2, exp_range=(5.0, 70.0))
    assert out_of_range["n_expected"] == 1 and out_of_range["score"] > 0.9, out_of_range
    print("fingerprint math OK")


def main():
    check_unit_math()
    path = os.path.abspath("_smoke_pattern.xye")
    make_pattern(path)

    app = QApplication(sys.argv)
    win = XRDMainWindow()
    win.show()
    ws = win.workspace

    ws.open_pattern_file(path)
    assert ws.session.has_pattern(), "pattern did not load"

    ws.process_stage.apply_processing()
    ws.process_stage.find_peaks()
    npeaks = len(ws.session.peaks["two_theta"])
    print(f"peaks found: {npeaks}")
    assert npeaks > 8

    # layout: control bars must fit without scrolling in a short bottom panel
    ws.show_bottom_tab("phases")
    app.processEvents()
    ctrl_h = ws.identify_stage.control_panel.sizeHint().height()
    peaks_h = ws.process_stage.peaks_panel.sizeHint().height()
    bg_h = ws.process_stage.background_panel.sizeHint().height()
    print(f"control heights — phases {ctrl_h}, peaks {peaks_h}, background {bg_h}")
    assert ctrl_h < 150 and peaks_h < 150

    # view toggles exist and re-plot cleanly
    for key in ("raw", "processed", "background", "peaks"):
        ws.view_toggles[key].setChecked(False)
        app.processEvents()
    ws.refresh_plot()
    for key in ("raw", "processed", "background", "peaks"):
        ws.view_toggles[key].setChecked(True)
    ws.refresh_plot()
    print("view toggles OK")

    # add known minerals by name
    stage = ws.identify_stage
    stage.mineral_search.setText("Quartz")
    stage.add_mineral_by_name()
    app.processEvents()
    print("status after add:", stage.status.text())
    assert ws.results_table.rowCount() >= 1
    checked = [
        i for i in range(ws.results_table.rowCount())
        if ws.results_table.cellWidget(i, 5) and ws.results_table.cellWidget(i, 5).isChecked()
    ]
    assert checked, "manually added mineral should be checked"

    # preview via row change (what arrow keys trigger)
    ws.results_table.setCurrentCell(0, 0)
    app.processEvents()
    print("preview:", (ws._preview or {}).get("name"), 
          len((ws._preview or {}).get("two_theta", [])))

    # details dialog
    ws.show_phase_details(0)
    app.processEvents()
    assert ws._details_dialog is not None
    print("details title:", ws._details_dialog.title.text(),
          "| peaks rows:", ws._details_dialog.peaks_table.rowCount())

    # fingerprint search against the real database
    stage.method_combo.setCurrentIndex(0)
    assert stage._method_key() == "fingerprint"
    stage.pool_size.setValue(120)
    stage.max_results.setValue(25)
    stage.start_search()
    app.processEvents()
    print("search status:", stage.status.text())
    print("candidates:", ws.results_table.rowCount())
    for i in range(min(8, ws.results_table.rowCount())):
        name = ws.results_table.item(i, 0).text()
        score = ws.results_table.item(i, 3).text()
        fp = ws.results_table.item(i, 4).text()
        print(f"   {i+1}. {name:28s} score={score} fp={fp}")

    # matching on a couple of checked candidates
    for i in range(min(3, ws.results_table.rowCount())):
        cb = ws.results_table.cellWidget(i, 5)
        if cb:
            cb.setChecked(True)
    stage.start_matching()
    if stage._match_thread is not None:
        stage._match_thread.wait(60000)
        app.processEvents()
    print("match status:", stage.status.text())
    print("match rows:", ws.results_table.rowCount(), "mode:", ws._results_mode)

    if ws._results_mode == "matches" and ws.results_table.rowCount():
        ws.results_table.setCurrentCell(0, 1)
        app.processEvents()
        print("match preview:", (ws._preview or {}).get("name"))
        cb = ws.results_table.cellWidget(0, 0)
        if cb:
            cb.setChecked(True)
        app.processEvents()
        print("selected phases:", len(ws.session.selected_phases))
        stage.search_residual()
        app.processEvents()
        print("residual status:", stage.status.text())

    # clear all (bypass the confirmation prompt)
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    ws.clear_all_phases()
    app.processEvents()
    assert ws.results_table.rowCount() == 0
    assert not ws.session.matched_phases and not ws.session.search_candidates
    print("clear all OK")

    # peaks tab clear
    ws.process_stage.clear_peaks()
    assert ws.peaks_table.rowCount() == 0 and not ws.session.has_peaks()
    print("clear peaks OK")

    os.remove(path)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
