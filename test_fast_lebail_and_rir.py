#!/usr/bin/env python3
"""Smoke tests for fast Le Bail trial mode, RIR parse, and Chung wt%."""

import time
import numpy as np
from utils.lebail_refinement import LeBailRefinement
from utils.multi_phase_analyzer import MultiPhaseAnalyzer
from utils.local_database import LocalCIFDatabase


def _synthetic_phase(name, peaks, cell=5.0):
    return {
        'phase': {
            'mineral': name,
            'formula': 'X',
            'id': name,
            'cell_a': cell,
            'cell_b': cell,
            'cell_c': cell,
            'cell_alpha': 90.0,
            'cell_beta': 90.0,
            'cell_gamma': 90.0,
            'rir': 2.5 if name == 'A' else 1.0,
        },
        'theoretical_peaks': {
            'two_theta': np.array(peaks, dtype=float),
            'intensity': np.array([100, 60, 40, 30][:len(peaks)], dtype=float),
            'd_spacing': np.ones(len(peaks)),
        }
    }


def test_trial_vs_polish_speed():
    two_theta = np.linspace(5, 50, 2250)
    peaks = [12.5, 17.7, 25.1, 30.9]
    inten = [100, 70, 50, 40]
    pattern = np.zeros_like(two_theta)
    for pos, intensity in zip(peaks, inten):
        fwhm = 0.12
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        pattern += intensity * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
    pattern += 5

    phase = _synthetic_phase('A', peaks)

    def run(mode):
        eng = LeBailRefinement()
        eng.set_experimental_data(two_theta, pattern)
        eng.add_phase(phase, {
            'refine_cell': mode == 'polish',
            'refine_profile': mode == 'polish',
            'refine_scale': True,
            'w_param': 0.015,
        })
        t0 = time.time()
        result = eng.refine_phases(max_iterations=3 if mode == 'trial' else 6,
                                   mode=mode, quiet=True)
        return time.time() - t0, result['final_r_factors']['Rwp']

    t_trial, rwp_trial = run('trial')
    t_polish, rwp_polish = run('polish')
    print(f"trial: {t_trial:.3f}s Rwp={rwp_trial:.2f}%")
    print(f"polish: {t_polish:.3f}s Rwp={rwp_polish:.2f}%")
    # Trial should complete quickly enough for iterative multiphase ID
    assert t_trial < 5.0, f"trial too slow: {t_trial:.3f}s"
    print("PASS trial/polish timing")


def test_rir_parse_and_chung():
    db = LocalCIFDatabase()
    # Ensure schema migration ran
    conn = __import__('sqlite3').connect(db.db_path)
    cols = {r[1] for r in conn.execute('PRAGMA table_info(minerals)')}
    conn.close()
    assert 'rir' in cols, "rir column missing"

    # Parse first blocks from DIF if present
    dif_path = __import__('pathlib').Path('data/difdata.dif')
    if dif_path.exists():
        with open(dif_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(50000)
        blocks = db._split_amcsd_dif_blocks(content)
        assert blocks, "no DIF blocks"
        parsed = db._parse_amcsd_single_block(blocks[0])
        assert parsed is not None
        assert parsed.get('rir') is not None, f"RIR not parsed: {parsed.keys()}"
        print(f"Parsed RIR={parsed['rir']} for {parsed['mineral_name']}")
    else:
        print("SKIP DIF parse (no data/difdata.dif)")

    analyzer = MultiPhaseAnalyzer()
    fake = {
        'identified_phases': [
            {'phase': {'mineral': 'A', 'rir': 2.0}, 'integrated_proxy': 100.0, 'rir': 2.0},
            {'phase': {'mineral': 'B', 'rir': 1.0}, 'integrated_proxy': 50.0, 'rir': 1.0},
        ]
    }
    wts = analyzer.calculate_rir_weight_percents(fake)
    # A: 100/2=50, B: 50/1=50 → 50/50 wt%
    assert abs(wts['A'] - 50.0) < 1e-6
    assert abs(wts['B'] - 50.0) < 1e-6
    print("PASS Chung RIR wt%", wts)


def test_joint_accept_reject_gate():
    two_theta = np.linspace(5, 40, 1750)
    pattern = np.zeros_like(two_theta)
    true_peaks = [10.0, 15.0, 22.0]
    for pos, intensity in zip(true_peaks, [100, 80, 50]):
        sigma = 0.05
        pattern += intensity * np.exp(-0.5 * ((two_theta - pos) / sigma) ** 2)
    pattern += 2

    good = _synthetic_phase('Good', true_peaks)
    bad = _synthetic_phase('Bad', [11.5, 18.0, 27.0], cell=6.0)
    bad['phase']['rir'] = 3.0

    analyzer = MultiPhaseAnalyzer()
    result = analyzer.joint_lebail_phase_identification(
        {'two_theta': two_theta, 'intensity': pattern, 'wavelength': 1.5406},
        [good, bad],
        max_phases=2,
        min_delta_rwp=1.0,
        min_scale=0.01,
        residual_research=False,
        polish=False,
    )
    names = [p['phase']['mineral'] for p in result['identified_phases']]
    print("Identified:", names, "Rwp", result.get('final_rwp'))
    assert 'Good' in names
    print("PASS joint gate")


if __name__ == '__main__':
    test_trial_vs_polish_speed()
    test_rir_parse_and_chung()
    test_joint_accept_reject_gate()
    print("\nAll smoke tests passed.")
