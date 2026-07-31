#!/usr/bin/env python3
"""Benchmark + accuracy check for Le Bail refinement on realistic multi-phase data."""

import time
import numpy as np

from utils.lebail_refinement import LeBailRefinement


def pseudo_voigt(x, center, fwhm, height, eta=0.5):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return height * ((1 - eta) * g + eta * l)


def make_phase(name, n_peaks, seed, cell):
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.uniform(8.0, 88.0, n_peaks))
    inten = rng.uniform(1.0, 100.0, n_peaks)
    inten[rng.integers(0, n_peaks)] = 100.0
    return {
        'phase': dict(mineral=name, id=name, **cell),
        'theoretical_peaks': {'two_theta': pos, 'intensity': inten},
    }


def main():
    two_theta = np.arange(5.0, 90.0, 0.02)  # 4250 pts, typical lab scan
    cell = dict(cell_a=5.0, cell_b=5.0, cell_c=7.0,
                cell_alpha=90.0, cell_beta=90.0, cell_gamma=90.0)

    phases = [
        make_phase('PhaseA', 180, 1, cell),
        make_phase('PhaseB', 150, 2, cell),
    ]

    obs = np.zeros_like(two_theta)
    for weight, ph in zip((1.0, 0.6), phases):
        tp = ph['theoretical_peaks']
        for p, i in zip(tp['two_theta'], tp['intensity']):
            obs += pseudo_voigt(two_theta, p, 0.10, weight * i)
    obs = np.maximum(obs + np.random.default_rng(0).normal(0, 0.3, obs.size), 0)

    print(f"points={len(two_theta)}  peaks={sum(len(p['theoretical_peaks']['two_theta']) for p in phases)}")

    lebail = LeBailRefinement()
    lebail.set_experimental_data(two_theta, obs)
    init = {
        'scale_factor': 1.0, 'u_param': 0.0005, 'v_param': 0.0,
        'w_param': 0.01, 'eta_param': 0.5, 'zero_shift': 0.0,
        'refine_cell': True, 'refine_profile': True, 'refine_scale': True,
        'refine_intensities': False,
    }
    for ph in phases:
        lebail.add_phase(ph, dict(init))

    t0 = time.time()
    res = lebail.refine_phases(max_iterations=10, convergence_threshold=1e-5, quiet=True)
    dt = time.time() - t0

    calc = res['calculated_pattern']
    exp = res['experimental_intensity']
    print(f"\nelapsed        : {dt:.2f} s  ({res['iterations']} iterations)")
    print(f"Rwp            : {res['final_r_factors']['Rwp']:.2f} %")
    print(f"max(exp)       : {exp.max():.1f}")
    print(f"max(calc)      : {calc.max():.1f}")
    print(f"sum(calc)/sum(exp) = {calc.sum() / exp.sum():.2f}   (should be ~1.0)")


if __name__ == '__main__':
    main()
