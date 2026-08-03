"""Sanity check: can a displaced phase still be found?"""

import numpy as np

from utils.fingerprint_search import fingerprint_score
from utils.two_theta_shift import (
    DISPLACEMENT, ZERO_SHIFT, apply_shift, fit_shift, shift_pattern, unshift_pattern,
)

# Quartz-like reference lines (Cu Kα)
REF_TT = np.array([20.86, 26.64, 36.54, 39.47, 40.30, 42.45, 45.79, 50.14, 59.96, 68.14])
REF_INT = np.array([16.0, 100.0, 8.0, 7.0, 4.0, 5.0, 3.0, 13.0, 9.0, 7.0])
EXP_RANGE = (5.0, 75.0)

# Other phases in the mixture, so the peak list is not just quartz
OTHER_TT = np.array([23.1, 29.4, 31.5, 43.2, 47.5, 48.5, 57.4, 64.7])
OTHER_INT = np.array([10.0, 100.0, 5.0, 14.0, 17.0, 18.0, 8.0, 6.0])


def make_peaks(true_shift, model):
    """Observed peak list for a mount displaced by `true_shift`."""
    tt = np.concatenate([apply_shift(REF_TT, true_shift, model), OTHER_TT])
    inten = np.concatenate([REF_INT * 0.4, OTHER_INT])
    order = np.argsort(tt)
    return tt[order], inten[order]


def score(exp_tt, exp_int, **kwargs):
    return fingerprint_score(
        exp_tt, exp_int, REF_TT, REF_INT,
        tolerance=0.2, exp_range=EXP_RANGE, **kwargs,
    )


print("=== undisplaced sample, no correction (baseline behaviour) ===")
tt, inten = make_peaks(0.0, DISPLACEMENT)
info = score(tt, inten)
print(f"  score {info['score']:.3f}  lines {info['n_found']}/{info['n_expected']}  "
      f"shift {info['shift']:+.3f}")
assert info["n_found"] == info["n_expected"], "clean case must match every line"
assert info["shift"] == 0.0

print("\n=== displaced sample, no correction (the reported failure) ===")
TRUE = -0.35
tt, inten = make_peaks(TRUE, DISPLACEMENT)
plain = score(tt, inten)
print(f"  score {plain['score']:.3f}  lines {plain['n_found']}/{plain['n_expected']}")

print("\n=== same pattern, auto fit +-0.5 ===")
fitted = score(tt, inten, shift_span=0.5, shift_model=DISPLACEMENT)
print(f"  score {fitted['score']:.3f}  lines {fitted['n_found']}/{fitted['n_expected']}  "
      f"shift {fitted['shift']:+.3f} (true {TRUE:+.3f})")
assert fitted["n_found"] > plain["n_found"], "auto fit must recover lines"
assert abs(fitted["shift"] - TRUE) < 0.03, f"shift off: {fitted['shift']}"
assert fitted["score"] > plain["score"]

print("\n=== manual shift, auto fit off ===")
manual = score(tt, inten, shift=TRUE, shift_model=DISPLACEMENT)
print(f"  score {manual['score']:.3f}  lines {manual['n_found']}/{manual['n_expected']}")
assert manual["n_found"] == manual["n_expected"]

print("\n=== zero-shift model on a zero-shifted pattern ===")
tt0, int0 = make_peaks(0.28, ZERO_SHIFT)
z = score(tt0, int0, shift_span=0.5, shift_model=ZERO_SHIFT)
print(f"  score {z['score']:.3f}  lines {z['n_found']}/{z['n_expected']}  "
      f"shift {z['shift']:+.3f} (true +0.280)")
assert abs(z["shift"] - 0.28) < 0.03

print("\n=== a wrong phase must not be rescued by the fit ===")
wrong_tt = np.array([12.3, 18.9, 24.1, 33.7, 41.9, 53.2, 62.8, 71.1])
wrong_int = np.array([40.0, 100.0, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0])
bad = fingerprint_score(
    tt, inten, wrong_tt, wrong_int,
    tolerance=0.2, exp_range=EXP_RANGE, shift_span=0.5, shift_model=DISPLACEMENT,
)
print(f"  score {bad['score']:.3f}  lines {bad['n_found']}/{bad['n_expected']}  "
      f"shift {bad['shift']:+.3f}")
assert bad["score"] < fitted["score"] / 2, "decoy scored too well"

print("\n=== shift_pattern is idempotent, unshift restores ===")
pat = {"two_theta": REF_TT.copy(), "intensity": REF_INT.copy(), "d_spacing": np.ones(10)}
once = shift_pattern(pat, -0.35, DISPLACEMENT)
twice = shift_pattern(once, -0.35, DISPLACEMENT)
assert np.allclose(once["two_theta"], twice["two_theta"])
assert np.allclose(unshift_pattern(twice)["two_theta"], REF_TT)
assert np.allclose(pat["two_theta"], REF_TT), "source pattern must not be mutated"
retuned = shift_pattern(once, 0.1, DISPLACEMENT)
assert np.allclose(retuned["two_theta"], apply_shift(REF_TT, 0.1, DISPLACEMENT))
print("  ok")

print("\n=== fit refuses to commit on too few lines ===")
sparse = np.array([26.29, 60.0])
val, n = fit_shift(sparse, REF_TT, REF_INT, tolerance=0.2, span=0.5)
print(f"  shift {val:+.3f} from {n} lines")
assert val == 0.0, "two lines should not pin a shift"

print("\nAll checks passed.")
