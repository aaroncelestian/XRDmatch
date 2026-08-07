"""
Reading the instrument profile off a single measured peak.

The Caglioti coefficients U, V and W describe an instrument, not a specimen, so
they are properly calibrated once against a standard and then left alone. In
practice they are usually left at whatever the program starts them at, and the
refinement is then asked to fit real peaks with a resolution curve that has
never seen the diffractometer. It cannot, so the sample terms take up the slack:
microstrain grows until the total width is right, and because that width now
arrives as a Lorentzian the peak acquires tails no lab peak has. The fit ends up
the wrong shape at every reflection while the width, taken alone, looks fine.

Fitting one isolated peak breaks that. A single peak gives a width and a mixing
parameter directly, and the mixing is exactly what says how much of the width is
Gaussian: inverting the Thompson-Cox-Hastings combination splits the measured
width into an instrument part, which is what W should have been, and a sample
part, which is what microstrain and crystallite size are for.

The same fit answers a second question. A lab source emits a Kα doublet, and if
those satellites are in the data but not in the model then every peak is a
little too wide and leans towards high 2θ. Fitting the satellite ratio as a free
parameter says whether they are there: a value near the nominal 0.5 means the
doublet is present, and one near zero means the beam was monochromated or the
satellites have already been stripped.

Nothing here is refinement. It produces starting values, and starting values
that come from the pattern in front of you beat defaults that came from somebody
else's instrument.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from utils import kalpha_filter as kalpha
from utils.profile_functions import (
    MAX_ASYMMETRY, pseudo_voigt, size_from_fwhm, strain_from_fwhm, tch_split,
)

# How much of the peak to fit, as a multiple of its apparent width. Far enough
# out to pin the flanks and the background, near enough to stay clear of the
# next reflection.
WINDOW_WIDTHS = 6.0
MIN_HALF_WINDOW = 0.15   # degrees
MAX_HALF_WINDOW = 3.0    # degrees

# A peak has to reach this fraction of the strongest one to be worth offering as
# a calibration line; below it the width is mostly noise.
MIN_CANDIDATE_HEIGHT = 0.10

# Clearance beyond which a peak counts as fully isolated. A neighbour a degree
# away does not distort a peak a few tenths of a degree wide.
FULL_ISOLATION = 1.0


def candidate_peaks(two_theta: Sequence[float], intensity: Sequence[float],
                    count: int = 6) -> List[Dict]:
    """
    The peaks in a pattern most worth calibrating on, best first.

    Wanted is a peak that is both strong and alone: strong so its shape is
    signal rather than counting noise, and alone so that what is being fitted is
    one reflection's profile and not the sum of two. Ranking on height times
    clearance balances the two, since the tallest peak in a pattern is often the
    one with a neighbour leaning on it.
    """
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if len(x) < 9 or len(x) != len(y):
        return []

    # Maxima are found on a lightly smoothed copy so that counting noise on the
    # top of a peak does not register as a dozen separate ones
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(y, kernel, mode='same')
    interior = smooth[1:-1]
    peaks = np.flatnonzero((interior > smooth[:-2]) & (interior >= smooth[2:])) + 1
    if not len(peaks):
        return []

    heights = smooth[peaks]
    tallest = float(np.max(heights))
    if tallest <= 0:
        return []
    peaks = peaks[heights >= MIN_CANDIDATE_HEIGHT * tallest]
    if not len(peaks):
        return []

    order = peaks[np.argsort(smooth[peaks])[::-1]]
    found: List[Dict] = []
    for index in order:
        stronger = order[smooth[order] > smooth[index]]
        isolation = (float(np.min(np.abs(x[stronger] - x[index])))
                     if len(stronger) else float('inf'))
        found.append({
            'two_theta': float(x[index]),
            'intensity': float(y[index]),
            'isolation': isolation,
        })

    found.sort(key=lambda peak: -(
        peak['intensity'] * min(peak['isolation'], FULL_ISOLATION)
    ))
    return found[:count]


def _apparent_width(x: np.ndarray, y: np.ndarray, centre_index: int,
                    baseline: float) -> float:
    """FWHM straight off the data, by walking down both flanks to half height."""
    peak = y[centre_index] - baseline
    if peak <= 0:
        return float(np.median(np.diff(x)) * 5.0)
    half = baseline + 0.5 * peak

    edges = []
    for step in (-1, 1):
        index = centre_index
        while 0 < index < len(y) - 1 and y[index] > half:
            index += step
        edges.append(x[int(np.clip(index, 0, len(x) - 1))])
    width = abs(edges[1] - edges[0])
    return float(max(width, np.median(np.diff(x)) * 2.0))


def fit_peak(two_theta: Sequence[float], intensity: Sequence[float],
             centre: float, wavelength: float, *,
             half_window: Optional[float] = None,
             fit_alpha2: bool = True) -> Dict:
    """
    Fit one peak with the profile the refinement itself uses, and read it out.

    The model is deliberately the same split pseudo-Voigt, over a linear
    background, with an optional Kα2 satellite at the position the doublet
    geometry puts it and a free fraction of the parent's height. Fitting
    something else here -- a plain Gaussian, say -- would hand back a width that
    means one thing to this function and another to the engine.

    Returns the fitted profile, the instrument and sample terms implied by it,
    and the arrays needed to show the fit.

    Raises ValueError when there is no usable peak at `centre`.
    """
    x_all = np.asarray(two_theta, dtype=float)
    y_all = np.asarray(intensity, dtype=float)
    if len(x_all) < 9 or len(x_all) != len(y_all):
        raise ValueError("Pattern is too short to fit a peak")

    centre = float(centre)
    start = int(np.argmin(np.abs(x_all - centre)))
    step = float(np.median(np.diff(x_all)))
    baseline = float(np.percentile(y_all, 10))
    apparent = _apparent_width(x_all, y_all, start, baseline)

    if half_window is None:
        half_window = float(np.clip(
            WINDOW_WIDTHS * apparent / 2.0, MIN_HALF_WINDOW, MAX_HALF_WINDOW
        ))
    window = np.abs(x_all - centre) <= half_window
    if int(np.count_nonzero(window)) < 8:
        raise ValueError(f"Too few points within {half_window:.2f}° of {centre:.3f}°")

    x = x_all[window]
    y = y_all[window]
    wavelength_ratio = kalpha.alpha2_ratio(float(wavelength))

    edge = max(2, len(x) // 10)
    background = float(np.median(np.concatenate([y[:edge], y[-edge:]])))
    height = float(np.max(y) - background)
    if height <= 0:
        raise ValueError(f"No peak above background at {centre:.3f}°")
    # A window whose largest value sits on its edge is not centred on a peak: the
    # intensity is still climbing towards one somewhere else. Fitting it anyway
    # returns a width belonging to whatever is over the horizon.
    if not edge <= int(np.argmax(np.convolve(y, np.ones(3) / 3.0, mode='same'))) < len(y) - edge:
        raise ValueError(f"No peak centred at {centre:.3f}°")

    def model(p):
        centre_, height_, fwhm_, eta_, skew_, ratio_, b0, b1 = p
        profile = pseudo_voigt(x - centre_, fwhm_, eta_, skew_)
        if ratio_ > 0.0:
            satellite = kalpha.alpha2_separation(centre_, wavelength_ratio)
            profile = profile + ratio_ * pseudo_voigt(
                x - centre_ - float(satellite), fwhm_, eta_, skew_
            )
        return height_ * profile + b0 + b1 * (x - centre_)

    ratio_start = 0.3 if fit_alpha2 else 0.0
    ratio_limit = kalpha.NOMINAL_INTENSITY_RATIO + 0.1 if fit_alpha2 else 0.0
    guess = [centre, height, apparent, 0.5, 0.0, ratio_start, background, 0.0]
    lower = [centre - apparent, 0.0, step, 0.0, -MAX_ASYMMETRY, 0.0,
             background - height, -height]
    upper = [centre + apparent, 10.0 * height, 4.0 * half_window, 1.0,
             MAX_ASYMMETRY, max(ratio_limit, 1e-9), background + height, height]
    guess = np.clip(guess, lower, upper)

    result = least_squares(
        lambda p: model(p) - y, guess, bounds=(lower, upper),
        method='trf', x_scale='jac', ftol=1e-12, xtol=1e-12, max_nfev=4000,
    )
    fitted = model(result.x)
    centre_, height_, fwhm_, eta_, skew_, ratio_, b0, b1 = (float(v) for v in result.x)

    gauss, lorentz = tch_split(fwhm_, eta_)
    # What is left of the peak after the fit, against the peak itself
    residual = float(np.sum((y - fitted) ** 2))
    signal = float(np.sum((y - background) ** 2))
    return {
        'centre': centre_,
        'height': height_,
        'fwhm': fwhm_,
        'eta': eta_,
        'skew': skew_,
        'alpha2_ratio': ratio_ if fit_alpha2 else 0.0,
        'background': (b0, b1),
        'gauss_fwhm': gauss,
        'lorentz_fwhm': lorentz,
        # U and V are the tan-theta terms of the resolution curve and one peak
        # says nothing about either; W alone reproduces the width measured here
        # and leaves the curve flat, which is honest about what was measured
        'u_param': 0.0,
        'v_param': 0.0,
        'w_param': gauss ** 2,
        # The Lorentzian belongs to the specimen. Which of the two sample terms
        # produced it is not decidable from one peak either, so both readings are
        # given: each is what that term alone would have to be.
        'microstrain': strain_from_fwhm(centre_, lorentz),
        'crystallite_size': size_from_fwhm(centre_, lorentz, float(wavelength)),
        # The instrument skew that would give this lean at this angle. A sample
        # skew would give it too; one peak cannot separate them.
        'axial_asymmetry': skew_ * float(np.tan(np.radians(centre_))),
        'half_window': float(half_window),
        'two_theta': x,
        'observed': y,
        'fitted': fitted,
        'misfit': (100.0 * float(np.sqrt(residual / signal))
                   if signal > 0 else float('nan')),
    }
