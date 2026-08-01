"""
Peak profile widths and shapes for Le Bail refinement.

The split here follows the standard separation of instrument from sample:

  Gaussian width   comes from the instrument, via the Caglioti equation
                   FWHM_G^2 = U tan^2(theta) + V tan(theta) + W
  Lorentzian width comes from the sample, as crystallite size (1/cos theta)
                   and microstrain (tan theta) terms

Those combine through the Thompson-Cox-Hastings approximation to a Voigt, which
yields both a total width and a mixing parameter. Fitting the mixing parameter
directly, as an independent variable, lets it trade against the widths; deriving
it removes that correlation.

Units are degrees 2-theta for widths, angstroms for wavelength, micrometres for
crystallite size, and dimensionless 1e-6 for microstrain.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# FWHM = 2 sqrt(2 ln 2) sigma for a Gaussian
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

_DEG = 180.0 / np.pi

# Scherrer shape factor. GSAS-II uses unity on the grounds that crystallite
# broadening is not a rigorously defined quantity, and a size that is accurate
# to 10% is as much as the method supports.
SCHERRER_K = 1.0

# Below this the Caglioti polynomial is allowed to go non-physical, so the
# width is floored rather than square-rooted into a NaN.
_MIN_FWHM_SQUARED = 1e-6


def caglioti_fwhm(two_theta: np.ndarray, u: float, v: float, w: float) -> np.ndarray:
    """
    Gaussian FWHM in degrees from the instrument resolution curve.

    FWHM^2 = U tan^2(theta) + V tan(theta) + W, the form Rietveld derived from
    Caglioti's neutron monochromator resolution work. It is a polynomial in
    tan(theta) and fits x-ray instruments as readily as neutron ones.
    """
    tan_theta = np.tan(np.radians(np.asarray(two_theta, dtype=float) / 2.0))
    fwhm_squared = u * tan_theta ** 2 + v * tan_theta + w
    return np.sqrt(np.maximum(fwhm_squared, _MIN_FWHM_SQUARED))


def lorentzian_fwhm_instrument(two_theta: np.ndarray, x: float = 0.0,
                               y: float = 0.0) -> np.ndarray:
    """
    Instrumental Lorentzian FWHM in degrees: X/cos(theta) + Y tan(theta).

    Instrumental broadening is mostly Gaussian, so X and Y are normally zero.
    They share their angular dependence with the sample size and strain terms
    and so cannot be refined alongside them.
    """
    theta = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    return x / np.maximum(np.cos(theta), 1e-9) + y * np.tan(theta)


def lorentzian_fwhm_size(two_theta: np.ndarray, size_um: float,
                         wavelength: float) -> np.ndarray:
    """
    Scherrer broadening in degrees from a mean crystallite size in micrometres.

    Size broadening is constant in Q, which transforms to a 1/cos(theta)
    dependence in 2-theta. Larger crystallites broaden less, and above roughly
    a micrometre the effect is below the resolution of most diffractometers.
    """
    size_um = float(size_um)
    if size_um <= 0:
        return np.zeros_like(np.asarray(two_theta, dtype=float))
    theta = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    wavelength_um = float(wavelength) * 1e-4
    return _DEG * SCHERRER_K * wavelength_um / (size_um * np.maximum(np.cos(theta), 1e-9))


def lorentzian_fwhm_strain(two_theta: np.ndarray, microstrain: float) -> np.ndarray:
    """
    Microstrain broadening in degrees for a strain of microstrain x 1e-6.

    A spread of lattice constants gives a constant delta-d/d, and differentiating
    Bragg's law turns that into delta(2 theta) = 2 (delta-d/d) tan(theta).
    """
    microstrain = float(microstrain)
    if microstrain <= 0:
        return np.zeros_like(np.asarray(two_theta, dtype=float))
    theta = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    return 2.0 * _DEG * microstrain * 1e-6 * np.tan(theta)


# Thompson-Cox-Hastings polynomial coefficients: the first set approximates the
# Voigt FWHM from its Gaussian and Lorentzian components, the second gives the
# pseudo-Voigt mixing parameter that reproduces it.
_TCH_WIDTH = (1.0, 2.69269, 2.42843, 4.47163, 0.07842, 1.0)
_TCH_ETA = (1.36603, -0.47719, 0.11116)


def tch_mix(gamma_gauss: np.ndarray, gamma_lorentz: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine Gaussian and Lorentzian widths into a pseudo-Voigt width and mixing.

    Returns (FWHM, eta), where eta is 0 for a pure Gaussian and 1 for a pure
    Lorentzian. This is the standard approximation to a true Voigt convolution,
    accurate to well under a percent and far cheaper to evaluate.
    """
    g = np.maximum(np.asarray(gamma_gauss, dtype=float), 0.0)
    l = np.maximum(np.asarray(gamma_lorentz, dtype=float), 0.0)
    a0, a1, a2, a3, a4, a5 = _TCH_WIDTH

    gamma = (
        a0 * g ** 5
        + a1 * g ** 4 * l
        + a2 * g ** 3 * l ** 2
        + a3 * g ** 2 * l ** 3
        + a4 * g * l ** 4
        + a5 * l ** 5
    ) ** 0.2
    gamma = np.maximum(gamma, 1e-6)

    q = np.clip(l / gamma, 0.0, 1.0)
    b1, b2, b3 = _TCH_ETA
    eta = np.clip(b1 * q + b2 * q ** 2 + b3 * q ** 3, 0.0, 1.0)
    return gamma, eta


def pseudo_voigt(offset: np.ndarray, fwhm: np.ndarray,
                 eta: np.ndarray) -> np.ndarray:
    """
    Unit-height pseudo-Voigt: a linear blend of Gaussian and Lorentzian.

    `offset` is the distance from the peak centre in degrees. `fwhm` and `eta`
    broadcast against it, so each reflection may carry its own width and mixing.
    """
    offset = np.asarray(offset, dtype=float)
    fwhm = np.maximum(np.asarray(fwhm, dtype=float), 1e-9)
    eta = np.asarray(eta, dtype=float)

    sigma = fwhm * FWHM_TO_SIGMA
    gaussian = np.exp(-0.5 * (offset / sigma) ** 2)
    lorentzian = 1.0 / (1.0 + (offset / (fwhm / 2.0)) ** 2)
    return (1.0 - eta) * gaussian + eta * lorentzian


def phase_widths(two_theta: np.ndarray, instrument: dict, sample: dict,
                 wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Total (FWHM, eta) per reflection for one phase.

    `instrument` supplies u, v, w and optionally x, y, shared by every phase.
    `sample` supplies crystallite_size (um) and microstrain (x 1e-6), plus an
    optional per-reflection `strain_extra` in degrees for anisotropic models.
    """
    two_theta = np.asarray(two_theta, dtype=float)

    gamma_g = caglioti_fwhm(
        two_theta,
        float(instrument.get('u_param', 0.0)),
        float(instrument.get('v_param', 0.0)),
        float(instrument.get('w_param', 0.0)),
    )

    gamma_l = lorentzian_fwhm_instrument(
        two_theta,
        float(instrument.get('x_param', 0.0)),
        float(instrument.get('y_param', 0.0)),
    )
    gamma_l = gamma_l + lorentzian_fwhm_size(
        two_theta, sample.get('crystallite_size', 0.0), wavelength
    )
    gamma_l = gamma_l + lorentzian_fwhm_strain(
        two_theta, sample.get('microstrain', 0.0)
    )

    extra = sample.get('strain_extra')
    if extra is not None:
        gamma_l = gamma_l + np.asarray(extra, dtype=float)

    return tch_mix(gamma_g, gamma_l)


def size_from_fwhm(two_theta: float, fwhm: float, wavelength: float) -> float:
    """Mean crystallite size in micrometres implied by a Lorentzian FWHM."""
    if fwhm <= 0:
        return float('inf')
    theta = np.radians(float(two_theta) / 2.0)
    wavelength_um = float(wavelength) * 1e-4
    return _DEG * SCHERRER_K * wavelength_um / (fwhm * np.cos(theta))


def strain_from_fwhm(two_theta: float, fwhm: float) -> float:
    """Microstrain (x 1e-6) implied by a Lorentzian FWHM at one angle."""
    theta = np.radians(float(two_theta) / 2.0)
    tan_theta = np.tan(theta)
    if tan_theta <= 0:
        return 0.0
    return float(fwhm) / (2.0 * _DEG * tan_theta) * 1e6
