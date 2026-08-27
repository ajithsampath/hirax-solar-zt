"""Core fitting and data-loading routines for the HIRAX solar-transit ZT
(Zernike-polynomial) fit.

This is a standalone, trimmed-down extraction of the pieces of the original
HIRAX_Analysis `lib.py` / `solar_beam_fit.py` actually used by the
combined-day, all-frequency ZT fit: a 1D Gaussian pre-fit, a Bessel-radial
1D Zernike basis fit in (time - transit) space, and the per-day data loader
(smoothed amplitude vs. time, rebinned onto a common frequency axis, with
the Sun's above-horizon mask from astropy). CST beam pattern code is
intentionally not included here -- this package only fits the observed
data, not the simulated beam.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy as sp
from scipy.special import jn
from scipy.optimize import curve_fit
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
import astropy.units as u


# -----------------------------------------------------------------------------
# Config / data loading
# -----------------------------------------------------------------------------

@dataclass
class Config:
    """Observation site and file-layout parameters.

    `freqs` defaults to the 81-channel grid used in the source HIRAX_Analysis
    notebooks (0.8 -> 0.4 GHz). `data_dir` must contain files named
    `Prod6_DAY{day}_sun_fft1d_filt.npz`.
    """

    data_dir: Path
    location: EarthLocation = field(
        default_factory=lambda: EarthLocation(lat=-31 * u.deg, lon=22 * u.deg, height=1300 * u.m)
    )
    freqs: np.ndarray = field(default_factory=lambda: np.round(np.linspace(0.8, 0.4, 81), 3))

    def day_file(self, day: int) -> Path:
        return Path(self.data_dir) / f"Prod6_DAY{day}_sun_fft1d_filt.npz"

    def available_days(self) -> list[int]:
        """Days for which a `Prod6_DAY{n}_sun_fft1d_filt.npz` file exists in `data_dir`."""
        days = []
        for p in sorted(Path(self.data_dir).glob("Prod6_DAY*_sun_fft1d_filt.npz")):
            m = re.search(r"DAY(\d+)", p.stem)
            if m:
                days.append(int(m.group(1)))
        return sorted(days)


@dataclass
class DayData:
    amp: np.ndarray       # (n_time, n_freq), rebinned onto cfg.freqs
    times: Time            # (n_time,)
    dt_hours: np.ndarray   # (n_time,) hours relative to transit


def rebin_axis_interp(arr: np.ndarray, new_size: int, axis: int = 0) -> np.ndarray:
    """Resample `arr` to `new_size` along `axis` by linear interpolation on a
    normalized [0, 1] index -- used to put each day's frequency axis onto the
    common `cfg.freqs` grid."""
    old_size = arr.shape[axis]
    old_x = np.linspace(0, 1, old_size)
    new_x = np.linspace(0, 1, new_size)
    return np.apply_along_axis(lambda y: np.interp(new_x, old_x, y), axis, arr)


def load_day_data(day: int, cfg: Config) -> DayData:
    data = np.load(cfg.day_file(day), allow_pickle=True)
    amp = rebin_axis_interp(data["absV"], len(cfg.freqs), axis=1)
    times = Time(data["day_utc"], scale="utc")
    dt_hours = data["dt_daytime"]
    return DayData(amp=amp, times=times, dt_hours=dt_hours)


def sun_altaz(times: Time, location: EarthLocation) -> tuple[np.ndarray, np.ndarray]:
    """Sun altitude/azimuth (degrees) at `times`, used to build the
    above-horizon visibility mask -- no beam pattern needed."""
    frame = AltAz(obstime=times, location=location)
    sun = get_sun(times).transform_to(frame)
    return sun.alt.degree, sun.az.degree


# -----------------------------------------------------------------------------
# Gaussian pre-fit
# -----------------------------------------------------------------------------

def gaussian_1d(x, amp, x0, sigma, offset):
    return amp * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + offset


def fit_gaussian_1d(x, data):
    """Simple 1D Gaussian fit via curve_fit. Returns (x0, sigma), used to seed
    the center and width of the Zernike basis below."""
    amp0 = data.max() - data.min()
    offset0 = data.min()
    x0_0 = x[np.argmax(data)]
    sigma0 = 0.2 * (x.max() - x.min())

    p0 = [amp0, x0_0, sigma0, offset0]
    bounds = (
        [0, x.min(), 0, -np.inf],
        [np.inf, x.max(), np.inf, np.inf],
    )

    popt, _ = curve_fit(gaussian_1d, x, data, p0=p0, bounds=bounds, maxfev=10000)
    amp, x0, sigma, offset = popt
    return x0, sigma


# -----------------------------------------------------------------------------
# 1D Zernike ("ZT") basis and fit
# -----------------------------------------------------------------------------

def noll_to_nm(j: int) -> tuple[int, int]:
    n = 0
    j1 = j
    while j1 > n:
        n += 1
        j1 -= n
    m = -n + 2 * j1
    return n, m


def theta_1d(x, x0):
    return np.where(x < x0, np.pi, 0.0)


def zernike_basis_1d_theta(x, x0, sigma, N_full):
    """Bessel-radial 1D Zernike basis in (x - x0)/sigma, Noll-indexed 0..N_full-1.

    Modes that are degenerate or numerically unstable at m < 0 / invalid
    (n, m) combinations are skipped, so the returned basis can have fewer
    than N_full rows.
    """
    x = np.asarray(x)

    sigma = max(sigma, 1e-6)
    r = np.abs(x - x0)
    rm = r / sigma

    rm = np.where(rm < 1e-8, 1e-8, rm)
    theta = theta_1d(x, x0)

    basis = []
    i_pow = {0: 1.0, 1: 1j, 2: -1.0, 3: -1j}

    for j in range(N_full):
        n, m = noll_to_nm(j)

        if m >= 0:
            if n < abs(m):
                continue
            if (n - abs(m)) % 2 != 0:
                continue

            Bes = jn(n + 1, rm) / rm
            if not np.all(np.isfinite(Bes)):
                continue

            nc = np.sqrt(np.abs((2 * n + 1) * (2 * n + 3) * (2 * n + 5)))
            im = i_pow[m % 4]

            Z = np.real(
                nc * np.exp(1j * m * theta) / (im * 2 * np.pi) * (-1) ** ((n - m) // 2) * Bes
            )

            if np.all(np.isfinite(Z)):
                basis.append(Z)

    return np.array(basis, dtype=float)


def fit_zernike_1d_theta(x, data, error, x0, sigma, N_full):
    """Variance-weighted least-squares fit of `data(x)` to the 1D Zernike
    basis. Returns (coef, model, chi2_red)."""
    error = np.asarray(error)
    error = np.where(error <= 0, np.min(error[error > 0]), error)

    Basis = zernike_basis_1d_theta(x, x0, sigma, N_full)
    if Basis.size == 0:
        raise RuntimeError("Zernike basis is empty")

    w = 1.0 / error ** 2
    Bw = Basis.T * np.sqrt(w[:, None])
    Dw = data * np.sqrt(w)

    mask = np.isfinite(Bw).all(axis=1) & np.isfinite(Dw)
    Bw = Bw[mask]
    Dw = Dw[mask]

    coef, _, _, _ = sp.linalg.lstsq(Bw, Dw)
    model = Basis.T @ coef

    resid = (data - model) / error
    chi2_red = np.vdot(resid, resid).real / (len(data) - len(coef))

    return coef, model, chi2_red


def zernike_eval_1d_theta(x, coef, x0, sigma, N_full):
    """Evaluate a fitted 1D Zernike model at arbitrary `x` (e.g. the full
    dt_grid), reusing the same (x0, sigma, N_full) the coefficients were fit
    with."""
    Basis = zernike_basis_1d_theta(x, x0, sigma, N_full)
    return Basis.T @ coef
