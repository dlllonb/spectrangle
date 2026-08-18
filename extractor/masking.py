"""
masking.py — Catalog-based point-source masking before trace detection.

Phase 3 fix (this session's investigation, `new_results.txt` Entries
71-75): a diffraction trace's angle gets corrupted when a nearby star's
point-source core lets `detection.detect_traces`'s gap-bridging blur
connect it to something it shouldn't — either several chance-aligned
faint stars merged into one fake trace (mechanism 1), or two genuinely
separate real traces bridged into one candidate by a third star sitting
between them (mechanism 2). Real-data forensics (catalog cross-match
against the disturbance location on corrupted traces) showed this is
almost always a real, identifiable, already-known star, not something
that needs to be inferred from the image itself.

The fix: given a solved WCS and a star catalog (both already needed
elsewhere in a real pipeline — the WCS to platesolve, the catalog to
confirm it), mask a small disk at every catalog star's predicted pixel
position *before* `detect_traces` runs, via its existing `exclude_mask`
parameter. Validated across all 18 fields in the project's simulated
test set (Entry 75) with a single fixed mask radius (no per-field
tuning needed) and confirmed under real multi-realization Monte Carlo
testing (Entry 76), not just a single lucky draw — collapses per-trace
calibration (std(z)) from ~6-8 (badly overconfident) to ~0.3-1.5
(honest to mildly conservative).

**Known limitations, not yet resolved** (Entries 77-82): a residual,
field-specific mean bias (~0.05-0.20 deg) remains in a majority of
tested fields, not fixable by radius tuning alone (a larger radius
causes a *different*, catastrophic failure — overlapping mask disks
tile the image and their leftover gaps get mistaken for real streaks,
Entry 79). This module implements the validated fix as-is; it does not
yet implement a targeted correction for the residual bias.
"""
from __future__ import annotations

import math

import numpy as np
from astropy.wcs import WCS
from scipy.optimize import curve_fit

from .stars import extract_stars


def _gaussian_2d(coords, amp, x0, y0, sigma, offset):
    x, y = coords
    return offset + amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def recover_empirical_psf_sigma(image: np.ndarray, n_stars: int = 15, half: int = 10) -> float:
    """Estimate the image's own stellar PSF sigma (pixels) by fitting a
    2-D Gaussian to a handful of the brightest detected point sources.

    Used to size the catalog star mask radius relative to how sharp
    stars actually are in THIS image, rather than assuming one fixed
    pixel radius transfers across different plate scales/seeing.

    Parameters
    ----------
    image : ndarray
        2-D image array.
    n_stars : int
        Number of successfully-fit stars to average (median) over.
    half : int
        Half-width (pixels) of the cutout fit around each candidate
        star position.

    Returns
    -------
    float
        Median fitted PSF sigma, in pixels.
    """
    xs, ys, _ = extract_stars(image.astype(np.float32), mask_spectra=True, max_sources=n_stars * 4)
    ny, nx = image.shape
    sigmas = []
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        xlo, xhi = max(0, xi - half), min(nx, xi + half + 1)
        ylo, yhi = max(0, yi - half), min(ny, yi + half + 1)
        cutout = image[ylo:yhi, xlo:xhi].astype(np.float64)
        yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
        p0 = [cutout.max() - np.median(cutout), x, y, 3.0, np.median(cutout)]
        try:
            popt, _ = curve_fit(_gaussian_2d, (xx.ravel(), yy.ravel()), cutout.ravel(), p0=p0, maxfev=2000)
            amp, _, _, sigma, _ = popt
            if 0.3 < sigma <= half and amp > 0:
                sigmas.append(sigma)
        except Exception:
            continue
        if len(sigmas) >= n_stars:
            break
    if not sigmas:
        raise ValueError("recover_empirical_psf_sigma: no star fit succeeded; "
                          "check the image has detectable point sources")
    return float(np.median(sigmas))


def build_catalog_star_mask(
    image_shape: tuple[int, int],
    wcs: WCS,
    catalog_ra_deg: np.ndarray,
    catalog_dec_deg: np.ndarray,
    catalog_mag: np.ndarray,
    radius_px: float,
    mag_cut: float = 13.0,
) -> np.ndarray:
    """Build a boolean mask with a disk of `radius_px` at every
    sufficiently-bright catalog star's predicted pixel position.

    Intended to be passed directly as `detection.detect_traces`'s
    `exclude_mask` (and thence `pipeline.measure_grating_angle`'s
    `star_catalog`/related parameters) — see this module's docstring
    for why (Entries 71-75).

    Parameters
    ----------
    image_shape : (ny, nx)
    wcs : astropy.wcs.WCS
        Solved WCS for this image (SIP/distortion-aware if available).
    catalog_ra_deg, catalog_dec_deg, catalog_mag : ndarray
        Parallel arrays describing the star catalog. Not a DataFrame
        to keep this module's only dependency astropy/numpy/scipy.
    radius_px : float
        Mask disk radius, in pixels. Entry 75 validated k=8 x the
        image's own `recover_empirical_psf_sigma` as a single fixed
        choice across all 18 test fields (roughly 4-45x range in local
        star density) — pass `k * recover_empirical_psf_sigma(image)`
        unless you have a specific reason to deviate. Entry 72/79:
        avoid radii much larger than this (roughly 2x already causes a
        distinct, catastrophic over-masking failure mode) — this is
        NOT "bigger is safer."
    mag_cut : float
        Only stars at or brighter than this magnitude are masked
        (default matches Entry 71-75's validated 13.0 — faint enough
        to be irrelevant at this project's tested exposure depth
        shouldn't cost mask coverage).

    Returns
    -------
    ndarray[bool], shape image_shape
    """
    ny, nx = image_shape
    bright = catalog_mag <= mag_cut
    ra, dec = np.asarray(catalog_ra_deg)[bright], np.asarray(catalog_dec_deg)[bright]
    if len(ra) == 0:
        return np.zeros(image_shape, dtype=bool)

    x_px, y_px = wcs.all_world2pix(ra, dec, 0)

    margin = radius_px + 2
    in_bounds = (x_px >= -margin) & (x_px < nx + margin) & (y_px >= -margin) & (y_px < ny + margin)
    x_px, y_px = x_px[in_bounds], y_px[in_bounds]

    mask = np.zeros(image_shape, dtype=bool)
    r_int = int(math.ceil(radius_px))
    yy, xx = np.mgrid[-r_int:r_int + 1, -r_int:r_int + 1]
    disk = (xx ** 2 + yy ** 2) <= radius_px ** 2
    dy, dx = np.where(disk)
    dy, dx = dy - r_int, dx - r_int
    for x0, y0 in zip(x_px, y_px):
        xi, yi = int(round(x0)), int(round(y0))
        ys, xs = yi + dy, xi + dx
        valid = (ys >= 0) & (ys < ny) & (xs >= 0) & (xs < nx)
        mask[ys[valid], xs[valid]] = True
    return mask
