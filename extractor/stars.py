"""
stars.py — Point-source extraction for plate-solving.

Two-stage pipeline:
  1. Build a mask of elongated diffraction features so they are not mistaken
     for stars and do not inflate the background noise estimate.
  2. Run DAOStarFinder on the cleaned, background-subtracted image.

Without stage 1, bright spectra raise the sigma-clipped std used as the
detection threshold, causing faint real stars to be missed, and elongated
features can slip through DAOStarFinder's shape filters as false positives.
"""

from __future__ import annotations

import io

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from astropy.io import fits


def extract_stars(
    image: np.ndarray,
    max_sources: int = 300,
    fwhm: float = 3.0,
    threshold_sigma: float = 5.0,
    bg_sigma: float = 50.0,
    min_separation: float = 24.0,
    mask_spectra: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect point sources and return (xs, ys, fluxes), sorted brightest first.

    Parameters
    ----------
    image : ndarray
        2-D float image array.
    max_sources : int
        Maximum number of sources to return after ranking by flux.
    fwhm : float
        Expected stellar FWHM in pixels for DAOStarFinder.
    threshold_sigma : float
        Detection threshold as a multiple of the background RMS.
    bg_sigma : float
        Sigma for the large-scale Gaussian used to estimate and subtract the
        sky background. Should be >> stellar PSF size. Set to 0 to skip.
    min_separation : float
        Minimum pixel separation between kept sources (greedy NMS).
    mask_spectra : bool
        If True (default), detect and zero out elongated diffraction features
        before running DAOStarFinder.

    Returns
    -------
    xs, ys, fluxes : ndarray
        Pixel coordinates (0-based) and flux estimates of detected sources,
        each of length ≤ max_sources, sorted brightest-first.
    """
    img = image.astype(np.float32)

    # Remove sky gradient and vignetting.
    if bg_sigma > 0:
        img = np.clip(img - gaussian_filter(img, sigma=bg_sigma), 0.0, None)

    # Zero out elongated features so they don't masquerade as stars.
    if mask_spectra:
        mask = _diffraction_mask(img)
        img = img.copy()
        img[mask] = 0.0

    xs, ys, fluxes = _detect(img, fwhm, threshold_sigma)

    if len(xs) == 0:
        return xs, ys, fluxes

    order = np.argsort(fluxes)[::-1]
    xs, ys, fluxes = xs[order], ys[order], fluxes[order]
    xs, ys, fluxes = xs[:max_sources], ys[:max_sources], fluxes[:max_sources]
    return _dedup(xs, ys, fluxes, min_separation)


def make_xylist(xs: np.ndarray, ys: np.ndarray) -> io.BytesIO:
    """Pack source positions into an in-memory FITS xylist for nova.astrometry.net.

    Uses float64 (FITS 'D' format) to preserve centroid precision.
    """
    col_x = fits.Column(name="X", format="D", array=xs.astype(np.float64))
    col_y = fits.Column(name="Y", format="D", array=ys.astype(np.float64))
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([col_x, col_y])])
    buf = io.BytesIO()
    hdul.writeto(buf)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Diffraction / spectrum masking
# ---------------------------------------------------------------------------

def _diffraction_mask(
    img: np.ndarray,
    threshold_pct: float = 75.0,
    min_pixels: int = 30,
    min_eccentricity: float = 0.85,
    min_major_length: float = 15.0,
) -> np.ndarray:
    """Return a boolean mask of elongated bright features in *img*.

    Algorithm
    ---------
    1. Downsample to ≤1200 px on the long side for speed.
    2. Median-filter (7×7) to suppress point sources while leaving extended
       features intact.
    3. Subtract a Gaussian background (σ=30) from the suppressed image.
    4. Threshold at the 75th percentile of non-zero residuals.
    5. Label connected components; keep only those that are elongated
       (eccentricity ≥ 0.85) and large enough.
    6. Upscale the result back to the original image shape.
    """
    from skimage.measure import label, regionprops

    # Work at reduced resolution for speed.
    factor = max(1, max(img.shape) // 1200)
    small = img[::factor, ::factor].astype(np.float32)

    # Star-suppressed image: median filter removes PSF-scale peaks.
    suppressed = median_filter(small, size=7)

    # Remove the large-scale background from the suppressed image.
    feature_img = np.clip(suppressed - gaussian_filter(suppressed, sigma=30.0), 0.0, None)

    nz = feature_img[feature_img > 0]
    if nz.size == 0:
        return np.zeros(img.shape, dtype=bool)

    label_img = label(feature_img >= np.percentile(nz, threshold_pct))

    mask_small = np.zeros(small.shape, dtype=bool)
    for region in regionprops(label_img, intensity_image=feature_img):
        if region.num_pixels < min_pixels:
            continue
        # `axis_major_length` (skimage ≥ 0.19) replaced `major_axis_length`.
        major = float(getattr(region, "axis_major_length", None) or region.major_axis_length)
        if major < min_major_length or region.eccentricity < min_eccentricity:
            continue
        mask_small[label_img == region.label] = True

    if factor == 1:
        return mask_small[: img.shape[0], : img.shape[1]]
    big = np.kron(mask_small, np.ones((factor, factor), dtype=bool))
    return big[: img.shape[0], : img.shape[1]]


# ---------------------------------------------------------------------------
# Detection and deduplication
# ---------------------------------------------------------------------------

def _detect(
    img: np.ndarray,
    fwhm: float,
    threshold_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma-clip background, then run DAOStarFinder with a find_peaks fallback."""
    try:
        import photutils
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder, find_peaks
    except ImportError as exc:
        raise ImportError("photutils and astropy are required for star extraction.") from exc

    _, median, std = sigma_clipped_stats(img, sigma=3.0)
    sub = img - median
    threshold = threshold_sigma * std

    # DAOStarFinder API changed in photutils 3.0.
    ver = tuple(int(x) for x in photutils.__version__.split(".")[:2])
    if ver >= (3, 0):
        finder = DAOStarFinder(fwhm=fwhm, threshold=threshold,
                               sharpness_range=(0.05, 2.0), roundness_range=(-2.0, 2.0),
                               peak_max=None)
    else:
        finder = DAOStarFinder(fwhm=fwhm, threshold=threshold,
                               sharplo=0.05, sharphi=2.0, roundlo=-2.0, roundhi=2.0,
                               peakmax=None)

    table = finder(sub)

    if table is None or len(table) == 0:
        table = find_peaks(sub, threshold=threshold, box_size=7)
        if table is None or len(table) == 0:
            return np.array([]), np.array([]), np.array([])
        return (table["x_peak"].data.astype(float),
                table["y_peak"].data.astype(float),
                table["peak_value"].data.astype(float))

    x_col = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
    y_col = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"
    return (table[x_col].data.astype(float),
            table[y_col].data.astype(float),
            table["flux"].data.astype(float))


def _dedup(
    xs: np.ndarray,
    ys: np.ndarray,
    fluxes: np.ndarray,
    min_sep: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy NMS: keep brightest source in each cluster. Input must be flux-sorted."""
    if len(xs) == 0:
        return xs, ys, fluxes
    kx, ky, kf = [], [], []
    for x, y, f in zip(xs, ys, fluxes):
        if kx:
            dx, dy = np.asarray(kx) - x, np.asarray(ky) - y
            if np.min(dx * dx + dy * dy) < min_sep ** 2:
                continue
        kx.append(x); ky.append(y); kf.append(f)
    return np.asarray(kx), np.asarray(ky), np.asarray(kf)
