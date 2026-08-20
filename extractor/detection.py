"""
detection.py — Diffraction/spectral trace candidate detection.

Phase 3 (spectral trace angle extraction): given a raw image, finds
candidate elongated trace segments -- threshold at a sigma-clipped
background level, label connected components, and keep only components
long AND elongated enough to plausibly be a piece of a diffraction trace
rather than a star core or noise clump.

This returns *candidates*, not final measurements -- each candidate still
needs a per-trace line fit and contamination removal (fitting.py) before
its angle is trustworthy.

Both filters were tuned empirically against known-truth simulated data and
cross-checked on a real image (project notebooks 02/03/05/06):
- Length alone lets short, statistically-unreliable "line-like" noise
  clumps through -- a handful of pixels can look elongated by chance.
- Eccentricity alone lets a long-but-round multi-star clump through, and
  is unreliable at low pixel counts.
Combined, neither failure mode passes alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops


@dataclass
class TraceCandidate:
    """One connected-component candidate from thresholding, before any
    per-trace fitting or contamination removal.

    Attributes
    ----------
    rows, cols, weights : ndarray
        Pixel coordinates (0-based; row=y, col=x) and intensity of every
        pixel in this candidate, in ORIGINAL image coordinates (already
        scaled back up from any internal downsampling).
    length_px : float
        Connected-component major-axis length, in original-image pixels.
    eccentricity : float
        `skimage.measure.regionprops` eccentricity of the component.
    n_pixels : int
    total_flux : float
    x_center, y_center : float
        Intensity-weighted centroid.
    original_length_px : float, optional
        Set only by `extension.extend_candidates` -- the candidate's
        `length_px` BEFORE the trace-extension walker ran, i.e. the
        length actually validated by ordinary connected-component
        detection. None for a candidate that was never extended.
        `fitting.measure_trace`/`remove_contamination` use this (when
        present) as the basis for ensemble WEIGHT instead of the
        (potentially extension-inflated) `length_px` -- see
        `extension.py`'s module docstring for why: extension genuinely
        improves the ANGLE estimate but its length gain isn't reliable
        enough to also inflate this trace's influence over the field
        combine, validated at full 18-field x 3-seed scale (this
        session's `new_results.txt` Entries 101-115).
    """
    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray
    length_px: float
    eccentricity: float
    n_pixels: int
    total_flux: float
    x_center: float
    y_center: float
    original_length_px: Optional[float] = None


def downsample_factor(image_shape: tuple[int, int], max_side: int = 1500) -> int:
    """The internal downsample factor `detect_traces` uses for a given
    image shape, exposed so `fitting.py` can derive the correct effective
    pixel-to-pixel correlation length for whatever image size was
    actually processed (see `fitting.trace_correlation_length_px`)."""
    return max(1, max(image_shape) // max_side)


def compute_background_residual(
    image: np.ndarray,
    bg_sigma: float = 50.0,
    smooth_sigma: float = 2.5,
    n_sigma: float = 6.0,
    exclude_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, float, float]:
    """Background-subtracted, gap-bridged residual and its sigma-clipped
    detection threshold -- the exact preprocessing `detect_traces` uses
    internally, factored out so `extension.py` can walk the SAME
    full-resolution residual (and use the SAME median/std/threshold) a
    trace was originally detected against, rather than recomputing a
    subtly different one.

    Returns
    -------
    resid : ndarray
        Full-resolution (not internally downsampled) background-
        subtracted, smoothed residual.
    median, std : float
        Sigma-clipped background statistics (computed on the internally
        downsampled residual, matching `detect_traces`).
    threshold : float
        `median + n_sigma * std`.
    """
    img = image.astype(np.float32)
    if exclude_mask is not None:
        img = img.copy()
        img[exclude_mask] = np.nan
        bg = gaussian_filter(np.nan_to_num(img, nan=np.nanmedian(img)), sigma=bg_sigma)
    else:
        bg = gaussian_filter(img, sigma=bg_sigma)
    resid = np.clip(image.astype(np.float32) - bg, 0.0, None)
    if exclude_mask is not None:
        resid[exclude_mask] = 0.0
    if smooth_sigma > 0:
        resid = gaussian_filter(resid, sigma=smooth_sigma)

    factor = downsample_factor(resid.shape)
    small = resid[::factor, ::factor] if factor > 1 else resid
    _, median, std = sigma_clipped_stats(small, sigma=3.0, maxiters=5)
    threshold = median + n_sigma * std
    return resid, float(median), float(std), float(threshold)


def detect_traces(
    image: np.ndarray,
    bg_sigma: float = 50.0,
    smooth_sigma: float = 2.5,
    n_sigma: float = 6.0,
    min_length_px: float = 25.0,
    min_eccentricity: float = 0.85,
    exclude_mask: Optional[np.ndarray] = None,
    background: Optional[tuple] = None,
) -> tuple[list[TraceCandidate], int, float]:
    """Detect candidate trace segments in a single image.

    Parameters
    ----------
    image : ndarray
        2-D image array (raw counts/electrons; not background-subtracted).
    bg_sigma : float
        Sigma of the large-scale Gaussian used to estimate and subtract
        sky background before thresholding.
    smooth_sigma : float
        Sigma of a small Gaussian smooth applied after background
        subtraction, to bridge small gaps along a trace. This smoothing,
        combined with the internal downsampling below, sets the effective
        pixel-to-pixel correlation length of the data that
        `fitting.trace_correlation_length_px` derives per-trace angular
        uncertainty from -- keep the two in sync if you change this.
    n_sigma : float
        Detection threshold, in units of the sigma-clipped background
        std, above the sigma-clipped median.
    min_length_px : float
        Minimum connected-component major-axis length (original-image
        pixels) to keep. This is the dominant, image-dependent knob --
        a clean, long-trace simulation and a faint real image do not
        share one good value. Use `sweep_min_length` to pick one for a
        new image rather than assuming this default transfers.
    min_eccentricity : float
        Minimum eccentricity to keep. 0.85 held up across both the
        validation simulation and the real test image without retuning
        (unlike min_length_px), since it's scale-free.
    exclude_mask : ndarray, optional
        Boolean array, same shape as `image`, marking pixels to zero out
        before detection (e.g. known bad pixels). None disables masking.
    background : tuple, optional
        Pre-computed `(resid, median, std, threshold)` from
        `compute_background_residual`, e.g. from a caller that also
        needs it for `extension.extend_candidates` and wants to avoid
        computing the same O(image-size) Gaussian-filtered residual
        twice (Entry 122 finding 8) -- must have been computed with the
        SAME bg_sigma/smooth_sigma/n_sigma/exclude_mask this call would
        otherwise use itself, that's the caller's responsibility. None
        (default) computes it fresh here, exactly as before.

    Returns
    -------
    candidates : list[TraceCandidate]
    n_raw_components : int
        Total connected components found before length/eccentricity
        filtering (diagnostic only).
    threshold : float
        The sigma-clipped detection threshold actually used.
    """
    if background is not None:
        resid, median, std, threshold = background
    else:
        resid, median, std, threshold = compute_background_residual(
            image, bg_sigma=bg_sigma, smooth_sigma=smooth_sigma, n_sigma=n_sigma, exclude_mask=exclude_mask,
        )
    factor = downsample_factor(resid.shape)
    small = resid[::factor, ::factor] if factor > 1 else resid
    labeled = label(small >= threshold)

    candidates: list[TraceCandidate] = []
    for reg in regionprops(labeled, intensity_image=small):
        length_px = float(reg.axis_major_length) * factor
        if length_px < min_length_px:
            continue
        if reg.eccentricity < min_eccentricity:
            continue
        coords = reg.coords
        rows = coords[:, 0].astype(float) * factor
        cols = coords[:, 1].astype(float) * factor
        weights = small[coords[:, 0], coords[:, 1]].astype(float)
        candidates.append(TraceCandidate(
            rows=rows, cols=cols, weights=weights,
            length_px=length_px, eccentricity=float(reg.eccentricity),
            n_pixels=len(rows), total_flux=float(weights.sum()),
            x_center=float(cols.mean()), y_center=float(rows.mean()),
        ))
    return candidates, int(labeled.max()), float(threshold)


def sweep_min_length(
    image: np.ndarray,
    length_values: list[float],
    **detect_kwargs,
) -> list[dict]:
    """Map candidate count vs `min_length_px` for a new image, so a
    sensible operating point can be picked instead of assuming one
    transfers from another image (see `detect_traces`'s min_length_px
    docstring). Returns a list of dicts with min_length_px and n_candidates
    for each tested value; combine with `fitting`/`combine` downstream to
    also see the resulting angle precision at each value."""
    rows = []
    for L in length_values:
        candidates, n_raw, threshold = detect_traces(image, min_length_px=L, **detect_kwargs)
        rows.append(dict(min_length_px=L, n_candidates=len(candidates), n_raw_components=n_raw))
    return rows
