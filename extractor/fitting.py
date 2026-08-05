"""
fitting.py — Per-trace line fitting and contamination removal.

Phase 3 (spectral trace angle extraction): fits a pixel-space direction
to each `detection.TraceCandidate`, and removes star contamination that
landed on a trace.

Two-stage design, validated in project notebooks 06 through 13:

1. `measure_trace` -- a plain intensity-weighted PCA line fit through all
   of a candidate's pixels. Axial (180-degree symmetric): never average
   raw angle values, see `combine.py`.

2. `remove_contamination` -- reprojects a trace's pixels into a frame
   aligned with an EXTERNAL reference angle (the current best ensemble
   estimate, not the trace's own fit), and strips per-pixel excursions
   beyond a robust, iteratively-estimated width around the (should-be-
   flat) median. This is deliberately not self-referential: a trace's
   own local fit can be dragged by a large contaminating clump and then
   "successfully" pass a sigma-clip against itself. Anchoring against an
   independent reference -- justified because all traces in a field share
   one true grating angle, a real physical fact, not an assumption being
   smuggled in -- lets a star sitting on/near a trace be identified by a
   real, independent structural signature (a localized excursion) rather
   than by whether it agrees with the answer.

Both per-trace angular uncertainty (`trace_correlation_length_px` /
`analytical_sigma_theta_deg`) use a Bevington-style straight-line-fit
formula. N_eff (effective independent samples along a trace) is derived
from the trace's own length divided by the pipeline's effective
pixel-to-pixel correlation length -- which, at the plate scales this
project has tested, is set by detection.py's own smoothing/downsampling,
not the telescope's optical resolution (project notebook 13).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detection import TraceCandidate, downsample_factor


# ---------------------------------------------------------------------------
# Angle arithmetic (axial: traces have 180-degree symmetry)
# ---------------------------------------------------------------------------

def norm_axial(angle_deg: float) -> float:
    """Wrap an angle to the axial range (-90, 90]."""
    a = float(angle_deg) % 180.0
    return a - 180.0 if a > 90.0 else a


def axial_diff(a_deg: float, b_deg: float) -> float:
    """Signed axial difference a-b, wrapped to (-90, 90]."""
    return float((a_deg - b_deg + 90.0) % 180.0 - 90.0)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TraceMeasurement:
    """A single trace's fitted direction and quality/uncertainty info.

    Attributes
    ----------
    x_center, y_center : float
        Intensity-weighted centroid (post-contamination-removal if
        `remove_contamination` has run; pre-removal otherwise).
    theta_pix_deg : float
        Pixel-space direction, degrees CCW from +x, origin='lower',
        axial (-90, 90].
    theta_pix_uncertainty_deg : float
        Per-trace analytical angular uncertainty (Bevington-style
        straight-line-fit formula). Population-level, not yet the
        ensemble uncertainty -- see `combine.py` for that.
    length_px : float
        Along-trace extent of the pixels actually used for the fit.
    perp_std_px : float
        Perpendicular (cross-trace) pixel scatter -- both the physical
        trace width and any residual noise.
    n_eff : float
        Effective independent sample count along the trace, used to
        derive theta_pix_uncertainty_deg.
    eccentricity : float
    total_flux : float
    weight : float
        `(length_px / perp_std_px) ** 2` -- an inverse-variance-like
        proxy for this trace's precision, validated (project notebook
        10) as a better ensemble weight than raw flux: it lets a long,
        thin, well-constrained trace dominate the combined result without
        discarding the rest, consistent with what the analytical
        uncertainty formula itself implies (weight ~ 1/sigma_theta^2).
    n_pixels_removed : int
        Pixels stripped as contamination by `remove_contamination` (0 if
        it hasn't run, or nothing was removed).
    quality_flags : list[str]
    """
    x_center: float
    y_center: float
    theta_pix_deg: float
    theta_pix_uncertainty_deg: float
    length_px: float
    perp_std_px: float
    n_eff: float
    eccentricity: float
    total_flux: float
    weight: float
    n_pixels_removed: int = 0
    quality_flags: list[str] = field(default_factory=list)

    # kept for remove_contamination / diagnostics -- the pixels this
    # measurement was actually fit from (post-removal if applicable)
    _rows: Optional[np.ndarray] = field(default=None, repr=False)
    _cols: Optional[np.ndarray] = field(default=None, repr=False)
    _weights: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Line fit
# ---------------------------------------------------------------------------

def pca_fit(rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Weighted PCA line fit through a pixel cloud.

    Returns
    -------
    angle_deg : float
        CCW from +x, origin='lower' convention, axial (-90, 90].
    minor_std_px, major_std_px : float
        sqrt of the covariance eigenvalues -- minor is the perpendicular
        ("how fat is this trace") spread, major is the along-trace
        spread.
    """
    w = weights / weights.sum()
    mr = np.dot(w, rows); mc = np.dot(w, cols)
    dr = rows - mr; dc = cols - mc
    cov = np.array([[np.dot(w, dc * dc), np.dot(w, dc * dr)],
                     [np.dot(w, dc * dr), np.dot(w, dr * dr)]])
    evals, evecs = np.linalg.eigh(cov)  # ascending
    v = evecs[:, 1]
    angle = norm_axial(np.degrees(np.arctan2(v[1], v[0])))
    return angle, float(np.sqrt(max(evals[0], 0))), float(np.sqrt(max(evals[1], 0)))


def trace_correlation_length_px(image_shape: tuple[int, int], smooth_sigma: float = 2.5) -> float:
    """Effective pixel-to-pixel correlation length of the data a trace
    was fit from -- i.e. what N_eff should divide length by.

    Not the telescope's optical resolution: at the plate scales this
    project has tested, the true PSF is sub-pixel, so detection.py's own
    smoothing (smooth_sigma) and internal downsampling (see
    `detection.downsample_factor`) are what actually set the correlation
    length in the fitted data (project notebook 13). Must be called with
    the same smooth_sigma actually used in `detection.detect_traces`.
    """
    return smooth_sigma * downsample_factor(image_shape)


def analytical_sigma_theta_deg(length_px, perp_std_px, n_eff) -> np.ndarray:
    """Per-trace angular uncertainty, Bevington-style straight-line-fit
    formula:

        sigma_theta ~ sqrt(12 / N_eff) * sigma_perp / L

    (radians for L, sigma_perp in the same units; returned in degrees).
    N_eff < 2 is treated as unconstrained (returns inf).
    """
    length_px = np.asarray(length_px, dtype=float)
    perp_std_px = np.asarray(perp_std_px, dtype=float)
    n_eff = np.asarray(n_eff, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_rad = np.sqrt(12.0 / np.maximum(n_eff, 1e-9)) * perp_std_px / np.maximum(length_px, 1e-9)
    sigma_rad = np.where(n_eff < 2, np.inf, sigma_rad)
    return np.degrees(sigma_rad)


def measure_trace(candidate: TraceCandidate, effective_resolution_px: float) -> TraceMeasurement:
    """Plain PCA fit of one candidate, with analytical uncertainty --
    no contamination removal yet (see `remove_contamination`)."""
    angle, minor_std, major_std = pca_fit(candidate.rows, candidate.cols, candidate.weights)
    n_eff = candidate.length_px / max(effective_resolution_px, 1e-9)
    sigma = float(analytical_sigma_theta_deg(candidate.length_px, minor_std, n_eff))
    weight = (candidate.length_px / max(minor_std, 1e-6)) ** 2
    return TraceMeasurement(
        x_center=candidate.x_center, y_center=candidate.y_center,
        theta_pix_deg=angle, theta_pix_uncertainty_deg=sigma,
        length_px=candidate.length_px, perp_std_px=minor_std, n_eff=n_eff,
        eccentricity=candidate.eccentricity, total_flux=candidate.total_flux,
        weight=weight, n_pixels_removed=0, quality_flags=[],
        _rows=candidate.rows, _cols=candidate.cols, _weights=candidate.weights,
    )


# ---------------------------------------------------------------------------
# Global-referenced contamination removal
# ---------------------------------------------------------------------------

def _trace_local_frame(rows, cols, weights, ref_angle_deg):
    """Project pixel coords into (s, d) relative to ref_angle_deg (not
    necessarily this trace's own fitted angle): s = position along the
    reference direction, d = perpendicular distance, both from the
    intensity-weighted centroid. A trace whose geometry agrees exactly
    with ref_angle_deg has all pixels at d~0."""
    w = weights / weights.sum()
    mc = np.dot(w, cols); mr = np.dot(w, rows)
    th = np.radians(ref_angle_deg)
    ux, uy = np.cos(th), np.sin(th)
    px, py = -uy, ux
    dc = cols - mc; dr = rows - mr
    s = dc * ux + dr * uy
    d = dc * px + dr * py
    return s, d


def remove_contamination(
    measurement: TraceMeasurement,
    reference_angle_deg: float,
    effective_resolution_px: float,
    bump_k: float = 2.5,
    min_width_px: float = 0.5,
    max_iter: int = 8,
) -> TraceMeasurement:
    """Strip star contamination from one trace, anchored against an
    external reference angle (typically the current ensemble estimate),
    and return a NEW TraceMeasurement refit from the surviving pixels.

    Algorithm: reproject to (s, d) vs reference_angle_deg, then flag any
    pixel beyond +/- bump_k * (robust width) from a single robust global
    median d. Iterated -- the median/width are recomputed from only the
    currently-kept pixels each round, so once a contaminating clump is
    excluded it stops influencing the estimate on the next round (a wide
    clump would otherwise inflate its own local width estimate and defeat
    a one-shot version of this check; validated by inspection across all
    traces in project notebook 12, not just a curated sample).

    Parameters
    ----------
    measurement : TraceMeasurement
        Must carry its source pixels (i.e. come from `measure_trace`,
        not reconstructed without _rows/_cols/_weights).
    reference_angle_deg : float
        External reference direction, e.g. the ensemble angle from an
        initial `combine.combine_traces` pass.
    effective_resolution_px : float
        Passed straight through to the refit's analytical uncertainty --
        see `trace_correlation_length_px`.
    bump_k : float
        Pixels beyond bump_k robust-widths from the median are flagged.
    """
    rows, cols, weights = measurement._rows, measurement._cols, measurement._weights
    if rows is None:
        raise ValueError("measurement has no source pixels attached (was it built by measure_trace?)")

    s, d = _trace_local_frame(rows, cols, weights, reference_angle_deg)

    mask = np.ones(len(s), dtype=bool)
    median_d = float(np.median(d))
    width_px = max(1.4826 * float(np.median(np.abs(d - median_d))), min_width_px)
    for _ in range(max_iter):
        dv = d[mask]
        median_d = float(np.median(dv))
        width_px = max(1.4826 * float(np.median(np.abs(dv - median_d))), min_width_px)
        new_keep = np.abs(d - median_d) <= bump_k * width_px
        if new_keep.sum() < 4:
            new_keep = mask  # don't over-clip into oblivion
            break
        if np.array_equal(new_keep, mask):
            mask = new_keep
            break
        mask = new_keep

    keep = mask
    n_removed = int((~keep).sum())
    flags = list(measurement.quality_flags)

    if 4 <= keep.sum() < len(keep):
        angle, minor_std, _ = pca_fit(rows[keep], cols[keep], weights[keep])
        s_kept = s[keep]
        length_px = float(s_kept.max() - s_kept.min())
        if n_removed > 0:
            flags = flags + ["contamination_removed"]
    else:
        angle, minor_std, length_px = measurement.theta_pix_deg, measurement.perp_std_px, measurement.length_px
        keep = np.ones(len(s), dtype=bool)
        n_removed = 0

    n_eff = length_px / max(effective_resolution_px, 1e-9)
    sigma = float(analytical_sigma_theta_deg(length_px, minor_std, n_eff))
    weight = (length_px / max(minor_std, 1e-6)) ** 2

    return TraceMeasurement(
        x_center=float(cols[keep].mean()), y_center=float(rows[keep].mean()),
        theta_pix_deg=angle, theta_pix_uncertainty_deg=sigma,
        length_px=length_px, perp_std_px=minor_std, n_eff=n_eff,
        eccentricity=measurement.eccentricity, total_flux=float(weights[keep].sum()),
        weight=weight, n_pixels_removed=n_removed, quality_flags=flags,
        _rows=rows[keep], _cols=cols[keep], _weights=weights[keep],
    )
