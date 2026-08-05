"""
pipeline.py — End-to-end grating-angle extraction.

Phase 3 entry point: detect_traces -> measure_trace (initial) ->
combine_traces (initial reference) -> remove_contamination (per trace,
against that reference) -> combine_traces (final) -> optional sky-frame
conversion via wcsangle.pixel_angle_to_sky_angle if a WCS is supplied.

The two-pass combine is intentional (project notebooks 09-13):
contamination removal needs an external reference angle that ISN'T the
trace's own fit (a self-referential check lets a subtly-wrong trace
validate itself), so an initial ensemble estimate has to exist before
per-trace cleanup can run, and the final combine re-weights and
re-combines the cleaned traces.

sigma_WCS is NOT computed here -- pass it in via sigma_wcs_deg if you have
it (e.g. from a WCS-orientation bootstrap; not yet implemented as
reusable code in this package -- see `measure_grating_angle`'s
docstring). Without it, only theta_sky_deg is returned, not a total
sky-frame uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.wcs import WCS

from .detection import detect_traces
from .fitting import measure_trace, remove_contamination, trace_correlation_length_px
from .combine import combine_traces, CombinedAngleResult
from .wcsangle import pixel_angle_to_sky_angle


@dataclass
class AngleExtractionResult:
    """End-to-end result of `measure_grating_angle`.

    Attributes
    ----------
    theta_pix_deg, theta_pix_uncertainty_deg : float
        Combined pixel-space grating angle and its bootstrap uncertainty
        (see `combine.py`'s module docstring for why bootstrap, not
        population scatter, is used here).
    theta_sky_deg : float or None
        East-of-north sky position angle, if a WCS was supplied; None
        otherwise. AXIAL, like theta_pix_deg (grating traces have 180 deg
        symmetry) -- this is theta and theta+180 indistinguishably, not
        an unambiguous 0-360 deg vector direction. Range (-180, 180].
    theta_sky_uncertainty_deg : float or None
        theta_pix_uncertainty_deg combined in quadrature with
        sigma_wcs_deg, only if BOTH a WCS and sigma_wcs_deg were
        supplied. Deliberately None (not silently approximated with
        theta_pix_uncertainty_deg alone) if sigma_wcs_deg is missing.
    n_traces_detected, n_traces_used : int
    quality : float
        Resultant length R in [0, 1] of the final combine.
    combined : CombinedAngleResult
        The full final-combine result (population scatter, analytical
        sigma, and the trace measurements themselves).
    initial_combined : CombinedAngleResult
        The pre-contamination-removal combine, kept for diagnostics.
    config : dict
        The parameters actually used, for reproducibility.
    """
    theta_pix_deg: float
    theta_pix_uncertainty_deg: float
    theta_sky_deg: Optional[float]
    theta_sky_uncertainty_deg: Optional[float]
    n_traces_detected: int
    n_traces_used: int
    quality: float
    combined: CombinedAngleResult
    initial_combined: CombinedAngleResult
    config: dict = field(default_factory=dict)


def measure_grating_angle(
    image: np.ndarray,
    wcs: Optional[WCS] = None,
    sigma_wcs_deg: Optional[float] = None,
    min_length_px: float = 25.0,
    min_eccentricity: float = 0.85,
    n_sigma: float = 6.0,
    bg_sigma: float = 50.0,
    smooth_sigma: float = 2.5,
    bump_k: float = 2.5,
    n_boot: int = 3000,
    seed: int = 42,
) -> AngleExtractionResult:
    """Measure the grating/diffraction-trace orientation angle in an
    image, optionally converted to a sky-frame position angle.

    Parameters
    ----------
    image : ndarray
        2-D image array (raw counts/electrons).
    wcs : astropy.wcs.WCS, optional
        Plate-solved WCS. If supplied, theta_sky_deg is computed via
        `wcsangle.pixel_angle_to_sky_angle` at the image center. If not
        supplied, only the pixel-space result is returned.
    sigma_wcs_deg : float, optional
        WCS-orientation uncertainty. NOT computed by this package -- the
        WCS-orientation bootstrap approach used earlier in this project
        exists only in exploratory notebooks, not as reusable
        `extractor` code yet. If supplied alongside `wcs`,
        theta_sky_uncertainty_deg is theta_pix_uncertainty_deg and
        sigma_wcs_deg combined in quadrature; otherwise it's None.
    min_length_px, min_eccentricity, n_sigma, bg_sigma, smooth_sigma :
        Passed to `detection.detect_traces`. min_length_px is the most
        image-dependent knob -- a clean, long-trace simulation and a
        faint real image do not share one good value. Consider
        `detection.sweep_min_length` for a new image rather than
        assuming this default transfers.
    bump_k : float
        Passed to `fitting.remove_contamination`.
    n_boot, seed :
        Passed to `combine.combine_traces`'s bootstrap uncertainty.

    Returns
    -------
    AngleExtractionResult
    """
    config = dict(min_length_px=min_length_px, min_eccentricity=min_eccentricity,
                  n_sigma=n_sigma, bg_sigma=bg_sigma, smooth_sigma=smooth_sigma,
                  bump_k=bump_k, n_boot=n_boot, seed=seed)

    candidates, n_raw, threshold = detect_traces(
        image, bg_sigma=bg_sigma, smooth_sigma=smooth_sigma, n_sigma=n_sigma,
        min_length_px=min_length_px, min_eccentricity=min_eccentricity,
    )

    eff_res_px = trace_correlation_length_px(image.shape, smooth_sigma=smooth_sigma)

    if not candidates:
        empty = combine_traces([], n_boot=n_boot, seed=seed)
        return AngleExtractionResult(
            theta_pix_deg=np.nan, theta_pix_uncertainty_deg=np.nan,
            theta_sky_deg=None, theta_sky_uncertainty_deg=None,
            n_traces_detected=0, n_traces_used=0, quality=np.nan,
            combined=empty, initial_combined=empty, config=config,
        )

    initial_measurements = [measure_trace(c, eff_res_px) for c in candidates]
    initial_combined = combine_traces(initial_measurements, n_boot=n_boot, seed=seed)

    final_measurements = [
        remove_contamination(m, initial_combined.theta_pix_deg, eff_res_px, bump_k=bump_k)
        for m in initial_measurements
    ]
    final_combined = combine_traces(final_measurements, n_boot=n_boot, seed=seed)

    theta_sky_deg = None
    theta_sky_uncertainty_deg = None
    if wcs is not None and np.isfinite(final_combined.theta_pix_deg):
        ny, nx = image.shape
        theta_sky_deg = pixel_angle_to_sky_angle(
            wcs, (nx - 1) / 2.0, (ny - 1) / 2.0, final_combined.theta_pix_deg,
        )
        if sigma_wcs_deg is not None:
            theta_sky_uncertainty_deg = float(np.hypot(
                final_combined.theta_pix_uncertainty_deg, sigma_wcs_deg,
            ))

    return AngleExtractionResult(
        theta_pix_deg=final_combined.theta_pix_deg,
        theta_pix_uncertainty_deg=final_combined.theta_pix_uncertainty_deg,
        theta_sky_deg=theta_sky_deg, theta_sky_uncertainty_deg=theta_sky_uncertainty_deg,
        n_traces_detected=len(candidates), n_traces_used=final_combined.n_used,
        quality=final_combined.quality,
        combined=final_combined, initial_combined=initial_combined, config=config,
    )
