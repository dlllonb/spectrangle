"""
pipeline.py — End-to-end grating-angle extraction.

Phase 3 entry point: detect_traces -> [optional: catalog star masking,
masking.py] -> [optional: mask-bridged fragment reunification,
fragments.py] -> measure_trace (initial) -> combine_traces (initial
reference) -> remove_contamination (per trace, against that reference)
-> combine_traces (final) -> optional sky-frame conversion via
wcsangle.pixel_angle_to_sky_angle if a WCS is supplied.

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
from .masking import build_catalog_star_mask, recover_empirical_psf_sigma, catalog_star_pixel_positions
from .fragments import merge_mask_bridged_fragments
from .extension import extend_candidates


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
    star_catalog_ra_deg: Optional[np.ndarray] = None,
    star_catalog_dec_deg: Optional[np.ndarray] = None,
    star_catalog_mag: Optional[np.ndarray] = None,
    mask_k: float = 8.0,
    mask_radius_px: Optional[float] = None,
    mask_mag_cut: float = 13.0,
    merge_fragments: bool = True,
    extend_traces: bool = True,
    ext_sigma: float = 4.0,
    ext_cut_factor: float = 1.5,
    ext_margin_steps: float = 2.0,
    sigma_clip: bool = True,
    sigma_clip_k: float = 2.0,
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
        supplied, only the pixel-space result is returned. Also
        required (alongside the star_catalog_* arrays) to enable
        catalog star masking -- see below.
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
    star_catalog_ra_deg, star_catalog_dec_deg, star_catalog_mag : ndarray, optional
        Parallel arrays describing a star catalog for this field. If
        ALL THREE are supplied alongside `wcs`, catalog point-source
        masking (`masking.build_catalog_star_mask`) is applied before
        `detect_traces` runs -- this is the validated fix (this
        session's `new_results.txt` Entries 71-75) for two confirmed
        contamination mechanisms: chance star-chain false candidates,
        and two real traces bridged by a third star's point-source
        core. Collapses per-trace calibration from badly overconfident
        (std(z)~6-8) to honest-to-conservative (~0.3-1.5), validated
        across 18 simulated fields and real multi-realization Monte
        Carlo testing. **Known limitation, not yet fixed**: a residual,
        field-specific mean bias (~0.05-0.20 deg) remains in a
        majority of tested fields even with masking applied (Entries
        76-82) -- see `masking.py`'s module docstring. If any of the
        three arrays is None, masking is skipped entirely (backward-
        compatible default).
    mask_k : float
        Mask disk radius as a multiple of this image's own empirically
        -recovered PSF sigma (`masking.recover_empirical_psf_sigma`).
        Default 8.0 is the single fixed value validated across all 18
        test fields (Entry 75) -- do not increase without re-reading
        Entry 79 and Entry 117 first: the over-masking/tiling failure
        (Entry 79) was originally characterized at k=12/16, but a full
        18-field x 2-seed sweep (Entry 117) found it ALREADY onsets
        catastrophically at k=10 on 2 of 18 fields (fieldC to 14 deg
        error, fieldB to 2 deg) -- the safe ceiling is tighter than
        previously documented, somewhere between 8 and 10, not "roughly
        2x". A {4,5,6,7,8} sweep (Entry 117) found no pooled improvement
        over 8.0 worth switching to (k=7 ties it within noise; k=5/6 are
        non-monotonically worse, likely a real but unexplained effect,
        not just 2-seed noise -- not yet investigated).
    mask_radius_px : float, optional
        Overrides `mask_k` with an explicit pixel radius, skipping the
        empirical PSF recovery step.
    mask_mag_cut : float
        Only catalog stars at or brighter than this magnitude are
        masked. Default 13.0 matches the validated setting.
    merge_fragments : bool
        Only has an effect when catalog masking is active (see above).
        If True (default), reunify pairs of detected candidates that
        are fragments of one real trace split by a masked star's disk
        (`fragments.merge_mask_bridged_fragments`) before measurement —
        the validated fix for Entry 87's fragmentation problem: masking
        frequently splits a real trace into two pieces, shortening the
        longest/highest-weight traces up to 3.5x. Validated on the full
        18-field Monte Carlo set (Entry 90): pooled |err| drops 10.5%,
        pooled bootstrap uncertainty drops 4.4%, no catastrophic
        per-field failures. Set False to reproduce pre-Entry-90 behavior
        exactly (e.g. for regression comparison).
    extend_traces : bool
        Only has an effect when catalog masking is active (see above --
        same scoping as `merge_fragments`; never validated without
        masking). If True (default), grow each detected candidate
        outward along its own fitted axis to recover faint,
        below-detection-threshold trace material (`extension.py`) --
        the fix for `detect_traces` truncating real traces early.
        Ensemble weight is based on the PRE-extension length, not the
        extended one (see `detection.TraceCandidate.original_length_px`)
        -- validated at full 18-field x 3-seed scale (this session's
        `new_results.txt` Entries 97-115): 16/18 fields beat-or-tie
        no-extension, pooled |err| drops 39% (0.051 -> 0.039 deg). Two
        more aggressive per-trace gating strategies were tried and both
        made results WORSE than this simpler length-based fix alone --
        see `extension.py`'s module docstring before attempting a
        per-trace accept/reject gate again. Set False to reproduce
        pre-extension behavior exactly.
    ext_sigma, ext_cut_factor, ext_margin_steps : float
        Passed to `extension.extend_candidates` -- see that module's
        docstring for what each controls and why the defaults (4.0,
        1.5, 2.0) are image-measured-quantity-derived rather than
        arbitrary. Not re-tuned without a proper multi-seed-averaged
        validation (a single-seed sweep was tried and produced an
        illusory "45% better" combo that failed on held-out seeds).
    sigma_clip : bool
        Only has an effect when catalog masking is active (same scoping
        as `merge_fragments`/`extend_traces` -- validated only on top of
        the masked+fragment-merged+extended pipeline; applying it to the
        bare/unmasked path clips far too aggressively, since there's no
        upstream cleanup removing the contamination/fragmentation that
        the masked pipeline already handles -- confirmed directly:
        unconditional clipping collapsed `n_traces_used` from 24 to 5 on
        the unmasked real-image regression test). If True (default) and
        masking is active, the FINAL combine (after `remove_contamination`)
        iteratively drops traces whose axial deviation from the current
        weighted circular mean exceeds `sigma_clip_k` sigma, recombining
        each round -- see `combine.combine_traces`'s docstring for the
        algorithm and validation. Applied only to the final combine, never
        the initial one (which only feeds `remove_contamination`'s
        reference angle and was never validated with clipping). Re-
        validated against this exact shipped baseline at full 18-field x
        3-seed scale (`new_results.txt` Entry 113): 33% pooled mean|err|
        reduction, 52% tighter bootstrap uncertainty, 14/18 fields beat-
        or-tie. Two fields (fieldB, fieldC) regress mildly -- root-caused
        (Entry 114) to clipping removing part of a real, fortuitous
        error-canceling trace cluster, not a clipping malfunction; both
        stay within their own quoted uncertainty even regressed. Set
        False to reproduce pre-clip behavior exactly.
    sigma_clip_k : float
        Clip threshold in units of the current weighted circular std.
        Default 2.0 is the validated best of {2.0, 2.5, 3.0} (Entry 113).

    Returns
    -------
    AngleExtractionResult
    """
    config = dict(min_length_px=min_length_px, min_eccentricity=min_eccentricity,
                  n_sigma=n_sigma, bg_sigma=bg_sigma, smooth_sigma=smooth_sigma,
                  bump_k=bump_k, n_boot=n_boot, seed=seed,
                  sigma_clip=sigma_clip, sigma_clip_k=sigma_clip_k)

    exclude_mask = None
    mask_star_x, mask_star_y, mask_radius_used = None, None, None
    catalog_supplied = (star_catalog_ra_deg is not None and star_catalog_dec_deg is not None
                         and star_catalog_mag is not None)
    if catalog_supplied and wcs is not None:
        radius_px = mask_radius_px if mask_radius_px is not None else mask_k * recover_empirical_psf_sigma(image)
        exclude_mask = build_catalog_star_mask(
            image.shape, wcs, star_catalog_ra_deg, star_catalog_dec_deg, star_catalog_mag,
            radius_px=radius_px, mag_cut=mask_mag_cut,
        )
        config.update(mask_radius_px=radius_px, mask_mag_cut=mask_mag_cut)
        mask_radius_used = radius_px
        if merge_fragments:
            mask_star_x, mask_star_y = catalog_star_pixel_positions(
                image.shape, wcs, star_catalog_ra_deg, star_catalog_dec_deg, star_catalog_mag,
                radius_px=radius_px, mag_cut=mask_mag_cut,
            )

    candidates, n_raw, threshold = detect_traces(
        image, bg_sigma=bg_sigma, smooth_sigma=smooth_sigma, n_sigma=n_sigma,
        min_length_px=min_length_px, min_eccentricity=min_eccentricity,
        exclude_mask=exclude_mask,
    )
    n_detected = len(candidates)
    if merge_fragments and mask_star_x is not None:
        candidates = merge_mask_bridged_fragments(candidates, mask_star_x, mask_star_y, mask_radius_used)
    if extend_traces and mask_radius_used is not None:
        candidates = extend_candidates(
            candidates, image, exclude_mask, mask_radius_used,
            bg_sigma=bg_sigma, smooth_sigma=smooth_sigma, n_sigma=n_sigma,
            ext_sigma=ext_sigma, cut_factor=ext_cut_factor, margin_steps=ext_margin_steps,
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
    # scoped like merge_fragments/extend_traces: only validated (Entry 113)
    # on top of the masked+fragment-merged+extended pipeline -- applying it
    # to the bare/unmasked path clips far too aggressively (regressed
    # tests/test_pipeline_real.py, tests/test_pipeline_sim.py, both
    # unmasked, when tried unconditionally).
    apply_sigma_clip = sigma_clip and mask_radius_used is not None
    final_combined = combine_traces(
        final_measurements, n_boot=n_boot, seed=seed,
        sigma_clip=apply_sigma_clip, sigma_clip_k=sigma_clip_k,
    )

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
        n_traces_detected=n_detected, n_traces_used=final_combined.n_used,
        quality=final_combined.quality,
        combined=final_combined, initial_combined=initial_combined, config=config,
    )
