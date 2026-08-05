"""
wcs_uncertainty.py — Empirical uncertainty on the WCS orientation (the
"north angle" used by wcsangle.py to convert pixel-space trace angles to
sky position angles), via a spatially-stratified bootstrap over
independent plate-solve re-solves.

This is sigma_WCS -- the piece `pipeline.measure_grating_angle` accepts
as `sigma_wcs_deg` but does not itself compute (see that module's
docstring). The method here was validated pre-`extractor` in exploratory
notebooks (`notebooks/archive/12_wcs_orientation_diagnostics.ipynb` and
neighbors): re-solving on the full source list gives one fiducial WCS;
re-solving many times on random SUBSETS of the same source list gives a
distribution of independently-derived `north_angle_deg` values, whose
scatter around the fiducial is the empirical orientation uncertainty.
Subsets are drawn **stratified** across a spatial grid (not uniformly at
random) so a bootstrap draw can't collapse to sources bunched in one
corner of the field, which would otherwise bias the derived WCS toward a
poorly-constrained rotation. The stratified approach measurably
outperformed (tighter, more stable across bootstrap fraction) a naive
random-subset bootstrap in the archived exploratory work.

This is slow (each bootstrap draw is a full plate-solve) and depends on
either a local `solve-field` install or network access to
nova.astrometry.net -- unlike the fast, dependency-light trace-angle code
in detection.py/fitting.py/combine.py. Expect ~1-30s per solve depending
on source count and backend; a default n_boot=50 run can take several
minutes.

Known failure mode (found empirically on the `110lmm50mm.fits` sim
image, cross-checked against that image's own injected ground-truth
rotation): astrometry.net occasionally locks onto a spuriously
high-confidence but WRONG quad match when given the *full* source list,
while most stratified subsets (missing just the handful of sources
responsible for the false match) correctly solve to the true answer. In
that case the "fiducial" full-list solve is itself the outlier, not the
bootstrap population -- trusting it unconditionally as the reference
would silently corrupt sigma_WCS and the final sky-angle. To guard
against this, the fiducial is cross-checked against the bootstrap
population's own median: if it disagrees by more than the bootstrap
scatter can explain (see `fiducial_is_outlier` below), the reported
`north_angle_deg`/`header` fall back to the bootstrap median / the
bootstrap draw closest to it, and the discrepancy is surfaced via
`fiducial_is_outlier` and `fiducial_offset_deg` rather than silently
swallowed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from .platesolve import solve_plate, PlatesolveResult
from .wcsangle import center_wcs_angle_metrics, angle_diff_deg


@dataclass
class WcsBootstrapResult:
    """Result of `bootstrap_wcs_orientation`.

    Attributes
    ----------
    north_angle_deg : float
        Recommended point estimate for the WCS orientation -- this is
        normally the fiducial (full-source-list) solve, but falls back
        to the bootstrap median if the fiducial is flagged as an outlier
        (see `fiducial_is_outlier`). Use this, not `fiducial_north_deg`,
        as "the" north angle.
    sigma_north_deg : float
        Standard deviation of bootstrap `north_angle_deg` values around
        `north_angle_deg` -- this is sigma_WCS, the number to pass as
        `pipeline.measure_grating_angle`'s `sigma_wcs_deg`.
    mad_north_deg : float
        Median absolute deviation around `north_angle_deg` (more robust
        to a few bad re-solves).
    header : astropy.io.fits.Header
        WCS header consistent with `north_angle_deg` -- a caller can
        build the WCS to pass into `measure_grating_angle` from this
        without re-solving. Normally the fiducial header; if the
        fiducial is an outlier, this is instead the header of whichever
        bootstrap draw landed closest to the bootstrap median.
    fiducial_north_deg : float
        The RAW north angle from the full-source-list solve, before any
        outlier check -- kept for diagnostics even when not used as the
        point estimate.
    fiducial_header : astropy.io.fits.Header
        The RAW full-source-list solve's WCS header (diagnostics only --
        use `header` for downstream WCS construction).
    fiducial_is_outlier : bool
        True if the fiducial solve disagreed with the bootstrap
        population by more than the population's own scatter can
        explain (repeatedly observed on sim data: astrometry.net can
        lock onto a spuriously confident but wrong match given the full
        source list -- see module docstring). When True, `north_angle_deg`
        and `header` are the bootstrap-median fallback, not the fiducial.
    fiducial_offset_deg : float
        Signed `fiducial_north_deg - north_angle_deg`, i.e. how far the
        raw fiducial sat from the (possibly-fallback) reported estimate.
        Near zero when `fiducial_is_outlier` is False.
    n_boot_requested, n_boot_ok : int
    north_angles_deg : ndarray
        Every successful bootstrap draw's north_angle_deg, UNFILTERED (for
        histogramming/diagnostics -- includes any draws excluded from
        `sigma_north_deg`/`mad_north_deg` by the per-draw outlier clip;
        see `bootstrap_kept_mask`).
    bootstrap_kept_mask : ndarray of bool
        Same length/order as `north_angles_deg`. False for any individual
        bootstrap draw that was itself a large outlier relative to the
        rest of the population (the same astrometry.net false-match
        failure mode that can hit the fiducial can independently hit a
        bootstrap subset -- see module docstring) and was therefore
        excluded from `sigma_north_deg`/`mad_north_deg` so one bad re-
        solve can't inflate the reported scatter by orders of magnitude.
    n_boot_clipped : int
        `sum(~bootstrap_kept_mask)` -- how many otherwise-successful
        draws were excluded as individual outliers. Non-zero is worth a
        second look even if small, since it's evidence the platesolve
        backend is not fully reliable for this source list.
    failures : list[str]
        Short reasons for any bootstrap draws that didn't produce a
        usable solve (solve failure, exception, etc.) -- if this is a
        large fraction of n_boot_requested, treat sigma_north_deg with
        suspicion (see this module's docstring on sim-data reliability).
    """
    north_angle_deg: float
    sigma_north_deg: float
    mad_north_deg: float
    header: fits.Header
    fiducial_north_deg: float
    fiducial_header: fits.Header
    fiducial_is_outlier: bool
    fiducial_offset_deg: float
    n_boot_requested: int
    n_boot_ok: int
    north_angles_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    bootstrap_kept_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    n_boot_clipped: int = 0
    failures: list = field(default_factory=list)


def _make_wcs(header: fits.Header) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        warnings.filterwarnings("ignore", message=".*datfix.*")
        return WCS(header)


def stratified_bootstrap_indices(
    xs: np.ndarray,
    ys: np.ndarray,
    image_shape: tuple[int, int],
    n_grid: tuple[int, int] = (3, 3),
    frac: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw a random subset of source indices, stratified across an
    n_grid[0] x n_grid[1] spatial grid over the image -- so a bootstrap
    draw keeps roughly even spatial coverage instead of risking a subset
    bunched in one region (which biases the derived rotation/orientation
    toward whatever that region's own solve constrains best)."""
    if rng is None:
        rng = np.random.default_rng()
    ny, nx = image_shape
    ngx, ngy = n_grid
    cx = np.floor(np.asarray(xs) / nx * ngx).astype(int).clip(0, ngx - 1)
    cy = np.floor(np.asarray(ys) / ny * ngy).astype(int).clip(0, ngy - 1)
    cell_id = cx + cy * ngx
    idx_parts = []
    for cid in range(ngx * ngy):
        members = np.where(cell_id == cid)[0]
        if len(members) == 0:
            continue
        n_take = max(1, int(round(len(members) * frac)))
        idx_parts.append(rng.choice(members, size=n_take, replace=False))
    return np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)


def bootstrap_wcs_orientation(
    xs: np.ndarray,
    ys: np.ndarray,
    image_shape: tuple[int, int],
    original_header: Optional[fits.Header] = None,
    n_boot: int = 50,
    boot_frac: float = 0.95,
    n_grid: tuple[int, int] = (3, 3),
    backend: str = "local",
    backend_config: Optional[str] = None,
    scale_units: str = "arcsecperpix",
    scale_low: Optional[float] = None,
    scale_high: Optional[float] = None,
    tweak_order: int = 5,
    timeout: int = 120,
    seed: int = 42,
    outlier_z: float = 5.0,
    outlier_floor_deg: float = 0.05,
    solve_fn: Optional[Callable[[np.ndarray, np.ndarray], Optional[PlatesolveResult]]] = None,
) -> WcsBootstrapResult:
    """Empirical WCS-orientation uncertainty via spatially-stratified
    bootstrap re-solving (see module docstring for the method).

    Parameters
    ----------
    xs, ys : array-like
        Full detected-source pixel positions (e.g. from
        `extractor.extract_stars`) -- the pool bootstrap subsets are
        drawn from.
    image_shape : (ny, nx)
    original_header, backend, backend_config, scale_units, scale_low,
    scale_high, tweak_order, timeout :
        Passed straight through to `platesolve.solve_plate` for both the
        fiducial solve and every bootstrap re-solve. See that function's
        docstring.
    n_boot : int
        Number of bootstrap re-solves.
    boot_frac : float
        Fraction of each spatial grid cell's sources kept per draw.
    n_grid : (int, int)
        Spatial stratification grid, (n_cols, n_rows).
    seed : int
    outlier_z : float
        How many robust-sigma (1.4826 * bootstrap MAD) the fiducial may
        sit from the bootstrap median before it's flagged as an outlier
        and replaced by the bootstrap-median fallback (see module
        docstring and `WcsBootstrapResult.fiducial_is_outlier`).
    outlier_floor_deg : float
        Absolute floor on the outlier threshold, so a near-zero bootstrap
        MAD (a suspiciously tight cluster, itself worth noting) doesn't
        flag a negligible fiducial offset purely from floating-point
        noise.
    solve_fn : callable, optional
        Override the actual solving call -- `solve_fn(sub_xs, sub_ys) ->
        PlatesolveResult or None`. Used for testing the sampling/
        statistics logic without invoking solve-field; production callers
        should leave this as None.

    Returns
    -------
    WcsBootstrapResult

    Raises
    ------
    RuntimeError
        If the fiducial (full-source-list) solve fails -- there is no
        reference angle to bootstrap around.
    """
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    def _default_solve(sub_x, sub_y):
        return solve_plate(
            xs=sub_x, ys=sub_y, image_width=image_shape[1], image_height=image_shape[0],
            original_header=original_header, backend=backend, backend_config=backend_config,
            scale_units=scale_units, scale_low=scale_low, scale_high=scale_high,
            tweak_order=tweak_order, timeout=timeout, verbose=False,
        )

    solve = solve_fn or _default_solve

    fid_result = solve(xs, ys)
    if fid_result is None:
        raise RuntimeError("fiducial (full-source-list) solve failed -- cannot bootstrap without a reference solution")
    fid_metrics = center_wcs_angle_metrics(_make_wcs(fid_result.header), image_shape, compute_east=False)
    fid_angle = fid_metrics.north_angle_deg

    angles = []
    headers = []
    failures: list = []
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(xs, ys, image_shape, n_grid=n_grid, frac=boot_frac, rng=rng)
        try:
            r = solve(xs[idx], ys[idx])
        except Exception as e:  # noqa: BLE001 -- record and continue bootstrapping
            failures.append(f"exception: {str(e)[:80]}")
            continue
        if r is None:
            failures.append("no_solution")
            continue
        m = center_wcs_angle_metrics(_make_wcs(r.header), image_shape, compute_east=False)
        angles.append(m.north_angle_deg)
        headers.append(r.header)

    angles_arr = np.array(angles, dtype=float)

    if len(angles_arr) >= 2:
        # deviations of every bootstrap draw from the fiducial, using the
        # fiducial purely as a numerically-convenient wrap-safe anchor --
        # NOT as an assumed-correct reference (see outlier check below).
        d_from_fid = angle_diff_deg(angles_arr, fid_angle)
        median_d = float(np.median(d_from_fid))
        mad_d = float(np.median(np.abs(d_from_fid - median_d)))
        robust_sigma_d = 1.4826 * mad_d
        threshold = max(outlier_z * robust_sigma_d, outlier_floor_deg)
        fiducial_is_outlier = abs(median_d) > threshold

        if fiducial_is_outlier:
            north_angle = (fid_angle + median_d + 180.0) % 360.0 - 180.0
            resid = d_from_fid - median_d  # deviation of each draw from north_angle
            closest_i = int(np.argmin(np.abs(resid)))
            header = headers[closest_i]
        else:
            north_angle = fid_angle
            resid = d_from_fid  # deviation of each draw from north_angle (== fid_angle here)
            header = fid_result.header

        # A single bootstrap draw can independently hit the same
        # astrometry.net false-match failure mode as the fiducial (found
        # empirically: one draw landed ~2.4 deg from a cluster of nine
        # that agreed to ~0.01 deg). Sigma-clip the population itself
        # before reporting sigma/mad so one bad re-solve can't inflate
        # sigma_WCS by 100x -- MAD alone would stay quiet about this since
        # it's already robust, but sigma_north_deg (plain std, the number
        # actually meant for sigma_wcs_deg) is not.
        resid_center = float(np.median(resid))
        resid_mad = float(np.median(np.abs(resid - resid_center)))
        clip_thresh = max(outlier_z * 1.4826 * resid_mad, outlier_floor_deg)
        keep_mask = np.abs(resid - resid_center) <= clip_thresh
        n_clipped = int(np.sum(~keep_mask))
        kept = resid[keep_mask]

        if len(kept) >= 2:
            sigma = float(np.std(kept))
            mad = float(np.median(np.abs(kept - np.median(kept))))
        else:
            sigma = float(np.std(resid))
            mad = resid_mad
    else:
        north_angle = fid_angle
        sigma = mad = float("nan")
        fiducial_is_outlier = False
        header = fid_result.header
        keep_mask = np.ones(len(angles_arr), dtype=bool)
        n_clipped = 0

    fiducial_offset = angle_diff_deg(fid_angle, north_angle)

    return WcsBootstrapResult(
        north_angle_deg=north_angle, sigma_north_deg=sigma, mad_north_deg=mad, header=header,
        fiducial_north_deg=fid_angle, fiducial_header=fid_result.header,
        fiducial_is_outlier=fiducial_is_outlier, fiducial_offset_deg=float(fiducial_offset),
        n_boot_requested=n_boot, n_boot_ok=len(angles_arr),
        north_angles_deg=angles_arr, bootstrap_kept_mask=keep_mask, n_boot_clipped=n_clipped,
        failures=failures,
    )
