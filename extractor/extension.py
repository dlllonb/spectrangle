"""
extension.py — Trace-extension walker: recover the faint, below-
detection-threshold continuation of an already-detected trace.

Phase 3 fix (this session's investigation, `new_results.txt` Entries
97-115): `detection.detect_traces` only keeps pixels that individually
clear a strict per-pixel significance threshold, which truncates real
traces well before their true faint end. This module grows each
candidate outward along its own fitted axis, testing perpendicular
flux-profile "cuts" against the SAME global background statistics
`detect_traces` itself thresholds against (never a fresh local/self-
referential estimate — this project has twice rejected that pattern,
`dev/blob_common.py`'s `robust_pca_fit` and its per-bin adaptive width).

**The "unproductive gap" bug and its fix (Entries 97-100)**: an earlier
prototype decided whether to CONTINUE walking using an aggregate flux
test over a whole window, but decided what to actually KEEP using a
much stricter per-pixel threshold — so a step could pass the aggregate
test (reset the walker's gap counter) while adding zero pixels, letting
the walker travel through long stretches of diffuse, non-significant
flux (observed up to 530px) before eventually latching onto something
unrelated. Fixed here: only a step that actually adds pixels resets the
gap counter; a "hit" that adds nothing counts toward the gap like a
real miss.

**Parameters, justified from image-measured quantities, not hardcoded**
(Entries 103-106): `cut_factor` (perpendicular sampling half-width) is
fixed at exactly a 3-sigma window (`cut_factor=1.5` gives `D =
cut_factor * 2 * minor_std = 3.0 * minor_std`) around the trace's OWN
measured perpendicular scatter — 99.73% Gaussian coverage, already
image-derived. `margin_steps` (along-track gap tolerance) is expressed
as `2 * mask_radius_px + margin_steps * step_px` — a masked star's full
physical diameter, plus a small number of walker steps of margin — both
terms already image-measured (`mask_radius_px` from the empirical PSF
sigma, `step_px` from the data's own correlation length), intended to
generalize to a different camera/lens/grating configuration rather than
transferring a single opaque pixel constant. `margin_steps=2.0` was
validated via a proper multi-seed-averaged search (not a single-seed
sweep, which was tried first and produced a combo that looked 45%
better but was WORSE than the untuned default on held-out seeds) to sit
in a stable, non-overfit performance plateau.

**Field-level validation and the weighting fix (Entries 107-115)**: a
full 18-field x 3-seed test found the per-trace effect of extension is
genuinely positive (net signed shift -55.8 deg, 49.5% improved vs 28.1%
worsened), but the properly-weighted FIELD-level combine was close to a
wash (pooled mean|err| 0.051 -> 0.057 deg) because a small number of
legitimate, correctly-reconstructed long extensions happen to measure a
real, deterministic wrong angle (traced to a mix of pre-existing bias
getting weight-amplified by `length^2`, and in some cases genuine new
bias introduced by extension) and dominate their field's combine via
inflated weight. Two per-trace gating attempts (built on a real,
validated-but-imperfect "does the added segment agree with the
original" discriminator) were tested and both made the field-level
result WORSE than doing nothing extra — the discriminator's error rate
isn't low enough to safely drive a binary, squared-consequence weight
decision. **The fix that actually works, validated at full 18-field x
3-seed scale (16/18 fields beat-or-tie no-extension, pooled mean|err|
0.051 -> 0.039 deg, 39% reduction)**: keep the extension-refined ANGLE,
but weight the trace in the ensemble combine by its ORIGINAL,
pre-extension length (see `detection.TraceCandidate.original_length_px`
and `fitting.measure_trace`) — decouples "how much should this trace's
angle benefit from extension" from "how much should extension be
allowed to inflate this trace's influence over the field combine."
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates

from .detection import TraceCandidate, compute_background_residual
from .fitting import pca_fit, trace_correlation_length_px


def _perp_cut_window(resid, exclude_mask, origin_row, origin_col, ux, uy, px, py, step_px, D):
    """Sample a rectangular window (along-track in [0, step_px), cross-
    track in [-D, D]) starting at (origin_row, origin_col) and extending
    along (ux, uy), via bilinear interpolation. Returns (values, rows,
    cols, masked_frac) as (n_s, n_d) grids (rows/cols in original-image
    pixel coordinates, for downstream per-pixel threshold inclusion)."""
    n_s = max(2, int(round(step_px)))
    n_d = max(3, int(round(2 * D)))
    s_vals = (np.arange(n_s) + 0.5) * (step_px / n_s)
    d_vals = np.linspace(-D, D, n_d)
    ss, dd = np.meshgrid(s_vals, d_vals, indexing='ij')
    rows = origin_row + ss * uy + dd * py
    cols = origin_col + ss * ux + dd * px
    values = map_coordinates(resid, [rows.ravel(), cols.ravel()], order=1, mode='constant', cval=0.0)
    if exclude_mask is not None:
        mvals = map_coordinates(exclude_mask.astype(np.float32), [rows.ravel(), cols.ravel()],
                                 order=0, mode='constant', cval=1.0)
        masked_frac = float(np.mean(mvals > 0.5))
    else:
        masked_frac = 0.0
    return (values.reshape(n_s, n_d), rows.reshape(n_s, n_d), cols.reshape(n_s, n_d), masked_frac)


def _hump_test(values_grid, ext_sigma, median, std):
    """values_grid: (n_s, n_d) sampled residual for one step's window.
    Central band = middle third of d-bins. HIT if the central band's
    summed flux clears ext_sigma above the global background AND the
    brightest of 3 equal d-thirds is the central one (rejects "elevated
    somewhere in the window but off the trace line")."""
    n_s, n_d = values_grid.shape
    third = max(1, n_d // 3)
    lo, hi = third, n_d - third
    central = values_grid[:, lo:hi]
    n_central = central.size
    if n_central == 0:
        return False
    z = (central.sum() - n_central * median) / (std * math.sqrt(n_central))
    thirds = [values_grid[:, :third].mean(), values_grid[:, third:hi].mean(), values_grid[:, hi:].mean()]
    centered = int(np.argmax(thirds)) == 1
    return bool(z >= ext_sigma and centered)


def extend_trace_candidate(
    candidate: TraceCandidate,
    resid: np.ndarray,
    exclude_mask: Optional[np.ndarray],
    median: float,
    std: float,
    threshold: float,
    image_shape: tuple[int, int],
    mask_radius_px: Optional[float] = None,
    ext_sigma: float = 4.0,
    cut_factor: float = 1.5,
    margin_steps: float = 2.0,
    step_px: Optional[float] = None,
    refit_every: int = 3,
    max_steps: int = 200,
) -> TraceCandidate:
    """Grow `candidate` outward from both ends along its own fitted
    axis. Returns a NEW `TraceCandidate` (never mutates the input) with
    `original_length_px` set to the input candidate's own `length_px`.

    A step is a HIT if its perpendicular-cut window's central band
    clears `ext_sigma` above background (see `_hump_test`); a hit is
    PRODUCTIVE only if at least one of its pixels also clears the
    strict per-pixel `threshold` (the same one `detect_traces` used).
    Only productive hits reset the along-track gap counter — this is
    the Issue-1 fix (see module docstring); a non-productive hit counts
    toward the gap exactly like a real miss.
    """
    if step_px is None:
        step_px = trace_correlation_length_px(image_shape, smooth_sigma=2.5)
    ny, nx = image_shape

    rows_acc = list(map(float, candidate.rows))
    cols_acc = list(map(float, candidate.cols))
    weights_acc = list(map(float, candidate.weights))

    max_gap_px = (2.0 * mask_radius_px + margin_steps * step_px) if mask_radius_px is not None \
        else (6.0 + margin_steps) * step_px
    max_consecutive_nonproductive = max(1, int(math.ceil(max_gap_px / step_px)))

    theta0_deg, minor_std0, _ = pca_fit(candidate.rows, candidate.cols, candidate.weights)
    th0 = math.radians(theta0_deg)
    ux0, uy0 = math.cos(th0), math.sin(th0)
    s0_all = candidate.cols * ux0 + candidate.rows * uy0

    for direction in (+1, -1):
        idx = int(np.argmax(s0_all)) if direction > 0 else int(np.argmin(s0_all))
        origin_row = float(candidate.rows[idx])
        origin_col = float(candidate.cols[idx])
        ux, uy = ux0 * direction, uy0 * direction
        px_, py_ = -uy0, ux0
        minor_std = minor_std0
        D = cut_factor * max(2.0 * minor_std, 1.0)

        consecutive_nonproductive = 0
        n_hits_since_refit = 0
        n_step = 0
        while n_step < max_steps:
            n_step += 1
            values_grid, rows_grid, cols_grid, masked_frac = _perp_cut_window(
                resid, exclude_mask, origin_row, origin_col, ux, uy, px_, py_, step_px, D)

            productive = False
            if masked_frac < 0.5:
                hit = _hump_test(values_grid, ext_sigma, median, std)
                if hit:
                    flat_vals = values_grid.ravel(); flat_rows = rows_grid.ravel(); flat_cols = cols_grid.ravel()
                    keep = flat_vals >= threshold
                    if keep.any():
                        rows_acc.extend(flat_rows[keep].tolist())
                        cols_acc.extend(flat_cols[keep].tolist())
                        weights_acc.extend(flat_vals[keep].tolist())
                        productive = True

            if productive:
                consecutive_nonproductive = 0
                n_hits_since_refit += 1
                if n_hits_since_refit >= refit_every:
                    rr, cc, ww = np.array(rows_acc), np.array(cols_acc), np.array(weights_acc)
                    theta_r, minor_std, _ = pca_fit(rr, cc, ww)
                    thr = math.radians(theta_r)
                    ux_r, uy_r = math.cos(thr), math.sin(thr)
                    if ux_r * ux + uy_r * uy < 0:  # keep pointing outward through the refit
                        ux_r, uy_r = -ux_r, -uy_r
                    ux, uy = ux_r, uy_r
                    px_, py_ = -uy_r, ux_r
                    D = cut_factor * max(2.0 * minor_std, 1.0)
                    n_hits_since_refit = 0
            else:
                consecutive_nonproductive += 1
                if consecutive_nonproductive > max_consecutive_nonproductive:
                    break

            origin_row += step_px * uy
            origin_col += step_px * ux
            if not (-50 <= origin_row <= ny + 50 and -50 <= origin_col <= nx + 50):
                break

    rows_f, cols_f, weights_f = np.array(rows_acc), np.array(cols_acc), np.array(weights_acc)
    theta_deg, minor_std, major_std = pca_fit(rows_f, cols_f, weights_f)
    th = math.radians(theta_deg); ux, uy = math.cos(th), math.sin(th)
    s = cols_f * ux + rows_f * uy
    return TraceCandidate(
        rows=rows_f, cols=cols_f, weights=weights_f,
        length_px=float(s.max() - s.min()),
        eccentricity=candidate.eccentricity,
        n_pixels=len(rows_f), total_flux=float(weights_f.sum()),
        x_center=float(np.average(cols_f, weights=weights_f)),
        y_center=float(np.average(rows_f, weights=weights_f)),
        original_length_px=candidate.length_px,
    )


def extend_candidates(
    candidates: list[TraceCandidate],
    image: np.ndarray,
    exclude_mask: Optional[np.ndarray],
    mask_radius_px: float,
    bg_sigma: float = 50.0,
    smooth_sigma: float = 2.5,
    n_sigma: float = 6.0,
    ext_sigma: float = 4.0,
    cut_factor: float = 1.5,
    margin_steps: float = 2.0,
    refit_every: int = 3,
    max_steps: int = 200,
) -> list[TraceCandidate]:
    """Extend every candidate, sharing ONE background-residual/threshold
    computation (`detection.compute_background_residual`, called with
    the exact same parameters `detect_traces` used) across all of them
    for consistency and to avoid recomputing it per-candidate."""
    resid, median, std, threshold = compute_background_residual(
        image, bg_sigma=bg_sigma, smooth_sigma=smooth_sigma, n_sigma=n_sigma, exclude_mask=exclude_mask,
    )
    return [
        extend_trace_candidate(
            c, resid, exclude_mask, median, std, threshold, image.shape,
            mask_radius_px=mask_radius_px, ext_sigma=ext_sigma, cut_factor=cut_factor,
            margin_steps=margin_steps, refit_every=refit_every, max_steps=max_steps,
        )
        for c in candidates
    ]
