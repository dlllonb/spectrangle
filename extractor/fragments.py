"""
fragments.py — Mechanism-grounded fragment reunification for traces
split by catalog star masking.

Phase 3 fix (this session's investigation, `new_results.txt` Entries
87-95): masking a star's point-source footprint before `detect_traces`
runs (`masking.py`) frequently splits one real trace into two
connected-component fragments when the masked disk happens to cross
the trace's path, shortening the longest/highest-weight traces by up
to 3.5x (Entry 87) and degrading their measurement quality.

A first fix attempt (Entry 88) merged fragments using pure geometric
collinearity (similar fitted angle, small along-track gap, small
perpendicular offset) and was a real, reported NEGATIVE result — it
measurably worsened some fields by merging unrelated traces, because
every real trace in a field shares nearly the same grating angle by
physical construction, so "similar angle" alone barely discriminates a
true split from two different stars' traces that happen to be nearby.

The fix that actually works (Entry 90): require ACTUAL EVIDENCE of the
causal mechanism, not just geometric similarity — only merge two
fragments when one of the SAME masked stars used to build the
detection mask geometrically sits in the gap between them, close
enough to the shared fit line to plausibly have clipped the trace
there. Validated on the full 180-draw (18-field x ~10-seed) Monte
Carlo confirmation set: pooled mean |err| drops 10.5%, pooled mean
bootstrap uncertainty drops 4.4%, with no catastrophic single-field
failures (unlike the first attempt). Two follow-up concerns were
tested and refuted before shipping: tightening the geometric
tolerances does NOT help (Entry 94 — it suppresses real merges faster
than false ones, net-worse pooled result), and a residual local
mask-seam contamination effect does NOT exist (Entry 95 — 0/102 real
merge events showed a significant local pixel offset near the seam).
The parameters below are the validated, NOT the tightened, ones — do
not tighten them further without re-reading Entries 94/95 first.
"""
from __future__ import annotations

import numpy as np

from .detection import TraceCandidate
from .fitting import pca_fit, norm_axial, axial_diff, axial_unit_vectors, project_onto_axis


def merge_mask_bridged_fragments(
    candidates: list[TraceCandidate],
    star_x: np.ndarray,
    star_y: np.ndarray,
    radius_px: float,
    angle_tol_deg: float = 1.0,
    max_perp_px: float = 2.5,
    gap_pad_px: float = 3.0,
    max_perp_std_growth: float = 1.3,
) -> list[TraceCandidate]:
    """Merge pairs of `TraceCandidate`s that look like fragments of one
    real trace split by a masked star, using evidence of the actual
    splitting mechanism rather than geometry alone.

    A pair is merged only if ALL of the following hold (validated,
    Entry 90 — do not loosen or tighten without re-testing on the full
    18-field Monte Carlo set, see this module's docstring):

    1. Individually-fitted angles agree within `angle_tol_deg` (axial).
    2. The perpendicular offset between the two fragments' centroids,
       relative to their shared fit line, is <= `max_perp_px`.
    3. The along-track gap between the fragments is short enough that
       a single star's mask disk could plausibly span it
       (<= 2*radius_px + 2*gap_pad_px).
    4. **The mechanism check**: at least one of the SAME masked stars
       passed in via `star_x`/`star_y` (i.e. one of the exact stars
       used to build this image's detection mask — see
       `masking.catalog_star_pixel_positions`) projects into the
       along-track gap interval (padded by `gap_pad_px`) AND sits
       within `radius_px` of the shared fit line. Without this, two
       unrelated same-angle traces from different stars are NOT
       merged just because they happen to be geometrically close —
       this is what makes the difference between this function and
       the naive version that was reported as a negative result
       (Entry 88).
    5. **Post-merge sanity check**: the merged point cloud's own
       perpendicular scatter must not exceed `max_perp_std_growth`
       times the larger of the two fragments' individual perpendicular
       scatter — catches the case where 1-4 pass by coincidence but
       the combined cloud doesn't actually look like one straight line.

    Greedy: repeatedly merges the best-scoring valid pair and re-checks
    (a trace can be split into more than two fragments by more than one
    masked star). Fragments/candidates with no qualifying merge are
    returned unchanged.

    Parameters
    ----------
    candidates : list[TraceCandidate]
        Typically the direct output of `detection.detect_traces` run
        with a catalog star exclude_mask.
    star_x, star_y : ndarray
        Pixel positions of the masked stars for THIS image — pass
        `masking.catalog_star_pixel_positions(...)`'s output directly
        (same catalog selection used to build the exclude_mask, so the
        mechanism check is checking against the actual cause).
    radius_px : float
        The mask disk radius actually used to build the exclude_mask
        (same value passed to `masking.build_catalog_star_mask`).
    angle_tol_deg, max_perp_px, gap_pad_px, max_perp_std_growth : float
        Validated defaults (Entry 90) — see module docstring before
        changing; Entry 94 showed tightening these makes results
        WORSE, not better.

    Returns
    -------
    list[TraceCandidate]
        Possibly shorter than the input list (merged fragments replace
        their two source candidates with one combined candidate).
    """
    if len(candidates) < 2 or len(star_x) == 0:
        return list(candidates)

    remaining = list(candidates)
    merged_any = True
    while merged_any:
        merged_any = False
        n = len(remaining)
        best = None
        for i in range(n):
            ci = remaining[i]
            theta_i, perp_std_i, _ = pca_fit(ci.rows, ci.cols, ci.weights)
            for j in range(i + 1, n):
                cj = remaining[j]
                theta_j, perp_std_j, _ = pca_fit(cj.rows, cj.cols, cj.weights)
                if abs(axial_diff(theta_i, theta_j)) > angle_tol_deg:
                    continue

                # Proper axial mean of two angles: fold theta_j onto the branch
                # nearest theta_i (axial_diff always returns a value in (-90,90]),
                # THEN average -- naive (theta_i+theta_j)/2 is wrong whenever the
                # two angles straddle the +/-90 branch cut (e.g. 89.5 and -89.5
                # describe nearly the same direction but naively average to 0.0,
                # not ~90) even though the angle_tol_deg gate above already
                # correctly recognizes them as within tolerance via axial_diff.
                theta_j_folded = theta_i + axial_diff(theta_j, theta_i)
                theta = norm_axial((theta_i + theta_j_folded) / 2.0)
                ux, uy, px, py = axial_unit_vectors(theta)

                s_i, d_i = project_onto_axis(ci.rows, ci.cols, ux, uy, px, py)
                s_j, d_j = project_onto_axis(cj.rows, cj.cols, ux, uy, px, py)

                perp_offset = abs(np.average(d_i, weights=ci.weights) - np.average(d_j, weights=cj.weights))
                if perp_offset > max_perp_px:
                    continue

                if s_i.max() <= s_j.min():
                    gap_lo, gap_hi = s_i.max(), s_j.min()
                elif s_j.max() <= s_i.min():
                    gap_lo, gap_hi = s_j.max(), s_i.min()
                else:
                    continue  # overlapping along-track -- not a simple end-to-end split, skip (ambiguous)

                if gap_hi - gap_lo > 2.0 * radius_px + 2 * gap_pad_px:
                    continue

                cx = float(np.average(np.concatenate([ci.cols, cj.cols]),
                                       weights=np.concatenate([ci.weights, cj.weights])))
                cy = float(np.average(np.concatenate([ci.rows, cj.rows]),
                                       weights=np.concatenate([ci.weights, cj.weights])))

                s_star = star_x * ux + star_y * uy
                perp_dist_star = np.abs((star_x - cx) * px + (star_y - cy) * py)
                in_gap = (s_star >= gap_lo - gap_pad_px) & (s_star <= gap_hi + gap_pad_px)
                bridges = in_gap & (perp_dist_star <= radius_px)
                if not np.any(bridges):
                    continue

                rows_ij = np.concatenate([ci.rows, cj.rows])
                cols_ij = np.concatenate([ci.cols, cj.cols])
                weights_ij = np.concatenate([ci.weights, cj.weights])
                _, merged_perp_std, _ = pca_fit(rows_ij, cols_ij, weights_ij)
                if merged_perp_std > max_perp_std_growth * max(perp_std_i, perp_std_j):
                    continue

                score = (gap_hi - gap_lo) + perp_offset
                if best is None or score < best[0]:
                    best = (score, i, j)

        if best is not None:
            _, i, j = best
            ci, cj = remaining[i], remaining[j]
            rows = np.concatenate([ci.rows, cj.rows])
            cols = np.concatenate([ci.cols, cj.cols])
            weights = np.concatenate([ci.weights, cj.weights])
            theta_deg, minor_std, major_std = pca_fit(rows, cols, weights)
            ux, uy, _, _ = axial_unit_vectors(theta_deg)
            s, _ = project_onto_axis(rows, cols, ux, uy, -uy, ux)
            merged = TraceCandidate(
                rows=rows, cols=cols, weights=weights,
                length_px=float(s.max() - s.min()),
                eccentricity=max(ci.eccentricity, cj.eccentricity),
                n_pixels=len(rows), total_flux=float(weights.sum()),
                x_center=float(np.average(cols, weights=weights)),
                y_center=float(np.average(rows, weights=weights)),
            )
            remaining = [remaining[k] for k in range(len(remaining)) if k not in (i, j)] + [merged]
            merged_any = True

    return remaining
