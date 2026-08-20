import numpy as np
import pytest

from extractor.detection import TraceCandidate
from extractor.fitting import pca_fit
from extractor.fragments import merge_mask_bridged_fragments


def _horizontal_fragment(col_start, col_end, row=100.0, step=2.0):
    cols = np.arange(col_start, col_end + 1e-9, step)
    rows = np.full_like(cols, row)
    weights = np.ones_like(cols)
    return TraceCandidate(
        rows=rows, cols=cols, weights=weights,
        length_px=float(col_end - col_start), eccentricity=0.98,
        n_pixels=len(cols), total_flux=float(weights.sum()),
        x_center=float(cols.mean()), y_center=float(rows.mean()),
    )


def _near_vertical_fragment(row_start, row_end, col0, seed, step=2.0, jitter_amp=0.06):
    """A near-vertical (theta close to +/-90) fragment with small per-point
    perpendicular jitter -- used to exercise the axial +/-90 branch-cut
    averaging bug (a real trace's own short-fragment PCA fit can land on
    EITHER side of the wrap purely from ordinary pixel noise)."""
    rows = np.arange(row_start, row_end + 1e-9, step)
    rng = np.random.default_rng(seed)
    cols = col0 + rng.normal(0.0, jitter_amp, size=rows.shape)
    weights = np.ones_like(cols)
    return TraceCandidate(
        rows=rows, cols=cols, weights=weights,
        length_px=float(row_end - row_start), eccentricity=0.98,
        n_pixels=len(cols), total_flux=float(weights.sum()),
        x_center=float(cols.mean()), y_center=float(rows.mean()),
    )


def test_merges_fragments_with_a_bridging_star_in_the_gap():
    frag_a = _horizontal_fragment(0.0, 50.0)
    frag_b = _horizontal_fragment(60.0, 110.0)
    # star sits in the 10px gap (col 50-60), right on the fit line (row 100)
    star_x, star_y = np.array([55.0]), np.array([100.0])

    merged = merge_mask_bridged_fragments([frag_a, frag_b], star_x, star_y, radius_px=8.0)

    assert len(merged) == 1
    result = merged[0]
    assert result.n_pixels == frag_a.n_pixels + frag_b.n_pixels
    assert result.length_px == pytest.approx(110.0, abs=1.0)


def test_does_not_merge_without_a_bridging_star():
    frag_a = _horizontal_fragment(0.0, 50.0)
    frag_b = _horizontal_fragment(60.0, 110.0)
    # star far away from the gap -- geometrically collinear fragments alone
    # must NOT be enough to merge (this is exactly the failure mode of the
    # earlier, refuted naive-geometry-only fix, Entry 88)
    star_x, star_y = np.array([2000.0]), np.array([2000.0])

    merged = merge_mask_bridged_fragments([frag_a, frag_b], star_x, star_y, radius_px=8.0)

    assert len(merged) == 2


def test_does_not_merge_with_empty_star_list():
    frag_a = _horizontal_fragment(0.0, 50.0)
    frag_b = _horizontal_fragment(60.0, 110.0)

    merged = merge_mask_bridged_fragments([frag_a, frag_b], np.array([]), np.array([]), radius_px=8.0)

    assert len(merged) == 2


def test_does_not_merge_fragments_with_large_perpendicular_offset():
    frag_a = _horizontal_fragment(0.0, 50.0, row=100.0)
    # same angle (both horizontal) but offset well off the shared line --
    # even with a bridging star present, a large perpendicular offset should block the merge
    frag_b = _horizontal_fragment(60.0, 110.0, row=140.0)
    star_x, star_y = np.array([55.0]), np.array([120.0])

    merged = merge_mask_bridged_fragments([frag_a, frag_b], star_x, star_y, radius_px=8.0)

    assert len(merged) == 2


def test_merges_correctly_across_the_axial_90deg_branch_cut():
    # Two fragments of one real near-vertical trace: frag_a's own short-
    # fragment PCA fit lands at theta~+90 (89.9999deg), frag_b's at
    # theta~-90 (-89.9999deg) -- the SAME physical direction, straddling
    # the axial (-90,90] wrap, purely from ordinary per-point pixel noise
    # (no artificial angle difference introduced). A naive
    # (theta_i+theta_j)/2 average computes ~0.0deg here (see the module's
    # git history / new_results.txt for the confirmed-then-fixed bug) --
    # the correct axial mean is ~90deg, which is what the fold-onto-
    # theta_i's-branch approach (extractor/fragments.py) must produce for
    # this merge to succeed with the right geometry.
    frag_a = _near_vertical_fragment(150.0, 200.0, col0=200.0, seed=25)
    frag_b = _near_vertical_fragment(210.0, 260.0, col0=200.0, seed=1133)
    theta_i, _, _ = pca_fit(frag_a.rows, frag_a.cols, frag_a.weights)
    theta_j, _, _ = pca_fit(frag_b.rows, frag_b.cols, frag_b.weights)
    assert theta_i > 89.0 and theta_j < -89.0, "fixture no longer straddles the wrap -- regenerate seeds"

    star_x, star_y = np.array([200.0]), np.array([205.0])
    merged = merge_mask_bridged_fragments([frag_a, frag_b], star_x, star_y, radius_px=8.0)

    assert len(merged) == 1
    result = merged[0]
    assert result.n_pixels == frag_a.n_pixels + frag_b.n_pixels
    merged_theta, _, _ = pca_fit(result.rows, result.cols, result.weights)
    # correct axial mean is ~90deg (equivalently ~-90deg); a naive average
    # of the raw theta_i/theta_j values would put this near 0deg instead
    assert abs(abs(merged_theta) - 90.0) < 1.0


def test_single_candidate_passes_through_unchanged():
    frag_a = _horizontal_fragment(0.0, 50.0)
    star_x, star_y = np.array([55.0]), np.array([100.0])

    result = merge_mask_bridged_fragments([frag_a], star_x, star_y, radius_px=8.0)

    assert len(result) == 1
    assert result[0] is frag_a
