import numpy as np
import pytest

from extractor.detection import TraceCandidate
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


def test_single_candidate_passes_through_unchanged():
    frag_a = _horizontal_fragment(0.0, 50.0)
    star_x, star_y = np.array([55.0]), np.array([100.0])

    result = merge_mask_bridged_fragments([frag_a], star_x, star_y, radius_px=8.0)

    assert len(result) == 1
    assert result[0] is frag_a
