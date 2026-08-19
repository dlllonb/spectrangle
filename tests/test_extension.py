import numpy as np

from extractor.detection import TraceCandidate, compute_background_residual, detect_traces
from extractor.extension import extend_trace_candidate, extend_candidates
from extractor.fitting import pca_fit, measure_trace, trace_correlation_length_px
from test_detection import _synthetic_line_image


def _mask_beyond(image, angle_deg, cx, cy, s_cut):
    """Zero out the part of `image` beyond along-track position `s_cut`
    (in the direction of increasing s) -- used to build a genuinely
    truncated INPUT image for `detect_traces`, so the resulting
    candidate is constructed exactly like real production code would,
    not via a hand-rolled pixel mask."""
    th = np.radians(angle_deg)
    ux, uy = np.cos(th), np.sin(th)
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    s = (xx - cx) * ux + (yy - cy) * uy
    out = image.copy()
    out[s > s_cut] = np.median(image)
    return out


def test_extend_trace_candidate_recovers_truncated_length():
    angle = 20.0
    cx, cy = 200.0, 150.0
    full_image = _synthetic_line_image(shape=(300, 400), angle_deg=angle, length=250.0,
                                        cx=cx, cy=cy, width_sigma=2.0, amplitude=600.0)
    # only the first ~90px of the 250px line is visible to detect_traces
    truncated_image = _mask_beyond(full_image, angle, cx, cy, s_cut=-35.0)

    candidates, _, _ = detect_traces(truncated_image, min_length_px=50.0, min_eccentricity=0.8,
                                      bg_sigma=40.0, smooth_sigma=1.0)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.length_px < 110.0  # sanity: genuinely truncated vs the full 250px line

    resid, median, std, threshold = compute_background_residual(full_image, bg_sigma=40.0, smooth_sigma=1.0)
    extended = extend_trace_candidate(
        candidate, resid, None, median, std, threshold, full_image.shape,
        mask_radius_px=10.0, step_px=trace_correlation_length_px(full_image.shape, smooth_sigma=1.0),
    )

    assert extended.length_px > candidate.length_px + 50.0  # recovered real extra length
    assert extended.original_length_px == candidate.length_px
    theta_ext, _, _ = pca_fit(extended.rows, extended.cols, extended.weights)
    assert abs(((theta_ext - angle + 90) % 180) - 90) < 1.0  # still the right angle


def test_extend_trace_candidate_does_not_bridge_a_large_empty_gap():
    angle = 0.0
    cx, cy = 100.0, 100.0
    # two separate segments on the same line, ~370px apart, nothing but
    # noise in between -- detect_traces sees only the first one
    image = _synthetic_line_image(shape=(200, 600), angle_deg=angle, length=60.0,
                                   cx=cx, cy=cy, width_sigma=2.0, amplitude=600.0)
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    second_segment = (np.abs(xx - 500) < 30) & (np.abs(yy - 100) < 3)
    image = image + np.where(second_segment, 600.0, 0.0)

    candidates, _, _ = detect_traces(image, min_length_px=50.0, min_eccentricity=0.8,
                                      bg_sigma=40.0, smooth_sigma=1.0)
    # both segments individually detected (not yet merged/extended)
    assert len(candidates) == 2
    candidate = min(candidates, key=lambda c: c.x_center)  # the one near cx=100

    resid, median, std, threshold = compute_background_residual(image, bg_sigma=40.0, smooth_sigma=1.0)
    extended = extend_trace_candidate(
        candidate, resid, None, median, std, threshold, image.shape,
        mask_radius_px=10.0, step_px=trace_correlation_length_px(image.shape, smooth_sigma=1.0),
    )
    # the second segment sits ~370px away -- far beyond any plausible
    # k=10px-derived gap tolerance, so it must NOT be bridged
    assert extended.length_px < 150.0


def test_extend_candidates_sets_original_length_on_every_candidate():
    angle = -10.0
    cx, cy = 175.0, 125.0
    full_image = _synthetic_line_image(shape=(250, 350), angle_deg=angle, length=200.0,
                                        cx=cx, cy=cy, width_sigma=2.0, amplitude=600.0)
    truncated_image = _mask_beyond(full_image, angle, cx, cy, s_cut=-25.0)
    candidates, _, _ = detect_traces(truncated_image, min_length_px=50.0, min_eccentricity=0.8,
                                      bg_sigma=40.0, smooth_sigma=1.0)
    assert len(candidates) == 1
    candidate = candidates[0]

    extended_list = extend_candidates([candidate], full_image, None, mask_radius_px=10.0,
                                        bg_sigma=40.0, smooth_sigma=1.0)
    assert len(extended_list) == 1
    assert extended_list[0].original_length_px == candidate.length_px
    assert extended_list[0].length_px > candidate.length_px


def test_measure_trace_weight_uses_original_length_when_set():
    eff_res = 5.0
    rng = np.random.default_rng(0)
    n = 60
    rows = np.linspace(0, 200, n)
    cols = np.zeros(n)
    weights = np.full(n, 100.0) + rng.normal(0, 1.0, n)

    plain = TraceCandidate(rows=rows, cols=cols, weights=weights, length_px=200.0,
                            eccentricity=0.99, n_pixels=n, total_flux=float(weights.sum()),
                            x_center=0.0, y_center=100.0)
    extended = TraceCandidate(rows=rows, cols=cols, weights=weights, length_px=200.0,
                               eccentricity=0.99, n_pixels=n, total_flux=float(weights.sum()),
                               x_center=0.0, y_center=100.0, original_length_px=50.0)

    m_plain = measure_trace(plain, eff_res)
    m_extended = measure_trace(extended, eff_res)

    # same pixels -> same angle/perp_std/n_eff, but weight must differ:
    # plain uses length_px=200 for weight, extended uses original_length_px=50
    assert m_plain.theta_pix_deg == m_extended.theta_pix_deg
    assert m_plain.length_px == m_extended.length_px  # reported length unchanged
    assert m_extended.original_length_px == 50.0
    expected_ratio = (50.0 / 200.0) ** 2
    assert abs(m_extended.weight / m_plain.weight - expected_ratio) < 1e-6
