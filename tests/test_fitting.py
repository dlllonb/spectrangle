import numpy as np
import pytest

from extractor.detection import detect_traces
from extractor.fitting import (
    TraceMeasurement, axial_diff, axial_unit_vectors, measure_trace, project_onto_axis,
    remove_contamination, trace_correlation_length_px,
)
from test_detection import _synthetic_line_image


def test_axial_unit_vectors_matches_manual_trig():
    for angle in (0.0, 30.0, 90.0, -45.0, 137.0):
        ux, uy, px, py = axial_unit_vectors(angle)
        th = np.radians(angle)
        assert ux == pytest.approx(np.cos(th))
        assert uy == pytest.approx(np.sin(th))
        # perpendicular: rotate (ux,uy) by +90deg, and unit vectors stay orthonormal
        assert px == pytest.approx(-uy)
        assert py == pytest.approx(ux)
        assert ux * px + uy * py == pytest.approx(0.0, abs=1e-9)


def test_project_onto_axis_is_identity_along_its_own_axis():
    # points laid out exactly along a 30deg axis should have all their
    # perpendicular (d) coordinate at ~0 relative to the same origin
    angle = 30.0
    ux, uy, px, py = axial_unit_vectors(angle)
    s_true = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    rows = s_true * uy
    cols = s_true * ux
    s, d = project_onto_axis(rows, cols, ux, uy, px, py)
    assert np.allclose(s, s_true, atol=1e-9)
    assert np.allclose(d, 0.0, atol=1e-9)


def test_project_onto_axis_respects_explicit_origin():
    ux, uy, px, py = axial_unit_vectors(0.0)  # along +x
    rows, cols = np.array([5.0]), np.array([12.0])
    s1, d1 = project_onto_axis(rows, cols, ux, uy, px, py, origin_row=5.0, origin_col=2.0)
    assert s1[0] == pytest.approx(10.0)  # 12 - 2 along +x
    assert d1[0] == pytest.approx(0.0)   # 5 - 5 perpendicular


def test_remove_contamination_weight_reflects_shrinkage_below_original_length():
    # simulate an extended trace (original_length_px=110, i.e. pre-extension)
    # whose extension "tips" turn out to be contamination -- a 100px clean
    # main body (cols 0-99) plus two off-axis tips (cols -15..-1 and
    # 100..114, offset well off the main line) that remove_contamination
    # correctly identifies and strips, leaving length_px=99 -- BELOW even
    # the pre-extension original_length_px=110. Weight must reflect the
    # shorter, post-removal length here, not stay pinned to the stale
    # original_length_px (Entry 122 finding 2) -- this is the one case
    # where the two differ; every extended-but-not-heavily-contaminated
    # trace keeps the exact pre-fix weight (min() is a no-op there).
    rng = np.random.default_rng(3)
    main_cols = np.arange(0, 100, 1.0)
    main_rows = 100.0 + rng.normal(0, 0.3, size=main_cols.shape)
    tip1_cols = np.arange(100, 115, 1.0)
    tip1_rows = np.full_like(tip1_cols, 120.0)
    tip2_cols = np.arange(-15, 0, 1.0)
    tip2_rows = np.full_like(tip2_cols, 80.0)
    cols = np.concatenate([tip2_cols, main_cols, tip1_cols])
    rows = np.concatenate([tip2_rows, main_rows, tip1_rows])
    weights = np.ones_like(cols) * 100.0

    measurement = TraceMeasurement(
        x_center=float(cols.mean()), y_center=float(rows.mean()),
        theta_pix_deg=0.0, theta_pix_uncertainty_deg=0.1,
        length_px=130.0, perp_std_px=3.0, n_eff=50.0,
        eccentricity=0.99, total_flux=float(weights.sum()),
        weight=1000.0, original_length_px=110.0,
        _rows=rows, _cols=cols, _weights=weights,
    )

    result = remove_contamination(measurement, reference_angle_deg=0.0, effective_resolution_px=5.0)

    assert result.n_pixels_removed == 30  # both 15-point tips stripped
    assert result.length_px < measurement.original_length_px  # the pathological case: shrunk below original
    expected_weight = (min(measurement.original_length_px, result.length_px) / result.perp_std_px) ** 2
    assert result.weight == pytest.approx(expected_weight)
    buggy_old_weight = (measurement.original_length_px / result.perp_std_px) ** 2
    assert result.weight < buggy_old_weight  # fix never INCREASES weight vs the old behavior


def test_measure_trace_recovers_known_angle():
    true_angle = -35.0
    image = _synthetic_line_image(angle_deg=true_angle, length=150.0)
    candidates, _, _ = detect_traces(image, min_length_px=50.0, min_eccentricity=0.8,
                                      bg_sigma=40.0, smooth_sigma=1.0)
    assert len(candidates) == 1
    eff_res = trace_correlation_length_px(image.shape, smooth_sigma=1.0)
    m = measure_trace(candidates[0], eff_res)
    assert abs(axial_diff(m.theta_pix_deg, true_angle)) < 1.0
    assert m.theta_pix_uncertainty_deg > 0
    assert np.isfinite(m.theta_pix_uncertainty_deg)


def test_remove_contamination_strips_injected_clump_and_improves_angle():
    true_angle = 15.0
    image = _synthetic_line_image(angle_deg=true_angle, length=200.0, cx=150.0, cy=100.0)

    # Inject a bright round clump straddling the line, offset to one side
    # partway along its length -- a synthetic "star landed on the trace".
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    th = np.radians(true_angle)
    ux, uy = np.cos(th), np.sin(th)
    px, py = -uy, ux
    clump_cx = 150.0 + 40.0 * ux + 6.0 * px
    clump_cy = 100.0 + 40.0 * uy + 6.0 * py
    r2 = (xx - clump_cx) ** 2 + (yy - clump_cy) ** 2
    image = image + 800.0 * np.exp(-0.5 * r2 / (4.0 ** 2))

    candidates, _, _ = detect_traces(image, min_length_px=50.0, min_eccentricity=0.5,
                                      bg_sigma=40.0, smooth_sigma=1.0)
    assert len(candidates) == 1

    eff_res = trace_correlation_length_px(image.shape, smooth_sigma=1.0)
    plain = measure_trace(candidates[0], eff_res)
    cleaned = remove_contamination(plain, reference_angle_deg=true_angle, effective_resolution_px=eff_res)

    plain_err = abs(axial_diff(plain.theta_pix_deg, true_angle))
    cleaned_err = abs(axial_diff(cleaned.theta_pix_deg, true_angle))

    assert cleaned.n_pixels_removed > 0
    assert "contamination_removed" in cleaned.quality_flags
    assert cleaned_err < plain_err


def test_remove_contamination_no_op_on_clean_trace():
    true_angle = 5.0
    image = _synthetic_line_image(angle_deg=true_angle, length=150.0)
    candidates, _, _ = detect_traces(image, min_length_px=50.0, min_eccentricity=0.8,
                                      bg_sigma=40.0, smooth_sigma=1.0)
    eff_res = trace_correlation_length_px(image.shape, smooth_sigma=1.0)
    plain = measure_trace(candidates[0], eff_res)
    cleaned = remove_contamination(plain, reference_angle_deg=true_angle, effective_resolution_px=eff_res)
    # a clean synthetic line shouldn't lose more than a handful of noise pixels
    assert cleaned.n_pixels_removed < 0.05 * plain._weights.size
