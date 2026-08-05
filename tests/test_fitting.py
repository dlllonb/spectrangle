import numpy as np

from extractor.detection import detect_traces
from extractor.fitting import (
    axial_diff, measure_trace, remove_contamination, trace_correlation_length_px,
)
from test_detection import _synthetic_line_image


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
