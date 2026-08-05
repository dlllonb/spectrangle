import numpy as np

from extractor.detection import detect_traces


def _synthetic_line_image(shape=(200, 300), angle_deg=30.0, length=150.0,
                           cx=150.0, cy=100.0, width_sigma=1.2, amplitude=500.0, seed=0):
    """A single straight line (Gaussian cross-section) at a known angle,
    on a flat noisy background -- for detection/fitting unit tests."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    image = rng.normal(loc=100.0, scale=5.0, size=shape).astype(np.float64)
    th = np.radians(angle_deg)
    ux, uy = np.cos(th), np.sin(th)
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - cx, yy - cy
    s = dx * ux + dy * uy
    d = -dx * uy + dy * ux
    on_segment = np.abs(s) <= length / 2.0
    profile = amplitude * np.exp(-0.5 * (d / width_sigma) ** 2)
    image += np.where(on_segment, profile, 0.0)
    return image


def test_detect_traces_finds_the_injected_line():
    image = _synthetic_line_image(angle_deg=30.0, length=150.0)
    candidates, n_raw, threshold = detect_traces(
        image, min_length_px=50.0, min_eccentricity=0.8, bg_sigma=40.0, smooth_sigma=1.0,
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.length_px > 100.0
    assert c.eccentricity > 0.9


def test_detect_traces_rejects_short_round_clump():
    ny, nx = 200, 300
    rng = np.random.default_rng(1)
    image = rng.normal(loc=100.0, scale=5.0, size=(ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - 150) ** 2 + (yy - 100) ** 2
    image = image + 500.0 * np.exp(-0.5 * r2 / (3.0 ** 2))  # round blob, not a line
    candidates, n_raw, threshold = detect_traces(
        image, min_length_px=50.0, min_eccentricity=0.8, bg_sigma=40.0, smooth_sigma=1.0,
    )
    assert len(candidates) == 0


def test_detect_traces_finds_nothing_in_pure_noise():
    rng = np.random.default_rng(2)
    image = rng.normal(loc=100.0, scale=5.0, size=(200, 300))
    candidates, n_raw, threshold = detect_traces(
        image, min_length_px=50.0, min_eccentricity=0.8, bg_sigma=40.0, smooth_sigma=1.0,
    )
    assert len(candidates) == 0
