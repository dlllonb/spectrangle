import numpy as np
from astropy.io import fits

from extractor.stars import make_xylist


def test_make_xylist_converts_0_based_to_1_based_fits_convention():
    """astrometry.net's xylist format follows the FITS standard (pixel
    coordinate (1,1) = center of the first pixel), but every other part of
    this package uses 0-based pixel coordinates (numpy/photutils
    convention, matching extract_stars' documented output). Before this
    conversion existed, every solve-field WCS was silently shifted by
    exactly 1 pixel in both x and y relative to the true source positions
    -- found via a ~14" constant position-error offset on the sim test
    image (see extractor/stars.py's make_xylist docstring for the full
    story). Regression-test the fix directly: read back the xylist FITS
    table and confirm the written X/Y are xs+1, ys+1, not xs, ys."""
    xs = np.array([0.0, 100.5, 3095.0])
    ys = np.array([0.0, 200.25, 2079.0])

    buf = make_xylist(xs, ys)
    with fits.open(buf) as hdul:
        table = hdul[1].data
        written_x = np.asarray(table["X"], dtype=float)
        written_y = np.asarray(table["Y"], dtype=float)

    np.testing.assert_allclose(written_x, xs + 1.0)
    np.testing.assert_allclose(written_y, ys + 1.0)


def test_make_xylist_preserves_float_precision():
    """The +1 conversion must not lose sub-pixel centroid precision."""
    xs = np.array([123.456789])
    ys = np.array([987.654321])

    buf = make_xylist(xs, ys)
    with fits.open(buf) as hdul:
        table = hdul[1].data
        written_x = float(np.asarray(table["X"], dtype=float)[0])
        written_y = float(np.asarray(table["Y"], dtype=float)[0])

    assert abs(written_x - (xs[0] + 1.0)) < 1e-9
    assert abs(written_y - (ys[0] + 1.0)) < 1e-9


def test_a_constant_pixel_offset_does_not_change_fitted_rotation():
    """Direct, from-first-principles confirmation that this bug's effect
    (a UNIFORM +1 pixel shift applied identically to every correspondence
    point) cannot change a fitted rotation/scale -- only the fitted
    reference point (CRPIX/CRVAL) absorbs a constant offset. This is the
    linear-algebra fact the fix's docstring relies on to claim "no
    previously-reported angle result changes" -- checked here directly
    with an ordinary least-squares fit rather than asserted.

    Model: true_sky = A @ true_pixel + b (A encodes rotation+scale). Fit A
    from noisy correspondences at pixel positions p, and separately from
    the SAME correspondences shifted by a constant delta (p + delta) --
    the recovered A (and therefore any angle derived from it) must match
    to numerical precision; only the fitted intercept differs."""
    rng = np.random.default_rng(0)
    n = 40
    true_pixels = rng.uniform(0, 3000, size=(n, 2))
    true_A = np.array([[0.9998, -0.0175], [0.0175, 0.9998]]) * 1e-3  # ~1 deg rotation, small scale
    true_b = np.array([10.0, 56.0])
    noise = rng.normal(0.0, 1e-7, size=(n, 2))
    true_sky = true_pixels @ true_A.T + true_b + noise

    def fit_affine(pixels, sky):
        design = np.column_stack([pixels, np.ones(len(pixels))])
        coeffs, *_ = np.linalg.lstsq(design, sky, rcond=None)
        return coeffs[:2].T, coeffs[2]  # A, b

    A_unshifted, b_unshifted = fit_affine(true_pixels, true_sky)

    delta = np.array([1.0, 1.0])  # the bug: every point shifted by the same (1,1) px
    A_shifted, b_shifted = fit_affine(true_pixels + delta, true_sky)

    np.testing.assert_allclose(A_shifted, A_unshifted, atol=1e-12)
    # only the intercept differs, by exactly -A @ delta (first-order Taylor,
    # exact here since the model is exactly affine)
    np.testing.assert_allclose(b_shifted - b_unshifted, -A_unshifted @ delta, atol=1e-9)

    def rotation_angle_deg(A):
        return np.degrees(np.arctan2(A[1, 0], A[0, 0]))

    assert abs(rotation_angle_deg(A_shifted) - rotation_angle_deg(A_unshifted)) < 1e-9
