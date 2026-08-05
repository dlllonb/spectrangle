import numpy as np
from astropy.io import fits

from extractor.platesolve import PlatesolveResult
from extractor.wcs_uncertainty import bootstrap_wcs_orientation, stratified_bootstrap_indices


def _fake_wcs_header(ra0=10.0, dec0=56.0, north_angle_deg=90.0, scale_deg_per_px=9.9 / 3600.0):
    """A minimal TAN-projection FITS header at a given (fixed) pixel-space
    north-angle orientation, for testing WITHOUT solve-field. north_angle_deg
    is engineered via CD-matrix rotation: 90 deg -> a pure +y=north,
    +x=east-ish orthogonal case is the simplest to reason about.

    Sign note: verified by direct round-trip through
    `wcsangle.center_wcs_angle_metrics` (the same finite-difference measurer
    `bootstrap_wcs_orientation` uses) that `-(north_angle_deg - 90.0)` is the
    correct rotation sign -- the naive `+(north_angle_deg - 90.0)` mirrors the
    requested angle about 90 deg. Round-trip-tested for all deviations from
    90 deg; earlier versions of this helper only tested angles within
    fractions of a degree of 90, where std-only assertions can't distinguish
    the two signs (a mirrored jitter distribution has the same scatter)."""
    theta = np.radians(-(north_angle_deg - 90.0))  # rotation of the CD matrix
    cd = scale_deg_per_px * np.array([[-np.cos(theta), np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]])
    h = fits.Header()
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = ra0
    h["CRVAL2"] = dec0
    h["CRPIX1"] = 1500.0
    h["CRPIX2"] = 1000.0
    h["CD1_1"] = cd[0, 0]
    h["CD1_2"] = cd[0, 1]
    h["CD2_1"] = cd[1, 0]
    h["CD2_2"] = cd[1, 1]
    return h


def test_stratified_bootstrap_indices_keeps_spatial_coverage():
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 3000, size=300)
    ys = rng.uniform(0, 2000, size=300)
    idx = stratified_bootstrap_indices(xs, ys, (2000, 3000), n_grid=(3, 3), frac=0.5, rng=rng)
    assert 100 < len(idx) < 300
    # every grid cell should still be represented after sampling
    sub_x, sub_y = xs[idx], ys[idx]
    cx = np.floor(sub_x / 3000 * 3).clip(0, 2)
    cy = np.floor(sub_y / 2000 * 3).clip(0, 2)
    cells_present = {(int(a), int(b)) for a, b in zip(cx, cy)}
    assert len(cells_present) == 9


def test_bootstrap_wcs_orientation_recovers_known_scatter():
    """Mocked solve_fn returns headers with an EXACTLY known jittered
    north angle -- confirms the sampling/statistics machinery (not the
    real solve-field integration) reproduces the injected scatter."""
    rng_truth = np.random.default_rng(1)
    xs = rng_truth.uniform(0, 3000, size=200)
    ys = rng_truth.uniform(0, 2000, size=200)
    image_shape = (2000, 3000)

    true_sigma = 0.3
    injected = rng_truth.normal(0.0, true_sigma, size=200)  # per-solve-call jitter pool

    call_count = {"n": 0}

    def fake_solve(sub_x, sub_y):
        i = call_count["n"]
        call_count["n"] += 1
        jitter = 0.0 if i == 0 else injected[i % len(injected)]
        header = _fake_wcs_header(north_angle_deg=90.0 + jitter)
        return PlatesolveResult(header=header)

    result = bootstrap_wcs_orientation(
        xs, ys, image_shape, n_boot=200, boot_frac=0.9, n_grid=(3, 3),
        seed=2, solve_fn=fake_solve,
    )

    assert result.n_boot_ok == 200
    # small residual expected: local_north_angle_deg is a finite-step (10 arcsec)
    # geodesic approximation, not an exact closed-form angle from the CD matrix
    assert abs(result.fiducial_north_deg - 90.0) < 0.01
    # recovered scatter should be close to the injected 0.3 deg (statistical, not exact)
    assert 0.15 < result.sigma_north_deg < 0.5


def test_bootstrap_wcs_orientation_falls_back_when_fiducial_is_outlier():
    """Reproduces the real failure mode found on the sim image
    (110lmm50mm.fits): the full-source-list "fiducial" solve locks onto a
    confidently-wrong answer 6 deg away from where the bootstrap
    population (and ground truth) actually sits. The result should flag
    this and report the bootstrap-median fallback, not the bad fiducial."""
    rng_truth = np.random.default_rng(3)
    xs = rng_truth.uniform(0, 3000, size=200)
    ys = rng_truth.uniform(0, 2000, size=200)
    image_shape = (2000, 3000)

    true_sigma = 0.05
    injected = rng_truth.normal(0.0, true_sigma, size=200)

    call_count = {"n": 0}

    def fake_solve(sub_x, sub_y):
        i = call_count["n"]
        call_count["n"] += 1
        if i == 0:
            # the fiducial call: full source list -> wrong, confident answer
            return PlatesolveResult(header=_fake_wcs_header(north_angle_deg=96.0))
        jitter = injected[i % len(injected)]
        header = _fake_wcs_header(north_angle_deg=90.0 + jitter)
        return PlatesolveResult(header=header)

    result = bootstrap_wcs_orientation(
        xs, ys, image_shape, n_boot=30, boot_frac=0.9, n_grid=(3, 3),
        seed=4, solve_fn=fake_solve,
    )

    assert result.n_boot_ok == 30
    assert abs(result.fiducial_north_deg - 96.0) < 0.01
    assert result.fiducial_is_outlier is True
    assert abs(result.fiducial_offset_deg - 6.0) < 0.1
    # the reported estimate should track the bootstrap population (~90), not the fiducial (96)
    assert abs(result.north_angle_deg - 90.0) < 0.1
    assert result.sigma_north_deg < 0.5
    # the returned header should be a "good" (~90 deg) one, not the bad fiducial header
    from extractor.wcsangle import center_wcs_angle_metrics
    m = center_wcs_angle_metrics(_make_wcs_for_test(result.header), image_shape, compute_east=False)
    assert abs(m.north_angle_deg - 90.0) < 0.5


def test_bootstrap_wcs_orientation_clips_a_single_bad_draw():
    """Reproduces the OTHER real failure mode found on the sim image: the
    fiducial was fine, but one of ten bootstrap subset re-solves
    independently landed ~2.4 deg away from a tight cluster of the other
    nine. Plain std over all draws would report sigma ~0.7 deg even
    though the population is really ~0.01 deg tight -- the single bad
    draw should be clipped out of sigma_north_deg/mad_north_deg (but kept
    in north_angles_deg for diagnostics)."""
    rng = np.random.default_rng(5)
    xs = rng.uniform(0, 3000, size=200)
    ys = rng.uniform(0, 2000, size=200)
    image_shape = (2000, 3000)

    tight_jitter = rng.normal(0.0, 0.01, size=9)
    call_count = {"n": 0}

    def fake_solve(sub_x, sub_y):
        i = call_count["n"]
        call_count["n"] += 1
        if i == 0:
            return PlatesolveResult(header=_fake_wcs_header(north_angle_deg=90.0))
        j = i - 1
        if j == 4:  # one bad draw, buried in the middle
            angle = 92.4
        else:
            angle = 90.0 + tight_jitter[j % len(tight_jitter)]
        return PlatesolveResult(header=_fake_wcs_header(north_angle_deg=angle))

    result = bootstrap_wcs_orientation(
        xs, ys, image_shape, n_boot=10, boot_frac=0.9, n_grid=(3, 3),
        seed=6, solve_fn=fake_solve,
    )

    assert result.n_boot_ok == 10
    assert result.fiducial_is_outlier is False
    assert result.n_boot_clipped == 1
    assert result.bootstrap_kept_mask.sum() == 9
    assert not result.bootstrap_kept_mask[4]
    # the bad draw is still visible in the raw array for diagnostics
    assert len(result.north_angles_deg) == 10
    # sigma should reflect the tight cluster, not be blown out by the one bad draw
    assert result.sigma_north_deg < 0.1
    assert result.mad_north_deg < 0.1


def _make_wcs_for_test(header):
    from astropy.wcs import WCS
    return WCS(header)


def test_bootstrap_wcs_orientation_raises_if_fiducial_fails():
    def always_fail(sub_x, sub_y):
        return None

    xs = np.linspace(0, 3000, 50)
    ys = np.linspace(0, 2000, 50)
    try:
        bootstrap_wcs_orientation(xs, ys, (2000, 3000), n_boot=5, solve_fn=always_fail)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_bootstrap_wcs_orientation_records_failures_without_crashing():
    def flaky_solve(sub_x, sub_y):
        if flaky_solve.calls == 0:
            flaky_solve.calls += 1
            return PlatesolveResult(header=_fake_wcs_header())
        flaky_solve.calls += 1
        return None  # every bootstrap draw fails after the fiducial
    flaky_solve.calls = 0

    xs = np.linspace(0, 3000, 50)
    ys = np.linspace(0, 2000, 50)
    result = bootstrap_wcs_orientation(xs, ys, (2000, 3000), n_boot=10, solve_fn=flaky_solve)
    assert result.n_boot_ok == 0
    assert len(result.failures) == 10
    assert np.isnan(result.sigma_north_deg)
