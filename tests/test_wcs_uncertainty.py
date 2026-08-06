import numpy as np
import pytest
from astropy.io import fits

from extractor.platesolve import PlatesolveResult
from extractor.wcs_uncertainty import (
    bootstrap_wcs_orientation, stratified_bootstrap_indices,
    run_one_bootstrap_draw, robust_bootstrap_summary,
    sip_order_achieved, sip_filter_indices,
)


def _fake_wcs_header(ra0=10.0, dec0=56.0, north_angle_deg=90.0, scale_deg_per_px=9.9 / 3600.0,
                      sip_order=None):
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
    the two signs (a mirrored jitter distribution has the same scatter).

    sip_order : if given, adds a trivial (all-zero, i.e. no actual distortion)
    SIP polynomial of that order, purely so `A_ORDER`/`B_ORDER` are present --
    used to test the SIP-achievement filter without needing a real distorted
    fit."""
    theta = np.radians(-(north_angle_deg - 90.0))  # rotation of the CD matrix
    cd = scale_deg_per_px * np.array([[-np.cos(theta), np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]])
    h = fits.Header()
    h["CTYPE1"] = "RA---TAN-SIP" if sip_order is not None else "RA---TAN"
    h["CTYPE2"] = "DEC--TAN-SIP" if sip_order is not None else "DEC--TAN"
    h["CRVAL1"] = ra0
    h["CRVAL2"] = dec0
    h["CRPIX1"] = 1500.0
    h["CRPIX2"] = 1000.0
    h["CD1_1"] = cd[0, 0]
    h["CD1_2"] = cd[0, 1]
    h["CD2_1"] = cd[1, 0]
    h["CD2_2"] = cd[1, 1]
    if sip_order is not None:
        h["A_ORDER"] = sip_order
        h["B_ORDER"] = sip_order
        h["A_0_0"] = 0.0
        h["B_0_0"] = 0.0
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


def test_run_one_bootstrap_draw_ok():
    xs = np.linspace(0, 3000, 50)
    ys = np.linspace(0, 2000, 50)
    rng = np.random.default_rng(1)

    def solve_fn(sub_x, sub_y):
        return PlatesolveResult(header=_fake_wcs_header(north_angle_deg=90.0))

    draw = run_one_bootstrap_draw(xs, ys, (2000, 3000), (3, 3), 0.9, rng, solve_fn)
    assert draw["status"] == "ok"
    assert draw["error"] is None
    assert abs(draw["north_angle_deg"] - 90.0) < 0.01
    assert draw["header"] is not None
    assert draw["n_sources"] > 0


def test_run_one_bootstrap_draw_no_solution():
    xs = np.linspace(0, 3000, 50)
    ys = np.linspace(0, 2000, 50)
    rng = np.random.default_rng(1)

    draw = run_one_bootstrap_draw(xs, ys, (2000, 3000), (3, 3), 0.9, rng, lambda sx, sy: None)
    assert draw["status"] == "no_solution"
    assert draw["north_angle_deg"] is None
    assert draw["header"] is None


def test_run_one_bootstrap_draw_exception():
    xs = np.linspace(0, 3000, 50)
    ys = np.linspace(0, 2000, 50)
    rng = np.random.default_rng(1)

    def bad_solve(sx, sy):
        raise ValueError("boom")

    draw = run_one_bootstrap_draw(xs, ys, (2000, 3000), (3, 3), 0.9, rng, bad_solve)
    assert draw["status"] == "exception"
    assert "boom" in draw["error"]
    assert draw["north_angle_deg"] is None


def test_robust_bootstrap_summary_recovers_tight_cluster_and_clips_outlier():
    """Same shape as the real overnight use-case: many draws tightly
    clustered around the anchor, one lone outlier that should be clipped
    out of sigma/mad but still counted in n_clipped."""
    rng = np.random.default_rng(7)
    tight = 90.0 + rng.normal(0.0, 0.01, size=99)
    angles = np.concatenate([tight, [95.0]])  # one 5 deg outlier among 100

    summary = robust_bootstrap_summary(angles, anchor_deg=90.0)
    assert abs(summary["center_deg"] - 90.0) < 0.01
    assert summary["sigma_deg"] < 0.1
    assert summary["n_clipped"] == 1
    assert summary["keep_mask"][-1] == False  # noqa: E712 -- the outlier is the last element
    assert summary["keep_mask"][:-1].all()


def test_robust_bootstrap_summary_handles_wraparound():
    """Angles near the +/-180 boundary must not be corrupted by naive
    subtraction -- angle_diff_deg handles this, robust_bootstrap_summary
    should inherit that correctness."""
    rng = np.random.default_rng(8)
    angles = 179.5 + rng.normal(0.0, 0.02, size=50)
    angles = (angles + 180.0) % 360.0 - 180.0  # wrap some into the -180 branch

    summary = robust_bootstrap_summary(angles, anchor_deg=179.5)
    # center should be close to 179.5 (mod 360, in (-180, 180] representation)
    from extractor.wcsangle import angle_diff_deg
    assert abs(angle_diff_deg(summary["center_deg"], 179.5)) < 0.05
    assert summary["sigma_deg"] < 0.1


def test_robust_bootstrap_summary_empty_and_single():
    empty = robust_bootstrap_summary(np.array([]), anchor_deg=90.0)
    assert empty["center_deg"] == 90.0
    assert np.isnan(empty["sigma_deg"])
    assert empty["n_clipped"] == 0

    single = robust_bootstrap_summary(np.array([90.2]), anchor_deg=90.0)
    assert abs(single["center_deg"] - 90.2) < 1e-9
    assert np.isnan(single["sigma_deg"])
    assert single["n_clipped"] == 0


# ---------------------------------------------------------------------------
# SIP-order-achievement filter (notebooks 18/19: real image's non-SIP-
# achieving majority was measurably biased relative to an independently
# trusted solve, by more than the existing outlier clip catches)
# ---------------------------------------------------------------------------

def test_sip_order_achieved():
    assert sip_order_achieved(_fake_wcs_header(sip_order=5), requested_order=5) is True
    assert sip_order_achieved(_fake_wcs_header(sip_order=3), requested_order=5) is False
    assert sip_order_achieved(_fake_wcs_header(sip_order=5), requested_order=3) is True  # >=, not ==
    assert sip_order_achieved(_fake_wcs_header(sip_order=None), requested_order=5) is False
    # plain dict (as loaded from JSONL) works the same as a fits.Header
    assert sip_order_achieved({"A_ORDER": 5, "B_ORDER": 5}, requested_order=5) is True
    assert sip_order_achieved({}, requested_order=5) is False


def test_sip_filter_indices_engages_with_enough_sip_draws():
    headers = [_fake_wcs_header(sip_order=5)] * 25 + [_fake_wcs_header(sip_order=None)] * 75
    with pytest.warns(UserWarning, match="restricting sigma_WCS/center"):
        idx, applied, n_full_sip = sip_filter_indices(headers, requested_tweak_order=5, min_full_sip_draws=20)
    assert applied is True
    assert n_full_sip == 25
    assert len(idx) == 25
    assert set(idx) == set(range(25))  # the SIP-achieving headers happen to be first 25


def test_sip_filter_indices_falls_back_below_min_full_sip_draws():
    headers = [_fake_wcs_header(sip_order=5)] * 5 + [_fake_wcs_header(sip_order=None)] * 95
    with pytest.warns(UserWarning, match="not enough to trust a SIP-only subset"):
        idx, applied, n_full_sip = sip_filter_indices(headers, requested_tweak_order=5, min_full_sip_draws=20)
    assert applied is False
    assert n_full_sip == 5
    assert len(idx) == 100  # falls back to every draw


def test_sip_filter_indices_disabled_emits_no_warning():
    headers = [_fake_wcs_header(sip_order=None)] * 10
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        idx, applied, n_full_sip = sip_filter_indices(headers, requested_tweak_order=5, prefer_full_sip=False)
    assert applied is False
    assert n_full_sip == 0
    assert len(idx) == 10


def test_bootstrap_wcs_orientation_prefers_sip_achieving_draws_when_biased():
    """Mirrors the real-image finding directly: a majority of draws that
    never achieve the requested SIP order are systematically biased away
    from a minority that does. With the filter engaged (default), the
    reported center should track the SIP-achieving (correct) subset, not
    get pulled toward the biased majority."""
    rng = np.random.default_rng(9)
    xs = rng.uniform(0, 3000, size=200)
    ys = rng.uniform(0, 2000, size=200)
    image_shape = (2000, 3000)

    TRUE_ANGLE = 90.0
    BIAS_DEG = 0.3  # small enough that the existing outlier clip would NOT catch it
    tight_jitter = rng.normal(0.0, 0.01, size=200)

    call_count = {"n": 0}

    def fake_solve(sub_x, sub_y):
        i = call_count["n"]
        call_count["n"] += 1
        if i == 0:
            # fiducial: full list, achieves SIP, correct
            return PlatesolveResult(header=_fake_wcs_header(north_angle_deg=TRUE_ANGLE, sip_order=5))
        j = (i - 1) % len(tight_jitter)
        if j < 30:  # minority: achieves SIP, clusters on the true answer
            header = _fake_wcs_header(north_angle_deg=TRUE_ANGLE + tight_jitter[j], sip_order=5)
        else:  # majority: no SIP, systematically biased but not outlier-large
            header = _fake_wcs_header(north_angle_deg=TRUE_ANGLE + BIAS_DEG + tight_jitter[j], sip_order=None)
        return PlatesolveResult(header=header)

    kwargs = dict(n_boot=100, boot_frac=0.9, n_grid=(3, 3), seed=10, solve_fn=fake_solve, tweak_order=5)

    with pytest.warns(UserWarning, match="restricting sigma_WCS/center"):
        filtered = bootstrap_wcs_orientation(xs, ys, image_shape, min_full_sip_draws=20, **kwargs)

    call_count["n"] = 0  # reset for the second, unfiltered run
    unfiltered = bootstrap_wcs_orientation(xs, ys, image_shape, prefer_full_sip=False, **kwargs)

    assert filtered.sip_filter_applied is True
    assert filtered.n_boot_full_sip == 30
    # filtered result should sit close to the true/SIP-cluster angle...
    assert abs(filtered.north_angle_deg - TRUE_ANGLE) < 0.05
    # ...and clearly closer to truth than the unfiltered (majority-biased) result
    assert abs(filtered.north_angle_deg - TRUE_ANGLE) < abs(unfiltered.north_angle_deg - TRUE_ANGLE)
    # diagnostics still cover every draw, not just the filtered subset
    assert len(filtered.north_angles_deg) == 100
    assert filtered.bootstrap_kept_mask.sum() <= 30
