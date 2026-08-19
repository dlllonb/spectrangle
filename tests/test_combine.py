import numpy as np

from extractor.fitting import TraceMeasurement
from extractor.combine import combine_traces


def _measurement(angle, weight=1.0, sigma=0.1):
    return TraceMeasurement(
        x_center=0.0, y_center=0.0, theta_pix_deg=angle,
        theta_pix_uncertainty_deg=sigma, length_px=100.0, perp_std_px=1.0,
        n_eff=50.0, eccentricity=0.99, total_flux=1000.0, weight=weight,
    )


def test_combine_traces_recovers_exact_agreement():
    ms = [_measurement(23.0) for _ in range(10)]
    result = combine_traces(ms, n_boot=200)
    assert abs(result.theta_pix_deg - 23.0) < 1e-6
    assert result.quality > 0.999
    assert result.population_scatter_deg < 1e-6
    assert result.n_used == 10


def test_combine_traces_high_weight_measurement_dominates():
    ms = [_measurement(0.0, weight=1.0) for _ in range(20)] + [_measurement(10.0, weight=1000.0)]
    result = combine_traces(ms, n_boot=200)
    assert abs(result.theta_pix_deg - 10.0) < 1.0


def test_combine_traces_axial_wrap():
    # angles straddling the +/-90 wrap should combine near the wrap point,
    # not get washed out by naive (non-axial) averaging
    ms = [_measurement(89.0), _measurement(-89.0)]
    result = combine_traces(ms, n_boot=200)
    assert abs(result.theta_pix_deg) > 85.0  # near +/-90, not near 0


def test_combine_traces_empty():
    result = combine_traces([])
    assert np.isnan(result.theta_pix_deg)
    assert result.n_used == 0


def test_combine_traces_bootstrap_uncertainty_is_finite_and_nonnegative():
    rng = np.random.default_rng(0)
    ms = [_measurement(23.0 + d) for d in rng.normal(0, 0.5, size=15)]
    result = combine_traces(ms, n_boot=500, seed=1)
    assert np.isfinite(result.theta_pix_uncertainty_deg)
    assert result.theta_pix_uncertainty_deg >= 0.0
    assert np.isfinite(result.analytical_uncertainty_deg)


def test_combine_traces_sigma_clip_off_by_default():
    # a clear outlier stays in unless sigma_clip is explicitly requested
    ms = [_measurement(23.0) for _ in range(10)] + [_measurement(60.0, weight=1.0)]
    result = combine_traces(ms, n_boot=200)
    assert result.n_used == 11
    assert result.n_before_clip == 11


def test_combine_traces_sigma_clip_drops_a_clear_outlier():
    ms = [_measurement(23.0 + d) for d in [-0.1, 0.05, -0.05, 0.1, 0.0, 0.02, -0.02, 0.08, -0.08, 0.03]]
    ms.append(_measurement(60.0, weight=1.0))  # one obvious outlier
    result = combine_traces(ms, n_boot=200, sigma_clip=True, sigma_clip_k=2.0)
    assert result.n_before_clip == 11
    assert result.n_used == 10  # the outlier should be dropped
    assert abs(result.theta_pix_deg - 23.0) < 0.5


def test_combine_traces_sigma_clip_never_drops_below_three():
    # only 3 measurements, one far off -- clipping must fall back rather
    # than leave < 3 traces
    ms = [_measurement(23.0), _measurement(23.1), _measurement(80.0, weight=1.0)]
    result = combine_traces(ms, n_boot=200, sigma_clip=True, sigma_clip_k=2.0)
    assert result.n_used >= 3
    assert result.n_before_clip == 3


def test_combine_traces_sigma_clip_pure_noise_stays_close_to_truth():
    # no genuine outlier here -- iterative clipping on pure Gaussian noise
    # legitimately trims some tail draws each round (that's the intended
    # behavior, not a bug), so this only checks it doesn't collapse the
    # population or bias the result, not that everyone survives
    rng = np.random.default_rng(3)
    ms = [_measurement(23.0 + d) for d in rng.normal(0, 0.1, size=12)]
    result = combine_traces(ms, n_boot=200, sigma_clip=True, sigma_clip_k=2.0)
    assert result.n_before_clip == 12
    assert result.n_used >= 8
    assert abs(result.theta_pix_deg - 23.0) < 0.1
