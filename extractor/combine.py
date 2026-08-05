"""
combine.py — Combine many per-trace angle measurements into one robust
pixel-space grating angle, with a defensible uncertainty.

Because traces are axial (180-degree symmetric), combination uses the
doubled-angle trick: average sin(2*theta)/cos(2*theta), halve the result
back. Never average raw angle values (project CLAUDE.md, and see
`fitting.norm_axial`/`axial_diff`).

**What "sigma" means here matters** (project notebook 13, and paper
sections on WCS-orientation / diffraction-angle precision): population
scatter (how much traces disagree with each other) is NOT the same
quantity as the precision of the combined mean, and only the latter
should be combined in quadrature with sigma_WCS for a final sky-angle
error budget. Three numbers are computed, deliberately kept distinct
rather than collapsed into one:

1. `population_scatter_deg` -- weighted circular std of the individual
   trace angles around the mean. A diagnostic (how much do traces
   disagree), not a precision-of-the-mean statistic.
2. `analytical_uncertainty_deg` -- inverse-variance combination of each
   trace's own analytical sigma_theta (paper Eq. ensemble_spectra_angle):
   `sigma_diff = (sum 1/sigma_i^2)^-1/2`. Assumes independent per-trace
   noise only -- the theoretical best case.
3. `bootstrap_uncertainty_deg` -- empirical, assumption-light: resample
   the trace list with replacement, recombine each time, take the spread.
   If it's notably larger than (2), that's evidence of correlated or
   systematic structure (e.g. residual lens distortion) that (2) would
   hide. **This is the number validated as the one to actually report**
   (project notebook 13) -- `theta_pix_uncertainty_deg` on the returned
   result is set to this, not the population scatter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .fitting import TraceMeasurement, norm_axial, axial_diff


@dataclass
class CombinedAngleResult:
    """Result of combining many TraceMeasurements into one pixel-space
    grating angle.

    Attributes
    ----------
    theta_pix_deg : float
        Weighted combined pixel-space angle, axial (-90, 90].
    theta_pix_uncertainty_deg : float
        Bootstrap uncertainty on theta_pix_deg -- see module docstring
        for why this, not population_scatter_deg, is the number to
        propagate into a total error budget.
    population_scatter_deg : float
        Weighted circular std of individual trace angles (diagnostic).
    analytical_uncertainty_deg : float
        Independent-noise-assumption inverse-variance combination
        (diagnostic / theoretical-best-case comparison point).
    quality : float
        Resultant length R in [0, 1] -- 1 means every trace agrees
        exactly; low R means the traces are scattered, not clustered.
    n_used : int
    trace_measurements : list[TraceMeasurement]
        The measurements actually combined (post contamination-removal,
        if that was run upstream).
    bootstrap_means_deg : ndarray
        Every resampled combined angle from the bootstrap used to compute
        `theta_pix_uncertainty_deg` (for histogramming/diagnostics only --
        empty if `combine_traces` was never called, e.g. the empty-input
        early return).
    """
    theta_pix_deg: float
    theta_pix_uncertainty_deg: float
    population_scatter_deg: float
    analytical_uncertainty_deg: float
    quality: float
    n_used: int
    trace_measurements: list[TraceMeasurement] = field(default_factory=list)
    bootstrap_means_deg: np.ndarray = field(default_factory=lambda: np.array([]))


def _weighted_circular_combine(angles_deg: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean_angle_deg, population_scatter_deg, R)."""
    s = np.dot(weights, np.sin(2 * np.radians(angles_deg)))
    c = np.dot(weights, np.cos(2 * np.radians(angles_deg)))
    mean_ang = norm_axial(np.degrees(np.arctan2(s, c)) / 2.0)
    R = float(np.hypot(s, c) / weights.sum())
    diffs = np.array([axial_diff(a, mean_ang) for a in angles_deg])
    sigma = float(np.sqrt(np.dot(weights / weights.sum(), diffs ** 2)))
    return mean_ang, sigma, R


def combine_traces(
    measurements: Sequence[TraceMeasurement],
    n_boot: int = 3000,
    seed: int = 42,
) -> CombinedAngleResult:
    """Combine trace measurements into one pixel-space angle + the three
    uncertainty statistics described in the module docstring.

    Weighting: each trace's own `weight` attribute (see
    `fitting.TraceMeasurement.weight` -- `(length/perp_std)^2`, validated
    in project notebook 10 as tracking a trace's actual precision far
    better than raw flux).
    """
    if not measurements:
        return CombinedAngleResult(
            theta_pix_deg=np.nan, theta_pix_uncertainty_deg=np.nan,
            population_scatter_deg=np.nan, analytical_uncertainty_deg=np.nan,
            quality=np.nan, n_used=0, trace_measurements=[],
        )

    angles = np.array([m.theta_pix_deg for m in measurements])
    weights = np.array([m.weight for m in measurements])
    mean_ang, pop_sigma, R = _weighted_circular_combine(angles, weights)

    analytical_sigma = _combine_analytical(measurements)
    boot_sigma, boot_means = _bootstrap_uncertainty(angles, weights, n_boot=n_boot, seed=seed)

    return CombinedAngleResult(
        theta_pix_deg=mean_ang, theta_pix_uncertainty_deg=boot_sigma,
        population_scatter_deg=pop_sigma, analytical_uncertainty_deg=analytical_sigma,
        quality=R, n_used=len(measurements), trace_measurements=list(measurements),
        bootstrap_means_deg=boot_means,
    )


def _combine_analytical(measurements: Sequence[TraceMeasurement]) -> float:
    """Inverse-variance combination of independent per-trace analytical
    sigmas: sigma_diff = (sum 1/sigma_i^2)^-1/2. Non-finite/non-positive
    per-trace sigmas are skipped."""
    sig = np.array([m.theta_pix_uncertainty_deg for m in measurements], dtype=float)
    valid = np.isfinite(sig) & (sig > 0)
    if not valid.any():
        return float("nan")
    inv_var_sum = np.sum(1.0 / sig[valid] ** 2)
    return float(np.sqrt(1.0 / inv_var_sum))


def _bootstrap_uncertainty(
    angles: np.ndarray, weights: np.ndarray, n_boot: int, seed: int
) -> tuple[float, np.ndarray]:
    """Empirical SE of the weighted combined angle: resample the trace
    list with replacement (uniform per-trace probability -- weighting is
    applied when recombining each resample, not when choosing which
    traces appear), recompute the weighted circular mean each time, and
    return the (axial-aware) std of the resulting distribution, plus the
    raw per-draw combined angles themselves (for diagnostics/plots)."""
    n = len(angles)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, w = angles[idx], weights[idx]
        s = np.dot(w, np.sin(2 * np.radians(a)))
        c = np.dot(w, np.cos(2 * np.radians(a)))
        boot_means[i] = norm_axial(np.degrees(np.arctan2(s, c)) / 2.0)
    center = float(np.median(boot_means))
    resid = np.array([axial_diff(m, center) for m in boot_means])
    return float(np.std(resid)), boot_means
