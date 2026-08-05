from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import extractor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_PATH = DATA_DIR / "fuji6_asi178_100_15s.fit"


@pytest.mark.skipif(not REAL_PATH.exists(), reason=f"{REAL_PATH} not present")
def test_pipeline_runs_on_real_image():
    """Smoke test, not a precision assertion -- there is no ground truth
    for this image. Catches silent breakage (crashes, NaNs, detection
    collapsing to ~0 traces), not correctness. See project notebook 13
    for the actual precision characterization of this image (theta_pix
    approx -22.6 deg, bootstrap sigma approx 0.4 deg as of that
    notebook) -- deliberately not asserted exactly here since minor,
    legitimate code changes could shift it slightly without being wrong.
    """
    with fits.open(REAL_PATH) as h:
        image = h[0].data.astype(np.float64)

    result = extractor.measure_grating_angle(image, min_length_px=25.0, min_eccentricity=0.85)

    assert result.n_traces_detected > 20
    assert result.n_traces_used > 20
    assert np.isfinite(result.theta_pix_deg)
    assert np.isfinite(result.theta_pix_uncertainty_deg)
    assert 0.0 < result.theta_pix_uncertainty_deg < 2.0  # generous sanity bound
    assert 0.0 < result.quality <= 1.0
