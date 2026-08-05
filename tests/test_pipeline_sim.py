from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import extractor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIM_PATH = DATA_DIR / "110lmm50mm.fits"


@pytest.mark.skipif(not SIM_PATH.exists(), reason=f"{SIM_PATH} not present")
def test_pipeline_recovers_known_sim_angle():
    """Regression test: locks in current working behavior against the one
    dataset with known ground truth (MASKANG=23 deg), validated across
    project notebooks 06-13. If this starts failing, something in
    detection/fitting/combine regressed -- not a flaky external
    dependency (no solve-field/network calls here)."""
    with fits.open(SIM_PATH) as h:
        image = h[0].data.astype(np.float64)
        truth = float(h[0].header["MASKANG"])

    result = extractor.measure_grating_angle(image, min_length_px=80.0, min_eccentricity=0.85)

    err = abs(((result.theta_pix_deg - truth + 90) % 180) - 90)

    assert result.n_traces_detected >= 20
    assert result.n_traces_used >= 20
    assert result.quality > 0.99
    # both a statistical check (within a handful of our own reported sigma)
    # and an absolute sanity bound matching the project's target precision
    assert err < 5 * result.theta_pix_uncertainty_deg
    assert err < 0.1
