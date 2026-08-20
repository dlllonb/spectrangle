from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import extractor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SIM_PATH = DATA_DIR / "110lmm50mm.fits"


def _simple_tan_wcs(crval=(30.0, 60.0), crpix=(150.5, 100.5), scale_deg_per_px=2.0 / 3600.0):
    w = WCS(naxis=2)
    w.wcs.crval = crval
    w.wcs.crpix = crpix
    w.wcs.cdelt = [-scale_deg_per_px, scale_deg_per_px]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


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


@pytest.mark.skipif(not SIM_PATH.exists(), reason=f"{SIM_PATH} not present")
def test_config_reflects_actually_applied_masking_gated_options():
    """No star catalog/WCS supplied here, so masking is inactive and
    sigma_clip/merge_fragments/extend_traces are all silently gated off
    regardless of their (all-True) defaults being requested -- config
    must report what actually ran, not the raw request (previously it
    echoed the raw sigma_clip argument unconditionally, misrepresenting
    its own docstring's "parameters actually used" promise)."""
    with fits.open(SIM_PATH) as h:
        image = h[0].data.astype(np.float64)

    result = extractor.measure_grating_angle(
        image, min_length_px=80.0, min_eccentricity=0.85,
        sigma_clip=True, merge_fragments=True, extend_traces=True,
    )

    assert result.config["sigma_clip"] is False
    assert result.config["merge_fragments"] is False
    assert result.config["extend_traces"] is False


def test_falls_back_to_unmasked_pipeline_when_psf_recovery_fails():
    """Catalog masking requested (WCS + catalog all supplied, so masking
    WOULD normally activate) on a pure-noise image with no real point
    sources -- recover_empirical_psf_sigma can't converge a PSF fit to
    anything, and previously this raised an unhandled ValueError,
    crashing the whole measurement even though the pre-masking pipeline
    would have run fine (Entry 122 finding 3). Must now warn and fall
    back instead of raising."""
    # perfectly flat (zero-variance) image -- DAOStarFinder can't find any
    # local peak at all in constant data, so extract_stars returns nothing
    # and recover_empirical_psf_sigma raises immediately, before even
    # attempting a Gaussian fit (more reliable than ordinary noise, which
    # can still produce a spurious peak that technically "converges")
    image = np.full((200, 300), 100.0)
    wcs = _simple_tan_wcs()
    ra, dec, mag = np.array([30.0]), np.array([60.0]), np.array([10.0])

    with pytest.warns(UserWarning, match="PSF sigma recovery failed"):
        result = extractor.measure_grating_angle(
            image, wcs=wcs, min_length_px=20.0, min_eccentricity=0.8,
            star_catalog_ra_deg=ra, star_catalog_dec_deg=dec, star_catalog_mag=mag,
        )

    # masking never activated -- config never gets the mask_radius_px/
    # mask_mag_cut keys at all (only set inside the masking-active branch)
    assert "mask_radius_px" not in result.config
    assert result.config["merge_fragments"] is False
    assert result.config["extend_traces"] is False
    assert result.config["sigma_clip"] is False
